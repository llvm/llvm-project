//===- SyntheticSections.cpp ----------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains linker-synthesized sections. Currently,
// synthetic sections are created either output sections or input sections,
// but we are rewriting code so that all synthetic sections are created as
// input sections.
//
//===----------------------------------------------------------------------===//

#include "SyntheticSections.h"
#include "Config.h"
#include "Error.h"
#include "InputFiles.h"
#include "Memory.h"
#include "OutputSections.h"
#include "Strings.h"
#include "SymbolTable.h"
#include "Target.h"
#include "Writer.h"

#include "lld/Config/Version.h"
#include "lld/Core/Parallel.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/MD5.h"
#include "llvm/Support/RandomNumberGenerator.h"
#include "llvm/Support/SHA1.h"
#include "llvm/Support/xxhash.h"
#include <cstdlib>

using namespace llvm;
using namespace llvm::ELF;
using namespace llvm::object;
using namespace llvm::support;
using namespace llvm::support::endian;

using namespace lld;
using namespace lld::elf;

template <class ELFT> static std::vector<DefinedCommon *> getCommonSymbols() {
  std::vector<DefinedCommon *> V;
  for (Symbol *S : Symtab<ELFT>::X->getSymbols())
    if (auto *B = dyn_cast<DefinedCommon>(S->body()))
      V.push_back(B);
  return V;
}

// Find all common symbols and allocate space for them.
template <class ELFT> InputSection<ELFT> *elf::createCommonSection() {
  auto *Ret = make<InputSection<ELFT>>(SHF_ALLOC | SHF_WRITE, SHT_NOBITS, 1,
                                       ArrayRef<uint8_t>(), "COMMON");
  Ret->Live = true;

  // Sort the common symbols by alignment as an heuristic to pack them better.
  std::vector<DefinedCommon *> Syms = getCommonSymbols<ELFT>();
  std::stable_sort(Syms.begin(), Syms.end(),
                   [](const DefinedCommon *A, const DefinedCommon *B) {
                     return A->Alignment > B->Alignment;
                   });

  // Assign offsets to symbols.
  size_t Size = 0;
  size_t Alignment = 1;
  for (DefinedCommon *Sym : Syms) {
    Alignment = std::max<size_t>(Alignment, Sym->Alignment);
    Size = alignTo(Size, Sym->Alignment);

    // Compute symbol offset relative to beginning of input section.
    Sym->Offset = Size;
    Size += Sym->Size;
  }
  Ret->Alignment = Alignment;
  Ret->Data = makeArrayRef<uint8_t>(nullptr, Size);
  return Ret;
}

// Returns an LLD version string.
static ArrayRef<uint8_t> getVersion() {
  // Check LLD_VERSION first for ease of testing.
  // You can get consitent output by using the environment variable.
  // This is only for testing.
  StringRef S = getenv("LLD_VERSION");
  if (S.empty())
    S = Saver.save(Twine("Linker: ") + getLLDVersion());

  // +1 to include the terminating '\0'.
  return {(const uint8_t *)S.data(), S.size() + 1};
}

// Creates a .comment section containing LLD version info.
// With this feature, you can identify LLD-generated binaries easily
// by "objdump -s -j .comment <file>".
// The returned object is a mergeable string section.
template <class ELFT> MergeInputSection<ELFT> *elf::createCommentSection() {
  typename ELFT::Shdr Hdr = {};
  Hdr.sh_flags = SHF_MERGE | SHF_STRINGS;
  Hdr.sh_type = SHT_PROGBITS;
  Hdr.sh_entsize = 1;
  Hdr.sh_addralign = 1;

  auto *Ret = make<MergeInputSection<ELFT>>(/*file=*/nullptr, &Hdr, ".comment");
  Ret->Data = getVersion();
  Ret->splitIntoPieces();
  return Ret;
}

// Iterate over sections of the specified type. For each section call
// provided function. After that "kill" the section by turning off
// "Live" flag, so that they won't be included in the final output.
template <class ELFT>
static void iterateSectionContents(
    uint32_t Type,
    std::function<void(elf::ObjectFile<ELFT> *, ArrayRef<uint8_t>)> F) {
  for (InputSectionBase<ELFT> *Sec : Symtab<ELFT>::X->Sections) {
    if (Sec && Sec->Live && Sec->Type == Type) {
      Sec->Live = false;
      F(Sec->getFile(), Sec->Data);
    }
  }
}

// .MIPS.abiflags section.
template <class ELFT>
MipsAbiFlagsSection<ELFT>::MipsAbiFlagsSection()
    : InputSection<ELFT>(SHF_ALLOC, SHT_MIPS_ABIFLAGS, 8, ArrayRef<uint8_t>(),
                         ".MIPS.abiflags") {
  auto Func = [this](ObjectFile<ELFT> *F, ArrayRef<uint8_t> D) {
    if (D.size() != sizeof(Elf_Mips_ABIFlags)) {
      error(getFilename(F) + ": invalid size of .MIPS.abiflags section");
      return;
    }
    auto *S = reinterpret_cast<const Elf_Mips_ABIFlags *>(D.data());
    if (S->version != 0) {
      error(getFilename(F) + ": unexpected .MIPS.abiflags version " +
            Twine(S->version));
      return;
    }
    // LLD checks ISA compatibility in getMipsEFlags(). Here we just
    // select the highest number of ISA/Rev/Ext.
    Flags.isa_level = std::max(Flags.isa_level, S->isa_level);
    Flags.isa_rev = std::max(Flags.isa_rev, S->isa_rev);
    Flags.isa_ext = std::max(Flags.isa_ext, S->isa_ext);
    Flags.gpr_size = std::max(Flags.gpr_size, S->gpr_size);
    Flags.cpr1_size = std::max(Flags.cpr1_size, S->cpr1_size);
    Flags.cpr2_size = std::max(Flags.cpr2_size, S->cpr2_size);
    Flags.ases |= S->ases;
    Flags.flags1 |= S->flags1;
    Flags.flags2 |= S->flags2;
    Flags.fp_abi =
        elf::getMipsFpAbiFlag(Flags.fp_abi, S->fp_abi, getFilename(F));
  };
  iterateSectionContents<ELFT>(SHT_MIPS_ABIFLAGS, Func);

  this->Data = ArrayRef<uint8_t>((const uint8_t *)&Flags, sizeof(Flags));
  this->Live = true;
}

// .MIPS.options section.
template <class ELFT>
MipsOptionsSection<ELFT>::MipsOptionsSection()
    : InputSection<ELFT>(SHF_ALLOC, SHT_MIPS_OPTIONS, 8, ArrayRef<uint8_t>(),
                         ".MIPS.options") {
  Buf.resize(sizeof(Elf_Mips_Options) + sizeof(Elf_Mips_RegInfo));
  getOptions()->kind = ODK_REGINFO;
  getOptions()->size = Buf.size();
  auto Func = [this](ObjectFile<ELFT> *F, ArrayRef<uint8_t> D) {
    while (!D.empty()) {
      if (D.size() < sizeof(Elf_Mips_Options)) {
        error(getFilename(F) + ": invalid size of .MIPS.options section");
        break;
      }
      auto *O = reinterpret_cast<const Elf_Mips_Options *>(D.data());
      if (O->kind == ODK_REGINFO) {
        if (Config->Relocatable && O->getRegInfo().ri_gp_value)
          error(getFilename(F) + ": unsupported non-zero ri_gp_value");
        getOptions()->getRegInfo().ri_gprmask |= O->getRegInfo().ri_gprmask;
        F->MipsGp0 = O->getRegInfo().ri_gp_value;
        break;
      }
      if (!O->size)
        fatal(getFilename(F) + ": zero option descriptor size");
      D = D.slice(O->size);
    }
  };
  iterateSectionContents<ELFT>(SHT_MIPS_OPTIONS, Func);

  this->Data = ArrayRef<uint8_t>(Buf);
  // Section should be alive for N64 ABI only.
  this->Live = ELFT::Is64Bits;
}

template <class ELFT> void MipsOptionsSection<ELFT>::finalize() {
  if (!Config->Relocatable)
    getOptions()->getRegInfo().ri_gp_value =
        In<ELFT>::MipsGot->getVA() + MipsGPOffset;
}

// MIPS .reginfo section.
template <class ELFT>
MipsReginfoSection<ELFT>::MipsReginfoSection()
    : InputSection<ELFT>(SHF_ALLOC, SHT_MIPS_REGINFO, 4, ArrayRef<uint8_t>(),
                         ".reginfo") {
  auto Func = [this](ObjectFile<ELFT> *F, ArrayRef<uint8_t> D) {
    if (D.size() != sizeof(Elf_Mips_RegInfo)) {
      error(getFilename(F) + ": invalid size of .reginfo section");
      return;
    }
    auto *R = reinterpret_cast<const Elf_Mips_RegInfo *>(D.data());
    if (Config->Relocatable && R->ri_gp_value)
      error(getFilename(F) + ": unsupported non-zero ri_gp_value");
    Reginfo.ri_gprmask |= R->ri_gprmask;
    F->MipsGp0 = R->ri_gp_value;
  };
  iterateSectionContents<ELFT>(SHT_MIPS_REGINFO, Func);

  this->Data = ArrayRef<uint8_t>((const uint8_t *)&Reginfo, sizeof(Reginfo));
  // Section should be alive for O32 and N32 ABIs only.
  this->Live = !ELFT::Is64Bits;
}

template <class ELFT> void MipsReginfoSection<ELFT>::finalize() {
  if (!Config->Relocatable)
    Reginfo.ri_gp_value = In<ELFT>::MipsGot->getVA() + MipsGPOffset;
}

static ArrayRef<uint8_t> createInterp() {
  // StringSaver guarantees that the returned string ends with '\0'.
  StringRef S = Saver.save(Config->DynamicLinker);
  return {(const uint8_t *)S.data(), S.size() + 1};
}

template <class ELFT> InputSection<ELFT> *elf::createInterpSection() {
  auto *Ret = make<InputSection<ELFT>>(SHF_ALLOC, SHT_PROGBITS, 1,
                                       createInterp(), ".interp");
  Ret->Live = true;
  return Ret;
}

template <class ELFT>
BuildIdSection<ELFT>::BuildIdSection(size_t HashSize)
    : InputSection<ELFT>(SHF_ALLOC, SHT_NOTE, 1, ArrayRef<uint8_t>(),
                         ".note.gnu.build-id"),
      HashSize(HashSize) {
  this->Live = true;

  Buf.resize(HeaderSize + HashSize);
  const endianness E = ELFT::TargetEndianness;
  write32<E>(Buf.data(), 4);                   // Name size
  write32<E>(Buf.data() + 4, HashSize);        // Content size
  write32<E>(Buf.data() + 8, NT_GNU_BUILD_ID); // Type
  memcpy(Buf.data() + 12, "GNU", 4);           // Name string
  this->Data = ArrayRef<uint8_t>(Buf);
}

// Returns the location of the build-id hash value in the output.
template <class ELFT>
uint8_t *BuildIdSection<ELFT>::getOutputLoc(uint8_t *Start) const {
  return Start + this->OutSec->Offset + this->OutSecOff + HeaderSize;
}

// Split one uint8 array into small pieces of uint8 arrays.
static std::vector<ArrayRef<uint8_t>> split(ArrayRef<uint8_t> Arr,
                                            size_t ChunkSize) {
  std::vector<ArrayRef<uint8_t>> Ret;
  while (Arr.size() > ChunkSize) {
    Ret.push_back(Arr.take_front(ChunkSize));
    Arr = Arr.drop_front(ChunkSize);
  }
  if (!Arr.empty())
    Ret.push_back(Arr);
  return Ret;
}

// Computes a hash value of Data using a given hash function.
// In order to utilize multiple cores, we first split data into 1MB
// chunks, compute a hash for each chunk, and then compute a hash value
// of the hash values.
template <class ELFT>
void BuildIdSection<ELFT>::computeHash(
    llvm::MutableArrayRef<uint8_t> Data,
    std::function<void(ArrayRef<uint8_t> Arr, uint8_t *Dest)> HashFn) {
  std::vector<ArrayRef<uint8_t>> Chunks = split(Data, 1024 * 1024);
  std::vector<uint8_t> HashList(Chunks.size() * HashSize);

  auto Fn = [&](ArrayRef<uint8_t> &Chunk) {
    size_t Idx = &Chunk - Chunks.data();
    HashFn(Chunk, HashList.data() + Idx * HashSize);
  };

  if (Config->Threads)
    parallel_for_each(Chunks.begin(), Chunks.end(), Fn);
  else
    std::for_each(Chunks.begin(), Chunks.end(), Fn);

  HashFn(HashList, this->getOutputLoc(Data.begin()));
}

template <class ELFT>
void BuildIdFastHash<ELFT>::writeBuildId(MutableArrayRef<uint8_t> Buf) {
  this->computeHash(Buf, [](ArrayRef<uint8_t> Arr, uint8_t *Dest) {
    write64le(Dest, xxHash64(toStringRef(Arr)));
  });
}

template <class ELFT>
void BuildIdMd5<ELFT>::writeBuildId(MutableArrayRef<uint8_t> Buf) {
  this->computeHash(Buf, [](ArrayRef<uint8_t> Arr, uint8_t *Dest) {
    MD5 Hash;
    Hash.update(Arr);
    MD5::MD5Result Res;
    Hash.final(Res);
    memcpy(Dest, Res, 16);
  });
}

template <class ELFT>
void BuildIdSha1<ELFT>::writeBuildId(MutableArrayRef<uint8_t> Buf) {
  this->computeHash(Buf, [](ArrayRef<uint8_t> Arr, uint8_t *Dest) {
    SHA1 Hash;
    Hash.update(Arr);
    memcpy(Dest, Hash.final().data(), 20);
  });
}

template <class ELFT>
void BuildIdUuid<ELFT>::writeBuildId(MutableArrayRef<uint8_t> Buf) {
  if (getRandomBytes(this->getOutputLoc(Buf.data()), this->HashSize))
    error("entropy source failure");
}

template <class ELFT>
BuildIdHexstring<ELFT>::BuildIdHexstring()
    : BuildIdSection<ELFT>(Config->BuildIdVector.size()) {}

template <class ELFT>
void BuildIdHexstring<ELFT>::writeBuildId(MutableArrayRef<uint8_t> Buf) {
  memcpy(this->getOutputLoc(Buf.data()), Config->BuildIdVector.data(),
         Config->BuildIdVector.size());
}

template <class ELFT>
GotSection<ELFT>::GotSection()
    : SyntheticSection<ELFT>(SHF_ALLOC | SHF_WRITE, SHT_PROGBITS,
                             Target->GotEntrySize, ".got") {}

template <class ELFT> void GotSection<ELFT>::addEntry(SymbolBody &Sym) {
  Sym.GotIndex = Entries.size();
  Entries.push_back(&Sym);
}

template <class ELFT> bool GotSection<ELFT>::addDynTlsEntry(SymbolBody &Sym) {
  if (Sym.GlobalDynIndex != -1U)
    return false;
  Sym.GlobalDynIndex = Entries.size();
  // Global Dynamic TLS entries take two GOT slots.
  Entries.push_back(nullptr);
  Entries.push_back(&Sym);
  return true;
}

// Reserves TLS entries for a TLS module ID and a TLS block offset.
// In total it takes two GOT slots.
template <class ELFT> bool GotSection<ELFT>::addTlsIndex() {
  if (TlsIndexOff != uint32_t(-1))
    return false;
  TlsIndexOff = Entries.size() * sizeof(uintX_t);
  Entries.push_back(nullptr);
  Entries.push_back(nullptr);
  return true;
}

template <class ELFT>
typename GotSection<ELFT>::uintX_t
GotSection<ELFT>::getGlobalDynAddr(const SymbolBody &B) const {
  return this->getVA() + B.GlobalDynIndex * sizeof(uintX_t);
}

template <class ELFT>
typename GotSection<ELFT>::uintX_t
GotSection<ELFT>::getGlobalDynOffset(const SymbolBody &B) const {
  return B.GlobalDynIndex * sizeof(uintX_t);
}

template <class ELFT> void GotSection<ELFT>::finalize() {
  Size = Entries.size() * sizeof(uintX_t);
}

template <class ELFT> void GotSection<ELFT>::writeTo(uint8_t *Buf) {
  for (const SymbolBody *B : Entries) {
    uint8_t *Entry = Buf;
    Buf += sizeof(uintX_t);
    if (!B)
      continue;
    if (B->isPreemptible())
      continue; // The dynamic linker will take care of it.
    uintX_t VA = B->getVA<ELFT>();
    write<uintX_t, ELFT::TargetEndianness, sizeof(uintX_t)>(Entry, VA);
  }
}

template <class ELFT>
MipsGotSection<ELFT>::MipsGotSection()
    : SyntheticSection<ELFT>(SHF_ALLOC | SHF_WRITE | SHF_MIPS_GPREL,
                             SHT_PROGBITS, Target->GotEntrySize, ".got") {}

template <class ELFT>
void MipsGotSection<ELFT>::addEntry(SymbolBody &Sym, uintX_t Addend,
                                    RelExpr Expr) {
  // For "true" local symbols which can be referenced from the same module
  // only compiler creates two instructions for address loading:
  //
  // lw   $8, 0($gp) # R_MIPS_GOT16
  // addi $8, $8, 0  # R_MIPS_LO16
  //
  // The first instruction loads high 16 bits of the symbol address while
  // the second adds an offset. That allows to reduce number of required
  // GOT entries because only one global offset table entry is necessary
  // for every 64 KBytes of local data. So for local symbols we need to
  // allocate number of GOT entries to hold all required "page" addresses.
  //
  // All global symbols (hidden and regular) considered by compiler uniformly.
  // It always generates a single `lw` instruction and R_MIPS_GOT16 relocation
  // to load address of the symbol. So for each such symbol we need to
  // allocate dedicated GOT entry to store its address.
  //
  // If a symbol is preemptible we need help of dynamic linker to get its
  // final address. The corresponding GOT entries are allocated in the
  // "global" part of GOT. Entries for non preemptible global symbol allocated
  // in the "local" part of GOT.
  //
  // See "Global Offset Table" in Chapter 5:
  // ftp://www.linux-mips.org/pub/linux/mips/doc/ABI/mipsabi.pdf
  if (Expr == R_MIPS_GOT_LOCAL_PAGE) {
    // At this point we do not know final symbol value so to reduce number
    // of allocated GOT entries do the following trick. Save all output
    // sections referenced by GOT relocations. Then later in the `finalize`
    // method calculate number of "pages" required to cover all saved output
    // section and allocate appropriate number of GOT entries.
    auto *OutSec = cast<DefinedRegular<ELFT>>(&Sym)->Section->OutSec;
    OutSections.insert(OutSec);
    return;
  }
  if (Sym.isTls()) {
    // GOT entries created for MIPS TLS relocations behave like
    // almost GOT entries from other ABIs. They go to the end
    // of the global offset table.
    Sym.GotIndex = TlsEntries.size();
    TlsEntries.push_back(&Sym);
    return;
  }
  auto AddEntry = [&](SymbolBody &S, uintX_t A, GotEntries &Items) {
    if (S.isInGot() && !A)
      return;
    size_t NewIndex = Items.size();
    if (!EntryIndexMap.insert({{&S, A}, NewIndex}).second)
      return;
    Items.emplace_back(&S, A);
    if (!A)
      S.GotIndex = NewIndex;
  };
  if (Sym.isPreemptible()) {
    // Ignore addends for preemptible symbols. They got single GOT entry anyway.
    AddEntry(Sym, 0, GlobalEntries);
    Sym.IsInGlobalMipsGot = true;
  } else if (Expr == R_MIPS_GOT_OFF32) {
    AddEntry(Sym, Addend, LocalEntries32);
    Sym.Is32BitMipsGot = true;
  } else {
    // Hold local GOT entries accessed via a 16-bit index separately.
    // That allows to write them in the beginning of the GOT and keep
    // their indexes as less as possible to escape relocation's overflow.
    AddEntry(Sym, Addend, LocalEntries);
  }
}

template <class ELFT> bool MipsGotSection<ELFT>::addDynTlsEntry(SymbolBody &Sym) {
  if (Sym.GlobalDynIndex != -1U)
    return false;
  Sym.GlobalDynIndex = TlsEntries.size();
  // Global Dynamic TLS entries take two GOT slots.
  TlsEntries.push_back(nullptr);
  TlsEntries.push_back(&Sym);
  return true;
}

// Reserves TLS entries for a TLS module ID and a TLS block offset.
// In total it takes two GOT slots.
template <class ELFT> bool MipsGotSection<ELFT>::addTlsIndex() {
  if (TlsIndexOff != uint32_t(-1))
    return false;
  TlsIndexOff = TlsEntries.size() * sizeof(uintX_t);
  TlsEntries.push_back(nullptr);
  TlsEntries.push_back(nullptr);
  return true;
}

template <class ELFT>
typename MipsGotSection<ELFT>::uintX_t
MipsGotSection<ELFT>::getPageEntryOffset(uintX_t EntryValue) {
  // Initialize the entry by the %hi(EntryValue) expression
  // but without right-shifting.
  EntryValue = (EntryValue + 0x8000) & ~0xffff;
  // Take into account MIPS GOT header.
  // See comment in the MipsGotSection::writeTo.
  size_t NewIndex = PageIndexMap.size() + 2;
  auto P = PageIndexMap.insert(std::make_pair(EntryValue, NewIndex));
  assert(!P.second || PageIndexMap.size() <= PageEntriesNum);
  return (uintX_t)P.first->second * sizeof(uintX_t) - MipsGPOffset;
}

template <class ELFT>
typename MipsGotSection<ELFT>::uintX_t
MipsGotSection<ELFT>::getBodyEntryOffset(const SymbolBody &B,
                                         uintX_t Addend) const {
  // Calculate offset of the GOT entries block: TLS, global, local.
  uintX_t GotBlockOff;
  if (B.isTls())
    GotBlockOff = getTlsOffset();
  else if (B.IsInGlobalMipsGot)
    GotBlockOff = getLocalEntriesNum() * sizeof(uintX_t);
  else if (B.Is32BitMipsGot)
    GotBlockOff = (PageEntriesNum + LocalEntries.size()) * sizeof(uintX_t);
  else
    GotBlockOff = PageEntriesNum * sizeof(uintX_t);
  // Calculate index of the GOT entry in the block.
  uintX_t GotIndex;
  if (B.isInGot())
    GotIndex = B.GotIndex;
  else {
    auto It = EntryIndexMap.find({&B, Addend});
    assert(It != EntryIndexMap.end());
    GotIndex = It->second;
  }
  return GotBlockOff + GotIndex * sizeof(uintX_t) - MipsGPOffset;
}

template <class ELFT>
typename MipsGotSection<ELFT>::uintX_t
MipsGotSection<ELFT>::getTlsOffset() const {
  return (getLocalEntriesNum() + GlobalEntries.size()) * sizeof(uintX_t);
}

template <class ELFT>
typename MipsGotSection<ELFT>::uintX_t
MipsGotSection<ELFT>::getGlobalDynOffset(const SymbolBody &B) const {
  return B.GlobalDynIndex * sizeof(uintX_t);
}

template <class ELFT>
const SymbolBody *MipsGotSection<ELFT>::getFirstGlobalEntry() const {
  return GlobalEntries.empty() ? nullptr : GlobalEntries.front().first;
}

template <class ELFT>
unsigned MipsGotSection<ELFT>::getLocalEntriesNum() const {
  return PageEntriesNum + LocalEntries.size() + LocalEntries32.size();
}

template <class ELFT> void MipsGotSection<ELFT>::finalize() {
  size_t EntriesNum = TlsEntries.size();
  // Take into account MIPS GOT header.
  // See comment in the MipsGotSection::writeTo.
  PageEntriesNum += 2;
  for (const OutputSectionBase *OutSec : OutSections) {
    // Calculate an upper bound of MIPS GOT entries required to store page
    // addresses of local symbols. We assume the worst case - each 64kb
    // page of the output section has at least one GOT relocation against it.
    // Add 0x8000 to the section's size because the page address stored
    // in the GOT entry is calculated as (value + 0x8000) & ~0xffff.
    PageEntriesNum += (OutSec->Size + 0x8000 + 0xfffe) / 0xffff;
  }
  EntriesNum += getLocalEntriesNum() + GlobalEntries.size();
  Size = EntriesNum * sizeof(uintX_t);
}

template <class ELFT>
static void writeUint(uint8_t *Buf, typename ELFT::uint Val) {
  typedef typename ELFT::uint uintX_t;
  write<uintX_t, ELFT::TargetEndianness, sizeof(uintX_t)>(Buf, Val);
}

template <class ELFT> void MipsGotSection<ELFT>::writeTo(uint8_t *Buf) {
  // Set the MSB of the second GOT slot. This is not required by any
  // MIPS ABI documentation, though.
  //
  // There is a comment in glibc saying that "The MSB of got[1] of a
  // gnu object is set to identify gnu objects," and in GNU gold it
  // says "the second entry will be used by some runtime loaders".
  // But how this field is being used is unclear.
  //
  // We are not really willing to mimic other linkers behaviors
  // without understanding why they do that, but because all files
  // generated by GNU tools have this special GOT value, and because
  // we've been doing this for years, it is probably a safe bet to
  // keep doing this for now. We really need to revisit this to see
  // if we had to do this.
  auto *P = reinterpret_cast<typename ELFT::Off *>(Buf);
  P[1] = uintX_t(1) << (ELFT::Is64Bits ? 63 : 31);
  // Write 'page address' entries to the local part of the GOT.
  for (std::pair<uintX_t, size_t> &L : PageIndexMap) {
    uint8_t *Entry = Buf + L.second * sizeof(uintX_t);
    writeUint<ELFT>(Entry, L.first);
  }
  Buf += PageEntriesNum * sizeof(uintX_t);
  auto AddEntry = [&](const GotEntry &SA) {
    uint8_t *Entry = Buf;
    Buf += sizeof(uintX_t);
    const SymbolBody *Body = SA.first;
    uintX_t VA = Body->template getVA<ELFT>(SA.second);
    writeUint<ELFT>(Entry, VA);
  };
  std::for_each(std::begin(LocalEntries), std::end(LocalEntries), AddEntry);
  std::for_each(std::begin(LocalEntries32), std::end(LocalEntries32), AddEntry);
  std::for_each(std::begin(GlobalEntries), std::end(GlobalEntries), AddEntry);
  // Initialize TLS-related GOT entries. If the entry has a corresponding
  // dynamic relocations, leave it initialized by zero. Write down adjusted
  // TLS symbol's values otherwise. To calculate the adjustments use offsets
  // for thread-local storage.
  // https://www.linux-mips.org/wiki/NPTL
  if (TlsIndexOff != -1U && !Config->Pic)
    writeUint<ELFT>(Buf + TlsIndexOff, 1);
  for (const SymbolBody *B : TlsEntries) {
    if (!B || B->isPreemptible())
      continue;
    uintX_t VA = B->getVA<ELFT>();
    if (B->GotIndex != -1U) {
      uint8_t *Entry = Buf + B->GotIndex * sizeof(uintX_t);
      writeUint<ELFT>(Entry, VA - 0x7000);
    }
    if (B->GlobalDynIndex != -1U) {
      uint8_t *Entry = Buf + B->GlobalDynIndex * sizeof(uintX_t);
      writeUint<ELFT>(Entry, 1);
      Entry += sizeof(uintX_t);
      writeUint<ELFT>(Entry, VA - 0x8000);
    }
  }
}

template <class ELFT>
GotPltSection<ELFT>::GotPltSection()
    : SyntheticSection<ELFT>(SHF_ALLOC | SHF_WRITE, SHT_PROGBITS,
                             Target->GotPltEntrySize, ".got.plt") {}

template <class ELFT> void GotPltSection<ELFT>::addEntry(SymbolBody &Sym) {
  Sym.GotPltIndex = Target->GotPltHeaderEntriesNum + Entries.size();
  Entries.push_back(&Sym);
}

template <class ELFT> bool GotPltSection<ELFT>::empty() const {
  return Entries.empty();
}

template <class ELFT> size_t GotPltSection<ELFT>::getSize() const {
  return (Target->GotPltHeaderEntriesNum + Entries.size()) *
         Target->GotPltEntrySize;
}

template <class ELFT> void GotPltSection<ELFT>::writeTo(uint8_t *Buf) {
  Target->writeGotPltHeader(Buf);
  Buf += Target->GotPltHeaderEntriesNum * Target->GotPltEntrySize;
  for (const SymbolBody *B : Entries) {
    Target->writeGotPlt(Buf, *B);
    Buf += sizeof(uintX_t);
  }
}

template <class ELFT>
StringTableSection<ELFT>::StringTableSection(StringRef Name, bool Dynamic)
    : SyntheticSection<ELFT>(Dynamic ? (uintX_t)SHF_ALLOC : 0, SHT_STRTAB, 1,
                             Name),
      Dynamic(Dynamic) {}

// Adds a string to the string table. If HashIt is true we hash and check for
// duplicates. It is optional because the name of global symbols are already
// uniqued and hashing them again has a big cost for a small value: uniquing
// them with some other string that happens to be the same.
template <class ELFT>
unsigned StringTableSection<ELFT>::addString(StringRef S, bool HashIt) {
  if (HashIt) {
    auto R = StringMap.insert(std::make_pair(S, this->Size));
    if (!R.second)
      return R.first->second;
  }
  unsigned Ret = this->Size;
  this->Size = this->Size + S.size() + 1;
  Strings.push_back(S);
  return Ret;
}

template <class ELFT> void StringTableSection<ELFT>::writeTo(uint8_t *Buf) {
  // ELF string tables start with NUL byte, so advance the pointer by one.
  ++Buf;
  for (StringRef S : Strings) {
    memcpy(Buf, S.data(), S.size());
    Buf += S.size() + 1;
  }
}

static unsigned getVerDefNum() { return Config->VersionDefinitions.size() + 1; }

template <class ELFT>
DynamicSection<ELFT>::DynamicSection()
    : SyntheticSection<ELFT>(SHF_ALLOC | SHF_WRITE, SHT_DYNAMIC,
                             sizeof(uintX_t), ".dynamic") {
  this->Entsize = ELFT::Is64Bits ? 16 : 8;
  // .dynamic section is not writable on MIPS.
  // See "Special Section" in Chapter 4 in the following document:
  // ftp://www.linux-mips.org/pub/linux/mips/doc/ABI/mipsabi.pdf
  if (Config->EMachine == EM_MIPS)
    this->Flags = SHF_ALLOC;

  addEntries();
}

// There are some dynamic entries that don't depend on other sections.
// Such entries can be set early.
template <class ELFT> void DynamicSection<ELFT>::addEntries() {
  // Add strings to .dynstr early so that .dynstr's size will be
  // fixed early.
  for (StringRef S : Config->AuxiliaryList)
    add({DT_AUXILIARY, In<ELFT>::DynStrTab->addString(S)});
  if (!Config->RPath.empty())
    add({Config->EnableNewDtags ? DT_RUNPATH : DT_RPATH,
         In<ELFT>::DynStrTab->addString(Config->RPath)});
  for (SharedFile<ELFT> *F : Symtab<ELFT>::X->getSharedFiles())
    if (F->isNeeded())
      add({DT_NEEDED, In<ELFT>::DynStrTab->addString(F->getSoName())});
  if (!Config->SoName.empty())
    add({DT_SONAME, In<ELFT>::DynStrTab->addString(Config->SoName)});

  // Set DT_FLAGS and DT_FLAGS_1.
  uint32_t DtFlags = 0;
  uint32_t DtFlags1 = 0;
  if (Config->Bsymbolic)
    DtFlags |= DF_SYMBOLIC;
  if (Config->ZNodelete)
    DtFlags1 |= DF_1_NODELETE;
  if (Config->ZNow) {
    DtFlags |= DF_BIND_NOW;
    DtFlags1 |= DF_1_NOW;
  }
  if (Config->ZOrigin) {
    DtFlags |= DF_ORIGIN;
    DtFlags1 |= DF_1_ORIGIN;
  }

  if (DtFlags)
    add({DT_FLAGS, DtFlags});
  if (DtFlags1)
    add({DT_FLAGS_1, DtFlags1});

  if (!Config->Entry.empty())
    add({DT_DEBUG, (uint64_t)0});
}

// Add remaining entries to complete .dynamic contents.
template <class ELFT> void DynamicSection<ELFT>::finalize() {
  if (this->Size)
    return; // Already finalized.

  this->Link = In<ELFT>::DynStrTab->OutSec->SectionIndex;

  if (In<ELFT>::RelaDyn->hasRelocs()) {
    bool IsRela = Config->Rela;
    add({IsRela ? DT_RELA : DT_REL, In<ELFT>::RelaDyn});
    add({IsRela ? DT_RELASZ : DT_RELSZ, In<ELFT>::RelaDyn->getSize()});
    add({IsRela ? DT_RELAENT : DT_RELENT,
         uintX_t(IsRela ? sizeof(Elf_Rela) : sizeof(Elf_Rel))});

    // MIPS dynamic loader does not support RELCOUNT tag.
    // The problem is in the tight relation between dynamic
    // relocations and GOT. So do not emit this tag on MIPS.
    if (Config->EMachine != EM_MIPS) {
      size_t NumRelativeRels = In<ELFT>::RelaDyn->getRelativeRelocCount();
      if (Config->ZCombreloc && NumRelativeRels)
        add({IsRela ? DT_RELACOUNT : DT_RELCOUNT, NumRelativeRels});
    }
  }
  if (In<ELFT>::RelaPlt->hasRelocs()) {
    add({DT_JMPREL, In<ELFT>::RelaPlt});
    add({DT_PLTRELSZ, In<ELFT>::RelaPlt->getSize()});
    add({Config->EMachine == EM_MIPS ? DT_MIPS_PLTGOT : DT_PLTGOT,
         In<ELFT>::GotPlt});
    add({DT_PLTREL, uint64_t(Config->Rela ? DT_RELA : DT_REL)});
  }

  add({DT_SYMTAB, In<ELFT>::DynSymTab});
  add({DT_SYMENT, sizeof(Elf_Sym)});
  add({DT_STRTAB, In<ELFT>::DynStrTab});
  add({DT_STRSZ, In<ELFT>::DynStrTab->getSize()});
  if (In<ELFT>::GnuHashTab)
    add({DT_GNU_HASH, In<ELFT>::GnuHashTab});
  if (In<ELFT>::HashTab)
    add({DT_HASH, In<ELFT>::HashTab});

  if (Out<ELFT>::PreinitArray) {
    add({DT_PREINIT_ARRAY, Out<ELFT>::PreinitArray});
    add({DT_PREINIT_ARRAYSZ, Out<ELFT>::PreinitArray, Entry::SecSize});
  }
  if (Out<ELFT>::InitArray) {
    add({DT_INIT_ARRAY, Out<ELFT>::InitArray});
    add({DT_INIT_ARRAYSZ, Out<ELFT>::InitArray, Entry::SecSize});
  }
  if (Out<ELFT>::FiniArray) {
    add({DT_FINI_ARRAY, Out<ELFT>::FiniArray});
    add({DT_FINI_ARRAYSZ, Out<ELFT>::FiniArray, Entry::SecSize});
  }

  if (SymbolBody *B = Symtab<ELFT>::X->find(Config->Init))
    add({DT_INIT, B});
  if (SymbolBody *B = Symtab<ELFT>::X->find(Config->Fini))
    add({DT_FINI, B});

  bool HasVerNeed = Out<ELFT>::VerNeed->getNeedNum() != 0;
  if (HasVerNeed || Out<ELFT>::VerDef)
    add({DT_VERSYM, Out<ELFT>::VerSym});
  if (Out<ELFT>::VerDef) {
    add({DT_VERDEF, Out<ELFT>::VerDef});
    add({DT_VERDEFNUM, getVerDefNum()});
  }
  if (HasVerNeed) {
    add({DT_VERNEED, Out<ELFT>::VerNeed});
    add({DT_VERNEEDNUM, Out<ELFT>::VerNeed->getNeedNum()});
  }

  if (Config->EMachine == EM_MIPS) {
    add({DT_MIPS_RLD_VERSION, 1});
    add({DT_MIPS_FLAGS, RHF_NOTPOT});
    add({DT_MIPS_BASE_ADDRESS, Config->ImageBase});
    add({DT_MIPS_SYMTABNO, In<ELFT>::DynSymTab->getNumSymbols()});
    add({DT_MIPS_LOCAL_GOTNO, In<ELFT>::MipsGot->getLocalEntriesNum()});
    if (const SymbolBody *B = In<ELFT>::MipsGot->getFirstGlobalEntry())
      add({DT_MIPS_GOTSYM, B->DynsymIndex});
    else
      add({DT_MIPS_GOTSYM, In<ELFT>::DynSymTab->getNumSymbols()});
    add({DT_PLTGOT, In<ELFT>::MipsGot});
    if (Out<ELFT>::MipsRldMap)
      add({DT_MIPS_RLD_MAP, Out<ELFT>::MipsRldMap});
  }

  this->OutSec->Entsize = this->Entsize;
  this->OutSec->Link = this->Link;

  // +1 for DT_NULL
  this->Size = (Entries.size() + 1) * this->Entsize;
}

template <class ELFT> void DynamicSection<ELFT>::writeTo(uint8_t *Buf) {
  auto *P = reinterpret_cast<Elf_Dyn *>(Buf);

  for (const Entry &E : Entries) {
    P->d_tag = E.Tag;
    switch (E.Kind) {
    case Entry::SecAddr:
      P->d_un.d_ptr = E.OutSec->Addr;
      break;
    case Entry::InSecAddr:
      P->d_un.d_ptr = E.InSec->OutSec->Addr + E.InSec->OutSecOff;
      break;
    case Entry::SecSize:
      P->d_un.d_val = E.OutSec->Size;
      break;
    case Entry::SymAddr:
      P->d_un.d_ptr = E.Sym->template getVA<ELFT>();
      break;
    case Entry::PlainInt:
      P->d_un.d_val = E.Val;
      break;
    }
    ++P;
  }
}

template <class ELFT>
typename ELFT::uint DynamicReloc<ELFT>::getOffset() const {
  if (OutputSec)
    return OutputSec->Addr + OffsetInSec;
  return InputSec->OutSec->Addr + InputSec->getOffset(OffsetInSec);
}

template <class ELFT>
typename ELFT::uint DynamicReloc<ELFT>::getAddend() const {
  if (UseSymVA)
    return Sym->getVA<ELFT>(Addend);
  return Addend;
}

template <class ELFT> uint32_t DynamicReloc<ELFT>::getSymIndex() const {
  if (Sym && !UseSymVA)
    return Sym->DynsymIndex;
  return 0;
}

template <class ELFT>
RelocationSection<ELFT>::RelocationSection(StringRef Name, bool Sort)
    : SyntheticSection<ELFT>(SHF_ALLOC, Config->Rela ? SHT_RELA : SHT_REL,
                             sizeof(uintX_t), Name),
      Sort(Sort) {
  this->Entsize = Config->Rela ? sizeof(Elf_Rela) : sizeof(Elf_Rel);
}

template <class ELFT>
void RelocationSection<ELFT>::addReloc(const DynamicReloc<ELFT> &Reloc) {
  if (Reloc.Type == Target->RelativeRel)
    ++NumRelativeRelocs;
  Relocs.push_back(Reloc);
}

template <class ELFT, class RelTy>
static bool compRelocations(const RelTy &A, const RelTy &B) {
  bool AIsRel = A.getType(Config->Mips64EL) == Target->RelativeRel;
  bool BIsRel = B.getType(Config->Mips64EL) == Target->RelativeRel;
  if (AIsRel != BIsRel)
    return AIsRel;

  return A.getSymbol(Config->Mips64EL) < B.getSymbol(Config->Mips64EL);
}

template <class ELFT> void RelocationSection<ELFT>::writeTo(uint8_t *Buf) {
  uint8_t *BufBegin = Buf;
  for (const DynamicReloc<ELFT> &Rel : Relocs) {
    auto *P = reinterpret_cast<Elf_Rela *>(Buf);
    Buf += Config->Rela ? sizeof(Elf_Rela) : sizeof(Elf_Rel);

    if (Config->Rela)
      P->r_addend = Rel.getAddend();
    P->r_offset = Rel.getOffset();
    if (Config->EMachine == EM_MIPS && Rel.getInputSec() == In<ELFT>::MipsGot)
      // Dynamic relocation against MIPS GOT section make deal TLS entries
      // allocated in the end of the GOT. We need to adjust the offset to take
      // in account 'local' and 'global' GOT entries.
      P->r_offset += In<ELFT>::MipsGot->getTlsOffset();
    P->setSymbolAndType(Rel.getSymIndex(), Rel.Type, Config->Mips64EL);
  }

  if (Sort) {
    if (Config->Rela)
      std::stable_sort((Elf_Rela *)BufBegin,
                       (Elf_Rela *)BufBegin + Relocs.size(),
                       compRelocations<ELFT, Elf_Rela>);
    else
      std::stable_sort((Elf_Rel *)BufBegin, (Elf_Rel *)BufBegin + Relocs.size(),
                       compRelocations<ELFT, Elf_Rel>);
  }
}

template <class ELFT> unsigned RelocationSection<ELFT>::getRelocOffset() {
  return this->Entsize * Relocs.size();
}

template <class ELFT> void RelocationSection<ELFT>::finalize() {
  this->Link = In<ELFT>::DynSymTab ? In<ELFT>::DynSymTab->OutSec->SectionIndex
                                   : In<ELFT>::SymTab->OutSec->SectionIndex;

  // Set required output section properties.
  this->OutSec->Link = this->Link;
  this->OutSec->Entsize = this->Entsize;
}

template <class ELFT>
SymbolTableSection<ELFT>::SymbolTableSection(
    StringTableSection<ELFT> &StrTabSec)
    : SyntheticSection<ELFT>(StrTabSec.isDynamic() ? (uintX_t)SHF_ALLOC : 0,
                             StrTabSec.isDynamic() ? SHT_DYNSYM : SHT_SYMTAB,
                             sizeof(uintX_t),
                             StrTabSec.isDynamic() ? ".dynsym" : ".symtab"),
      StrTabSec(StrTabSec) {
  this->Entsize = sizeof(Elf_Sym);
}

// Orders symbols according to their positions in the GOT,
// in compliance with MIPS ABI rules.
// See "Global Offset Table" in Chapter 5 in the following document
// for detailed description:
// ftp://www.linux-mips.org/pub/linux/mips/doc/ABI/mipsabi.pdf
static bool sortMipsSymbols(const SymbolBody *L, const SymbolBody *R) {
  // Sort entries related to non-local preemptible symbols by GOT indexes.
  // All other entries go to the first part of GOT in arbitrary order.
  bool LIsInLocalGot = !L->IsInGlobalMipsGot;
  bool RIsInLocalGot = !R->IsInGlobalMipsGot;
  if (LIsInLocalGot || RIsInLocalGot)
    return !RIsInLocalGot;
  return L->GotIndex < R->GotIndex;
}

static uint8_t getSymbolBinding(SymbolBody *Body) {
  Symbol *S = Body->symbol();
  if (Config->Relocatable)
    return S->Binding;
  uint8_t Visibility = S->Visibility;
  if (Visibility != STV_DEFAULT && Visibility != STV_PROTECTED)
    return STB_LOCAL;
  if (Config->NoGnuUnique && S->Binding == STB_GNU_UNIQUE)
    return STB_GLOBAL;
  return S->Binding;
}

template <class ELFT> void SymbolTableSection<ELFT>::finalize() {
  this->OutSec->Link = this->Link = StrTabSec.OutSec->SectionIndex;
  this->OutSec->Info = this->Info = NumLocals + 1;
  this->OutSec->Entsize = this->Entsize;

  if (Config->Relocatable) {
    size_t I = NumLocals;
    for (const SymbolTableEntry &S : Symbols)
      S.Symbol->DynsymIndex = ++I;
    return;
  }

  if (!StrTabSec.isDynamic()) {
    std::stable_sort(Symbols.begin(), Symbols.end(),
                     [](const SymbolTableEntry &L, const SymbolTableEntry &R) {
                       return getSymbolBinding(L.Symbol) == STB_LOCAL &&
                              getSymbolBinding(R.Symbol) != STB_LOCAL;
                     });
    return;
  }
  if (In<ELFT>::GnuHashTab)
    // NB: It also sorts Symbols to meet the GNU hash table requirements.
    In<ELFT>::GnuHashTab->addSymbols(Symbols);
  else if (Config->EMachine == EM_MIPS)
    std::stable_sort(Symbols.begin(), Symbols.end(),
                     [](const SymbolTableEntry &L, const SymbolTableEntry &R) {
                       return sortMipsSymbols(L.Symbol, R.Symbol);
                     });
  size_t I = 0;
  for (const SymbolTableEntry &S : Symbols)
    S.Symbol->DynsymIndex = ++I;
}

template <class ELFT> void SymbolTableSection<ELFT>::addSymbol(SymbolBody *B) {
  Symbols.push_back({B, StrTabSec.addString(B->getName(), false)});
}

template <class ELFT> void SymbolTableSection<ELFT>::writeTo(uint8_t *Buf) {
  Buf += sizeof(Elf_Sym);

  // All symbols with STB_LOCAL binding precede the weak and global symbols.
  // .dynsym only contains global symbols.
  if (Config->Discard != DiscardPolicy::All && !StrTabSec.isDynamic())
    writeLocalSymbols(Buf);

  writeGlobalSymbols(Buf);
}

template <class ELFT>
void SymbolTableSection<ELFT>::writeLocalSymbols(uint8_t *&Buf) {
  // Iterate over all input object files to copy their local symbols
  // to the output symbol table pointed by Buf.
  for (ObjectFile<ELFT> *File : Symtab<ELFT>::X->getObjectFiles()) {
    for (const std::pair<const DefinedRegular<ELFT> *, size_t> &P :
         File->KeptLocalSyms) {
      const DefinedRegular<ELFT> &Body = *P.first;
      InputSectionBase<ELFT> *Section = Body.Section;
      auto *ESym = reinterpret_cast<Elf_Sym *>(Buf);

      if (!Section) {
        ESym->st_shndx = SHN_ABS;
        ESym->st_value = Body.Value;
      } else {
        const OutputSectionBase *OutSec = Section->OutSec;
        ESym->st_shndx = OutSec->SectionIndex;
        ESym->st_value = OutSec->Addr + Section->getOffset(Body);
      }
      ESym->st_name = P.second;
      ESym->st_size = Body.template getSize<ELFT>();
      ESym->setBindingAndType(STB_LOCAL, Body.Type);
      Buf += sizeof(*ESym);
    }
  }
}

template <class ELFT>
void SymbolTableSection<ELFT>::writeGlobalSymbols(uint8_t *Buf) {
  // Write the internal symbol table contents to the output symbol table
  // pointed by Buf.
  auto *ESym = reinterpret_cast<Elf_Sym *>(Buf);
  for (const SymbolTableEntry &S : Symbols) {
    SymbolBody *Body = S.Symbol;
    size_t StrOff = S.StrTabOffset;

    uint8_t Type = Body->Type;
    uintX_t Size = Body->getSize<ELFT>();

    ESym->setBindingAndType(getSymbolBinding(Body), Type);
    ESym->st_size = Size;
    ESym->st_name = StrOff;
    ESym->setVisibility(Body->symbol()->Visibility);
    ESym->st_value = Body->getVA<ELFT>();

    if (const OutputSectionBase *OutSec = getOutputSection(Body))
      ESym->st_shndx = OutSec->SectionIndex;
    else if (isa<DefinedRegular<ELFT>>(Body))
      ESym->st_shndx = SHN_ABS;

    if (Config->EMachine == EM_MIPS) {
      // On MIPS we need to mark symbol which has a PLT entry and requires
      // pointer equality by STO_MIPS_PLT flag. That is necessary to help
      // dynamic linker distinguish such symbols and MIPS lazy-binding stubs.
      // https://sourceware.org/ml/binutils/2008-07/txt00000.txt
      if (Body->isInPlt() && Body->NeedsCopyOrPltAddr)
        ESym->st_other |= STO_MIPS_PLT;
      if (Config->Relocatable) {
        auto *D = dyn_cast<DefinedRegular<ELFT>>(Body);
        if (D && D->isMipsPIC())
          ESym->st_other |= STO_MIPS_PIC;
      }
    }
    ++ESym;
  }
}

template <class ELFT>
const OutputSectionBase *
SymbolTableSection<ELFT>::getOutputSection(SymbolBody *Sym) {
  switch (Sym->kind()) {
  case SymbolBody::DefinedSyntheticKind:
    return cast<DefinedSynthetic<ELFT>>(Sym)->Section;
  case SymbolBody::DefinedRegularKind: {
    auto &D = cast<DefinedRegular<ELFT>>(*Sym);
    if (D.Section)
      return D.Section->OutSec;
    break;
  }
  case SymbolBody::DefinedCommonKind:
    return In<ELFT>::Common->OutSec;
  case SymbolBody::SharedKind:
    if (cast<SharedSymbol<ELFT>>(Sym)->needsCopy())
      return Out<ELFT>::Bss;
    break;
  case SymbolBody::UndefinedKind:
  case SymbolBody::LazyArchiveKind:
  case SymbolBody::LazyObjectKind:
    break;
  }
  return nullptr;
}

template <class ELFT>
GnuHashTableSection<ELFT>::GnuHashTableSection()
    : SyntheticSection<ELFT>(SHF_ALLOC, SHT_GNU_HASH, sizeof(uintX_t),
                             ".gnu.hash") {
  this->Entsize = ELFT::Is64Bits ? 0 : 4;
}

template <class ELFT>
unsigned GnuHashTableSection<ELFT>::calcNBuckets(unsigned NumHashed) {
  if (!NumHashed)
    return 0;

  // These values are prime numbers which are not greater than 2^(N-1) + 1.
  // In result, for any particular NumHashed we return a prime number
  // which is not greater than NumHashed.
  static const unsigned Primes[] = {
      1,   1,    3,    3,    7,    13,    31,    61,    127,   251,
      509, 1021, 2039, 4093, 8191, 16381, 32749, 65521, 131071};

  return Primes[std::min<unsigned>(Log2_32_Ceil(NumHashed),
                                   array_lengthof(Primes) - 1)];
}

// Bloom filter estimation: at least 8 bits for each hashed symbol.
// GNU Hash table requirement: it should be a power of 2,
//   the minimum value is 1, even for an empty table.
// Expected results for a 32-bit target:
//   calcMaskWords(0..4)   = 1
//   calcMaskWords(5..8)   = 2
//   calcMaskWords(9..16)  = 4
// For a 64-bit target:
//   calcMaskWords(0..8)   = 1
//   calcMaskWords(9..16)  = 2
//   calcMaskWords(17..32) = 4
template <class ELFT>
unsigned GnuHashTableSection<ELFT>::calcMaskWords(unsigned NumHashed) {
  if (!NumHashed)
    return 1;
  return NextPowerOf2((NumHashed - 1) / sizeof(Elf_Off));
}

template <class ELFT> void GnuHashTableSection<ELFT>::finalize() {
  unsigned NumHashed = Symbols.size();
  NBuckets = calcNBuckets(NumHashed);
  MaskWords = calcMaskWords(NumHashed);
  // Second hash shift estimation: just predefined values.
  Shift2 = ELFT::Is64Bits ? 6 : 5;

  this->OutSec->Entsize = this->Entsize;
  this->OutSec->Link = this->Link = In<ELFT>::DynSymTab->OutSec->SectionIndex;
  this->Size = sizeof(Elf_Word) * 4            // Header
               + sizeof(Elf_Off) * MaskWords   // Bloom Filter
               + sizeof(Elf_Word) * NBuckets   // Hash Buckets
               + sizeof(Elf_Word) * NumHashed; // Hash Values
}

template <class ELFT> void GnuHashTableSection<ELFT>::writeTo(uint8_t *Buf) {
  writeHeader(Buf);
  if (Symbols.empty())
    return;
  writeBloomFilter(Buf);
  writeHashTable(Buf);
}

template <class ELFT>
void GnuHashTableSection<ELFT>::writeHeader(uint8_t *&Buf) {
  auto *P = reinterpret_cast<Elf_Word *>(Buf);
  *P++ = NBuckets;
  *P++ = In<ELFT>::DynSymTab->getNumSymbols() - Symbols.size();
  *P++ = MaskWords;
  *P++ = Shift2;
  Buf = reinterpret_cast<uint8_t *>(P);
}

template <class ELFT>
void GnuHashTableSection<ELFT>::writeBloomFilter(uint8_t *&Buf) {
  unsigned C = sizeof(Elf_Off) * 8;

  auto *Masks = reinterpret_cast<Elf_Off *>(Buf);
  for (const SymbolData &Sym : Symbols) {
    size_t Pos = (Sym.Hash / C) & (MaskWords - 1);
    uintX_t V = (uintX_t(1) << (Sym.Hash % C)) |
                (uintX_t(1) << ((Sym.Hash >> Shift2) % C));
    Masks[Pos] |= V;
  }
  Buf += sizeof(Elf_Off) * MaskWords;
}

template <class ELFT>
void GnuHashTableSection<ELFT>::writeHashTable(uint8_t *Buf) {
  Elf_Word *Buckets = reinterpret_cast<Elf_Word *>(Buf);
  Elf_Word *Values = Buckets + NBuckets;

  int PrevBucket = -1;
  int I = 0;
  for (const SymbolData &Sym : Symbols) {
    int Bucket = Sym.Hash % NBuckets;
    assert(PrevBucket <= Bucket);
    if (Bucket != PrevBucket) {
      Buckets[Bucket] = Sym.Body->DynsymIndex;
      PrevBucket = Bucket;
      if (I > 0)
        Values[I - 1] |= 1;
    }
    Values[I] = Sym.Hash & ~1;
    ++I;
  }
  if (I > 0)
    Values[I - 1] |= 1;
}

static uint32_t hashGnu(StringRef Name) {
  uint32_t H = 5381;
  for (uint8_t C : Name)
    H = (H << 5) + H + C;
  return H;
}

// Add symbols to this symbol hash table. Note that this function
// destructively sort a given vector -- which is needed because
// GNU-style hash table places some sorting requirements.
template <class ELFT>
void GnuHashTableSection<ELFT>::addSymbols(std::vector<SymbolTableEntry> &V) {
  // Ideally this will just be 'auto' but GCC 6.1 is not able
  // to deduce it correctly.
  std::vector<SymbolTableEntry>::iterator Mid =
      std::stable_partition(V.begin(), V.end(), [](const SymbolTableEntry &S) {
        return S.Symbol->isUndefined();
      });
  if (Mid == V.end())
    return;
  for (auto I = Mid, E = V.end(); I != E; ++I) {
    SymbolBody *B = I->Symbol;
    size_t StrOff = I->StrTabOffset;
    Symbols.push_back({B, StrOff, hashGnu(B->getName())});
  }

  unsigned NBuckets = calcNBuckets(Symbols.size());
  std::stable_sort(Symbols.begin(), Symbols.end(),
                   [&](const SymbolData &L, const SymbolData &R) {
                     return L.Hash % NBuckets < R.Hash % NBuckets;
                   });

  V.erase(Mid, V.end());
  for (const SymbolData &Sym : Symbols)
    V.push_back({Sym.Body, Sym.STName});
}

template <class ELFT>
HashTableSection<ELFT>::HashTableSection()
    : SyntheticSection<ELFT>(SHF_ALLOC, SHT_HASH, sizeof(Elf_Word), ".hash") {
  this->Entsize = sizeof(Elf_Word);
}

template <class ELFT> void HashTableSection<ELFT>::finalize() {
  this->OutSec->Link = this->Link = In<ELFT>::DynSymTab->OutSec->SectionIndex;
  this->OutSec->Entsize = this->Entsize;

  unsigned NumEntries = 2;                            // nbucket and nchain.
  NumEntries += In<ELFT>::DynSymTab->getNumSymbols(); // The chain entries.

  // Create as many buckets as there are symbols.
  // FIXME: This is simplistic. We can try to optimize it, but implementing
  // support for SHT_GNU_HASH is probably even more profitable.
  NumEntries += In<ELFT>::DynSymTab->getNumSymbols();
  this->Size = NumEntries * sizeof(Elf_Word);
}

template <class ELFT> void HashTableSection<ELFT>::writeTo(uint8_t *Buf) {
  unsigned NumSymbols = In<ELFT>::DynSymTab->getNumSymbols();
  auto *P = reinterpret_cast<Elf_Word *>(Buf);
  *P++ = NumSymbols; // nbucket
  *P++ = NumSymbols; // nchain

  Elf_Word *Buckets = P;
  Elf_Word *Chains = P + NumSymbols;

  for (const SymbolTableEntry &S : In<ELFT>::DynSymTab->getSymbols()) {
    SymbolBody *Body = S.Symbol;
    StringRef Name = Body->getName();
    unsigned I = Body->DynsymIndex;
    uint32_t Hash = hashSysV(Name) % NumSymbols;
    Chains[I] = Buckets[Hash];
    Buckets[Hash] = I;
  }
}

template InputSection<ELF32LE> *elf::createCommonSection();
template InputSection<ELF32BE> *elf::createCommonSection();
template InputSection<ELF64LE> *elf::createCommonSection();
template InputSection<ELF64BE> *elf::createCommonSection();

template InputSection<ELF32LE> *elf::createInterpSection();
template InputSection<ELF32BE> *elf::createInterpSection();
template InputSection<ELF64LE> *elf::createInterpSection();
template InputSection<ELF64BE> *elf::createInterpSection();

template MergeInputSection<ELF32LE> *elf::createCommentSection();
template MergeInputSection<ELF32BE> *elf::createCommentSection();
template MergeInputSection<ELF64LE> *elf::createCommentSection();
template MergeInputSection<ELF64BE> *elf::createCommentSection();

template class elf::MipsAbiFlagsSection<ELF32LE>;
template class elf::MipsAbiFlagsSection<ELF32BE>;
template class elf::MipsAbiFlagsSection<ELF64LE>;
template class elf::MipsAbiFlagsSection<ELF64BE>;

template class elf::MipsOptionsSection<ELF32LE>;
template class elf::MipsOptionsSection<ELF32BE>;
template class elf::MipsOptionsSection<ELF64LE>;
template class elf::MipsOptionsSection<ELF64BE>;

template class elf::MipsReginfoSection<ELF32LE>;
template class elf::MipsReginfoSection<ELF32BE>;
template class elf::MipsReginfoSection<ELF64LE>;
template class elf::MipsReginfoSection<ELF64BE>;

template class elf::BuildIdSection<ELF32LE>;
template class elf::BuildIdSection<ELF32BE>;
template class elf::BuildIdSection<ELF64LE>;
template class elf::BuildIdSection<ELF64BE>;

template class elf::BuildIdFastHash<ELF32LE>;
template class elf::BuildIdFastHash<ELF32BE>;
template class elf::BuildIdFastHash<ELF64LE>;
template class elf::BuildIdFastHash<ELF64BE>;

template class elf::BuildIdMd5<ELF32LE>;
template class elf::BuildIdMd5<ELF32BE>;
template class elf::BuildIdMd5<ELF64LE>;
template class elf::BuildIdMd5<ELF64BE>;

template class elf::BuildIdSha1<ELF32LE>;
template class elf::BuildIdSha1<ELF32BE>;
template class elf::BuildIdSha1<ELF64LE>;
template class elf::BuildIdSha1<ELF64BE>;

template class elf::BuildIdUuid<ELF32LE>;
template class elf::BuildIdUuid<ELF32BE>;
template class elf::BuildIdUuid<ELF64LE>;
template class elf::BuildIdUuid<ELF64BE>;

template class elf::BuildIdHexstring<ELF32LE>;
template class elf::BuildIdHexstring<ELF32BE>;
template class elf::BuildIdHexstring<ELF64LE>;
template class elf::BuildIdHexstring<ELF64BE>;

template class elf::GotSection<ELF32LE>;
template class elf::GotSection<ELF32BE>;
template class elf::GotSection<ELF64LE>;
template class elf::GotSection<ELF64BE>;

template class elf::MipsGotSection<ELF32LE>;
template class elf::MipsGotSection<ELF32BE>;
template class elf::MipsGotSection<ELF64LE>;
template class elf::MipsGotSection<ELF64BE>;

template class elf::GotPltSection<ELF32LE>;
template class elf::GotPltSection<ELF32BE>;
template class elf::GotPltSection<ELF64LE>;
template class elf::GotPltSection<ELF64BE>;

template class elf::StringTableSection<ELF32LE>;
template class elf::StringTableSection<ELF32BE>;
template class elf::StringTableSection<ELF64LE>;
template class elf::StringTableSection<ELF64BE>;

template class elf::DynamicSection<ELF32LE>;
template class elf::DynamicSection<ELF32BE>;
template class elf::DynamicSection<ELF64LE>;
template class elf::DynamicSection<ELF64BE>;

template class elf::RelocationSection<ELF32LE>;
template class elf::RelocationSection<ELF32BE>;
template class elf::RelocationSection<ELF64LE>;
template class elf::RelocationSection<ELF64BE>;

template class elf::SymbolTableSection<ELF32LE>;
template class elf::SymbolTableSection<ELF32BE>;
template class elf::SymbolTableSection<ELF64LE>;
template class elf::SymbolTableSection<ELF64BE>;

template class elf::GnuHashTableSection<ELF32LE>;
template class elf::GnuHashTableSection<ELF32BE>;
template class elf::GnuHashTableSection<ELF64LE>;
template class elf::GnuHashTableSection<ELF64BE>;

template class elf::HashTableSection<ELF32LE>;
template class elf::HashTableSection<ELF32BE>;
template class elf::HashTableSection<ELF64LE>;
template class elf::HashTableSection<ELF64BE>;
