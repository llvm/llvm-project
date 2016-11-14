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
        In<ELFT>::Got->getVA() + MipsGPOffset;
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
    Reginfo.ri_gp_value = In<ELFT>::Got->getVA() + MipsGPOffset;
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
                             Target->GotEntrySize, ".got") {
  if (Config->EMachine == EM_MIPS)
    this->Flags |= SHF_MIPS_GPREL;
}

template <class ELFT> void GotSection<ELFT>::addEntry(SymbolBody &Sym) {
  Sym.GotIndex = Entries.size();
  Entries.push_back(&Sym);
}

template <class ELFT>
void GotSection<ELFT>::addMipsEntry(SymbolBody &Sym, uintX_t Addend,
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
    MipsOutSections.insert(OutSec);
    return;
  }
  if (Sym.isTls()) {
    // GOT entries created for MIPS TLS relocations behave like
    // almost GOT entries from other ABIs. They go to the end
    // of the global offset table.
    Sym.GotIndex = Entries.size();
    Entries.push_back(&Sym);
    return;
  }
  auto AddEntry = [&](SymbolBody &S, uintX_t A, MipsGotEntries &Items) {
    if (S.isInGot() && !A)
      return;
    size_t NewIndex = Items.size();
    if (!MipsGotMap.insert({{&S, A}, NewIndex}).second)
      return;
    Items.emplace_back(&S, A);
    if (!A)
      S.GotIndex = NewIndex;
  };
  if (Sym.isPreemptible()) {
    // Ignore addends for preemptible symbols. They got single GOT entry anyway.
    AddEntry(Sym, 0, MipsGlobal);
    Sym.IsInGlobalMipsGot = true;
  } else if (Expr == R_MIPS_GOT_OFF32) {
    AddEntry(Sym, Addend, MipsLocal32);
    Sym.Is32BitMipsGot = true;
  } else {
    // Hold local GOT entries accessed via a 16-bit index separately.
    // That allows to write them in the beginning of the GOT and keep
    // their indexes as less as possible to escape relocation's overflow.
    AddEntry(Sym, Addend, MipsLocal);
  }
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
GotSection<ELFT>::getMipsLocalPageOffset(uintX_t EntryValue) {
  // Initialize the entry by the %hi(EntryValue) expression
  // but without right-shifting.
  EntryValue = (EntryValue + 0x8000) & ~0xffff;
  // Take into account MIPS GOT header.
  // See comment in the GotSection::writeTo.
  size_t NewIndex = MipsLocalGotPos.size() + 2;
  auto P = MipsLocalGotPos.insert(std::make_pair(EntryValue, NewIndex));
  assert(!P.second || MipsLocalGotPos.size() <= MipsPageEntries);
  return (uintX_t)P.first->second * sizeof(uintX_t) - MipsGPOffset;
}

template <class ELFT>
typename GotSection<ELFT>::uintX_t
GotSection<ELFT>::getMipsGotOffset(const SymbolBody &B, uintX_t Addend) const {
  // Calculate offset of the GOT entries block: TLS, global, local.
  uintX_t GotBlockOff;
  if (B.isTls())
    GotBlockOff = getMipsTlsOffset();
  else if (B.IsInGlobalMipsGot)
    GotBlockOff = getMipsLocalEntriesNum() * sizeof(uintX_t);
  else if (B.Is32BitMipsGot)
    GotBlockOff = (MipsPageEntries + MipsLocal.size()) * sizeof(uintX_t);
  else
    GotBlockOff = MipsPageEntries * sizeof(uintX_t);
  // Calculate index of the GOT entry in the block.
  uintX_t GotIndex;
  if (B.isInGot())
    GotIndex = B.GotIndex;
  else {
    auto It = MipsGotMap.find({&B, Addend});
    assert(It != MipsGotMap.end());
    GotIndex = It->second;
  }
  return GotBlockOff + GotIndex * sizeof(uintX_t) - MipsGPOffset;
}

template <class ELFT>
typename GotSection<ELFT>::uintX_t GotSection<ELFT>::getMipsTlsOffset() const {
  return (getMipsLocalEntriesNum() + MipsGlobal.size()) * sizeof(uintX_t);
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

template <class ELFT>
const SymbolBody *GotSection<ELFT>::getMipsFirstGlobalEntry() const {
  return MipsGlobal.empty() ? nullptr : MipsGlobal.front().first;
}

template <class ELFT>
unsigned GotSection<ELFT>::getMipsLocalEntriesNum() const {
  return MipsPageEntries + MipsLocal.size() + MipsLocal32.size();
}

template <class ELFT> void GotSection<ELFT>::finalize() {
  size_t EntriesNum = Entries.size();
  if (Config->EMachine == EM_MIPS) {
    // Take into account MIPS GOT header.
    // See comment in the GotSection::writeTo.
    MipsPageEntries += 2;
    for (const OutputSectionBase *OutSec : MipsOutSections) {
      // Calculate an upper bound of MIPS GOT entries required to store page
      // addresses of local symbols. We assume the worst case - each 64kb
      // page of the output section has at least one GOT relocation against it.
      // Add 0x8000 to the section's size because the page address stored
      // in the GOT entry is calculated as (value + 0x8000) & ~0xffff.
      MipsPageEntries += (OutSec->Size + 0x8000 + 0xfffe) / 0xffff;
    }
    EntriesNum += getMipsLocalEntriesNum() + MipsGlobal.size();
  }
  Size = EntriesNum * sizeof(uintX_t);
}

template <class ELFT>
static void writeUint(uint8_t *Buf, typename ELFT::uint Val) {
  typedef typename ELFT::uint uintX_t;
  write<uintX_t, ELFT::TargetEndianness, sizeof(uintX_t)>(Buf, Val);
}

template <class ELFT> void GotSection<ELFT>::writeMipsGot(uint8_t *Buf) {
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
  for (std::pair<uintX_t, size_t> &L : MipsLocalGotPos) {
    uint8_t *Entry = Buf + L.second * sizeof(uintX_t);
    writeUint<ELFT>(Entry, L.first);
  }
  Buf += MipsPageEntries * sizeof(uintX_t);
  auto AddEntry = [&](const MipsGotEntry &SA) {
    uint8_t *Entry = Buf;
    Buf += sizeof(uintX_t);
    const SymbolBody *Body = SA.first;
    uintX_t VA = Body->template getVA<ELFT>(SA.second);
    writeUint<ELFT>(Entry, VA);
  };
  std::for_each(std::begin(MipsLocal), std::end(MipsLocal), AddEntry);
  std::for_each(std::begin(MipsLocal32), std::end(MipsLocal32), AddEntry);
  std::for_each(std::begin(MipsGlobal), std::end(MipsGlobal), AddEntry);
  // Initialize TLS-related GOT entries. If the entry has a corresponding
  // dynamic relocations, leave it initialized by zero. Write down adjusted
  // TLS symbol's values otherwise. To calculate the adjustments use offsets
  // for thread-local storage.
  // https://www.linux-mips.org/wiki/NPTL
  if (TlsIndexOff != -1U && !Config->Pic)
    writeUint<ELFT>(Buf + TlsIndexOff, 1);
  for (const SymbolBody *B : Entries) {
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

template <class ELFT> void GotSection<ELFT>::writeTo(uint8_t *Buf) {
  if (Config->EMachine == EM_MIPS) {
    writeMipsGot(Buf);
    return;
  }
  for (const SymbolBody *B : Entries) {
    uint8_t *Entry = Buf;
    Buf += sizeof(uintX_t);
    if (!B)
      continue;
    if (B->isPreemptible())
      continue; // The dynamic linker will take care of it.
    uintX_t VA = B->getVA<ELFT>();
    writeUint<ELFT>(Entry, VA);
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

template class elf::GotPltSection<ELF32LE>;
template class elf::GotPltSection<ELF32BE>;
template class elf::GotPltSection<ELF64LE>;
template class elf::GotPltSection<ELF64BE>;

template class elf::StringTableSection<ELF32LE>;
template class elf::StringTableSection<ELF32BE>;
template class elf::StringTableSection<ELF64LE>;
template class elf::StringTableSection<ELF64BE>;
