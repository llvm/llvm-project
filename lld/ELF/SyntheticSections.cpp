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
#include "LinkerScript.h"
#include "Memory.h"
#include "OutputSections.h"
#include "Strings.h"
#include "SymbolTable.h"
#include "Target.h"
#include "Threads.h"
#include "Writer.h"
#include "lld/Config/Version.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/MD5.h"
#include "llvm/Support/RandomNumberGenerator.h"
#include "llvm/Support/SHA1.h"
#include "llvm/Support/xxhash.h"
#include <cstdlib>

using namespace llvm;
using namespace llvm::dwarf;
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

  if (!Config->DefineCommon)
    return Ret;

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

// .MIPS.abiflags section.
template <class ELFT>
MipsAbiFlagsSection<ELFT>::MipsAbiFlagsSection(Elf_Mips_ABIFlags Flags)
    : SyntheticSection<ELFT>(SHF_ALLOC, SHT_MIPS_ABIFLAGS, 8, ".MIPS.abiflags"),
      Flags(Flags) {}

template <class ELFT> void MipsAbiFlagsSection<ELFT>::writeTo(uint8_t *Buf) {
  memcpy(Buf, &Flags, sizeof(Flags));
}

template <class ELFT>
MipsAbiFlagsSection<ELFT> *MipsAbiFlagsSection<ELFT>::create() {
  Elf_Mips_ABIFlags Flags = {};
  bool Create = false;

  for (InputSectionBase<ELFT> *Sec : Symtab<ELFT>::X->Sections) {
    if (!Sec->Live || Sec->Type != SHT_MIPS_ABIFLAGS)
      continue;
    Sec->Live = false;
    Create = true;

    std::string Filename = toString(Sec->getFile());
    const size_t Size = Sec->Data.size();
    // Older version of BFD (such as the default FreeBSD linker) concatenate
    // .MIPS.abiflags instead of merging. To allow for this case (or potential
    // zero padding) we ignore everything after the first Elf_Mips_ABIFlags
    if (Size < sizeof(Elf_Mips_ABIFlags)) {
      error(Filename + ": invalid size of .MIPS.abiflags section: got " +
            Twine(Size) + " instead of " + Twine(sizeof(Elf_Mips_ABIFlags)));
      return nullptr;
    }
    auto *S = reinterpret_cast<const Elf_Mips_ABIFlags *>(Sec->Data.data());
    if (S->version != 0) {
      error(Filename + ": unexpected .MIPS.abiflags version " +
            Twine(S->version));
      return nullptr;
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
    Flags.fp_abi = elf::getMipsFpAbiFlag(Flags.fp_abi, S->fp_abi, Filename);
  };

  if (Create)
    return make<MipsAbiFlagsSection<ELFT>>(Flags);
  return nullptr;
}

// .MIPS.options section.
template <class ELFT>
MipsOptionsSection<ELFT>::MipsOptionsSection(Elf_Mips_RegInfo Reginfo)
    : SyntheticSection<ELFT>(SHF_ALLOC, SHT_MIPS_OPTIONS, 8, ".MIPS.options"),
      Reginfo(Reginfo) {}

template <class ELFT> void MipsOptionsSection<ELFT>::writeTo(uint8_t *Buf) {
  auto *Options = reinterpret_cast<Elf_Mips_Options *>(Buf);
  Options->kind = ODK_REGINFO;
  Options->size = getSize();

  if (!Config->Relocatable)
    Reginfo.ri_gp_value = In<ELFT>::MipsGot->getGp();
  memcpy(Buf + sizeof(Elf_Mips_Options), &Reginfo, sizeof(Reginfo));
}

template <class ELFT>
MipsOptionsSection<ELFT> *MipsOptionsSection<ELFT>::create() {
  // N64 ABI only.
  if (!ELFT::Is64Bits)
    return nullptr;

  Elf_Mips_RegInfo Reginfo = {};
  bool Create = false;

  for (InputSectionBase<ELFT> *Sec : Symtab<ELFT>::X->Sections) {
    if (!Sec->Live || Sec->Type != SHT_MIPS_OPTIONS)
      continue;
    Sec->Live = false;
    Create = true;

    std::string Filename = toString(Sec->getFile());
    ArrayRef<uint8_t> D = Sec->Data;

    while (!D.empty()) {
      if (D.size() < sizeof(Elf_Mips_Options)) {
        error(Filename + ": invalid size of .MIPS.options section");
        break;
      }

      auto *Opt = reinterpret_cast<const Elf_Mips_Options *>(D.data());
      if (Opt->kind == ODK_REGINFO) {
        if (Config->Relocatable && Opt->getRegInfo().ri_gp_value)
          error(Filename + ": unsupported non-zero ri_gp_value");
        Reginfo.ri_gprmask |= Opt->getRegInfo().ri_gprmask;
        Sec->getFile()->MipsGp0 = Opt->getRegInfo().ri_gp_value;
        break;
      }

      if (!Opt->size)
        fatal(Filename + ": zero option descriptor size");
      D = D.slice(Opt->size);
    }
  };

  if (Create)
    return make<MipsOptionsSection<ELFT>>(Reginfo);
  return nullptr;
}

// MIPS .reginfo section.
template <class ELFT>
MipsReginfoSection<ELFT>::MipsReginfoSection(Elf_Mips_RegInfo Reginfo)
    : SyntheticSection<ELFT>(SHF_ALLOC, SHT_MIPS_REGINFO, 4, ".reginfo"),
      Reginfo(Reginfo) {}

template <class ELFT> void MipsReginfoSection<ELFT>::writeTo(uint8_t *Buf) {
  if (!Config->Relocatable)
    Reginfo.ri_gp_value = In<ELFT>::MipsGot->getGp();
  memcpy(Buf, &Reginfo, sizeof(Reginfo));
}

template <class ELFT>
MipsReginfoSection<ELFT> *MipsReginfoSection<ELFT>::create() {
  // Section should be alive for O32 and N32 ABIs only.
  if (ELFT::Is64Bits)
    return nullptr;

  Elf_Mips_RegInfo Reginfo = {};
  bool Create = false;

  for (InputSectionBase<ELFT> *Sec : Symtab<ELFT>::X->Sections) {
    if (!Sec->Live || Sec->Type != SHT_MIPS_REGINFO)
      continue;
    Sec->Live = false;
    Create = true;

    if (Sec->Data.size() != sizeof(Elf_Mips_RegInfo)) {
      error(toString(Sec->getFile()) + ": invalid size of .reginfo section");
      return nullptr;
    }
    auto *R = reinterpret_cast<const Elf_Mips_RegInfo *>(Sec->Data.data());
    if (Config->Relocatable && R->ri_gp_value)
      error(toString(Sec->getFile()) + ": unsupported non-zero ri_gp_value");

    Reginfo.ri_gprmask |= R->ri_gprmask;
    Sec->getFile()->MipsGp0 = R->ri_gp_value;
  };

  if (Create)
    return make<MipsReginfoSection<ELFT>>(Reginfo);
  return nullptr;
}

template <class ELFT> InputSection<ELFT> *elf::createInterpSection() {
  auto *Ret = make<InputSection<ELFT>>(SHF_ALLOC, SHT_PROGBITS, 1,
                                       ArrayRef<uint8_t>(), ".interp");
  Ret->Live = true;

  // StringSaver guarantees that the returned string ends with '\0'.
  StringRef S = Saver.save(Config->DynamicLinker);
  Ret->Data = {(const uint8_t *)S.data(), S.size() + 1};
  return Ret;
}

template <class ELFT>
SymbolBody *elf::addSyntheticLocal(StringRef Name, uint8_t Type,
                                   typename ELFT::uint Value,
                                   typename ELFT::uint Size,
                                   InputSectionBase<ELFT> *Section) {
  auto *S = make<DefinedRegular<ELFT>>(Name, /*IsLocal*/ true, STV_DEFAULT,
                                       Type, Value, Size, Section, nullptr);
  if (In<ELFT>::SymTab)
    In<ELFT>::SymTab->addLocal(S);
  return S;
}

static size_t getHashSize() {
  switch (Config->BuildId) {
  case BuildIdKind::Fast:
    return 8;
  case BuildIdKind::Md5:
  case BuildIdKind::Uuid:
    return 16;
  case BuildIdKind::Sha1:
    return 20;
  case BuildIdKind::Hexstring:
    return Config->BuildIdVector.size();
  default:
    llvm_unreachable("unknown BuildIdKind");
  }
}

template <class ELFT>
BuildIdSection<ELFT>::BuildIdSection()
    : SyntheticSection<ELFT>(SHF_ALLOC, SHT_NOTE, 1, ".note.gnu.build-id"),
      HashSize(getHashSize()) {}

template <class ELFT> void BuildIdSection<ELFT>::writeTo(uint8_t *Buf) {
  const endianness E = ELFT::TargetEndianness;
  write32<E>(Buf, 4);                   // Name size
  write32<E>(Buf + 4, HashSize);        // Content size
  write32<E>(Buf + 8, NT_GNU_BUILD_ID); // Type
  memcpy(Buf + 12, "GNU", 4);           // Name string
  HashBuf = Buf + 16;
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
    llvm::ArrayRef<uint8_t> Data,
    std::function<void(uint8_t *Dest, ArrayRef<uint8_t> Arr)> HashFn) {
  std::vector<ArrayRef<uint8_t>> Chunks = split(Data, 1024 * 1024);
  std::vector<uint8_t> Hashes(Chunks.size() * HashSize);

  // Compute hash values.
  forLoop(0, Chunks.size(),
          [&](size_t I) { HashFn(Hashes.data() + I * HashSize, Chunks[I]); });

  // Write to the final output buffer.
  HashFn(HashBuf, Hashes);
}

template <class ELFT>
CopyRelSection<ELFT>::CopyRelSection(bool ReadOnly, uintX_t AddrAlign, size_t S)
    : SyntheticSection<ELFT>(SHF_ALLOC, SHT_NOBITS, AddrAlign,
                             ReadOnly ? ".bss.rel.ro" : ".bss"),
      Size(S) {
  if (!ReadOnly)
    this->Flags |= SHF_WRITE;
}

template <class ELFT>
void BuildIdSection<ELFT>::writeBuildId(ArrayRef<uint8_t> Buf) {
  switch (Config->BuildId) {
  case BuildIdKind::Fast:
    computeHash(Buf, [](uint8_t *Dest, ArrayRef<uint8_t> Arr) {
      write64le(Dest, xxHash64(toStringRef(Arr)));
    });
    break;
  case BuildIdKind::Md5:
    computeHash(Buf, [](uint8_t *Dest, ArrayRef<uint8_t> Arr) {
      memcpy(Dest, MD5::hash(Arr).data(), 16);
    });
    break;
  case BuildIdKind::Sha1:
    computeHash(Buf, [](uint8_t *Dest, ArrayRef<uint8_t> Arr) {
      memcpy(Dest, SHA1::hash(Arr).data(), 20);
    });
    break;
  case BuildIdKind::Uuid:
    if (getRandomBytes(HashBuf, HashSize))
      error("entropy source failure");
    break;
  case BuildIdKind::Hexstring:
    memcpy(HashBuf, Config->BuildIdVector.data(), Config->BuildIdVector.size());
    break;
  default:
    llvm_unreachable("unknown BuildIdKind");
  }
}

template <class ELFT>
GotSection<ELFT>::GotSection()
    : SyntheticSection<ELFT>(SHF_ALLOC | SHF_WRITE, SHT_PROGBITS,
                             Target->GotEntrySize, ".got") {}

template <class ELFT> void GotSection<ELFT>::addEntry(SymbolBody &Sym) {
  Sym.GotIndex = NumEntries;
  ++NumEntries;
}

template <class ELFT> bool GotSection<ELFT>::addDynTlsEntry(SymbolBody &Sym) {
  if (Sym.GlobalDynIndex != -1U)
    return false;
  Sym.GlobalDynIndex = NumEntries;
  // Global Dynamic TLS entries take two GOT slots.
  NumEntries += 2;
  return true;
}

// Reserves TLS entries for a TLS module ID and a TLS block offset.
// In total it takes two GOT slots.
template <class ELFT> bool GotSection<ELFT>::addTlsIndex() {
  if (TlsIndexOff != uint32_t(-1))
    return false;
  TlsIndexOff = NumEntries * sizeof(uintX_t);
  NumEntries += 2;
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
  Size = NumEntries * sizeof(uintX_t);
}

template <class ELFT> bool GotSection<ELFT>::empty() const {
  // If we have a relocation that is relative to GOT (such as GOTOFFREL),
  // we need to emit a GOT even if it's empty.
  return NumEntries == 0 && !HasGotOffRel;
}

template <class ELFT> void GotSection<ELFT>::writeTo(uint8_t *Buf) {
  this->relocate(Buf, Buf + Size);
}

template <class ELFT>
MipsGotSection<ELFT>::MipsGotSection()
    : SyntheticSection<ELFT>(SHF_ALLOC | SHF_WRITE | SHF_MIPS_GPREL,
                             SHT_PROGBITS, 16, ".got") {}

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
    auto *DefSym = cast<DefinedRegular<ELFT>>(&Sym);
    PageIndexMap.insert({DefSym->Section->getOutputSection(), 0});
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

template <class ELFT>
bool MipsGotSection<ELFT>::addDynTlsEntry(SymbolBody &Sym) {
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

static uint64_t getMipsPageAddr(uint64_t Addr) {
  return (Addr + 0x8000) & ~0xffff;
}

static uint64_t getMipsPageCount(uint64_t Size) {
  return (Size + 0xfffe) / 0xffff + 1;
}

template <class ELFT>
typename MipsGotSection<ELFT>::uintX_t
MipsGotSection<ELFT>::getPageEntryOffset(const SymbolBody &B,
                                         uintX_t Addend) const {
  const OutputSectionBase *OutSec =
      cast<DefinedRegular<ELFT>>(&B)->Section->getOutputSection();
  uintX_t SecAddr = getMipsPageAddr(OutSec->Addr);
  uintX_t SymAddr = getMipsPageAddr(B.getVA<ELFT>(Addend));
  uintX_t Index = PageIndexMap.lookup(OutSec) + (SymAddr - SecAddr) / 0xffff;
  assert(Index < PageEntriesNum);
  return (HeaderEntriesNum + Index) * sizeof(uintX_t);
}

template <class ELFT>
typename MipsGotSection<ELFT>::uintX_t
MipsGotSection<ELFT>::getBodyEntryOffset(const SymbolBody &B,
                                         uintX_t Addend) const {
  // Calculate offset of the GOT entries block: TLS, global, local.
  uintX_t Index = HeaderEntriesNum + PageEntriesNum;
  if (B.isTls())
    Index += LocalEntries.size() + LocalEntries32.size() + GlobalEntries.size();
  else if (B.IsInGlobalMipsGot)
    Index += LocalEntries.size() + LocalEntries32.size();
  else if (B.Is32BitMipsGot)
    Index += LocalEntries.size();
  // Calculate offset of the GOT entry in the block.
  if (B.isInGot())
    Index += B.GotIndex;
  else {
    auto It = EntryIndexMap.find({&B, Addend});
    assert(It != EntryIndexMap.end());
    Index += It->second;
  }
  return Index * sizeof(uintX_t);
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
  return HeaderEntriesNum + PageEntriesNum + LocalEntries.size() +
         LocalEntries32.size();
}

template <class ELFT> void MipsGotSection<ELFT>::finalize() {
  PageEntriesNum = 0;
  for (std::pair<const OutputSectionBase *, size_t> &P : PageIndexMap) {
    // For each output section referenced by GOT page relocations calculate
    // and save into PageIndexMap an upper bound of MIPS GOT entries required
    // to store page addresses of local symbols. We assume the worst case -
    // each 64kb page of the output section has at least one GOT relocation
    // against it. And take in account the case when the section intersects
    // page boundaries.
    P.second = PageEntriesNum;
    PageEntriesNum += getMipsPageCount(P.first->Size);
  }
  Size = (getLocalEntriesNum() + GlobalEntries.size() + TlsEntries.size()) *
         sizeof(uintX_t);
}

template <class ELFT> bool MipsGotSection<ELFT>::empty() const {
  // We add the .got section to the result for dynamic MIPS target because
  // its address and properties are mentioned in the .dynamic section.
  return Config->Relocatable;
}

template <class ELFT>
typename MipsGotSection<ELFT>::uintX_t MipsGotSection<ELFT>::getGp() const {
  return ElfSym<ELFT>::MipsGp->template getVA<ELFT>(0);
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
  Buf += HeaderEntriesNum * sizeof(uintX_t);
  // Write 'page address' entries to the local part of the GOT.
  for (std::pair<const OutputSectionBase *, size_t> &L : PageIndexMap) {
    size_t PageCount = getMipsPageCount(L.first->Size);
    uintX_t FirstPageAddr = getMipsPageAddr(L.first->Addr);
    for (size_t PI = 0; PI < PageCount; ++PI) {
      uint8_t *Entry = Buf + (L.second + PI) * sizeof(uintX_t);
      writeUint<ELFT>(Entry, FirstPageAddr + PI * 0x10000);
    }
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
  if (TlsIndexOff != -1U && !Config->pic())
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

// On ARM the IgotPltSection is part of the GotSection, on other Targets it is
// part of the .got.plt
template <class ELFT>
IgotPltSection<ELFT>::IgotPltSection()
    : SyntheticSection<ELFT>(SHF_ALLOC | SHF_WRITE, SHT_PROGBITS,
                             Target->GotPltEntrySize,
                             Config->EMachine == EM_ARM ? ".got" : ".got.plt") {
}

template <class ELFT> void IgotPltSection<ELFT>::addEntry(SymbolBody &Sym) {
  Sym.IsInIgot = true;
  Sym.GotPltIndex = Entries.size();
  Entries.push_back(&Sym);
}

template <class ELFT> size_t IgotPltSection<ELFT>::getSize() const {
  return Entries.size() * Target->GotPltEntrySize;
}

template <class ELFT> void IgotPltSection<ELFT>::writeTo(uint8_t *Buf) {
  for (const SymbolBody *B : Entries) {
    Target->writeIgotPlt(Buf, *B);
    Buf += sizeof(uintX_t);
  }
}

template <class ELFT>
StringTableSection<ELFT>::StringTableSection(StringRef Name, bool Dynamic)
    : SyntheticSection<ELFT>(Dynamic ? (uintX_t)SHF_ALLOC : 0, SHT_STRTAB, 1,
                             Name),
      Dynamic(Dynamic) {
  // ELF string tables start with a NUL byte.
  addString("");
}

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
  for (StringRef S : Strings) {
    memcpy(Buf, S.data(), S.size());
    Buf += S.size() + 1;
  }
}

// Returns the number of version definition entries. Because the first entry
// is for the version definition itself, it is the number of versioned symbols
// plus one. Note that we don't support multiple versions yet.
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

  if (!Config->Shared && !Config->Relocatable)
    add({DT_DEBUG, (uint64_t)0});
}

// Add remaining entries to complete .dynamic contents.
template <class ELFT> void DynamicSection<ELFT>::finalize() {
  if (this->Size)
    return; // Already finalized.

  this->Link = In<ELFT>::DynStrTab->OutSec->SectionIndex;
  if (In<ELFT>::RelaDyn->OutSec->Size > 0) {
    bool IsRela = Config->Rela;
    add({IsRela ? DT_RELA : DT_REL, In<ELFT>::RelaDyn});
    add({IsRela ? DT_RELASZ : DT_RELSZ, In<ELFT>::RelaDyn->OutSec->Size});
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
  if (In<ELFT>::RelaPlt->OutSec->Size > 0) {
    add({DT_JMPREL, In<ELFT>::RelaPlt});
    add({DT_PLTRELSZ, In<ELFT>::RelaPlt->OutSec->Size});
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

  if (SymbolBody *B = Symtab<ELFT>::X->findInCurrentDSO(Config->Init))
    add({DT_INIT, B});
  if (SymbolBody *B = Symtab<ELFT>::X->findInCurrentDSO(Config->Fini))
    add({DT_FINI, B});

  bool HasVerNeed = In<ELFT>::VerNeed->getNeedNum() != 0;
  if (HasVerNeed || In<ELFT>::VerDef)
    add({DT_VERSYM, In<ELFT>::VerSym});
  if (In<ELFT>::VerDef) {
    add({DT_VERDEF, In<ELFT>::VerDef});
    add({DT_VERDEFNUM, getVerDefNum()});
  }
  if (HasVerNeed) {
    add({DT_VERNEED, In<ELFT>::VerNeed});
    add({DT_VERNEEDNUM, In<ELFT>::VerNeed->getNeedNum()});
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
    if (In<ELFT>::MipsRldMap)
      add({DT_MIPS_RLD_MAP, In<ELFT>::MipsRldMap});
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

template <class ELFT> void SymbolTableSection<ELFT>::finalize() {
  this->OutSec->Link = this->Link = StrTabSec.OutSec->SectionIndex;
  this->OutSec->Info = this->Info = NumLocals + 1;
  this->OutSec->Entsize = this->Entsize;

  if (Config->Relocatable)
    return;

  if (!StrTabSec.isDynamic()) {
    auto GlobBegin = Symbols.begin() + NumLocals;
    auto It = std::stable_partition(
        GlobBegin, Symbols.end(), [](const SymbolTableEntry &S) {
          return S.Symbol->symbol()->computeBinding() == STB_LOCAL;
        });
    // update sh_info with number of Global symbols output with computed
    // binding of STB_LOCAL
    this->OutSec->Info = this->Info = 1 + (It - Symbols.begin());
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

template <class ELFT> void SymbolTableSection<ELFT>::addGlobal(SymbolBody *B) {
  Symbols.push_back({B, StrTabSec.addString(B->getName(), false)});
}

template <class ELFT> void SymbolTableSection<ELFT>::addLocal(SymbolBody *B) {
  assert(!StrTabSec.isDynamic());
  ++NumLocals;
  Symbols.push_back({B, StrTabSec.addString(B->getName())});
}

template <class ELFT>
size_t SymbolTableSection<ELFT>::getSymbolIndex(SymbolBody *Body) {
  auto I = llvm::find_if(Symbols, [&](const SymbolTableEntry &E) {
    if (E.Symbol == Body)
      return true;
    // This is used for -r, so we have to handle multiple section
    // symbols being combined.
    if (Body->Type == STT_SECTION && E.Symbol->Type == STT_SECTION)
      return cast<DefinedRegular<ELFT>>(Body)->Section->OutSec ==
             cast<DefinedRegular<ELFT>>(E.Symbol)->Section->OutSec;
    return false;
  });
  if (I == Symbols.end())
    return 0;
  return I - Symbols.begin() + 1;
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

  for (auto I = Symbols.begin(); I != Symbols.begin() + NumLocals; ++I) {
    const DefinedRegular<ELFT> &Body = *cast<DefinedRegular<ELFT>>(I->Symbol);
    InputSectionBase<ELFT> *Section = Body.Section;
    auto *ESym = reinterpret_cast<Elf_Sym *>(Buf);

    if (!Section) {
      ESym->st_shndx = SHN_ABS;
      ESym->st_value = Body.Value;
    } else {
      const OutputSectionBase *OutSec = Section->getOutputSection();
      ESym->st_shndx = OutSec->SectionIndex;
      ESym->st_value = OutSec->Addr + Section->getOffset(Body);
    }
    ESym->st_name = I->StrTabOffset;
    ESym->st_size = Body.template getSize<ELFT>();
    ESym->setBindingAndType(STB_LOCAL, Body.Type);
    Buf += sizeof(*ESym);
  }
}

template <class ELFT>
void SymbolTableSection<ELFT>::writeGlobalSymbols(uint8_t *Buf) {
  // Write the internal symbol table contents to the output symbol table
  // pointed by Buf.
  auto *ESym = reinterpret_cast<Elf_Sym *>(Buf);

  for (auto I = Symbols.begin() + NumLocals; I != Symbols.end(); ++I) {
    const SymbolTableEntry &S = *I;
    SymbolBody *Body = S.Symbol;
    size_t StrOff = S.StrTabOffset;

    uint8_t Type = Body->Type;
    uintX_t Size = Body->getSize<ELFT>();

    ESym->setBindingAndType(Body->symbol()->computeBinding(), Type);
    ESym->st_size = Size;
    ESym->st_name = StrOff;
    ESym->setVisibility(Body->symbol()->Visibility);
    ESym->st_value = Body->getVA<ELFT>();

    if (const OutputSectionBase *OutSec = getOutputSection(Body)) {
      ESym->st_shndx = OutSec->SectionIndex;
    } else if (isa<DefinedRegular<ELFT>>(Body)) {
      ESym->st_shndx = SHN_ABS;
    } else if (isa<DefinedCommon>(Body)) {
      ESym->st_shndx = SHN_COMMON;
      ESym->st_value = cast<DefinedCommon>(Body)->Alignment;
    }

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
    return cast<DefinedSynthetic>(Sym)->Section;
  case SymbolBody::DefinedRegularKind: {
    auto &D = cast<DefinedRegular<ELFT>>(*Sym);
    if (D.Section)
      return D.Section->getOutputSection();
    break;
  }
  case SymbolBody::DefinedCommonKind:
    if (!Config->DefineCommon)
      return nullptr;
    return In<ELFT>::Common->OutSec;
  case SymbolBody::SharedKind: {
    auto &SS = cast<SharedSymbol<ELFT>>(*Sym);
    if (SS.needsCopy())
      return SS.getBssSectionForCopy()->OutSec;
    break;
  }
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

template <class ELFT>
PltSection<ELFT>::PltSection(size_t S)
    : SyntheticSection<ELFT>(SHF_ALLOC | SHF_EXECINSTR, SHT_PROGBITS, 16,
                             ".plt"),
      HeaderSize(S) {}

template <class ELFT> void PltSection<ELFT>::writeTo(uint8_t *Buf) {
  // At beginning of PLT but not the IPLT, we have code to call the dynamic
  // linker to resolve dynsyms at runtime. Write such code.
  if (HeaderSize != 0)
    Target->writePltHeader(Buf);
  size_t Off = HeaderSize;
  // The IPlt is immediately after the Plt, account for this in RelOff
  unsigned PltOff = getPltRelocOff();

  for (auto &I : Entries) {
    const SymbolBody *B = I.first;
    unsigned RelOff = I.second + PltOff;
    uint64_t Got = B->getGotPltVA<ELFT>();
    uint64_t Plt = this->getVA() + Off;
    Target->writePlt(Buf + Off, Got, Plt, B->PltIndex, RelOff);
    Off += Target->PltEntrySize;
  }
}

template <class ELFT> void PltSection<ELFT>::addEntry(SymbolBody &Sym) {
  Sym.PltIndex = Entries.size();
  RelocationSection<ELFT> *PltRelocSection = In<ELFT>::RelaPlt;
  if (HeaderSize == 0) {
    PltRelocSection = In<ELFT>::RelaIplt;
    Sym.IsInIplt = true;
  }
  unsigned RelOff = PltRelocSection->getRelocOffset();
  Entries.push_back(std::make_pair(&Sym, RelOff));
}

template <class ELFT> size_t PltSection<ELFT>::getSize() const {
  return HeaderSize + Entries.size() * Target->PltEntrySize;
}

// Some architectures such as additional symbols in the PLT section. For
// example ARM uses mapping symbols to aid disassembly
template <class ELFT> void PltSection<ELFT>::addSymbols() {
  // The PLT may have symbols defined for the Header, the IPLT has no header
  if (HeaderSize != 0)
    Target->addPltHeaderSymbols(this);
  size_t Off = HeaderSize;
  for (size_t I = 0; I < Entries.size(); ++I) {
    Target->addPltSymbols(this, Off);
    Off += Target->PltEntrySize;
  }
}

template <class ELFT> unsigned PltSection<ELFT>::getPltRelocOff() const {
  return (HeaderSize == 0) ? In<ELFT>::Plt->getSize() : 0;
}

template <class ELFT>
GdbIndexSection<ELFT>::GdbIndexSection()
    : SyntheticSection<ELFT>(0, SHT_PROGBITS, 1, ".gdb_index"),
      StringPool(llvm::StringTableBuilder::ELF) {}

template <class ELFT> void GdbIndexSection<ELFT>::parseDebugSections() {
  for (InputSectionBase<ELFT> *S : Symtab<ELFT>::X->Sections)
    if (InputSection<ELFT> *IS = dyn_cast<InputSection<ELFT>>(S))
      if (IS->OutSec && IS->Name == ".debug_info")
        readDwarf(IS);
}

// Iterative hash function for symbol's name is described in .gdb_index format
// specification. Note that we use one for version 5 to 7 here, it is different
// for version 4.
static uint32_t hash(StringRef Str) {
  uint32_t R = 0;
  for (uint8_t C : Str)
    R = R * 67 + tolower(C) - 113;
  return R;
}

template <class ELFT>
void GdbIndexSection<ELFT>::readDwarf(InputSection<ELFT> *I) {
  GdbIndexBuilder<ELFT> Builder(I);
  if (ErrorCount)
    return;

  size_t CuId = CompilationUnits.size();
  std::vector<std::pair<uintX_t, uintX_t>> CuList = Builder.readCUList();
  CompilationUnits.insert(CompilationUnits.end(), CuList.begin(), CuList.end());

  std::vector<AddressEntry<ELFT>> AddrArea = Builder.readAddressArea(CuId);
  AddressArea.insert(AddressArea.end(), AddrArea.begin(), AddrArea.end());

  std::vector<std::pair<StringRef, uint8_t>> NamesAndTypes =
      Builder.readPubNamesAndTypes();

  for (std::pair<StringRef, uint8_t> &Pair : NamesAndTypes) {
    uint32_t Hash = hash(Pair.first);
    size_t Offset = StringPool.add(Pair.first);

    bool IsNew;
    GdbSymbol *Sym;
    std::tie(IsNew, Sym) = SymbolTable.add(Hash, Offset);
    if (IsNew) {
      Sym->CuVectorIndex = CuVectors.size();
      CuVectors.push_back({{CuId, Pair.second}});
      continue;
    }

    std::vector<std::pair<uint32_t, uint8_t>> &CuVec =
        CuVectors[Sym->CuVectorIndex];
    CuVec.push_back({CuId, Pair.second});
  }
}

template <class ELFT> void GdbIndexSection<ELFT>::finalize() {
  if (Finalized)
    return;
  Finalized = true;

  parseDebugSections();

  // GdbIndex header consist from version fields
  // and 5 more fields with different kinds of offsets.
  CuTypesOffset = CuListOffset + CompilationUnits.size() * CompilationUnitSize;
  SymTabOffset = CuTypesOffset + AddressArea.size() * AddressEntrySize;

  ConstantPoolOffset =
      SymTabOffset + SymbolTable.getCapacity() * SymTabEntrySize;

  for (std::vector<std::pair<uint32_t, uint8_t>> &CuVec : CuVectors) {
    CuVectorsOffset.push_back(CuVectorsSize);
    CuVectorsSize += OffsetTypeSize * (CuVec.size() + 1);
  }
  StringPoolOffset = ConstantPoolOffset + CuVectorsSize;

  StringPool.finalizeInOrder();
}

template <class ELFT> size_t GdbIndexSection<ELFT>::getSize() const {
  const_cast<GdbIndexSection<ELFT> *>(this)->finalize();
  return StringPoolOffset + StringPool.getSize();
}

template <class ELFT> void GdbIndexSection<ELFT>::writeTo(uint8_t *Buf) {
  write32le(Buf, 7);                       // Write version.
  write32le(Buf + 4, CuListOffset);        // CU list offset.
  write32le(Buf + 8, CuTypesOffset);       // Types CU list offset.
  write32le(Buf + 12, CuTypesOffset);      // Address area offset.
  write32le(Buf + 16, SymTabOffset);       // Symbol table offset.
  write32le(Buf + 20, ConstantPoolOffset); // Constant pool offset.
  Buf += 24;

  // Write the CU list.
  for (std::pair<uintX_t, uintX_t> CU : CompilationUnits) {
    write64le(Buf, CU.first);
    write64le(Buf + 8, CU.second);
    Buf += 16;
  }

  // Write the address area.
  for (AddressEntry<ELFT> &E : AddressArea) {
    uintX_t BaseAddr = E.Section->OutSec->Addr + E.Section->getOffset(0);
    write64le(Buf, BaseAddr + E.LowAddress);
    write64le(Buf + 8, BaseAddr + E.HighAddress);
    write32le(Buf + 16, E.CuIndex);
    Buf += 20;
  }

  // Write the symbol table.
  for (size_t I = 0; I < SymbolTable.getCapacity(); ++I) {
    GdbSymbol *Sym = SymbolTable.getSymbol(I);
    if (Sym) {
      size_t NameOffset =
          Sym->NameOffset + StringPoolOffset - ConstantPoolOffset;
      size_t CuVectorOffset = CuVectorsOffset[Sym->CuVectorIndex];
      write32le(Buf, NameOffset);
      write32le(Buf + 4, CuVectorOffset);
    }
    Buf += 8;
  }

  // Write the CU vectors into the constant pool.
  for (std::vector<std::pair<uint32_t, uint8_t>> &CuVec : CuVectors) {
    write32le(Buf, CuVec.size());
    Buf += 4;
    for (std::pair<uint32_t, uint8_t> &P : CuVec) {
      uint32_t Index = P.first;
      uint8_t Flags = P.second;
      Index |= Flags << 24;
      write32le(Buf, Index);
      Buf += 4;
    }
  }

  StringPool.write(Buf);
}

template <class ELFT> bool GdbIndexSection<ELFT>::empty() const {
  return !Out<ELFT>::DebugInfo;
}

template <class ELFT>
EhFrameHeader<ELFT>::EhFrameHeader()
    : SyntheticSection<ELFT>(SHF_ALLOC, SHT_PROGBITS, 1, ".eh_frame_hdr") {}

// .eh_frame_hdr contains a binary search table of pointers to FDEs.
// Each entry of the search table consists of two values,
// the starting PC from where FDEs covers, and the FDE's address.
// It is sorted by PC.
template <class ELFT> void EhFrameHeader<ELFT>::writeTo(uint8_t *Buf) {
  const endianness E = ELFT::TargetEndianness;

  // Sort the FDE list by their PC and uniqueify. Usually there is only
  // one FDE for a PC (i.e. function), but if ICF merges two functions
  // into one, there can be more than one FDEs pointing to the address.
  auto Less = [](const FdeData &A, const FdeData &B) { return A.Pc < B.Pc; };
  std::stable_sort(Fdes.begin(), Fdes.end(), Less);
  auto Eq = [](const FdeData &A, const FdeData &B) { return A.Pc == B.Pc; };
  Fdes.erase(std::unique(Fdes.begin(), Fdes.end(), Eq), Fdes.end());

  Buf[0] = 1;
  Buf[1] = DW_EH_PE_pcrel | DW_EH_PE_sdata4;
  Buf[2] = DW_EH_PE_udata4;
  Buf[3] = DW_EH_PE_datarel | DW_EH_PE_sdata4;
  write32<E>(Buf + 4, Out<ELFT>::EhFrame->Addr - this->getVA() - 4);
  write32<E>(Buf + 8, Fdes.size());
  Buf += 12;

  uintX_t VA = this->getVA();
  for (FdeData &Fde : Fdes) {
    write32<E>(Buf, Fde.Pc - VA);
    write32<E>(Buf + 4, Fde.FdeVA - VA);
    Buf += 8;
  }
}

template <class ELFT> size_t EhFrameHeader<ELFT>::getSize() const {
  // .eh_frame_hdr has a 12 bytes header followed by an array of FDEs.
  return 12 + Out<ELFT>::EhFrame->NumFdes * 8;
}

template <class ELFT>
void EhFrameHeader<ELFT>::addFde(uint32_t Pc, uint32_t FdeVA) {
  Fdes.push_back({Pc, FdeVA});
}

template <class ELFT> bool EhFrameHeader<ELFT>::empty() const {
  return Out<ELFT>::EhFrame->empty();
}

template <class ELFT>
VersionDefinitionSection<ELFT>::VersionDefinitionSection()
    : SyntheticSection<ELFT>(SHF_ALLOC, SHT_GNU_verdef, sizeof(uint32_t),
                             ".gnu.version_d") {}

static StringRef getFileDefName() {
  if (!Config->SoName.empty())
    return Config->SoName;
  return Config->OutputFile;
}

template <class ELFT> void VersionDefinitionSection<ELFT>::finalize() {
  FileDefNameOff = In<ELFT>::DynStrTab->addString(getFileDefName());
  for (VersionDefinition &V : Config->VersionDefinitions)
    V.NameOff = In<ELFT>::DynStrTab->addString(V.Name);

  this->OutSec->Link = this->Link = In<ELFT>::DynStrTab->OutSec->SectionIndex;

  // sh_info should be set to the number of definitions. This fact is missed in
  // documentation, but confirmed by binutils community:
  // https://sourceware.org/ml/binutils/2014-11/msg00355.html
  this->OutSec->Info = this->Info = getVerDefNum();
}

template <class ELFT>
void VersionDefinitionSection<ELFT>::writeOne(uint8_t *Buf, uint32_t Index,
                                              StringRef Name, size_t NameOff) {
  auto *Verdef = reinterpret_cast<Elf_Verdef *>(Buf);
  Verdef->vd_version = 1;
  Verdef->vd_cnt = 1;
  Verdef->vd_aux = sizeof(Elf_Verdef);
  Verdef->vd_next = sizeof(Elf_Verdef) + sizeof(Elf_Verdaux);
  Verdef->vd_flags = (Index == 1 ? VER_FLG_BASE : 0);
  Verdef->vd_ndx = Index;
  Verdef->vd_hash = hashSysV(Name);

  auto *Verdaux = reinterpret_cast<Elf_Verdaux *>(Buf + sizeof(Elf_Verdef));
  Verdaux->vda_name = NameOff;
  Verdaux->vda_next = 0;
}

template <class ELFT>
void VersionDefinitionSection<ELFT>::writeTo(uint8_t *Buf) {
  writeOne(Buf, 1, getFileDefName(), FileDefNameOff);

  for (VersionDefinition &V : Config->VersionDefinitions) {
    Buf += sizeof(Elf_Verdef) + sizeof(Elf_Verdaux);
    writeOne(Buf, V.Id, V.Name, V.NameOff);
  }

  // Need to terminate the last version definition.
  Elf_Verdef *Verdef = reinterpret_cast<Elf_Verdef *>(Buf);
  Verdef->vd_next = 0;
}

template <class ELFT> size_t VersionDefinitionSection<ELFT>::getSize() const {
  return (sizeof(Elf_Verdef) + sizeof(Elf_Verdaux)) * getVerDefNum();
}

template <class ELFT>
VersionTableSection<ELFT>::VersionTableSection()
    : SyntheticSection<ELFT>(SHF_ALLOC, SHT_GNU_versym, sizeof(uint16_t),
                             ".gnu.version") {}

template <class ELFT> void VersionTableSection<ELFT>::finalize() {
  this->OutSec->Entsize = this->Entsize = sizeof(Elf_Versym);
  // At the moment of june 2016 GNU docs does not mention that sh_link field
  // should be set, but Sun docs do. Also readelf relies on this field.
  this->OutSec->Link = this->Link = In<ELFT>::DynSymTab->OutSec->SectionIndex;
}

template <class ELFT> size_t VersionTableSection<ELFT>::getSize() const {
  return sizeof(Elf_Versym) * (In<ELFT>::DynSymTab->getSymbols().size() + 1);
}

template <class ELFT> void VersionTableSection<ELFT>::writeTo(uint8_t *Buf) {
  auto *OutVersym = reinterpret_cast<Elf_Versym *>(Buf) + 1;
  for (const SymbolTableEntry &S : In<ELFT>::DynSymTab->getSymbols()) {
    OutVersym->vs_index = S.Symbol->symbol()->VersionId;
    ++OutVersym;
  }
}

template <class ELFT> bool VersionTableSection<ELFT>::empty() const {
  return !In<ELFT>::VerDef && In<ELFT>::VerNeed->empty();
}

template <class ELFT>
VersionNeedSection<ELFT>::VersionNeedSection()
    : SyntheticSection<ELFT>(SHF_ALLOC, SHT_GNU_verneed, sizeof(uint32_t),
                             ".gnu.version_r") {
  // Identifiers in verneed section start at 2 because 0 and 1 are reserved
  // for VER_NDX_LOCAL and VER_NDX_GLOBAL.
  // First identifiers are reserved by verdef section if it exist.
  NextIndex = getVerDefNum() + 1;
}

template <class ELFT>
void VersionNeedSection<ELFT>::addSymbol(SharedSymbol<ELFT> *SS) {
  if (!SS->Verdef) {
    SS->symbol()->VersionId = VER_NDX_GLOBAL;
    return;
  }
  SharedFile<ELFT> *F = SS->file();
  // If we don't already know that we need an Elf_Verneed for this DSO, prepare
  // to create one by adding it to our needed list and creating a dynstr entry
  // for the soname.
  if (F->VerdefMap.empty())
    Needed.push_back({F, In<ELFT>::DynStrTab->addString(F->getSoName())});
  typename SharedFile<ELFT>::NeededVer &NV = F->VerdefMap[SS->Verdef];
  // If we don't already know that we need an Elf_Vernaux for this Elf_Verdef,
  // prepare to create one by allocating a version identifier and creating a
  // dynstr entry for the version name.
  if (NV.Index == 0) {
    NV.StrTab = In<ELFT>::DynStrTab->addString(
        SS->file()->getStringTable().data() + SS->Verdef->getAux()->vda_name);
    NV.Index = NextIndex++;
  }
  SS->symbol()->VersionId = NV.Index;
}

template <class ELFT> void VersionNeedSection<ELFT>::writeTo(uint8_t *Buf) {
  // The Elf_Verneeds need to appear first, followed by the Elf_Vernauxs.
  auto *Verneed = reinterpret_cast<Elf_Verneed *>(Buf);
  auto *Vernaux = reinterpret_cast<Elf_Vernaux *>(Verneed + Needed.size());

  for (std::pair<SharedFile<ELFT> *, size_t> &P : Needed) {
    // Create an Elf_Verneed for this DSO.
    Verneed->vn_version = 1;
    Verneed->vn_cnt = P.first->VerdefMap.size();
    Verneed->vn_file = P.second;
    Verneed->vn_aux =
        reinterpret_cast<char *>(Vernaux) - reinterpret_cast<char *>(Verneed);
    Verneed->vn_next = sizeof(Elf_Verneed);
    ++Verneed;

    // Create the Elf_Vernauxs for this Elf_Verneed. The loop iterates over
    // VerdefMap, which will only contain references to needed version
    // definitions. Each Elf_Vernaux is based on the information contained in
    // the Elf_Verdef in the source DSO. This loop iterates over a std::map of
    // pointers, but is deterministic because the pointers refer to Elf_Verdef
    // data structures within a single input file.
    for (auto &NV : P.first->VerdefMap) {
      Vernaux->vna_hash = NV.first->vd_hash;
      Vernaux->vna_flags = 0;
      Vernaux->vna_other = NV.second.Index;
      Vernaux->vna_name = NV.second.StrTab;
      Vernaux->vna_next = sizeof(Elf_Vernaux);
      ++Vernaux;
    }

    Vernaux[-1].vna_next = 0;
  }
  Verneed[-1].vn_next = 0;
}

template <class ELFT> void VersionNeedSection<ELFT>::finalize() {
  this->OutSec->Link = this->Link = In<ELFT>::DynStrTab->OutSec->SectionIndex;
  this->OutSec->Info = this->Info = Needed.size();
}

template <class ELFT> size_t VersionNeedSection<ELFT>::getSize() const {
  unsigned Size = Needed.size() * sizeof(Elf_Verneed);
  for (const std::pair<SharedFile<ELFT> *, size_t> &P : Needed)
    Size += P.first->VerdefMap.size() * sizeof(Elf_Vernaux);
  return Size;
}

template <class ELFT> bool VersionNeedSection<ELFT>::empty() const {
  return getNeedNum() == 0;
}

template <class ELFT>
MergeSyntheticSection<ELFT>::MergeSyntheticSection(StringRef Name,
                                                   uint32_t Type, uintX_t Flags,
                                                   uintX_t Alignment)
    : SyntheticSection<ELFT>(Flags, Type, Alignment, Name),
      Builder(StringTableBuilder::RAW, Alignment) {}

template <class ELFT>
void MergeSyntheticSection<ELFT>::addSection(MergeInputSection<ELFT> *MS) {
  assert(!Finalized);
  MS->MergeSec = this;
  Sections.push_back(MS);
}

template <class ELFT> void MergeSyntheticSection<ELFT>::writeTo(uint8_t *Buf) {
  Builder.write(Buf);
}

template <class ELFT>
bool MergeSyntheticSection<ELFT>::shouldTailMerge() const {
  return (this->Flags & SHF_STRINGS) && Config->Optimize >= 2;
}

template <class ELFT> void MergeSyntheticSection<ELFT>::finalizeTailMerge() {
  // Add all string pieces to the string table builder to create section
  // contents.
  for (MergeInputSection<ELFT> *Sec : Sections)
    for (size_t I = 0, E = Sec->Pieces.size(); I != E; ++I)
      if (Sec->Pieces[I].Live)
        Builder.add(Sec->getData(I));

  // Fix the string table content. After this, the contents will never change.
  Builder.finalize();

  // finalize() fixed tail-optimized strings, so we can now get
  // offsets of strings. Get an offset for each string and save it
  // to a corresponding StringPiece for easy access.
  for (MergeInputSection<ELFT> *Sec : Sections)
    for (size_t I = 0, E = Sec->Pieces.size(); I != E; ++I)
      if (Sec->Pieces[I].Live)
        Sec->Pieces[I].OutputOff = Builder.getOffset(Sec->getData(I));
}

template <class ELFT> void MergeSyntheticSection<ELFT>::finalizeNoTailMerge() {
  // Add all string pieces to the string table builder to create section
  // contents. Because we are not tail-optimizing, offsets of strings are
  // fixed when they are added to the builder (string table builder contains
  // a hash table from strings to offsets).
  for (MergeInputSection<ELFT> *Sec : Sections)
    for (size_t I = 0, E = Sec->Pieces.size(); I != E; ++I)
      if (Sec->Pieces[I].Live)
        Sec->Pieces[I].OutputOff = Builder.add(Sec->getData(I));

  Builder.finalizeInOrder();
}

template <class ELFT> void MergeSyntheticSection<ELFT>::finalize() {
  if (Finalized)
    return;
  Finalized = true;
  if (shouldTailMerge())
    finalizeTailMerge();
  else
    finalizeNoTailMerge();
}

template <class ELFT> size_t MergeSyntheticSection<ELFT>::getSize() const {
  // We should finalize string builder to know the size.
  const_cast<MergeSyntheticSection<ELFT> *>(this)->finalize();
  return Builder.getSize();
}

template <class ELFT>
MipsRldMapSection<ELFT>::MipsRldMapSection()
    : SyntheticSection<ELFT>(SHF_ALLOC | SHF_WRITE, SHT_PROGBITS,
                             sizeof(typename ELFT::uint), ".rld_map") {}

template <class ELFT> void MipsRldMapSection<ELFT>::writeTo(uint8_t *Buf) {
  // Apply filler from linker script.
  uint64_t Filler = Script<ELFT>::X->getFiller(this->Name);
  Filler = (Filler << 32) | Filler;
  memcpy(Buf, &Filler, getSize());
}

template <class ELFT>
ARMExidxSentinelSection<ELFT>::ARMExidxSentinelSection()
    : SyntheticSection<ELFT>(SHF_ALLOC | SHF_LINK_ORDER, SHT_ARM_EXIDX,
                             sizeof(typename ELFT::uint), ".ARM.exidx") {}

// Write a terminating sentinel entry to the end of the .ARM.exidx table.
// This section will have been sorted last in the .ARM.exidx table.
// This table entry will have the form:
// | PREL31 upper bound of code that has exception tables | EXIDX_CANTUNWIND |
template <class ELFT>
void ARMExidxSentinelSection<ELFT>::writeTo(uint8_t *Buf) {
  // Get the InputSection before us, we are by definition last
  auto RI = cast<OutputSection<ELFT>>(this->OutSec)->Sections.rbegin();
  InputSection<ELFT> *LE = *(++RI);
  InputSection<ELFT> *LC = cast<InputSection<ELFT>>(LE->getLinkOrderDep());
  uint64_t S = LC->OutSec->Addr + LC->getOffset(LC->getSize());
  uint64_t P = this->getVA();
  Target->relocateOne(Buf, R_ARM_PREL31, S - P);
  write32le(Buf + 4, 0x1);
}

template <class ELFT>
ThunkSection<ELFT>::ThunkSection(OutputSectionBase *OS, uint64_t Off)
    : SyntheticSection<ELFT>(SHF_ALLOC | SHF_EXECINSTR, SHT_PROGBITS,
                             sizeof(typename ELFT::uint), ".text.thunk") {
  this->OutSec = OS;
  this->OutSecOff = Off;
}

template <class ELFT> void ThunkSection<ELFT>::addThunk(Thunk<ELFT> *T) {
  uint64_t Off = alignTo(Size, T->alignment);
  T->Offset = Off;
  Thunks.push_back(T);
  T->addSymbols(*this);
  Size = Off + T->size();
}

template <class ELFT> void ThunkSection<ELFT>::writeTo(uint8_t *Buf) {
  for (const Thunk<ELFT> *T : Thunks)
    T->writeTo(Buf + T->Offset, *this);
}

template <class ELFT>
InputSection<ELFT> *ThunkSection<ELFT>::getTargetInputSection() const {
  const Thunk<ELFT> *T = Thunks.front();
  return T->getTargetInputSection();
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

template SymbolBody *
elf::addSyntheticLocal<ELF32LE>(StringRef, uint8_t, ELF32LE::uint,
                                ELF32LE::uint, InputSectionBase<ELF32LE> *);
template SymbolBody *
elf::addSyntheticLocal<ELF32BE>(StringRef, uint8_t, ELF32BE::uint,
                                ELF32BE::uint, InputSectionBase<ELF32BE> *);
template SymbolBody *
elf::addSyntheticLocal<ELF64LE>(StringRef, uint8_t, ELF64LE::uint,
                                ELF64LE::uint, InputSectionBase<ELF64LE> *);
template SymbolBody *
elf::addSyntheticLocal<ELF64BE>(StringRef, uint8_t, ELF64BE::uint,
                                ELF64BE::uint, InputSectionBase<ELF64BE> *);

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

template class elf::CopyRelSection<ELF32LE>;
template class elf::CopyRelSection<ELF32BE>;
template class elf::CopyRelSection<ELF64LE>;
template class elf::CopyRelSection<ELF64BE>;

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

template class elf::IgotPltSection<ELF32LE>;
template class elf::IgotPltSection<ELF32BE>;
template class elf::IgotPltSection<ELF64LE>;
template class elf::IgotPltSection<ELF64BE>;

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

template class elf::PltSection<ELF32LE>;
template class elf::PltSection<ELF32BE>;
template class elf::PltSection<ELF64LE>;
template class elf::PltSection<ELF64BE>;

template class elf::GdbIndexSection<ELF32LE>;
template class elf::GdbIndexSection<ELF32BE>;
template class elf::GdbIndexSection<ELF64LE>;
template class elf::GdbIndexSection<ELF64BE>;

template class elf::EhFrameHeader<ELF32LE>;
template class elf::EhFrameHeader<ELF32BE>;
template class elf::EhFrameHeader<ELF64LE>;
template class elf::EhFrameHeader<ELF64BE>;

template class elf::VersionTableSection<ELF32LE>;
template class elf::VersionTableSection<ELF32BE>;
template class elf::VersionTableSection<ELF64LE>;
template class elf::VersionTableSection<ELF64BE>;

template class elf::VersionNeedSection<ELF32LE>;
template class elf::VersionNeedSection<ELF32BE>;
template class elf::VersionNeedSection<ELF64LE>;
template class elf::VersionNeedSection<ELF64BE>;

template class elf::VersionDefinitionSection<ELF32LE>;
template class elf::VersionDefinitionSection<ELF32BE>;
template class elf::VersionDefinitionSection<ELF64LE>;
template class elf::VersionDefinitionSection<ELF64BE>;

template class elf::MergeSyntheticSection<ELF32LE>;
template class elf::MergeSyntheticSection<ELF32BE>;
template class elf::MergeSyntheticSection<ELF64LE>;
template class elf::MergeSyntheticSection<ELF64BE>;

template class elf::MipsRldMapSection<ELF32LE>;
template class elf::MipsRldMapSection<ELF32BE>;
template class elf::MipsRldMapSection<ELF64LE>;
template class elf::MipsRldMapSection<ELF64BE>;

template class elf::ARMExidxSentinelSection<ELF32LE>;
template class elf::ARMExidxSentinelSection<ELF32BE>;
template class elf::ARMExidxSentinelSection<ELF64LE>;
template class elf::ARMExidxSentinelSection<ELF64BE>;

template class elf::ThunkSection<ELF32LE>;
template class elf::ThunkSection<ELF32BE>;
template class elf::ThunkSection<ELF64LE>;
template class elf::ThunkSection<ELF64BE>;
