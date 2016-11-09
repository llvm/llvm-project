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

#include "lld/Core/Parallel.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/MD5.h"
#include "llvm/Support/RandomNumberGenerator.h"
#include "llvm/Support/SHA1.h"
#include "llvm/Support/xxhash.h"

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

  Buf.resize(16 + HashSize);
  const endianness E = ELFT::TargetEndianness;
  write32<E>(Buf.data(), 4);                   // Name size
  write32<E>(Buf.data() + 4, HashSize);        // Content size
  write32<E>(Buf.data() + 8, NT_GNU_BUILD_ID); // Type
  memcpy(Buf.data() + 12, "GNU", 4);           // Name string
  this->Data = ArrayRef<uint8_t>(Buf);
}

template <class ELFT>
uint8_t *BuildIdSection<ELFT>::getOutputLoc(uint8_t *Start) const {
  return Start + this->OutSec->getFileOffset() + this->OutSecOff;
}

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

template <class ELFT>
void BuildIdSection<ELFT>::computeHash(
    llvm::MutableArrayRef<uint8_t> Data,
    std::function<void(ArrayRef<uint8_t> Arr, uint8_t *Hash)> Hash) {
  std::vector<ArrayRef<uint8_t>> Chunks = split(Data, 1024 * 1024);
  std::vector<uint8_t> HashList(Chunks.size() * HashSize);

  if (Config->Threads)
    parallel_for_each(Chunks.begin(), Chunks.end(),
                      [&](ArrayRef<uint8_t> &Chunk) {
                        size_t Id = &Chunk - Chunks.data();
                        Hash(Chunk, HashList.data() + Id * HashSize);
                      });
  else
    std::for_each(Chunks.begin(), Chunks.end(), [&](ArrayRef<uint8_t> &Chunk) {
      size_t Id = &Chunk - Chunks.data();
      Hash(Chunk, HashList.data() + Id * HashSize);
    });

  Hash(HashList, this->getOutputLoc(Data.begin()) + 16);
}

template <class ELFT>
void BuildIdFastHash<ELFT>::writeBuildId(MutableArrayRef<uint8_t> Buf) {
  this->computeHash(Buf, [](ArrayRef<uint8_t> Arr, uint8_t *Dest) {
    uint64_t Hash = xxHash64(toStringRef(Arr));
    write64<ELFT::TargetEndianness>(Dest, Hash);
  });
}

template <class ELFT>
void BuildIdMd5<ELFT>::writeBuildId(MutableArrayRef<uint8_t> Buf) {
  this->computeHash(Buf, [&](ArrayRef<uint8_t> Arr, uint8_t *Dest) {
    MD5 Hash;
    Hash.update(Arr);
    MD5::MD5Result Res;
    Hash.final(Res);
    memcpy(Dest, Res, this->HashSize);
  });
}

template <class ELFT>
void BuildIdSha1<ELFT>::writeBuildId(MutableArrayRef<uint8_t> Buf) {
  this->computeHash(Buf, [&](ArrayRef<uint8_t> Arr, uint8_t *Dest) {
    SHA1 Hash;
    Hash.update(Arr);
    memcpy(Dest, Hash.final().data(), this->HashSize);
  });
}

template <class ELFT>
void BuildIdUuid<ELFT>::writeBuildId(MutableArrayRef<uint8_t> Buf) {
  if (getRandomBytes(this->getOutputLoc(Buf.begin()) + 16, 16))
    error("entropy source failure");
}

template <class ELFT>
BuildIdHexstring<ELFT>::BuildIdHexstring()
    : BuildIdSection<ELFT>(Config->BuildIdVector.size()) {}

template <class ELFT>
void BuildIdHexstring<ELFT>::writeBuildId(MutableArrayRef<uint8_t> Buf) {
  memcpy(this->getOutputLoc(Buf.begin()) + 16, Config->BuildIdVector.data(),
         Config->BuildIdVector.size());
}

template InputSection<ELF32LE> *elf::createCommonSection();
template InputSection<ELF32BE> *elf::createCommonSection();
template InputSection<ELF64LE> *elf::createCommonSection();
template InputSection<ELF64BE> *elf::createCommonSection();

template InputSection<ELF32LE> *elf::createInterpSection();
template InputSection<ELF32BE> *elf::createInterpSection();
template InputSection<ELF64LE> *elf::createInterpSection();
template InputSection<ELF64BE> *elf::createInterpSection();

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
