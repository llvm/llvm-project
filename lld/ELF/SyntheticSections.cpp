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
#include "OutputSections.h"
#include "Strings.h"

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

template <class ELFT>
BuildIdSection<ELFT>::BuildIdSection(size_t HashSize)
    : InputSection<ELFT>(SHF_ALLOC, SHT_NOTE, 1, ArrayRef<uint8_t>(),
                         ".note.gnu.build-id") {
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

template <class ELFT>
void BuildIdFastHash<ELFT>::writeBuildId(MutableArrayRef<uint8_t> Buf) {
  const endianness E = ELFT::TargetEndianness;

  // 64-bit xxhash
  uint64_t Hash = xxHash64(toStringRef(Buf));
  write64<E>(this->getOutputLoc(Buf.begin()) + 16, Hash);
}

template <class ELFT>
void BuildIdMd5<ELFT>::writeBuildId(MutableArrayRef<uint8_t> Buf) {
  MD5 Hash;
  Hash.update(Buf);
  MD5::MD5Result Res;
  Hash.final(Res);
  memcpy(this->getOutputLoc(Buf.begin()) + 16, Res, 16);
}

template <class ELFT>
void BuildIdSha1<ELFT>::writeBuildId(MutableArrayRef<uint8_t> Buf) {
  SHA1 Hash;
  Hash.update(Buf);
  memcpy(this->getOutputLoc(Buf.begin()) + 16, Hash.final().data(), 20);
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
