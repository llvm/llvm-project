//===- SyntheticSection.h ---------------------------------------*- C++ -*-===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_ELF_SYNTHETIC_SECTION_H
#define LLD_ELF_SYNTHETIC_SECTION_H

#include "InputSection.h"

namespace lld {
namespace elf {

template <class ELFT> class InterpSection final : public InputSection<ELFT> {
public:
  InterpSection();
};

template <class ELFT> class BuildIdSection : public InputSection<ELFT> {
public:
  virtual void writeBuildId(llvm::MutableArrayRef<uint8_t> Buf) = 0;
  virtual ~BuildIdSection() = default;

  uint8_t *getOutputLoc(uint8_t *Start) const;

protected:
  BuildIdSection(size_t HashSize);
  std::vector<uint8_t> Buf;
};

template <class ELFT>
class BuildIdFastHash final : public BuildIdSection<ELFT> {
public:
  BuildIdFastHash() : BuildIdSection<ELFT>(8) {}
  void writeBuildId(llvm::MutableArrayRef<uint8_t> Buf) override;
};

template <class ELFT> class BuildIdMd5 final : public BuildIdSection<ELFT> {
public:
  BuildIdMd5() : BuildIdSection<ELFT>(16) {}
  void writeBuildId(llvm::MutableArrayRef<uint8_t> Buf) override;
};

template <class ELFT> class BuildIdSha1 final : public BuildIdSection<ELFT> {
public:
  BuildIdSha1() : BuildIdSection<ELFT>(20) {}
  void writeBuildId(llvm::MutableArrayRef<uint8_t> Buf) override;
};

template <class ELFT> class BuildIdUuid final : public BuildIdSection<ELFT> {
public:
  BuildIdUuid() : BuildIdSection<ELFT>(16) {}
  void writeBuildId(llvm::MutableArrayRef<uint8_t> Buf) override;
};

template <class ELFT>
class BuildIdHexstring final : public BuildIdSection<ELFT> {
public:
  BuildIdHexstring();
  void writeBuildId(llvm::MutableArrayRef<uint8_t>) override;
};

// Linker generated sections which can be used as inputs.
template <class ELFT> struct In {
  static BuildIdSection<ELFT> *BuildId;
  static InterpSection<ELFT> *Interp;
  static std::vector<InputSection<ELFT> *> Sections;
};

template <class ELFT> BuildIdSection<ELFT> *In<ELFT>::BuildId;
template <class ELFT> InterpSection<ELFT> *In<ELFT>::Interp;
template <class ELFT> std::vector<InputSection<ELFT> *> In<ELFT>::Sections;

} // namespace elf
} // namespace lld

#endif
