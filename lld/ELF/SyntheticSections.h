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

// .MIPS.abiflags section.
template <class ELFT>
class MipsAbiFlagsSection final : public InputSection<ELFT> {
  typedef llvm::object::Elf_Mips_ABIFlags<ELFT> Elf_Mips_ABIFlags;

public:
  MipsAbiFlagsSection();

private:
  Elf_Mips_ABIFlags Flags = {};
};

// .MIPS.options section.
template <class ELFT>
class MipsOptionsSection final : public InputSection<ELFT> {
  typedef llvm::object::Elf_Mips_Options<ELFT> Elf_Mips_Options;
  typedef llvm::object::Elf_Mips_RegInfo<ELFT> Elf_Mips_RegInfo;

public:
  MipsOptionsSection();
  void finalize();

private:
  std::vector<uint8_t> Buf;

  Elf_Mips_Options *getOptions() {
    return reinterpret_cast<Elf_Mips_Options *>(Buf.data());
  }
};

// MIPS .reginfo section.
template <class ELFT>
class MipsReginfoSection final : public InputSection<ELFT> {
  typedef llvm::object::Elf_Mips_RegInfo<ELFT> Elf_Mips_RegInfo;

public:
  MipsReginfoSection();
  void finalize();

private:
  Elf_Mips_RegInfo Reginfo = {};
};

template <class ELFT> class SyntheticSection : public InputSection<ELFT> {
  typedef typename ELFT::uint uintX_t;

public:
  SyntheticSection(uintX_t Flags, uint32_t Type, uintX_t Addralign,
                   StringRef Name)
      : InputSection<ELFT>(Flags, Type, Addralign, ArrayRef<uint8_t>(), Name,
                           InputSectionData::Synthetic) {}

  virtual void writeTo(uint8_t *Buf) = 0;
  virtual size_t getSize() const { return this->Data.size(); }

  static bool classof(const InputSectionData *D) {
    return D->kind() == InputSectionData::Synthetic;
  }

protected:
  ~SyntheticSection() = default;
};

// .note.gnu.build-id section.
template <class ELFT> class BuildIdSection : public InputSection<ELFT> {
public:
  virtual void writeBuildId(llvm::MutableArrayRef<uint8_t> Buf) = 0;
  virtual ~BuildIdSection() = default;

  uint8_t *getOutputLoc(uint8_t *Start) const;

protected:
  BuildIdSection(size_t HashSize);
  std::vector<uint8_t> Buf;

  void
  computeHash(llvm::MutableArrayRef<uint8_t> Buf,
              std::function<void(ArrayRef<uint8_t> Arr, uint8_t *Hash)> Hash);

  size_t HashSize;
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

template <class ELFT>
class GotPltSection final : public SyntheticSection<ELFT> {
  typedef typename ELFT::uint uintX_t;

public:
  GotPltSection();
  void addEntry(SymbolBody &Sym);
  bool empty() const;
  size_t getSize() const override;
  void writeTo(uint8_t *Buf) override;
  uintX_t getVA() { return this->OutSec->Addr + this->OutSecOff; }

private:
  std::vector<const SymbolBody *> Entries;
};

template <class ELFT> InputSection<ELFT> *createCommonSection();
template <class ELFT> InputSection<ELFT> *createInterpSection();

// Linker generated sections which can be used as inputs.
template <class ELFT> struct In {
  static BuildIdSection<ELFT> *BuildId;
  static InputSection<ELFT> *Common;
  static GotPltSection<ELFT> *GotPlt;
  static InputSection<ELFT> *Interp;
  static MipsAbiFlagsSection<ELFT> *MipsAbiFlags;
  static MipsOptionsSection<ELFT> *MipsOptions;
  static MipsReginfoSection<ELFT> *MipsReginfo;
};

template <class ELFT> BuildIdSection<ELFT> *In<ELFT>::BuildId;
template <class ELFT> InputSection<ELFT> *In<ELFT>::Common;
template <class ELFT> GotPltSection<ELFT> *In<ELFT>::GotPlt;
template <class ELFT> InputSection<ELFT> *In<ELFT>::Interp;
template <class ELFT> MipsAbiFlagsSection<ELFT> *In<ELFT>::MipsAbiFlags;
template <class ELFT> MipsOptionsSection<ELFT> *In<ELFT>::MipsOptions;
template <class ELFT> MipsReginfoSection<ELFT> *In<ELFT>::MipsReginfo;

} // namespace elf
} // namespace lld

#endif
