//===- lib/ReaderWriter/ELF/AArch64/AArch64TargetHandler.h ----------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_AARCH64_AARCH64_TARGET_HANDLER_H
#define LLD_READER_WRITER_ELF_AARCH64_AARCH64_TARGET_HANDLER_H

#include "AArch64RelocationHandler.h"
#include "ELFReader.h"
#include "TargetLayout.h"
#include "lld/Core/Simple.h"

namespace lld {
namespace elf {

class AArch64LinkingContext;
class AArch64GOTSection;

class AArch64TargetLayout final : public TargetLayout<ELF64LE> {
  typedef llvm::object::Elf_Shdr_Impl<ELF64LE> Elf_Shdr;

public:
  AArch64TargetLayout(ELFLinkingContext &ctx);

  AtomSection<ELF64LE> *
  createSection(StringRef name, int32_t type,
                DefinedAtom::ContentPermissions permissions,
                TargetLayout<ELF64LE>::SectionOrder order) override;

  const std::vector<AArch64GOTSection *> &getGOTSections() const {
    return _gotSections;
  }

  uint64_t getTPOffset() {
    std::call_once(_tpOffOnce, [this]() {
      for (const auto &phdr : *_programHeader) {
        if (phdr->p_type == llvm::ELF::PT_TLS) {
          _tpOff = llvm::alignTo(TCB_SIZE, phdr->p_align);
          break;
        }
      }
      assert(_tpOff != 0 && "TLS segment not found");
    });
    return _tpOff;
  }

private:
  enum {
    TCB_SIZE = 16,
  };

private:
  std::vector<AArch64GOTSection *> _gotSections;
  uint64_t _tpOff = 0;
  std::once_flag _tpOffOnce;
};

class AArch64TargetHandler final : public TargetHandler {
public:
  AArch64TargetHandler(AArch64LinkingContext &ctx);

  const TargetRelocationHandler &getRelocationHandler() const override {
    return *_relocationHandler;
  }

  std::unique_ptr<Reader> getObjReader() override {
    return llvm::make_unique<ELFReader<ELFFile<ELF64LE>>>(_ctx);
  }

  std::unique_ptr<Reader> getDSOReader() override {
    return llvm::make_unique<ELFReader<DynamicFile<ELF64LE>>>(_ctx);
  }

  std::unique_ptr<Writer> getWriter() override;

private:
  AArch64LinkingContext &_ctx;
  std::unique_ptr<AArch64TargetLayout> _targetLayout;
  std::unique_ptr<AArch64TargetRelocationHandler> _relocationHandler;
};

} // end namespace elf
} // end namespace lld

#endif
