//===- lib/ReaderWriter/ELF/AMDGPU/AMDGPUTargetHandler.cpp -------*- C++ -*-===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "TargetLayout.h"
#include "AMDGPUExecutableWriter.h"
#include "AMDGPULinkingContext.h"
#include "AMDGPUTargetHandler.h"
#include "llvm/Support/ELF.h"

namespace lld {
namespace elf {

AMDGPUTargetHandler::AMDGPUTargetHandler(AMDGPULinkingContext &ctx)
    : _ctx(ctx), _targetLayout(new AMDGPUTargetLayout(ctx)),
      _relocationHandler(new AMDGPUTargetRelocationHandler(*_targetLayout)) {}

std::unique_ptr<Writer> AMDGPUTargetHandler::getWriter() {
  switch (_ctx.getOutputELFType()) {
  case llvm::ELF::ET_EXEC:
    return llvm::make_unique<AMDGPUExecutableWriter>(_ctx, *_targetLayout);
  case llvm::ELF::ET_DYN:
    llvm_unreachable("TODO: support dynamic libraries");
  case llvm::ELF::ET_REL:
    llvm_unreachable("TODO: support -r mode");
  default:
    llvm_unreachable("unsupported output type");
  }
}

HSATextSection::HSATextSection(const ELFLinkingContext &ctx)
    : AtomSection(ctx, ".hsatext", DefinedAtom::typeCode, 0, 0) {
  _type = SHT_PROGBITS;
  _flags = SHF_ALLOC | SHF_WRITE | SHF_EXECINSTR | SHF_AMDGPU_HSA_AGENT |
           SHF_AMDGPU_HSA_CODE;

  // FIXME: What alignment should we use here?
  _alignment = 4096;
}

void AMDGPUTargetLayout::assignSectionsToSegments() {

  TargetLayout::assignSectionsToSegments();
  for (OutputSection<ELF64LE> *osi : _outputSections) {
    for (Section<ELF64LE> *section : osi->sections()) {
      StringRef InputSectionName = section->inputSectionName();
      if (InputSectionName != ".hsatext")
        continue;

      auto *segment = new (_allocator) Segment<ELF64LE>(
          _ctx, "PT_AMDGPU_HSA_LOAD_CODE_AGENT", PT_AMDGPU_HSA_LOAD_CODE_AGENT);
      _segments.push_back(segment);
      assert(segment);
      segment->append(section);
    }
  }
}

} // namespace elf
} // namespace lld
