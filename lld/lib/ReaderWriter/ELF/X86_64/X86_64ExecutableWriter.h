//===- lib/ReaderWriter/ELF/X86/X86_64ExecutableWriter.h ------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef X86_64_EXECUTABLE_WRITER_H
#define X86_64_EXECUTABLE_WRITER_H

#include "ExecutableWriter.h"
#include "X86_64LinkingContext.h"

namespace lld {
namespace elf {

class X86_64ExecutableWriter : public ExecutableWriter<ELF64LE> {
public:
  X86_64ExecutableWriter(X86_64LinkingContext &ctx, X86_64TargetLayout &layout)
      : ExecutableWriter(ctx, layout), _targetLayout(layout) {}

protected:
  // Add any runtime files and their atoms to the output
  void
  createImplicitFiles(std::vector<std::unique_ptr<File>> &result) override {
    ExecutableWriter::createImplicitFiles(result);
    auto gotFile = llvm::make_unique<SimpleFile>("GOTFile",
                                                 File::kindELFObject);
    gotFile->addAtom(*new (gotFile->allocator())
                         GlobalOffsetTableAtom(*gotFile));
    if (this->_ctx.isDynamic())
      gotFile->addAtom(*new (gotFile->allocator()) DynamicAtom(*gotFile));
    result.push_back(std::move(gotFile));
  }

  void buildDynamicSymbolTable(const File &file) override {
    for (auto sec : this->_layout.sections()) {
      if (auto section = dyn_cast<AtomSection<ELF64LE>>(sec)) {
        for (const auto &atom : section->atoms()) {
          if (_targetLayout.getGOTSection().hasGlobalGOTEntry(atom->_atom)) {
            this->_dynamicSymbolTable->addSymbol(atom->_atom, section->ordinal(),
                                                 atom->_virtualAddr, atom);
          }
        }
      }
    }

    ExecutableWriter<ELF64LE>::buildDynamicSymbolTable(file);
  }

  X86_64TargetLayout &_targetLayout;
};

} // namespace elf
} // namespace lld

#endif
