//===- lib/ReaderWriter/ELF/X86/X86_64DynamicLibraryWriter.h ---------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef X86_64_DYNAMIC_LIBRARY_WRITER_H
#define X86_64_DYNAMIC_LIBRARY_WRITER_H

#include "DynamicLibraryWriter.h"
#include "X86_64LinkingContext.h"
#include "X86_64TargetHandler.h"

namespace lld {
namespace elf {

class X86_64DynamicLibraryWriter : public DynamicLibraryWriter<ELF64LE> {
public:
  X86_64DynamicLibraryWriter(X86_64LinkingContext &ctx,
                             X86_64TargetLayout &layout);

protected:
  // Add any runtime files and their atoms to the output
  void createImplicitFiles(std::vector<std::unique_ptr<File>> &) override;
};

X86_64DynamicLibraryWriter::X86_64DynamicLibraryWriter(
    X86_64LinkingContext &ctx, X86_64TargetLayout &layout)
    : DynamicLibraryWriter(ctx, layout) {}

void X86_64DynamicLibraryWriter::createImplicitFiles(
    std::vector<std::unique_ptr<File>> &result) {
  DynamicLibraryWriter::createImplicitFiles(result);
  auto gotFile = llvm::make_unique<SimpleFile>("GOTFile", File::kindELFObject);
  gotFile->addAtom(*new (gotFile->allocator()) GlobalOffsetTableAtom(*gotFile));
  gotFile->addAtom(*new (gotFile->allocator()) DynamicAtom(*gotFile));
  result.push_back(std::move(gotFile));
}

} // namespace elf
} // namespace lld

#endif
