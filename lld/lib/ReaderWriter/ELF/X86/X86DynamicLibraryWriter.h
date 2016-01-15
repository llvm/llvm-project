//===- lib/ReaderWriter/ELF/X86/X86DynamicLibraryWriter.h -----------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef X86_X86_DYNAMIC_LIBRARY_WRITER_H
#define X86_X86_DYNAMIC_LIBRARY_WRITER_H

#include "DynamicLibraryWriter.h"
#include "X86LinkingContext.h"

namespace lld {
namespace elf {

class X86DynamicLibraryWriter : public DynamicLibraryWriter<ELF32LE> {
public:
  X86DynamicLibraryWriter(X86LinkingContext &ctx,
                          TargetLayout<ELF32LE> &layout);

protected:
  // Add any runtime files and their atoms to the output
  void createImplicitFiles(std::vector<std::unique_ptr<File>> &) override;
};

X86DynamicLibraryWriter::X86DynamicLibraryWriter(X86LinkingContext &ctx,
                                                 TargetLayout<ELF32LE> &layout)
    : DynamicLibraryWriter(ctx, layout) {}

void X86DynamicLibraryWriter::createImplicitFiles(
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
