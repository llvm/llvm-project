//===- lib/ReaderWriter/ELF/AArch64/AArch64DynamicLibraryWriter.h ---------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef AARCH64_DYNAMIC_LIBRARY_WRITER_H
#define AARCH64_DYNAMIC_LIBRARY_WRITER_H

#include "AArch64LinkingContext.h"
#include "AArch64TargetHandler.h"
#include "DynamicLibraryWriter.h"

namespace lld {
namespace elf {

class AArch64DynamicLibraryWriter : public DynamicLibraryWriter<ELF64LE> {
public:
  AArch64DynamicLibraryWriter(AArch64LinkingContext &ctx,
                              TargetLayout<ELF64LE> &layout);

protected:
  // Add any runtime files and their atoms to the output
  void createImplicitFiles(std::vector<std::unique_ptr<File>> &) override;
};

AArch64DynamicLibraryWriter::AArch64DynamicLibraryWriter(
    AArch64LinkingContext &ctx, TargetLayout<ELF64LE> &layout)
    : DynamicLibraryWriter(ctx, layout) {}

void AArch64DynamicLibraryWriter::createImplicitFiles(
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
