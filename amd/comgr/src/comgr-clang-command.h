//===- comgr-clang-command.h - ClangCommand implementation ----------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef COMGR_CLANG_COMMAND_H
#define COMGR_CLANG_COMMAND_H

#include "comgr-cache-command.h"

#include <llvm/Support/VirtualFileSystem.h>

namespace clang {
class DiagnosticOptions;
namespace driver {
class Command;
} // namespace driver
} // namespace clang

namespace COMGR {
class ClangCommand final : public CachedCommandAdaptor {
public:
  using ExecuteFnTy = std::function<amd_comgr_status_t(
      clang::driver::Command &, llvm::raw_ostream &, clang::DiagnosticOptions &,
      llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem>)>;

private:
  clang::driver::Command &Command;
  clang::DiagnosticOptions &DiagOpts;
  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> VFS;
  ExecuteFnTy ExecuteImpl;

  // To avoid copies, store the output of execute, such that readExecuteOutput
  // can return a reference.
  std::unique_ptr<llvm::MemoryBuffer> Output;

public:
  ClangCommand(clang::driver::Command &Command,
               clang::DiagnosticOptions &DiagOpts,
               llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> VFS,
               ExecuteFnTy &&ExecuteImpl);

  bool canCache() const override;
  llvm::Error writeExecuteOutput(llvm::StringRef CachedBuffer) override;
  llvm::Expected<llvm::StringRef> readExecuteOutput() override;
  amd_comgr_status_t execute(llvm::raw_ostream &LogS) override;

  ~ClangCommand() override = default;

protected:
  ActionClass getClass() const override;
  void addOptionsIdentifier(HashAlgorithm &) const override;
  llvm::Error addInputIdentifier(HashAlgorithm &) const override;
};
} // namespace COMGR

#endif
