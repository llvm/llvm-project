//===- comgr-unbundle-command.h - UnbundleCommand implementation ----------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef COMGR_BUNDLER_COMMAND_H
#define COMGR_BUNDLER_COMMAND_H

#include <comgr-cache-command.h>

namespace clang {
class OffloadBundlerConfig;
} // namespace clang

namespace COMGR {
class UnbundleCommand final : public CachedCommandAdaptor {
private:
  amd_comgr_data_kind_t Kind;
  const clang::OffloadBundlerConfig &Config;

  // To avoid copies, store the output of execute, such that readExecuteOutput
  // can return a reference.
  llvm::SmallString<64> OutputBuffer;

public:
  UnbundleCommand(amd_comgr_data_kind_t Kind,
                  const clang::OffloadBundlerConfig &Config)
      : Kind(Kind), Config(Config) {}

  bool canCache() const override;
  llvm::Error writeExecuteOutput(llvm::StringRef CachedBuffer) override;
  llvm::Expected<llvm::StringRef> readExecuteOutput() override;
  amd_comgr_status_t execute(llvm::raw_ostream &LogS) override;

  ~UnbundleCommand() override = default;

protected:
  ActionClass getClass() const override;
  void addOptionsIdentifier(HashAlgorithm &) const override;
  llvm::Error addInputIdentifier(HashAlgorithm &) const override;
};
} // namespace COMGR

#endif
