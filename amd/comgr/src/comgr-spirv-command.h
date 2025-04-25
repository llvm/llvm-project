//===- comgr-spirv-command.h - SPIRVCommand implementation ----------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef COMGR_SPIRV_COMMAND_H
#define COMGR_SPIRV_COMMAND_H

#include "comgr-cache-command.h"
#include "comgr.h"

namespace COMGR {
class SPIRVCommand : public CachedCommandAdaptor {
public:
  llvm::StringRef InputBuffer;
  llvm::SmallVectorImpl<char> &OutputBuffer;

public:
  SPIRVCommand(DataObject *Input, llvm::SmallVectorImpl<char> &OutputBuffer)
      : InputBuffer(Input->Data, Input->Size), OutputBuffer(OutputBuffer) {}

  bool canCache() const final { return true; }
  llvm::Error writeExecuteOutput(llvm::StringRef CachedBuffer) final;
  llvm::Expected<llvm::StringRef> readExecuteOutput() final;
  amd_comgr_status_t execute(llvm::raw_ostream &LogS) final;

  ~SPIRVCommand() override = default;

protected:
  ActionClass getClass() const override;
  void addOptionsIdentifier(HashAlgorithm &) const override;
  llvm::Error addInputIdentifier(HashAlgorithm &) const override;
};
} // namespace COMGR

#endif
