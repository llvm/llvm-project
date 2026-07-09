//===- comgr-package-command.h - UnpackageCommand implementation ----------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef COMGR_PACKAGER_COMMAND_H
#define COMGR_PACKAGER_COMMAND_H

#include "amd_comgr.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/Object/OffloadBinary.h>

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace COMGR {
class UnpackageCommand {
private:
  const llvm::SmallVector<llvm::object::OffloadFile> &Files;
  const llvm::SmallVector<llvm::object::OffloadFile::TargetID> &TargetIDs;
  const llvm::SmallVector<std::string> &OutputFileNames;

public:
  UnpackageCommand(
      const llvm::SmallVector<llvm::object::OffloadFile> &Files,
      const llvm::SmallVector<llvm::object::OffloadFile::TargetID> &TargetIDs,
      const llvm::SmallVector<std::string> &OutputFileNames)
      : Files(Files), TargetIDs(TargetIDs), OutputFileNames(OutputFileNames) {}

  amd_comgr_status_t execute(llvm::raw_ostream &LogS);
};
} // namespace COMGR

#endif
