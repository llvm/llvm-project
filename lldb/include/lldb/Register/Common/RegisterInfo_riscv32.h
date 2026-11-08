//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_REGISTER_COMMON_REGISTERINFO_RISCV32_H
#define LLDB_REGISTER_COMMON_REGISTERINFO_RISCV32_H
#include "lldb/Register/RegisterDescription.h"
#include "lldb/Utility/Flags.h"
#include "llvm/ADT/SmallVector.h"

namespace lldb_private {

class RegisterInfo_riscv32 {
public:
  enum { eRegsetMaskDefault = 0, eRegsetMaskFP = 1, eRegsetMaskAll = -1 };
  static std::shared_ptr<const RegisterDescription>
  CreateDescription();

  static void BuildCSRegInfos(
      llvm::ArrayRef<std::string> features,
      llvm::SmallVectorImpl<RegisterDescriptor> &infos);

private:
  static void ConfigureCSRegInfos(
      llvm::StringRef feature,
      llvm::SmallVectorImpl<RegisterDescriptor> &infos);
};
} // namespace lldb_private

#endif
