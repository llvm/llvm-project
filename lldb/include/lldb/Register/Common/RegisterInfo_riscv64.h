//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_REGISTER_COMMON_REGISTERINFO_RISCV64_H
#define LLDB_REGISTER_COMMON_REGISTERINFO_RISCV64_H

#include "lldb/Register/RegisterDescription.h"
#include "lldb/Utility/Flags.h"

namespace lldb_private {

class RegisterInfo_riscv64 {
public:
  enum { GPRegSet = 0, FPRegSet = 1 };
  enum { eRegsetMaskDefault = 0, eRegsetMaskFP = 1, eRegsetMaskAll = -1 };

  static std::shared_ptr<const RegisterDescription>
  CreateDescription(Flags features);
};

} // namespace lldb_private

#endif
