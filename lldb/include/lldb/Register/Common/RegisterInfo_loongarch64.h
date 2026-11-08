//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_REGISTER_COMMON_REGISTERINFO_LOONGARCH64_H
#define LLDB_REGISTER_COMMON_REGISTERINFO_LOONGARCH64_H
#include "lldb/Register/RegisterDescription.h"

namespace lldb_private {

class RegisterInfo_loongarch64 {
public:
  enum RegSetKind { GPRegSet, FPRegSet, LSXRegSet, LASXRegSet };
  static std::shared_ptr<const RegisterDescription>
  CreateDescription();
};
} // namespace lldb_private

#endif
