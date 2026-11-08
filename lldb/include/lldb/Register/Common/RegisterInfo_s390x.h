//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_REGISTER_COMMON_REGISTERINFO_S390X_H
#define LLDB_REGISTER_COMMON_REGISTERINFO_S390X_H
#include "lldb/Register/RegisterDescription.h"

namespace lldb_private {

class RegisterInfo_s390x {
public:
  static std::shared_ptr<const RegisterDescription>
  CreateDescription();
};
} // namespace lldb_private

#endif
