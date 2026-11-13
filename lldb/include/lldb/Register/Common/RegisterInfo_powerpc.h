//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_REGISTER_COMMON_REGISTERINFO_POWERPC_H
#define LLDB_REGISTER_COMMON_REGISTERINFO_POWERPC_H
#include "lldb/Register/RegisterDescription.h"
/// ELF PowerPC numbering/value profile, independent of the saved-state layout.
namespace lldb_private {

class RegisterInfo_powerpc {
public:
  static std::shared_ptr<const RegisterDescription>
  CreateDescription(uint32_t value_bytes);
};
} // namespace lldb_private

#endif
