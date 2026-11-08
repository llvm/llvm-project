//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_REGISTER_COMMON_REGISTERINFO_PPC64_H
#define LLDB_REGISTER_COMMON_REGISTERINFO_PPC64_H
#include "lldb/Register/RegisterDescription.h"
#include "lldb/Utility/ArchSpec.h"
namespace lldb_private {

class RegisterInfo_ppc64 {
public:
  static std::shared_ptr<const RegisterDescription>
  CreateDescription(const ArchSpec &arch);
};
} // namespace lldb_private

#endif
