//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_REGISTER_LAYOUTS_COMMON_REGISTERLAYOUT_PPC32BE_H
#define LLDB_REGISTER_LAYOUTS_COMMON_REGISTERLAYOUT_PPC32BE_H

#include "lldb/Register/Descriptions/RegisterDescription_ppc.h"

namespace lldb_private {

/// 32-bit PPC numbering/value profile, independent of the saved-state layout.
class RegisterDescription_ppc32be : public RegisterDescription_ppc {
public:
  RegisterDescription_ppc32be();

  static std::shared_ptr<const RegisterDescription> GetShared();
};

} // namespace lldb_private

#endif
