//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_REGISTER_LAYOUTS_COMMON_REGISTERLAYOUT_S390X_H
#define LLDB_REGISTER_LAYOUTS_COMMON_REGISTERLAYOUT_S390X_H

#include "lldb/Register/RegisterDescription.h"
#include <utility>

namespace lldb_private {

class RegisterDescription_s390x : public RegisterDescription {
public:
  RegisterDescription_s390x();

  static std::shared_ptr<const RegisterDescription> GetShared();
};

} // namespace lldb_private

#endif
