//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_REGISTER_LAYOUTS_COMMON_REGISTERLAYOUT_PPC64BE_H
#define LLDB_REGISTER_LAYOUTS_COMMON_REGISTERLAYOUT_PPC64BE_H

#include "lldb/Register/Descriptions/RegisterDescription_ppc.h"

namespace lldb_private {

class RegisterDescription_ppc64be : public RegisterDescription_ppc {
public:
  /// Select external DWARF/EH-frame numbers, not logical IDs or storage.
  /// The PPC64 catalogs assign LR 108 for AIX and 65 for ELF.
  enum class Numbering { AIX, ELF };

  explicit RegisterDescription_ppc64be(Numbering numbering);

  static std::shared_ptr<const RegisterDescription>
  GetShared(Numbering numbering);
};

} // namespace lldb_private

#endif
