//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_REGISTER_LAYOUTS_COMMON_REGISTERLAYOUT_PPC_H
#define LLDB_REGISTER_LAYOUTS_COMMON_REGISTERLAYOUT_PPC_H

#include "lldb/Register/RegisterDescription.h"
#include <utility>

namespace lldb_private {

/// Common group construction and explicit external numbering for PPC variants.
class RegisterDescription_ppc : public RegisterDescription {
protected:
  using NumberingMap =
      llvm::ArrayRef<std::pair<RegisterID, RegisterDescriptor::Numbering>>;

  static RegisterDescription CreateDescription(
      llvm::ArrayRef<RegisterDescriptor> definitions,
      llvm::ArrayRef<RegisterID> gpr, llvm::ArrayRef<RegisterID> fpr,
      llvm::ArrayRef<RegisterID> vmx, llvm::ArrayRef<RegisterID> vsx = {},
      NumberingMap numbering = {});
};

} // namespace lldb_private

#endif
