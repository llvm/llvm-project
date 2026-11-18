//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_REGISTER_LAYOUTS_COMMON_REGISTERLAYOUT_X86_32_H
#define LLDB_REGISTER_LAYOUTS_COMMON_REGISTERLAYOUT_X86_32_H

#include "lldb/Register/RegisterDescription.h"
#include "lldb/Utility/Flags.h"
#include <utility>

namespace lldb_private {

/// Architectural descriptions for 32-bit x86 values and ABI roles.
class RegisterDescription_x86_32 : public RegisterDescription {
public:
  /// Select architectural values and ABI metadata without choosing storage.
  struct Config {
    enum class ABI { SystemV, Windows };
    enum {
      eRegsetMaskAliases = 1 << 0,
      eRegsetMaskSegments = 1 << 1,
      eRegsetMaskX87 = 1 << 2,
      eRegsetMaskSIMD = 1 << 3,
      eRegsetMaskAVX = 1 << 4,
      eRegsetMaskMPX = 1 << 5,
      eRegsetMaskDebug = 1 << 6,
      eRegsetMaskAll = eRegsetMaskAliases | eRegsetMaskSegments |
          eRegsetMaskX87 | eRegsetMaskSIMD | eRegsetMaskAVX | eRegsetMaskMPX |
          eRegsetMaskDebug
    };

    ABI abi = ABI::SystemV;
    Flags features = Flags(eRegsetMaskAll);
  };

  /// Apply optional maximum widths from the selected OS save format.
  explicit RegisterDescription_x86_32(
      const Config &config,
      llvm::ArrayRef<std::pair<RegisterID, uint32_t>> byte_sizes = {});
};

} // namespace lldb_private

#endif
