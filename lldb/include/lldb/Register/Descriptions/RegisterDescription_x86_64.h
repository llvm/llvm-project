//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_REGISTER_LAYOUTS_COMMON_REGISTERLAYOUT_X86_64_H
#define LLDB_REGISTER_LAYOUTS_COMMON_REGISTERLAYOUT_X86_64_H

#include "lldb/Register/RegisterDescription.h"
#include "lldb/Utility/Flags.h"
#include <utility>

namespace lldb_private {

/// Architectural values, aliases, and ABI roles; saved buffers belong to
/// layouts.
class RegisterDescription_x86_64 : public RegisterDescription {
public:
  /// Select architectural values and ABI metadata without choosing storage.
  /// SIMD values are always part of the x86_64 description.
  struct Config {
    enum class ABI { SystemV, Windows };
    enum {
      eRegsetMaskAliases = 1 << 0,
      eRegsetMaskSegments = 1 << 1,
      eRegsetMaskX87 = 1 << 2,
      eRegsetMaskAVX = 1 << 3,
      eRegsetMaskMPX = 1 << 4,
      eRegsetMaskDebug = 1 << 5,
      eRegsetMaskAll = eRegsetMaskAliases | eRegsetMaskSegments |
          eRegsetMaskX87 | eRegsetMaskAVX | eRegsetMaskMPX | eRegsetMaskDebug
    };

    ABI abi = ABI::SystemV;
    Flags features = Flags(eRegsetMaskAll);

    bool segment_bases = false;
  };

  /// Apply optional maximum widths from the selected OS save format.
  explicit RegisterDescription_x86_64(
      const Config &config,
      llvm::ArrayRef<std::pair<RegisterID, uint32_t>> byte_sizes = {});
};

} // namespace lldb_private

#endif
