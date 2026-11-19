//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_REGISTER_LAYOUTS_FREEBSD_REGISTERLAYOUTFREEBSD_PPC_H
#define LLDB_REGISTER_LAYOUTS_FREEBSD_REGISTERLAYOUTFREEBSD_PPC_H

#include "lldb/Register/RegisterLayout.h"
#include <optional>

namespace lldb_private {

class ArchSpec;

/// Common FreeBSD PPC save formats; native acquisition belongs to the plugin.
class RegisterLayoutFreeBSD_ppc final : public RegisterLayout {
public:
  struct VSX {
    // PT_GETVSRREGS supplies doubleword 1 of VSR0-31.
    uint64_t extra[32];
  };

  static_assert(sizeof(VSX) == 256);

  /// Use the target's GPR slot width unless the tracer supplies an override.
  /// A 64-bit tracer stores 32-bit GPR values in the low half of each slot.
  explicit RegisterLayoutFreeBSD_ppc(
      const ArchSpec &arch, bool has_vsx = false,
      std::optional<uint32_t> gpr_slot_size = std::nullopt);

  RegisterLayoutFreeBSD_ppc() = delete;
  RegisterLayoutFreeBSD_ppc(const RegisterLayoutFreeBSD_ppc &) = delete;
  RegisterLayoutFreeBSD_ppc(RegisterLayoutFreeBSD_ppc &&) = delete;
};

} // namespace lldb_private

#endif
