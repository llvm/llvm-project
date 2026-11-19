//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_REGISTER_LAYOUTS_FREEBSD_REGISTERLAYOUTFREEBSD_ARM32_H
#define LLDB_REGISTER_LAYOUTS_FREEBSD_REGISTERLAYOUTFREEBSD_ARM32_H

#include "lldb/Register/RegisterLayout.h"
#include "lldb/Utility/ArchSpec.h"

namespace lldb_private {

class RegisterLayoutFreeBSD_arm32 final : public RegisterLayout {
public:
  enum Bank : RegisterBankID { GP, FP, TLSBank };

  struct GPR {
    uint32_t r[16]; // R0-R15
    uint32_t cpsr;  // CPSR
  };

  struct QReg {
    uint8_t bytes[16];
  };

  struct FPU {
    union {
      uint32_t s[32];
      uint64_t d[32];
      QReg q[16]; // the 128-bit NEON registers
    } floats;
    uint32_t fpscr;
  };

  struct TLS {
    uint32_t tpidruro;
  };

  explicit RegisterLayoutFreeBSD_arm32(const ArchSpec &arch, bool tls = false);

  RegisterLayoutFreeBSD_arm32() = delete;
  RegisterLayoutFreeBSD_arm32(const RegisterLayoutFreeBSD_arm32 &) = delete;
  RegisterLayoutFreeBSD_arm32(RegisterLayoutFreeBSD_arm32 &&) = delete;
};

} // namespace lldb_private

#endif
