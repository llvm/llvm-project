//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_REGISTER_LAYOUTS_COMMON_REGISTERLAYOUT_RISCV32_H
#define LLDB_REGISTER_LAYOUTS_COMMON_REGISTERLAYOUT_RISCV32_H

#include "lldb/Register/Descriptions/RegisterDescription_riscv.h"
#include "lldb/Utility/Flags.h"

namespace lldb_private {

class RegisterDescription_riscv32 : public RegisterDescription_riscv {
public:
  /// Architectural groups and saved CSR addresses requested by a consumer.
  /// The borrowed arrays are consumed during construction; descriptions own all
  /// names and values. Invalid addresses are omitted and repeated addresses are
  /// described once, in request order. No storage offsets are selected here.
  struct Config {
    Flags features =
        Flags(1); // Floating-point registers are enabled by default.
    llvm::ArrayRef<uint32_t> csr_addresses = {};
    llvm::ArrayRef<std::string> extensions = {};
  };

  explicit RegisterDescription_riscv32(const Config &config);

  /// Describe the selected groups and CSRs through the ordinary lookup API.
  static std::shared_ptr<const RegisterDescription>
  GetShared(const Config &config);
};

} // namespace lldb_private

#endif
