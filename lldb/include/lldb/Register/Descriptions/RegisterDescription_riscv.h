//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_REGISTER_LAYOUTS_COMMON_REGISTERLAYOUT_RISCV_H
#define LLDB_REGISTER_LAYOUTS_COMMON_REGISTERLAYOUT_RISCV_H

#include "lldb/Register/RegisterDescription.h"
#include "lldb/Utility/Flags.h"
#include <optional>
#include <utility>

namespace lldb_private {

/// Common register groups and CSR construction for both RISC-V widths.
class RegisterDescription_riscv : public RegisterDescription {
public:
  enum {
    eRegsetMaskDefault = 0,
    eRegsetMaskFP = 1,
    eRegsetMaskVector = 2,
    eRegsetMaskAll = -1
  };

protected:
  RegisterDescription_riscv(llvm::ArrayRef<RegisterDescriptor> definitions,
                            llvm::ArrayRef<RegisterID> gpr,
                            llvm::ArrayRef<RegisterID> fpr,
                            llvm::ArrayRef<RegisterID> vector, Flags groups,
                            llvm::ArrayRef<uint32_t> csr_addresses,
                            uint32_t byte_size,
                            llvm::ArrayRef<std::string> extensions);

private:
  static std::optional<RegisterDescriptor>
  CreateCSRDescriptor(uint32_t address, uint32_t byte_size,
                      llvm::ArrayRef<std::string> features);
};

} // namespace lldb_private

#endif
