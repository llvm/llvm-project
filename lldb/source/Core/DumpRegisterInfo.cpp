//===-- DumpRegisterInfo.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/DumpRegisterInfo.h"
#include "lldb/Target/ThreadRegisterContext.h"
#include "lldb/Register/RegisterType.h"
#include "lldb/Register/RegisterTypeFlags.h"
#include "lldb/Utility/Stream.h"

#include "llvm/Support/Casting.h"

#include <cinttypes>

using namespace lldb;
using namespace lldb_private;

using SetInfo = std::pair<const char *, uint32_t>;

void lldb_private::DumpRegisterInfo(Stream &strm, ThreadRegisterContext &ctx,
                                    const RegisterInfo &info,
                                    uint32_t terminal_width) {
  std::vector<const char *> invalidates;
  if (info.invalidate_regs) {
    for (uint32_t *inv_regs = info.invalidate_regs;
         *inv_regs != LLDB_INVALID_REGNUM; ++inv_regs) {
      const RegisterInfo *inv_info =
          ctx.GetRegisterInfo(lldb::eRegisterKindLLDB, *inv_regs);
      assert(
          inv_info &&
          "Register invalidate list refers to a register that does not exist.");
      invalidates.push_back(inv_info->name);
    }
  }

  // We include the index here so that you can use it with "register read -s".
  std::vector<SetInfo> in_sets;
  for (uint32_t set_idx = 0; set_idx < ctx.GetRegisterSetCount(); ++set_idx) {
    const RegisterSet *set = ctx.GetRegisterSet(set_idx);
    assert(set && "Register set should be valid.");
    for (uint32_t reg_idx = 0; reg_idx < set->num_registers; ++reg_idx) {
      const RegisterInfo *set_reg_info =
          ctx.GetRegisterInfoAtIndex(set->registers[reg_idx]);
      assert(set_reg_info && "Register info should be valid.");

      if (set_reg_info == &info) {
        in_sets.push_back({set->name, set_idx});
        break;
      }
    }
  }

  std::vector<const char *> read_from;
  if (info.value_regs) {
    for (uint32_t *read_regs = info.value_regs;
         *read_regs != LLDB_INVALID_REGNUM; ++read_regs) {
      const RegisterInfo *read_info =
          ctx.GetRegisterInfo(lldb::eRegisterKindLLDB, *read_regs);
      assert(read_info && "Register value registers list refers to a register "
                          "that does not exist.");
      read_from.push_back(read_info->name);
    }
  }

  DoDumpRegisterInfo(strm, info.name, info.alt_name, info.byte_size,
                     invalidates, read_from, in_sets, info.register_type,
                     terminal_width);
}
