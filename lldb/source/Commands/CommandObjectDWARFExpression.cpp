//===-- CommandObjectDWARFExpression.cpp
//-----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Utility/RegisterValue.h"
#include "lldb/Utility/Status.h"
#include "lldb/lldb-private-types.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include <variant>

namespace lldb_private {

struct DWARF6Value {
  uint64_t IntegerLiteral;
};

struct DWARF6Location {
  enum Kind { MemoryAddress, Register } Type;
  uint64_t Payload;
};

using StackItem = std::variant<DWARF6Value, DWARF6Location>;

class DWARF6Evaluator {
public:
  std::vector<StackItem> Stack;

  Status Run(const std::vector<uint8_t> &opcodes, ExecutionContext &exe_ctx) {
    size_t pc = 0;
    RegisterContext *reg_ctx = exe_ctx.GetRegisterContext();

    while (pc < opcodes.size()) {
      uint8_t op = opcodes[pc++];
      switch (op) {
      case llvm::dwarf::DW_OP_lit0:
      case llvm::dwarf::DW_OP_lit1:
      case llvm::dwarf::DW_OP_lit2:
      case llvm::dwarf::DW_OP_lit3:
      case llvm::dwarf::DW_OP_lit4: {
        Stack.push_back(
            DWARF6Value{static_cast<uint64_t>(op - llvm::dwarf::DW_OP_lit0)});
        break;
      }

      case llvm::dwarf::DW_OP_reg0: {
        Stack.push_back(DWARF6Location{DWARF6Location::Register, 0});
        break;
      }

      case llvm::dwarf::DW_OP_deref: {
        if (Stack.empty())
          return Status::FromErrorString("Stack underflow on DW_OP_deref");

        StackItem top = Stack.back();
        Stack.pop_back();

        if (std::holds_alternative<DWARF6Location>(top)) {
          DWARF6Location loc = std::get<DWARF6Location>(top);
          if (loc.Type == DWARF6Location::Register) {
            if (!reg_ctx)
              return Status::FromErrorString("Missing Register Context");

            const RegisterInfo *reg_info =
                reg_ctx->GetRegisterInfoAtIndex(loc.Payload);
            if (!reg_info)
              return Status::FromErrorString("Failed to get Register Info");

            RegisterValue reg_val;
            if (!reg_ctx->ReadRegister(reg_info, reg_val))
              return Status::FromErrorString(
                  "Failed to read register value from frame");

            Stack.push_back(DWARF6Value{reg_val.GetAsUInt64()});
          } else if (loc.Type == DWARF6Location::MemoryAddress) {
            Process *process = exe_ctx.GetProcessPtr();
            if (!process)
              return Status::FromErrorString(
                  "No active target process context available");

            uint64_t mem_val = 0;
            Status error;
            process->ReadMemory(loc.Payload, &mem_val, sizeof(mem_val), error);

            Stack.push_back(DWARF6Value{mem_val});
          }
        } else {
          return Status::FromErrorString(
              "DWARF 6 Type Mismatch: Cannot dereference");
        }
        break;
      }

      case llvm::dwarf::DW_OP_plus: {
        if (Stack.size() < 2)
          return Status::FromErrorString("Stack underflow on DW_OP_plus");

        if (!std::holds_alternative<DWARF6Value>(Stack.back()))
          return Status::FromErrorString(
              "Right-hand operand must be a Value type");
        auto right = std::get<DWARF6Value>(Stack.back());
        Stack.pop_back();

        if (!std::holds_alternative<DWARF6Value>(Stack.back()))
          return Status::FromErrorString(
              "Left-hand operand must be a Value type");
        auto left = std::get<DWARF6Value>(Stack.back());
        Stack.pop_back();

        Stack.push_back(
            DWARF6Value{left.IntegerLiteral + right.IntegerLiteral});
        break;
      }

      default:
        return Status::FromErrorStringWithFormat(
            "Opcode 0x%02X is currently unimplemented", op);
      }
    }
    return Status();
  }
};
} // namespace lldb_private
