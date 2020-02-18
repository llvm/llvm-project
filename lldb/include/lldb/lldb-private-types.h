//===-- lldb-private-types.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_LLDB_PRIVATE_TYPES_H
#define LLDB_LLDB_PRIVATE_TYPES_H

#if defined(__cplusplus)

#include "lldb/lldb-private.h"

#include "llvm/ADT/ArrayRef.h"

namespace llvm {
namespace sys {
class DynamicLibrary;
}
}

namespace lldb_private {
class Platform;
class ExecutionContext;

typedef llvm::sys::DynamicLibrary (*LoadPluginCallbackType)(
    const lldb::DebuggerSP &debugger_sp, const FileSpec &spec, Status &error);

/// Every register is described in detail including its name, alternate name
/// (optional), encoding, size in bytes and the default display format.
struct RegisterInfo {
  /// Name of this register, can't be NULL.
  const char *name;
  /// Alternate name of this register, can be NULL.
  const char *alt_name;
  /// Size in bytes of the register.
  uint32_t byte_size;
  /// The byte offset in the register context data where this register's
  /// value is found.
  /// This is optional, and can be 0 if a particular RegisterContext does not
  /// need to address its registers by byte offset.
  uint32_t byte_offset;
  /// Encoding of the register bits.
  lldb::Encoding encoding;
  /// Default display format.
  lldb::Format format;
  /// Holds all of the various register numbers for all register kinds.
  uint32_t kinds[lldb::kNumRegisterKinds]; //
  /// List of registers (terminated with LLDB_INVALID_REGNUM). If this value is
  /// not null, all registers in this list will be read first, at which point
  /// the value for this register will be valid. For example, the value list
  /// for ah would be eax (x86) or rax (x64).
  uint32_t *value_regs; //
  /// List of registers (terminated with LLDB_INVALID_REGNUM). If this value is
  /// not null, all registers in this list will be invalidated when the value of
  /// this register changes. For example, the invalidate list for eax would be
  /// rax ax, ah, and al.
  uint32_t *invalidate_regs;
  /// A DWARF expression that when evaluated gives the byte size of this
  /// register.
  const uint8_t *dynamic_size_dwarf_expr_bytes;
  /// The length of the DWARF expression in bytes in the
  /// dynamic_size_dwarf_expr_bytes member.
  size_t dynamic_size_dwarf_len;

  llvm::ArrayRef<uint8_t> data(const uint8_t *context_base) const {
    return llvm::ArrayRef<uint8_t>(context_base + byte_offset, byte_size);
  }

  llvm::MutableArrayRef<uint8_t> mutable_data(uint8_t *context_base) const {
    return llvm::MutableArrayRef<uint8_t>(context_base + byte_offset,
                                          byte_size);
  }
};

/// Registers are grouped into register sets
struct RegisterSet {
  /// Name of this register set.
  const char *name;
  /// A short name for this register set.
  const char *short_name;
  /// The number of registers in REGISTERS array below.
  size_t num_registers;
  /// An array of register indices in this set. The values in this array are
  /// *indices* (not register numbers) into a particular RegisterContext's
  /// register array.  For example, if eax is defined at index 4 for a
  /// particular RegisterContext, eax would be included in this RegisterSet by
  /// adding the value 4.  Not by adding the value lldb_eax_i386.
  const uint32_t *registers;
};

struct OptionEnumValueElement {
  int64_t value;
  const char *string_value;
  const char *usage;
};

using OptionEnumValues = llvm::ArrayRef<OptionEnumValueElement>;

struct OptionValidator {
  virtual ~OptionValidator() {}
  virtual bool IsValid(Platform &platform,
                       const ExecutionContext &target) const = 0;
  virtual const char *ShortConditionString() const = 0;
  virtual const char *LongConditionString() const = 0;
};

struct OptionDefinition {
  /// Used to mark options that can be used together.  If
  /// `(1 << n & usage_mask) != 0` then this option belongs to option set n.
  uint32_t usage_mask;
  /// This option is required (in the current usage level).
  bool required;
  /// Full name for this option.
  const char *long_option;
  /// Single character for this option.
  int short_option;
  /// no_argument, required_argument or optional_argument
  int option_has_arg;
  /// If non-NULL, option is valid iff |validator->IsValid()|, otherwise
  /// always valid.
  OptionValidator *validator;
  /// If not empty, an array of enum values.
  OptionEnumValues enum_values;
  /// The kind of completion for this option.
  /// Contains values of the CommandCompletions::CommonCompletionTypes enum.
  uint32_t completion_type;
  /// Type of argument this option takes.
  lldb::CommandArgumentType argument_type;
  /// Full text explaining what this options does and what (if any) argument to
  /// pass it.
  const char *usage_text;
};

typedef struct type128 { uint64_t x[2]; } type128;
typedef struct type256 { uint64_t x[4]; } type256;

} // namespace lldb_private

#endif // #if defined(__cplusplus)

#endif // LLDB_LLDB_PRIVATE_TYPES_H
