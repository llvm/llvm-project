//===-- lldb-private-types.h ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_lldb_private_types_h_
#define liblldb_lldb_private_types_h_

#if defined(__cplusplus)

#include "lldb/lldb-private.h"

namespace lldb_private
{
    //----------------------------------------------------------------------
    // Every register is described in detail including its name, alternate
    // name (optional), encoding, size in bytes and the default display
    // format.
    //----------------------------------------------------------------------
    typedef struct
    {
        const char *name;        // Name of this register, can't be NULL
        const char *alt_name;    // Alternate name of this register, can be NULL
        uint32_t byte_size;      // Size in bytes of the register
        uint32_t byte_offset;    // The byte offset in the register context data where this register's value is found
        lldb::Encoding encoding; // Encoding of the register bits
        lldb::Format format;     // Default display format
        uint32_t kinds[lldb::kNumRegisterKinds]; // Holds all of the various register numbers for all register kinds
        uint32_t *value_regs;    // List of registers that must be terminated with LLDB_INVALID_REGNUM
        uint32_t *invalidate_regs; // List of registers that must be invalidated when this register is modified, list must be terminated with LLDB_INVALID_REGNUM
    } RegisterInfo;

    //----------------------------------------------------------------------
    // Registers are grouped into register sets
    //----------------------------------------------------------------------
    typedef struct
    {
        const char *name;           // Name of this register set
        const char *short_name;     // A short name for this register set
        size_t num_registers;       // The number of registers in REGISTERS array below
        const uint32_t *registers;  // An array of register numbers in this set
    } RegisterSet;

    typedef struct
    {
        int64_t value;
        const char *string_value;
        const char *usage;
    } OptionEnumValueElement;
    
    typedef struct
    {
        uint32_t usage_mask;                     // Used to mark options that can be used together.  If (1 << n & usage_mask) != 0
                                                 // then this option belongs to option set n.
        bool required;                           // This option is required (in the current usage level)
        const char *long_option;                 // Full name for this option.
        int short_option;                        // Single character for this option.
        int option_has_arg;                      // no_argument, required_argument or optional_argument
        OptionEnumValueElement *enum_values;     // If non-NULL an array of enum values.
        uint32_t completion_type;                // Cookie the option class can use to do define the argument completion.
        lldb::CommandArgumentType argument_type; // Type of argument this option takes
        const char *usage_text;                  // Full text explaining what this options does and what (if any) argument to
                                                 // pass it.
    } OptionDefinition;

} // namespace lldb_private

#endif  // #if defined(__cplusplus)

#endif  // liblldb_lldb_private_types_h_
