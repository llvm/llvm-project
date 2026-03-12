//===-- DumpRegisterValue.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/DumpRegisterValue.h"
#include "lldb/Core/DumpDataExtractor.h"
#include "lldb/DataFormatters/DumpValueObjectOptions.h"
#include "lldb/Target/RegisterTypeFlags.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/Endian.h"
#include "lldb/Utility/RegisterValue.h"
#include "lldb/Utility/StreamString.h"
#include "lldb/ValueObject/ValueObject.h"
#include "lldb/ValueObject/ValueObjectConstResult.h"
#include "lldb/lldb-private-types.h"
#include "llvm/ADT/bit.h"

using namespace lldb;

static void dump_type_value(lldb_private::CompilerType &fields_type,
                            lldb_private::RegisterValue reg_val,
                            const lldb_private::RegisterInfo &reg_info,
                            lldb_private::ExecutionContextScope *exe_scope,
                            lldb_private::Stream &strm) {
  auto heap_buf_sp =
      std::make_shared<lldb_private::DataBufferHeap>(reg_val.GetByteSize(), 0);
  lldb::ByteOrder target_order = exe_scope->CalculateProcess()->GetByteOrder();
  lldb_private::Status err;
  uint32_t wrote =
      reg_val.GetAsMemoryData(reg_info, heap_buf_sp->GetBytes(),
                              reg_val.GetByteSize(), target_order, err);
  if (wrote != reg_val.GetByteSize() || err.Fail())
    return;

  lldb_private::DataExtractor data_extractor(heap_buf_sp);
  data_extractor.SetByteOrder(target_order);
  lldb::ValueObjectSP vobj_sp = lldb_private::ValueObjectConstResult::Create(
      exe_scope, fields_type, lldb_private::ConstString(), data_extractor);
  lldb_private::DumpValueObjectOptions dump_options;
  dump_options.SetHideRootType(true);

  if (llvm::Error error = vobj_sp->Dump(strm, dump_options))
    strm << "error: " << toString(std::move(error));
}

void lldb_private::DumpRegisterValue(const RegisterValue &reg_val, Stream &s,
                                     const RegisterInfo &reg_info,
                                     bool prefix_with_name,
                                     bool prefix_with_alt_name, Format format,
                                     uint32_t reg_name_right_align_at,
                                     ExecutionContextScope *exe_scope,
                                     bool print_flags, TargetSP target_sp) {
  DataExtractor data;
  if (!reg_val.GetData(data))
    return;

  bool name_printed = false;
  // For simplicity, alignment of the register name printing applies only in
  // the most common case where:
  //
  //     prefix_with_name^prefix_with_alt_name is true
  //
  StreamString format_string;
  if (reg_name_right_align_at && (prefix_with_name ^ prefix_with_alt_name))
    format_string.Printf("%%%us", reg_name_right_align_at);
  else
    format_string.Printf("%%s");
  std::string fmt = std::string(format_string.GetString());
  if (prefix_with_name) {
    if (reg_info.name) {
      s.Printf(fmt.c_str(), reg_info.name);
      name_printed = true;
    } else if (reg_info.alt_name) {
      s.Printf(fmt.c_str(), reg_info.alt_name);
      prefix_with_alt_name = false;
      name_printed = true;
    }
  }
  if (prefix_with_alt_name) {
    if (name_printed)
      s.PutChar('/');
    if (reg_info.alt_name) {
      s.Printf(fmt.c_str(), reg_info.alt_name);
      name_printed = true;
    } else if (!name_printed) {
      // No alternate name but we were asked to display a name, so show the
      // main name
      s.Printf(fmt.c_str(), reg_info.name);
      name_printed = true;
    }
  }
  if (name_printed)
    s.PutCString(" = ");

  if (format == eFormatDefault)
    format = reg_info.format;

  DumpDataExtractor(data, &s,
                    0,                    // Offset in "data"
                    format,               // Format to use when dumping
                    reg_info.byte_size,   // item_byte_size
                    1,                    // item_count
                    UINT32_MAX,           // num_per_line
                    LLDB_INVALID_ADDRESS, // base_addr
                    0,                    // item_bit_size
                    0,                    // item_bit_offset
                    exe_scope);

  if (!print_flags || !reg_info.register_type || !exe_scope || !target_sp)
    return;

  CompilerType register_type =
      target_sp->GetRegisterType(*reg_info.register_type, reg_info.byte_size);
  if (!register_type.IsValid())
    return;

  // Use a new stream so we can remove a trailing newline later.
  StreamString register_type_stream;

  dump_type_value(register_type, reg_val, reg_info, exe_scope,
                  register_type_stream);

  // Registers are indented like:
  // (lldb) register read foo
  //     foo = 0x12345678
  // So we need to indent to match that.

  // First drop the extra newline that the value printer added. The register
  // command will add one itself.
  llvm::StringRef register_type_str =
      register_type_stream.GetString().drop_back();

  // End the line that contains "    foo = 0x12345678".
  s.EOL();

  // Then split the value lines and indent each one.
  bool first = true;
  while (register_type_str.size()) {
    std::pair<llvm::StringRef, llvm::StringRef> split =
        register_type_str.split('\n');
    register_type_str = split.second;
    // Indent as far as the register name did.
    s.Printf(fmt.c_str(), "");

    // Lines after the first won't have " = " so compensate for that.
    if (!first)
      s << "   ";
    first = false;

    s << split.first;

    // On the last line we don't want a newline because the command will add
    // one too.
    if (register_type_str.size())
      s.EOL();
  }
}
