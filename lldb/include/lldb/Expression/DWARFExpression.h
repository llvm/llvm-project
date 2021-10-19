//===-- DWARFExpression.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_EXPRESSION_DWARFEXPRESSION_H
#define LLDB_EXPRESSION_DWARFEXPRESSION_H

#include "lldb/Core/Address.h"
#include "lldb/Core/Disassembler.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/Scalar.h"
#include "lldb/Utility/Status.h"
#include "lldb/lldb-private.h"
#include <functional>

class DWARFUnit;

namespace lldb_private {

/// \class DWARFExpression DWARFExpression.h
/// "lldb/Expression/DWARFExpression.h" Encapsulates a DWARF location
/// expression and interprets it.
///
/// DWARF location expressions are used in two ways by LLDB.  The first
/// use is to find entities specified in the debug information, since their
/// locations are specified in precisely this language.  The second is to
/// interpret expressions without having to run the target in cases where the
/// overhead from copying JIT-compiled code into the target is too high or
/// where the target cannot be run.  This class encapsulates a single DWARF
/// location expression or a location list and interprets it.
class DWARFExpression {
public:
  DWARFExpression();

  /// Constructor
  ///
  /// \param[in] data
  ///     A data extractor configured to read the DWARF location expression's
  ///     bytecode.
  DWARFExpression(lldb::ModuleSP module, const DataExtractor &data,
                  const DWARFUnit *dwarf_cu);

  /// Destructor
  virtual ~DWARFExpression();

  /// Print the description of the expression to a stream
  ///
  /// \param[in] s
  ///     The stream to print to.
  ///
  /// \param[in] level
  ///     The level of verbosity to use.
  ///
  /// \param[in] location_list_base_addr
  ///     If this is a location list based expression, this is the
  ///     address of the object that owns it. NOTE: this value is
  ///     different from the DWARF version of the location list base
  ///     address which is compile unit relative. This base address
  ///     is the address of the object that owns the location list.
  ///
  /// \param[in] abi
  ///     An optional ABI plug-in that can be used to resolve register
  ///     names.
  void GetDescription(Stream *s, lldb::DescriptionLevel level,
                      lldb::addr_t location_list_base_addr, ABI *abi) const;

  /// Return true if the location expression contains data
  bool IsValid() const;

  /// Return true if a location list was provided
  bool IsLocationList() const;

  /// Search for a load address in the location list
  ///
  /// \param[in] func_load_addr
  ///     The actual address of the function containing this location list.
  ///
  /// \param[in] addr
  ///     The address to resolve
  ///
  /// \return
  ///     True if IsLocationList() is true and the address was found;
  ///     false otherwise.
  //    bool
  //    LocationListContainsLoadAddress (Process* process, const Address &addr)
  //    const;
  //
  bool LocationListContainsAddress(lldb::addr_t func_load_addr,
                                   lldb::addr_t addr) const;

  /// If a location is not a location list, return true if the location
  /// contains a DW_OP_addr () opcode in the stream that matches \a file_addr.
  /// If file_addr is LLDB_INVALID_ADDRESS, the this function will return true
  /// if the variable there is any DW_OP_addr in a location that (yet still is
  /// NOT a location list). This helps us detect if a variable is a global or
  /// static variable since there is no other indication from DWARF debug
  /// info.
  ///
  /// \param[in] op_addr_idx
  ///     The DW_OP_addr index to retrieve in case there is more than
  ///     one DW_OP_addr opcode in the location byte stream.
  ///
  /// \param[out] error
  ///     If the location stream contains unknown DW_OP opcodes or the
  ///     data is missing, \a error will be set to \b true.
  ///
  /// \return
  ///     LLDB_INVALID_ADDRESS if the location doesn't contain a
  ///     DW_OP_addr for \a op_addr_idx, otherwise a valid file address
  lldb::addr_t GetLocation_DW_OP_addr(uint32_t op_addr_idx, bool &error) const;

  bool Update_DW_OP_addr(lldb::addr_t file_addr);

  void UpdateValue(uint64_t const_value, lldb::offset_t const_value_byte_size,
                   uint8_t addr_byte_size);

  void SetModule(const lldb::ModuleSP &module) { m_module_wp = module; }

  bool ContainsThreadLocalStorage() const;

  bool LinkThreadLocalStorage(
      lldb::ModuleSP new_module_sp,
      std::function<lldb::addr_t(lldb::addr_t file_addr)> const
          &link_address_callback);

  /// Tells the expression that it refers to a location list.
  ///
  /// \param[in] cu_file_addr
  ///     The base address to use for interpreting relative location list
  ///     entries.
  /// \param[in] func_file_addr
  ///     The file address of the function containing this location list. This
  ///     address will be used to relocate the location list on the fly (in
  ///     conjuction with the func_load_addr arguments).
  void SetLocationListAddresses(lldb::addr_t cu_file_addr,
                                lldb::addr_t func_file_addr);

  /// Return the call-frame-info style register kind
  int GetRegisterKind();

  /// Set the call-frame-info style register kind
  ///
  /// \param[in] reg_kind
  ///     The register kind.
  void SetRegisterKind(lldb::RegisterKind reg_kind);

  /// Wrapper for the static evaluate function that accepts an
  /// ExecutionContextScope instead of an ExecutionContext and uses member
  /// variables to populate many operands
  bool Evaluate(ExecutionContextScope *exe_scope, lldb::addr_t func_load_addr,
                const Value *initial_value_ptr, const Value *object_address_ptr,
                Value &result, Status *error_ptr) const;

  /// Wrapper for the static evaluate function that uses member variables to
  /// populate many operands
  bool Evaluate(ExecutionContext *exe_ctx, RegisterContext *reg_ctx,
                lldb::addr_t loclist_base_load_addr,
                const Value *initial_value_ptr, const Value *object_address_ptr,
                Value &result, Status *error_ptr) const;

  /// Evaluate a DWARF location expression in a particular context
  ///
  /// \param[in] exe_ctx
  ///     The execution context in which to evaluate the location
  ///     expression.  The location expression may access the target's
  ///     memory, especially if it comes from the expression parser.
  ///
  /// \param[in] opcode_ctx
  ///     The module which defined the expression.
  ///
  /// \param[in] opcodes
  ///     This is a static method so the opcodes need to be provided
  ///     explicitly.
  ///
  ///  \param[in] reg_ctx
  ///     An optional parameter which provides a RegisterContext for use
  ///     when evaluating the expression (i.e. for fetching register values).
  ///     Normally this will come from the ExecutionContext's StackFrame but
  ///     in the case where an expression needs to be evaluated while building
  ///     the stack frame list, this short-cut is available.
  ///
  /// \param[in] reg_set
  ///     The call-frame-info style register kind.
  ///
  /// \param[in] initial_value_ptr
  ///     A value to put on top of the interpreter stack before evaluating
  ///     the expression, if the expression is parametrized.  Can be NULL.
  ///
  /// \param[in] result
  ///     A value into which the result of evaluating the expression is
  ///     to be placed.
  ///
  /// \param[in] error_ptr
  ///     If non-NULL, used to report errors in expression evaluation.
  ///
  /// \return
  ///     True on success; false otherwise.  If error_ptr is non-NULL,
  ///     details of the failure are provided through it.
  static bool Evaluate(ExecutionContext *exe_ctx, RegisterContext *reg_ctx,
                       lldb::ModuleSP opcode_ctx, const DataExtractor &opcodes,
                       const DWARFUnit *dwarf_cu,
                       const lldb::RegisterKind reg_set,
                       const Value *initial_value_ptr,
                       const Value *object_address_ptr, Value &result,
                       Status *error_ptr);

  bool GetExpressionData(DataExtractor &data) const {
    data = m_data;
    return data.GetByteSize() > 0;
  }

  bool DumpLocationForAddress(Stream *s, lldb::DescriptionLevel level,
                              lldb::addr_t func_load_addr, lldb::addr_t address,
                              ABI *abi);

  bool MatchesOperand(StackFrame &frame, const Instruction::Operand &op);

  llvm::Optional<DataExtractor>
  GetLocationExpression(lldb::addr_t load_function_start,
                        lldb::addr_t addr) const;

private:
  /// Pretty-prints the location expression to a stream
  ///
  /// \param[in] s
  ///     The stream to use for pretty-printing.
  ///
  /// \param[in] data
  ///     The data extractor.
  ///
  /// \param[in] level
  ///     The level of detail to use in pretty-printing.
  ///
  /// \param[in] abi
  ///     An optional ABI plug-in that can be used to resolve register
  ///     names.
  void DumpLocation(Stream *s, const DataExtractor &data,
                    lldb::DescriptionLevel level, ABI *abi) const;

  /// Module which defined this expression.
  lldb::ModuleWP m_module_wp;

  /// A data extractor capable of reading opcode bytes
  DataExtractor m_data;

  /// The DWARF compile unit this expression belongs to. It is used to evaluate
  /// values indexing into the .debug_addr section (e.g. DW_OP_GNU_addr_index,
  /// DW_OP_GNU_const_index)
  const DWARFUnit *m_dwarf_cu = nullptr;

  /// One of the defines that starts with LLDB_REGKIND_
  lldb::RegisterKind m_reg_kind = lldb::eRegisterKindDWARF;

  struct LoclistAddresses {
    lldb::addr_t cu_file_addr;
    lldb::addr_t func_file_addr;
  };
  llvm::Optional<LoclistAddresses> m_loclist_addresses;
};

} // namespace lldb_private

#endif // LLDB_EXPRESSION_DWARFEXPRESSION_H
