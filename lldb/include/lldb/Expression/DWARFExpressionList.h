//===-- DWARFExpressionList.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_EXPRESSION_DWARFEXPRESSIONLIST_H
#define LLDB_EXPRESSION_DWARFEXPRESSIONLIST_H

#include "lldb/Expression/DWARFExpression.h"
#include "lldb/Utility/RangeMap.h"
#include "lldb/lldb-private.h"
#include "llvm/ADT/Optional.h"

class DWARFUnit;

namespace lldb_private {

/// \class DWARFExpressionList DWARFExpressionList.h
/// "lldb/Expression/DWARFExpressionList.h" Encapsulates a range map from file
/// address range to a single DWARF location expression.
class DWARFExpressionList {
public:
  DWARFExpressionList() = default;

  DWARFExpressionList(lldb::ModuleSP module_sp, const DWARFUnit *dwarf_cu)
      : m_module_wp(module_sp), m_dwarf_cu(dwarf_cu) {}

  DWARFExpressionList(lldb::ModuleSP module_sp, DWARFExpression expr,
                      const DWARFUnit *dwarf_cu)
      : DWARFExpressionList(module_sp, dwarf_cu) {
    AddExpression(0, LLDB_INVALID_ADDRESS, expr);
  }

  /// Return true if the location expression contains data
  bool IsValid() const { return !m_exprs.IsEmpty(); }

  void Clear() { m_exprs.Clear(); }

  // Return true if the location expression is always valid.
  bool IsAlwaysValidSingleExpr() const;

  bool AddExpression(lldb::addr_t base, lldb::addr_t end, DWARFExpression expr);

  /// Get the expression data at the file address.
  bool GetExpressionData(DataExtractor &data, lldb::addr_t file_addr = 0) const;

  /// Sort m_expressions.
  void Sort() { m_exprs.Sort(); }

  const DWARFExpression *GetExpressionAtAddress(lldb::addr_t file_addr) const;

  const DWARFExpression *GetAlwaysValidExpr() const {
    return GetExpressionAtAddress(0);
  }

  DWARFExpression *GetMutableExpressionAtAddress(lldb::addr_t file_addr = 0);

  size_t GetSize() const { return m_exprs.GetSize(); }

  bool ContainsThreadLocalStorage() const;

  bool LinkThreadLocalStorage(
      lldb::ModuleSP new_module_sp,
      std::function<lldb::addr_t(lldb::addr_t file_addr)> const
          &link_address_callback);

  bool MatchesOperand(StackFrame &frame,
                      const Instruction::Operand &operand) const;

  /// Dump locations that contains file_addr if it's valid. Otherwise. dump all
  /// locations.
  bool DumpLocations(Stream *s, lldb::DescriptionLevel level,
                     lldb::addr_t file_addr, ABI *abi) const;

  /// Dump all locaitons with each seperated by new line.
  void GetDescription(Stream *s, lldb::DescriptionLevel level, ABI *abi) const;

  /// Search for a load address in the location list
  ///
  /// \param[in] file_addr
  ///     The file address to resolve
  ///
  /// \return
  ///     True if IsLocationList() is true and the address was found;
  ///     false otherwise.
  //    bool
  //    LocationListContainsLoadAddress (Process* process, const Address &addr)
  //    const;
  //
  bool ContainsAddress(lldb::addr_t file_addr) const;

  void SetModule(const lldb::ModuleSP &module) { m_module_wp = module; }

  bool Evaluate(ExecutionContext *exe_ctx, RegisterContext *reg_ctx,
                const Value *initial_value_ptr, const Value *object_address_ptr,
                Value &result, Status *error_ptr) const;

private:
  // RangeDataVector requires a comparator for DWARFExpression, but it doesn't
  // make sense to do so.
  struct DWARFExpressionCompare {
  public:
    bool operator()(const DWARFExpression &lhs,
                    const DWARFExpression &rhs) const {
      return false;
    }
  };
  using ExprVec = RangeDataVector<lldb::addr_t, lldb::addr_t, DWARFExpression,
                                  0, DWARFExpressionCompare>;
  using Entry = ExprVec::Entry;

  // File address range mapping to single dwarf expression.
  ExprVec m_exprs;

  /// Module which defined this expression.
  lldb::ModuleWP m_module_wp;

  /// The DWARF compile unit this expression belongs to. It is used to evaluate
  /// values indexing into the .debug_addr section (e.g. DW_OP_GNU_addr_index,
  /// DW_OP_GNU_const_index)
  const DWARFUnit *m_dwarf_cu = nullptr;

  using const_iterator = ExprVec::Collection::const_iterator;
  const_iterator begin() const { return m_exprs.begin(); }
  const_iterator end() const { return m_exprs.end(); }
};
} // namespace lldb_private

#endif // LLDB_EXPRESSION_DWARFEXPRESSIONLIST_H
