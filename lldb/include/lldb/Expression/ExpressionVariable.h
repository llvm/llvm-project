//===-- ExpressionVariable.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_EXPRESSION_EXPRESSIONVARIABLE_H
#define LLDB_EXPRESSION_EXPRESSIONVARIABLE_H

#include <memory>
#include <optional>
#include <vector>

#include "llvm/ADT/DenseMap.h"

#include "lldb/Utility/ConstString.h"
#include "lldb/ValueObject/ValueObject.h"
#include "lldb/lldb-public.h"
#include "llvm/Support/ExtensibleRTTI.h"

namespace lldb_private {

class ExpressionVariable
    : public std::enable_shared_from_this<ExpressionVariable>,
      public llvm::RTTIExtends<ExpressionVariable, llvm::RTTIRoot> {
public:
  /// LLVM RTTI support
  static char ID;

  ExpressionVariable();

  virtual ~ExpressionVariable() = default;

  llvm::Expected<uint64_t> GetByteSize() {
    return GetValueObject()->GetByteSize();
  }

  ConstString GetName() { return m_frozen_sp->GetName(); }

  lldb::ValueObjectSP GetValueObject() {
    lldb::ValueObjectSP dyn_sp =
        m_frozen_sp->GetDynamicValue(lldb::eDynamicDontRunTarget);
    if (dyn_sp && dyn_sp->UpdateValueIfNeeded())
      return dyn_sp;
    else
      return m_frozen_sp;
  }

  uint8_t *GetValueBytes();

  void ValueUpdated() { m_frozen_sp->ValueUpdated(); }

  RegisterInfo *GetRegisterInfo() {
    return m_frozen_sp->GetValue().GetRegisterInfo();
  }

  void SetRegisterInfo(const RegisterInfo *reg_info) {
    return m_frozen_sp->GetValue().SetContext(
        Value::ContextType::RegisterInfo, const_cast<RegisterInfo *>(reg_info));
  }

  CompilerType GetCompilerType() { return GetValueObject()->GetCompilerType(); }

  void SetCompilerType(const CompilerType &compiler_type) {
    m_frozen_sp->GetValue().SetCompilerType(compiler_type);
  }

  void SetName(ConstString name) { m_frozen_sp->SetName(name); }

  // This function is used to copy the address-of m_live_sp into m_frozen_sp.
  // This is necessary because the results of certain cast and pointer-
  // arithmetic operations (such as those described in bugzilla issues 11588
  // and 11618) generate frozen objects that do not have a valid address-of,
  // which can be troublesome when using synthetic children providers.
  // Transferring the address-of the live object solves these issues and
  // provides the expected user-level behavior.
  // The other job we do in TransferAddress is adjust the value in the live
  // address slot in the target for the "offset to top" in multiply inherited
  // class hierarchies.
  void TransferAddress(bool force = false);

  // When we build an expression variable we know whether we're going to use the
  // static or dynamic result.  If we present the dynamic value once, we should
  // use the dynamic value in future references to the variable, so we record
  // that fact here.
  void PreserveDynamicOption(lldb::DynamicValueType dyn_type) {
    m_dyn_option = dyn_type;
  }
  // We don't try to get the dynamic value of the live object when we fetch
  // it here.  The live object describes the container of the value in the
  // target, but it's type is of the object for convenience.  So it can't
  // produce the dynamic value.  Instead, we use TransferAddress to adjust the
  // value held by the LiveObject.
  lldb::ValueObjectSP GetLiveObject() { return m_live_sp; }

  enum Flags {
    EVNone = 0,
    EVIsLLDBAllocated = 1 << 0, ///< This variable is resident in a location
                                ///specifically allocated for it by LLDB in the
                                ///target process
    EVIsProgramReference = 1 << 1, ///< This variable is a reference to a
                                   ///(possibly invalid) area managed by the
                                   ///target program
    EVNeedsAllocation = 1 << 2,    ///< Space for this variable has yet to be
                                   ///allocated in the target process
    EVIsFreezeDried = 1 << 3, ///< This variable's authoritative version is in
                              ///m_frozen_sp (for example, for
                              ///statically-computed results)
    EVNeedsFreezeDry =
        1 << 4, ///< Copy from m_live_sp to m_frozen_sp during dematerialization
    EVKeepInTarget = 1 << 5, ///< Keep the allocation after the expression is
                             ///complete rather than freeze drying its contents
                             ///and freeing it
    EVTypeIsReference = 1 << 6, ///< The original type of this variable is a
                                ///reference, so materialize the value rather
                                ///than the location
    EVBareRegister = 1 << 7 ///< This variable is a direct reference to $pc or
                            ///some other entity.
  };

  typedef uint16_t FlagType;

  FlagType m_flags; // takes elements of Flags

  /// These members should be private.
  /// @{
  /// A value object whose value's data lives in host (lldb's) memory.
  /// The m_frozen_sp holds the data & type of the expression variable or result
  /// in the host.  The m_frozen_sp also can present a dynamic value if one is
  /// available.
  /// The m_frozen_sp manages the copy of this value in m_frozen_sp that we
  /// insert in the target so that it can be referred to in future expressions.
  /// We don't actually use the contents of the live_sp to create the value in
  /// the target, that comes from the frozen sp.  The live_sp is mostly to track
  /// the target-side of the value.
  lldb::ValueObjectSP m_frozen_sp;
  /// The ValueObject counterpart to m_frozen_sp that tracks the value in
  /// inferior memory. This object may not always exist; its presence depends on
  /// whether it is logical for the value to exist in the inferior memory. For
  /// example, when evaluating a C++ expression that generates an r-value, such
  /// as a single function call, there is no memory address in the inferior to
  /// track.
  lldb::ValueObjectSP m_live_sp;
  /// @}

  // LLVMCastKind m_kind;
  lldb::DynamicValueType m_dyn_option = lldb::eNoDynamicValues;
};

/// \class ExpressionVariableList ExpressionVariable.h
/// "lldb/Expression/ExpressionVariable.h"
/// A list of variable references.
///
/// This class stores variables internally, acting as the permanent store.
class ExpressionVariableList {
public:
  /// Implementation of methods in ExpressionVariableListBase
  size_t GetSize() { return m_variables.size(); }

  lldb::ExpressionVariableSP GetVariableAtIndex(size_t index) {
    lldb::ExpressionVariableSP var_sp;
    if (index < m_variables.size())
      var_sp = m_variables[index];
    return var_sp;
  }

  size_t AddVariable(const lldb::ExpressionVariableSP &var_sp) {
    m_variables.push_back(var_sp);
    return m_variables.size() - 1;
  }

  lldb::ExpressionVariableSP
  AddNewlyConstructedVariable(ExpressionVariable *var) {
    lldb::ExpressionVariableSP var_sp(var);
    m_variables.push_back(var_sp);
    return m_variables.back();
  }

  bool ContainsVariable(const lldb::ExpressionVariableSP &var_sp) {
    const size_t size = m_variables.size();
    for (size_t index = 0; index < size; ++index) {
      if (m_variables[index].get() == var_sp.get())
        return true;
    }
    return false;
  }

  /// Finds a variable by name in the list.
  ///
  /// \param[in] name
  ///     The name of the requested variable.
  ///
  /// \return
  ///     The variable requested, or nullptr if that variable is not in the
  ///     list.
  lldb::ExpressionVariableSP GetVariable(ConstString name) {
    lldb::ExpressionVariableSP var_sp;
    for (size_t index = 0, size = GetSize(); index < size; ++index) {
      var_sp = GetVariableAtIndex(index);
      if (var_sp->GetName() == name)
        return var_sp;
    }
    var_sp.reset();
    return var_sp;
  }

  lldb::ExpressionVariableSP GetVariable(llvm::StringRef name) {
    if (name.empty())
      return nullptr;

    for (size_t index = 0, size = GetSize(); index < size; ++index) {
      auto var_sp = GetVariableAtIndex(index);
      llvm::StringRef var_name_str = var_sp->GetName().GetStringRef();
      if (var_name_str == name)
        return var_sp;
    }
    return nullptr;
  }

  void RemoveVariable(lldb::ExpressionVariableSP var_sp) {
    for (std::vector<lldb::ExpressionVariableSP>::iterator
             vi = m_variables.begin(),
             ve = m_variables.end();
         vi != ve; ++vi) {
      if (vi->get() == var_sp.get()) {
        m_variables.erase(vi);
        return;
      }
    }
  }

  void Clear() { m_variables.clear(); }

private:
  std::vector<lldb::ExpressionVariableSP> m_variables;
};

class PersistentExpressionState
    : public ExpressionVariableList,
      public llvm::RTTIExtends<PersistentExpressionState, llvm::RTTIRoot> {
public:
  /// LLVM RTTI support
  static char ID;

  PersistentExpressionState();

  virtual ~PersistentExpressionState();

  virtual lldb::ExpressionVariableSP
  CreatePersistentVariable(const lldb::ValueObjectSP &valobj_sp) = 0;

  virtual lldb::ExpressionVariableSP
  CreatePersistentVariable(ExecutionContextScope *exe_scope,
                           ConstString name, const CompilerType &type,
                           lldb::ByteOrder byte_order,
                           uint32_t addr_byte_size) = 0;

  /// Return a new persistent variable name with the specified prefix.
  virtual ConstString GetNextPersistentVariableName(bool is_error = false) = 0;

  virtual void
  RemovePersistentVariable(lldb::ExpressionVariableSP variable) = 0;

  virtual std::optional<CompilerType>
  GetCompilerTypeFromPersistentDecl(ConstString type_name) = 0;

  virtual lldb::addr_t LookupSymbol(ConstString name);

  void RegisterExecutionUnit(lldb::IRExecutionUnitSP &execution_unit_sp);

protected:
  virtual llvm::StringRef
  GetPersistentVariablePrefix(bool is_error = false) const = 0;

private:
  typedef std::set<lldb::IRExecutionUnitSP> ExecutionUnitSet;
  ExecutionUnitSet
      m_execution_units; ///< The execution units that contain valuable symbols.

  typedef llvm::DenseMap<const char *, lldb::addr_t> SymbolMap;
  SymbolMap
      m_symbol_map; ///< The addresses of the symbols in m_execution_units.
};

} // namespace lldb_private

#endif // LLDB_EXPRESSION_EXPRESSIONVARIABLE_H
