//===-- ValueObjectVariable.h -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ValueObjectVariable_h_
#define liblldb_ValueObjectVariable_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/ValueObject.h"

namespace lldb_private {

//----------------------------------------------------------------------
// A ValueObject that contains a root variable that may or may not
// have children.
//----------------------------------------------------------------------
class ValueObjectVariable : public ValueObject {
public:
  ~ValueObjectVariable() override;

  static lldb::ValueObjectSP Create(ExecutionContextScope *exe_scope,
                                    const lldb::VariableSP &var_sp);

  uint64_t GetByteSize() override;

  ConstString GetTypeName() override;

  ConstString GetQualifiedTypeName() override;

  ConstString GetDisplayTypeName() override;

  size_t CalculateNumChildren(uint32_t max) override;

  lldb::ValueType GetValueType() const override;

  bool IsInScope() override;

  lldb::ModuleSP GetModule() override;

  SymbolContextScope *GetSymbolContextScope() override;

  bool GetDeclaration(Declaration &decl) override;

  const char *GetLocationAsCString() override;

  bool SetValueFromCString(const char *value_str, Error &error) override;

  bool SetData(DataExtractor &data, Error &error) override;

  virtual lldb::VariableSP GetVariable() override { return m_variable_sp; }

protected:
  bool UpdateValue() override;

  CompilerType GetCompilerTypeImpl() override;

  lldb::VariableSP
      m_variable_sp;      ///< The variable that this value object is based upon
  Value m_resolved_value; ///< The value that DWARFExpression resolves this
                          ///variable to before we patch it up

private:
  ValueObjectVariable(ExecutionContextScope *exe_scope,
                      const lldb::VariableSP &var_sp);
  //------------------------------------------------------------------
  // For ValueObject only
  //------------------------------------------------------------------
  DISALLOW_COPY_AND_ASSIGN(ValueObjectVariable);
};

} // namespace lldb_private

#endif // liblldb_ValueObjectVariable_h_
