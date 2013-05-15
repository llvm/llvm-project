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
class ValueObjectVariable : public ValueObject
{
public:
    static lldb::ValueObjectSP
    Create (ExecutionContextScope *exe_scope, const lldb::VariableSP &var_sp);

    virtual
    ~ValueObjectVariable();

    virtual uint64_t
    GetByteSize();

    virtual ConstString
    GetTypeName();

    virtual ConstString
    GetQualifiedTypeName();

    virtual size_t
    CalculateNumChildren();

    virtual lldb::ValueType
    GetValueType() const;

    virtual bool
    IsInScope ();

    virtual lldb::ModuleSP
    GetModule();
    
    virtual SymbolContextScope *
    GetSymbolContextScope();

    virtual bool
    GetDeclaration (Declaration &decl);
    
    virtual const char *
    GetLocationAsCString ();
    
    virtual bool
    SetValueFromCString (const char *value_str, Error& error);

    virtual bool
    SetData (DataExtractor &data, Error &error);
    
protected:
    virtual bool
    UpdateValue ();
    
    virtual clang::ASTContext *
    GetClangASTImpl ();
    
    virtual lldb::clang_type_t
    GetClangTypeImpl ();

    lldb::VariableSP  m_variable_sp;  ///< The variable that this value object is based upon
    Value m_resolved_value;           ///< The value that DWARFExpression resolves this variable to before we patch it up

private:
    ValueObjectVariable (ExecutionContextScope *exe_scope, const lldb::VariableSP &var_sp);
    //------------------------------------------------------------------
    // For ValueObject only
    //------------------------------------------------------------------
    DISALLOW_COPY_AND_ASSIGN (ValueObjectVariable);
};

} // namespace lldb_private

#endif  // liblldb_ValueObjectVariable_h_
