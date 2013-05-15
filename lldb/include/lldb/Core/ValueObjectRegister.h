//===-- ValueObjectRegister.h -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ValueObjectRegister_h_
#define liblldb_ValueObjectRegister_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/lldb-private.h"
#include "lldb/Core/RegisterValue.h"
#include "lldb/Core/ValueObject.h"

namespace lldb_private {

//----------------------------------------------------------------------
// A ValueObject that contains a root variable that may or may not
// have children.
//----------------------------------------------------------------------
class ValueObjectRegisterContext : public ValueObject
{
public:

    virtual
    ~ValueObjectRegisterContext();

    virtual uint64_t
    GetByteSize();

    virtual lldb::ValueType
    GetValueType () const
    {
        return lldb::eValueTypeRegisterSet;
    }

    virtual ConstString
    GetTypeName();
    
    virtual ConstString
    GetQualifiedTypeName();

    virtual size_t
    CalculateNumChildren();

    virtual ValueObject *
    CreateChildAtIndex (size_t idx, bool synthetic_array_member, int32_t synthetic_index);

protected:
    virtual bool
    UpdateValue ();
    
    virtual clang::ASTContext *
    GetClangASTImpl ();
    
    virtual lldb::clang_type_t
    GetClangTypeImpl ();

    lldb::RegisterContextSP m_reg_ctx_sp;

private:
    ValueObjectRegisterContext (ValueObject &parent, lldb::RegisterContextSP &reg_ctx_sp);
    //------------------------------------------------------------------
    // For ValueObject only
    //------------------------------------------------------------------
    DISALLOW_COPY_AND_ASSIGN (ValueObjectRegisterContext);
};

class ValueObjectRegisterSet : public ValueObject
{
public:
    static lldb::ValueObjectSP
    Create (ExecutionContextScope *exe_scope, lldb::RegisterContextSP &reg_ctx_sp, uint32_t set_idx);

    virtual
    ~ValueObjectRegisterSet();

    virtual uint64_t
    GetByteSize();

    virtual lldb::ValueType
    GetValueType () const
    {
        return lldb::eValueTypeRegisterSet;
    }

    virtual ConstString
    GetTypeName();
    
    virtual ConstString
    GetQualifiedTypeName();

    virtual size_t
    CalculateNumChildren();

    virtual ValueObject *
    CreateChildAtIndex (size_t idx, bool synthetic_array_member, int32_t synthetic_index);
    
    virtual lldb::ValueObjectSP
    GetChildMemberWithName (const ConstString &name, bool can_create);

    virtual size_t
    GetIndexOfChildWithName (const ConstString &name);


protected:
    virtual bool
    UpdateValue ();
    
    virtual clang::ASTContext *
    GetClangASTImpl ();
    
    virtual lldb::clang_type_t
    GetClangTypeImpl ();

    lldb::RegisterContextSP m_reg_ctx_sp;
    const RegisterSet *m_reg_set;
    uint32_t m_reg_set_idx;

private:
    friend class ValueObjectRegisterContext;
    ValueObjectRegisterSet (ExecutionContextScope *exe_scope, lldb::RegisterContextSP &reg_ctx_sp, uint32_t set_idx);

    //------------------------------------------------------------------
    // For ValueObject only
    //------------------------------------------------------------------
    DISALLOW_COPY_AND_ASSIGN (ValueObjectRegisterSet);
};

class ValueObjectRegister : public ValueObject
{
public:
    static lldb::ValueObjectSP
    Create (ExecutionContextScope *exe_scope, lldb::RegisterContextSP &reg_ctx_sp, uint32_t reg_num);

    virtual
    ~ValueObjectRegister();

    virtual uint64_t
    GetByteSize();

    virtual lldb::ValueType
    GetValueType () const
    {
        return lldb::eValueTypeRegister;
    }

    virtual ConstString
    GetTypeName();

    virtual size_t
    CalculateNumChildren();
    
    virtual bool
    SetValueFromCString (const char *value_str, Error& error);
    
    virtual bool
    SetData (DataExtractor &data, Error &error);

    virtual bool
    ResolveValue (Scalar &scalar);
    
    virtual void
    GetExpressionPath (Stream &s, bool qualify_cxx_base_classes, GetExpressionPathFormat epformat = eGetExpressionPathFormatDereferencePointers);

protected:
    virtual bool
    UpdateValue ();
    
    virtual clang::ASTContext *
    GetClangASTImpl ();
    
    virtual lldb::clang_type_t
    GetClangTypeImpl ();

    lldb::RegisterContextSP m_reg_ctx_sp;
    RegisterInfo m_reg_info;
    RegisterValue m_reg_value;
    ConstString m_type_name;
    void *m_clang_type;

private:
    void
    ConstructObject (uint32_t reg_num);
    
    friend class ValueObjectRegisterSet;
    ValueObjectRegister (ValueObject &parent, lldb::RegisterContextSP &reg_ctx_sp, uint32_t reg_num);
    ValueObjectRegister (ExecutionContextScope *exe_scope, lldb::RegisterContextSP &reg_ctx_sp, uint32_t reg_num);

    //------------------------------------------------------------------
    // For ValueObject only
    //------------------------------------------------------------------
    DISALLOW_COPY_AND_ASSIGN (ValueObjectRegister);
};

} // namespace lldb_private

#endif  // liblldb_ValueObjectRegister_h_
