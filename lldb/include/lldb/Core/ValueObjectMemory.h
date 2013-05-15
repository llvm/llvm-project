//===-- ValueObjectMemory.h -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ValueObjectMemory_h_
#define liblldb_ValueObjectMemory_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/ValueObject.h"
#include "lldb/Symbol/ClangASTType.h"

namespace lldb_private {

//----------------------------------------------------------------------
// A ValueObject that represents memory at a given address, viewed as some 
// set lldb type.
//----------------------------------------------------------------------
class ValueObjectMemory : public ValueObject
{
public:
    static lldb::ValueObjectSP
    Create (ExecutionContextScope *exe_scope, 
            const char *name,
            const Address &address, 
            lldb::TypeSP &type_sp);

    static lldb::ValueObjectSP
    Create (ExecutionContextScope *exe_scope, 
            const char *name,
            const Address &address, 
            const ClangASTType &ast_type);

    virtual
    ~ValueObjectMemory();

    virtual uint64_t
    GetByteSize();

    virtual ConstString
    GetTypeName();

    virtual size_t
    CalculateNumChildren();

    virtual lldb::ValueType
    GetValueType() const;

    virtual bool
    IsInScope ();

    virtual lldb::ModuleSP
    GetModule();

protected:
    virtual bool
    UpdateValue ();
    
    virtual clang::ASTContext *
    GetClangASTImpl ();
    
    virtual lldb::clang_type_t
    GetClangTypeImpl ();

    Address  m_address;  ///< The variable that this value object is based upon
    lldb::TypeSP m_type_sp;
    ClangASTType m_clang_type;

private:
    ValueObjectMemory (ExecutionContextScope *exe_scope, 
                       const char *name,
                       const Address &address, 
                       lldb::TypeSP &type_sp);

    ValueObjectMemory (ExecutionContextScope *exe_scope,
                       const char *name, 
                       const Address &address,
                       const ClangASTType &ast_type);
    //------------------------------------------------------------------
    // For ValueObject only
    //------------------------------------------------------------------
    DISALLOW_COPY_AND_ASSIGN (ValueObjectMemory);
};

} // namespace lldb_private

#endif  // liblldb_ValueObjectMemory_h_
