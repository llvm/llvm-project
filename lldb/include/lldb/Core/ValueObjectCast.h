//===-- ValueObjectDynamicValue.h -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ValueObjectCast_h_
#define liblldb_ValueObjectCast_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/ValueObject.h"

namespace lldb_private {

//---------------------------------------------------------------------------------
// A ValueObject that represents a given value represented as a different type.
//---------------------------------------------------------------------------------
class ValueObjectCast : public ValueObject
{
public:
    static lldb::ValueObjectSP
    Create (ValueObject &parent, 
            const ConstString &name, 
            const ClangASTType &cast_type);

    virtual
    ~ValueObjectCast();
    
    virtual size_t
    GetByteSize();
    
    virtual uint32_t
    CalculateNumChildren();
    
    virtual lldb::ValueType
    GetValueType() const;
    
    virtual bool
    IsInScope ();
    
    virtual ValueObject *
    GetParent()
    {
        if (m_parent)
            return m_parent->GetParent();
        else
            return NULL;
    }
    
    virtual const ValueObject *
    GetParent() const
    {
        if (m_parent)
            return m_parent->GetParent();
        else
            return NULL;
    }
    
protected:
    virtual bool
    UpdateValue ();
    
    virtual clang::ASTContext *
    GetClangASTImpl ();
    
    virtual lldb::clang_type_t
    GetClangTypeImpl ();
    
    ClangASTType m_cast_type;
    
private:
    ValueObjectCast (ValueObject &parent, 
                     const ConstString &name, 
                     const ClangASTType &cast_type);
    
    //------------------------------------------------------------------
    // For ValueObject only
    //------------------------------------------------------------------
    DISALLOW_COPY_AND_ASSIGN (ValueObjectCast);
};

} // namespace lldb_private

#endif  // liblldb_ValueObjectCast_h_
