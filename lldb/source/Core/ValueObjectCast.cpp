//===-- ValueObjectDynamicValue.cpp ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


#include "lldb/Core/ValueObjectCast.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/Log.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ValueObjectList.h"
#include "lldb/Core/Value.h"
#include "lldb/Core/ValueObject.h"

#include "lldb/Symbol/ClangASTType.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Symbol/Type.h"
#include "lldb/Symbol/Variable.h"

#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/LanguageRuntime.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"

using namespace lldb_private;

lldb::ValueObjectSP
ValueObjectCast::Create (ValueObject &parent, 
                         const ConstString &name, 
                         const ClangASTType &cast_type)
{
    ValueObjectCast *cast_valobj_ptr = new ValueObjectCast (parent, name, cast_type);
    return cast_valobj_ptr->GetSP();
}

ValueObjectCast::ValueObjectCast
(
    ValueObject &parent, 
    const ConstString &name, 
    const ClangASTType &cast_type
) :
    ValueObject(parent),
    m_cast_type (cast_type)
{
    SetName (name);
    m_value.SetContext (Value::eContextTypeClangType, cast_type.GetOpaqueQualType());
}

ValueObjectCast::~ValueObjectCast()
{
}

lldb::clang_type_t
ValueObjectCast::GetClangTypeImpl ()
{
    return m_cast_type.GetOpaqueQualType();
}

size_t
ValueObjectCast::CalculateNumChildren()
{
    return ClangASTContext::GetNumChildren (GetClangAST (), GetClangType(), true);
}

clang::ASTContext *
ValueObjectCast::GetClangASTImpl ()
{
    return m_cast_type.GetASTContext();
}

uint64_t
ValueObjectCast::GetByteSize()
{
    return m_value.GetValueByteSize(GetClangAST(), NULL);
}

lldb::ValueType
ValueObjectCast::GetValueType() const
{
    // Let our parent answer global, local, argument, etc...
    return m_parent->GetValueType();
}

bool
ValueObjectCast::UpdateValue ()
{
    SetValueIsValid (false);
    m_error.Clear();
    
    if (m_parent->UpdateValueIfNeeded(false))
    {
        Value old_value(m_value);
        m_update_point.SetUpdated();
        m_value = m_parent->GetValue();
        m_value.SetContext (Value::eContextTypeClangType, GetClangType());
        SetAddressTypeOfChildren(m_parent->GetAddressTypeOfChildren());
        if (ClangASTContext::IsAggregateType (GetClangType()))
        {
            // this value object represents an aggregate type whose
            // children have values, but this object does not. So we
            // say we are changed if our location has changed.
            SetValueDidChange (m_value.GetValueType() != old_value.GetValueType() || m_value.GetScalar() != old_value.GetScalar());
        } 
        ExecutionContext exe_ctx (GetExecutionContextRef());
        m_error = m_value.GetValueAsData(&exe_ctx, GetClangAST(), m_data, 0, GetModule().get());
        SetValueDidChange (m_parent->GetValueDidChange());
        return true;
    }
    
    // The dynamic value failed to get an error, pass the error along
    if (m_error.Success() && m_parent->GetError().Fail())
        m_error = m_parent->GetError();
    SetValueIsValid (false);
    return false;
}

bool
ValueObjectCast::IsInScope ()
{
    return m_parent->IsInScope();
}
