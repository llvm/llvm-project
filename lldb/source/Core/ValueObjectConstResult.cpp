//===-- ValueObjectConstResult.cpp ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/ValueObjectConstResult.h"

#include "lldb/Core/ValueObjectChild.h"
#include "lldb/Core/ValueObjectConstResultChild.h"
#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ValueObjectDynamicValue.h"
#include "lldb/Core/ValueObjectList.h"

#include "lldb/Symbol/ClangASTType.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Symbol/Type.h"
#include "lldb/Symbol/Variable.h"

#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"

using namespace lldb;
using namespace lldb_private;

ValueObjectSP
ValueObjectConstResult::Create
(
    ExecutionContextScope *exe_scope,
    ByteOrder byte_order, 
     uint32_t addr_byte_size,
     lldb::addr_t address
)
{
    return (new ValueObjectConstResult (exe_scope,
                                        byte_order,
                                        addr_byte_size,
                                        address))->GetSP();
}

ValueObjectConstResult::ValueObjectConstResult
(
    ExecutionContextScope *exe_scope,
    ByteOrder byte_order, 
    uint32_t addr_byte_size,
    lldb::addr_t address
) :
    ValueObject (exe_scope),
    m_clang_ast (NULL),
    m_type_name (),
    m_byte_size (0),
    m_impl(this, address)
{
    SetIsConstant ();
    SetValueIsValid(true);
    m_data.SetByteOrder(byte_order);
    m_data.SetAddressByteSize(addr_byte_size);
    SetAddressTypeOfChildren(eAddressTypeLoad);
}

ValueObjectSP
ValueObjectConstResult::Create
(
    ExecutionContextScope *exe_scope,
    clang::ASTContext *clang_ast,
    void *clang_type,
    const ConstString &name,
    const DataExtractor &data,
    lldb::addr_t address
)
{
    return (new ValueObjectConstResult (exe_scope,
                                        clang_ast,
                                        clang_type,
                                        name,
                                        data,
                                        address))->GetSP();
}

ValueObjectConstResult::ValueObjectConstResult
(
    ExecutionContextScope *exe_scope,
    clang::ASTContext *clang_ast,
    void *clang_type,
    const ConstString &name,
    const DataExtractor &data,
    lldb::addr_t address
) :
    ValueObject (exe_scope),
    m_clang_ast (clang_ast),
    m_type_name (),
    m_byte_size (0),
    m_impl(this, address)
{
    m_data = data;
    
    if (!m_data.GetSharedDataBuffer())
    {
        DataBufferSP shared_data_buffer(new DataBufferHeap(data.GetDataStart(), data.GetByteSize()));
        m_data.SetData(shared_data_buffer);
    }
    
    m_value.GetScalar() = (uintptr_t)m_data.GetDataStart();
    m_value.SetValueType(Value::eValueTypeHostAddress);
    m_value.SetContext(Value::eContextTypeClangType, clang_type);
    m_name = name;
    SetIsConstant ();
    SetValueIsValid(true);
    SetAddressTypeOfChildren(eAddressTypeLoad);
}

ValueObjectSP
ValueObjectConstResult::Create
(
    ExecutionContextScope *exe_scope,
    clang::ASTContext *clang_ast,
    void *clang_type,
    const ConstString &name,
    const lldb::DataBufferSP &data_sp,
    lldb::ByteOrder data_byte_order, 
    uint32_t data_addr_size,
    lldb::addr_t address
)
{
    return (new ValueObjectConstResult (exe_scope,
                                        clang_ast,
                                        clang_type,
                                        name,
                                        data_sp,
                                        data_byte_order,
                                        data_addr_size,
                                        address))->GetSP();
}

ValueObjectSP
ValueObjectConstResult::Create (ExecutionContextScope *exe_scope,
        clang::ASTContext *clang_ast,
        Value &value,
        const ConstString &name)
{
    return (new ValueObjectConstResult (exe_scope, clang_ast, value, name))->GetSP();
}

ValueObjectConstResult::ValueObjectConstResult
(
    ExecutionContextScope *exe_scope,
    clang::ASTContext *clang_ast,
    void *clang_type,
    const ConstString &name,
    const lldb::DataBufferSP &data_sp,
    lldb::ByteOrder data_byte_order, 
    uint32_t data_addr_size,
    lldb::addr_t address
) :
    ValueObject (exe_scope),
    m_clang_ast (clang_ast),
    m_type_name (),
    m_byte_size (0),
    m_impl(this, address)
{
    m_data.SetByteOrder(data_byte_order);
    m_data.SetAddressByteSize(data_addr_size);
    m_data.SetData(data_sp);
    m_value.GetScalar() = (uintptr_t)data_sp->GetBytes();
    m_value.SetValueType(Value::eValueTypeHostAddress);
    m_value.SetContext(Value::eContextTypeClangType, clang_type);
    m_name = name;
    SetIsConstant ();
    SetValueIsValid(true);
    SetAddressTypeOfChildren(eAddressTypeLoad);
}

ValueObjectSP
ValueObjectConstResult::Create
(
    ExecutionContextScope *exe_scope,
    clang::ASTContext *clang_ast,
    void *clang_type,
    const ConstString &name,
    lldb::addr_t address,
    AddressType address_type,
    uint32_t addr_byte_size
)
{
    return (new ValueObjectConstResult (exe_scope,
                                        clang_ast,
                                        clang_type,
                                        name,
                                        address,
                                        address_type,
                                        addr_byte_size))->GetSP();
}

ValueObjectConstResult::ValueObjectConstResult 
(
    ExecutionContextScope *exe_scope,
    clang::ASTContext *clang_ast,
    void *clang_type,
    const ConstString &name,
    lldb::addr_t address,
    AddressType address_type,
    uint32_t addr_byte_size
) :
    ValueObject (exe_scope),
    m_clang_ast (clang_ast),
    m_type_name (),
    m_byte_size (0),
    m_impl(this, address)
{
    m_value.GetScalar() = address;
    m_data.SetAddressByteSize(addr_byte_size);
    m_value.GetScalar().GetData (m_data, addr_byte_size);
    //m_value.SetValueType(Value::eValueTypeHostAddress); 
    switch (address_type)
    {
    case eAddressTypeInvalid:   m_value.SetValueType(Value::eValueTypeScalar);      break;
    case eAddressTypeFile:      m_value.SetValueType(Value::eValueTypeFileAddress); break;
    case eAddressTypeLoad:      m_value.SetValueType(Value::eValueTypeLoadAddress); break;    
    case eAddressTypeHost:      m_value.SetValueType(Value::eValueTypeHostAddress); break;
    }
    m_value.SetContext(Value::eContextTypeClangType, clang_type);
    m_name = name;
    SetIsConstant ();
    SetValueIsValid(true);
    SetAddressTypeOfChildren(eAddressTypeLoad);
}

ValueObjectSP
ValueObjectConstResult::Create
(
    ExecutionContextScope *exe_scope,
    const Error& error
)
{
    return (new ValueObjectConstResult (exe_scope,
                                        error))->GetSP();
}

ValueObjectConstResult::ValueObjectConstResult (
    ExecutionContextScope *exe_scope,
    const Error& error) :
    ValueObject (exe_scope),
    m_clang_ast (NULL),
    m_type_name (),
    m_byte_size (0),
    m_impl(this)
{
    m_error = error;
    SetIsConstant ();
}

ValueObjectConstResult::ValueObjectConstResult (
    ExecutionContextScope *exe_scope,
    clang::ASTContext *clang_ast,
    const Value &value,
    const ConstString &name) :
    ValueObject (exe_scope),
    m_clang_ast (clang_ast),
    m_type_name (),
    m_byte_size (0),
    m_impl(this)
{
    m_value = value;
    m_value.GetData(m_data);
}

ValueObjectConstResult::~ValueObjectConstResult()
{
}

lldb::clang_type_t
ValueObjectConstResult::GetClangTypeImpl()
{
    return m_value.GetClangType();
}

lldb::ValueType
ValueObjectConstResult::GetValueType() const
{
    return eValueTypeConstResult;
}

uint64_t
ValueObjectConstResult::GetByteSize()
{
    if (m_byte_size == 0)
        m_byte_size = ClangASTType::GetTypeByteSize(GetClangAST(), GetClangType());
    return m_byte_size;
}

void
ValueObjectConstResult::SetByteSize (size_t size)
{
    m_byte_size = size;
}

size_t
ValueObjectConstResult::CalculateNumChildren()
{
    return ClangASTContext::GetNumChildren (GetClangAST (), GetClangType(), true);
}

clang::ASTContext *
ValueObjectConstResult::GetClangASTImpl ()
{
    return m_clang_ast;
}

ConstString
ValueObjectConstResult::GetTypeName()
{
    if (m_type_name.IsEmpty())
        m_type_name = ClangASTType::GetConstTypeName (GetClangAST(), GetClangType());
    return m_type_name;
}

bool
ValueObjectConstResult::UpdateValue ()
{
    // Const value is always valid
    SetValueIsValid (true);
    return true;
}


bool
ValueObjectConstResult::IsInScope ()
{
    // A const result value is always in scope since it serializes all 
    // information needed to contain the constant value.
    return true;
}

lldb::ValueObjectSP
ValueObjectConstResult::Dereference (Error &error)
{
    return m_impl.Dereference(error);
}

lldb::ValueObjectSP
ValueObjectConstResult::GetSyntheticChildAtOffset(uint32_t offset, const ClangASTType& type, bool can_create)
{
    return m_impl.GetSyntheticChildAtOffset(offset, type, can_create);
}

lldb::ValueObjectSP
ValueObjectConstResult::AddressOf (Error &error)
{
    return m_impl.AddressOf(error);
}

lldb::addr_t
ValueObjectConstResult::GetAddressOf (bool scalar_is_load_address,
                                      AddressType *address_type)
{
    return m_impl.GetAddressOf(scalar_is_load_address, address_type);
}

ValueObject *
ValueObjectConstResult::CreateChildAtIndex (size_t idx, bool synthetic_array_member, int32_t synthetic_index)
{
    return m_impl.CreateChildAtIndex(idx, synthetic_array_member, synthetic_index);
}

size_t
ValueObjectConstResult::GetPointeeData (DataExtractor& data,
                                        uint32_t item_idx,
                                        uint32_t item_count)
{
    return m_impl.GetPointeeData(data, item_idx, item_count);
}

lldb::ValueObjectSP
ValueObjectConstResult::GetDynamicValue (lldb::DynamicValueType use_dynamic)
{
    // Always recalculate dynamic values for const results as the memory that
    // they might point to might have changed at any time.
    if (use_dynamic != eNoDynamicValues)
    {
        if (!IsDynamic())
        {
            ExecutionContext exe_ctx (GetExecutionContextRef());
            Process *process = exe_ctx.GetProcessPtr();
            if (process && process->IsPossibleDynamicValue(*this))
                m_dynamic_value = new ValueObjectDynamicValue (*this, use_dynamic);
        }
        if (m_dynamic_value)
            return m_dynamic_value->GetSP();
    }
    return ValueObjectSP();
}

