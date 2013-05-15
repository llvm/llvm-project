//===-- ValueObjectConstResultImpl.cpp ---------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/ValueObjectConstResultImpl.h"

#include "lldb/Core/ValueObjectChild.h"
#include "lldb/Core/ValueObjectConstResult.h"
#include "lldb/Core/ValueObjectConstResultChild.h"
#include "lldb/Core/ValueObjectMemory.h"
#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/Module.h"
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

// this macro enables a simpler implementation for some method calls in this object that relies only upon
// ValueObject knowning how to set the address type of its children correctly. the alternative implementation
// relies on being able to create a target copy of the frozen object, which makes it less bug-prone but less
// efficient as well. once we are confident the faster implementation is bug-free, this macro (and the slower
// implementations) can go
#define TRIVIAL_IMPL 1

ValueObjectConstResultImpl::ValueObjectConstResultImpl (ValueObject* valobj,
                                                        lldb::addr_t live_address) :
    m_impl_backend(valobj),
    m_live_address(live_address),
    m_live_address_type(eAddressTypeLoad),
    m_load_addr_backend(),
    m_address_of_backend()
{
}

lldb::ValueObjectSP
ValueObjectConstResultImpl::DerefOnTarget()
{
    if (m_load_addr_backend.get() == NULL)
    {
        lldb::addr_t tgt_address = m_impl_backend->GetPointerValue();
        ExecutionContext exe_ctx (m_impl_backend->GetExecutionContextRef());
        m_load_addr_backend = ValueObjectConstResult::Create (exe_ctx.GetBestExecutionContextScope(),
                                                              m_impl_backend->GetClangAST(),
                                                              m_impl_backend->GetClangType(),
                                                              m_impl_backend->GetName(),
                                                              tgt_address,
                                                              eAddressTypeLoad,
                                                              exe_ctx.GetAddressByteSize());
    }
    return m_load_addr_backend;
}

lldb::ValueObjectSP
ValueObjectConstResultImpl::Dereference (Error &error)
{
    if (m_impl_backend == NULL)
        return lldb::ValueObjectSP();
    
#if defined (TRIVIAL_IMPL) && TRIVIAL_IMPL == 1
    return m_impl_backend->ValueObject::Dereference(error);
#else
    m_impl_backend->UpdateValueIfNeeded(false);
        
    if (NeedsDerefOnTarget())
        return DerefOnTarget()->Dereference(error);
    else
        return m_impl_backend->ValueObject::Dereference(error);
#endif
}

ValueObject *
ValueObjectConstResultImpl::CreateChildAtIndex (size_t idx, bool synthetic_array_member, int32_t synthetic_index)
{
    if (m_impl_backend == NULL)
        return NULL;

    m_impl_backend->UpdateValueIfNeeded(false);
    
    ValueObjectConstResultChild *valobj = NULL;
    
    bool omit_empty_base_classes = true;
    bool ignore_array_bounds = synthetic_array_member;
    std::string child_name_str;
    uint32_t child_byte_size = 0;
    int32_t child_byte_offset = 0;
    uint32_t child_bitfield_bit_size = 0;
    uint32_t child_bitfield_bit_offset = 0;
    bool child_is_base_class = false;
    bool child_is_deref_of_parent = false;
    
    const bool transparent_pointers = synthetic_array_member == false;
    clang::ASTContext *clang_ast = m_impl_backend->GetClangAST();
    lldb::clang_type_t clang_type = m_impl_backend->GetClangType();
    lldb::clang_type_t child_clang_type;
    
    ExecutionContext exe_ctx (m_impl_backend->GetExecutionContextRef());
    
    child_clang_type = ClangASTContext::GetChildClangTypeAtIndex (&exe_ctx,
                                                                  clang_ast,
                                                                  m_impl_backend->GetName().GetCString(),
                                                                  clang_type,
                                                                  idx,
                                                                  transparent_pointers,
                                                                  omit_empty_base_classes,
                                                                  ignore_array_bounds,
                                                                  child_name_str,
                                                                  child_byte_size,
                                                                  child_byte_offset,
                                                                  child_bitfield_bit_size,
                                                                  child_bitfield_bit_offset,
                                                                  child_is_base_class,
                                                                  child_is_deref_of_parent);
    if (child_clang_type && child_byte_size)
    {
        if (synthetic_index)
            child_byte_offset += child_byte_size * synthetic_index;
        
        ConstString child_name;
        if (!child_name_str.empty())
            child_name.SetCString (child_name_str.c_str());
        
        valobj = new ValueObjectConstResultChild (*m_impl_backend,
                                                  clang_ast,
                                                  child_clang_type,
                                                  child_name,
                                                  child_byte_size,
                                                  child_byte_offset,
                                                  child_bitfield_bit_size,
                                                  child_bitfield_bit_offset,
                                                  child_is_base_class,
                                                  child_is_deref_of_parent);
        valobj->m_impl.SetLiveAddress(m_live_address+child_byte_offset);
    }
    
    return valobj;
}

lldb::ValueObjectSP
ValueObjectConstResultImpl::GetSyntheticChildAtOffset (uint32_t offset, const ClangASTType& type, bool can_create)
{
    if (m_impl_backend == NULL)
        return lldb::ValueObjectSP();

#if defined (TRIVIAL_IMPL) && TRIVIAL_IMPL == 1
    return m_impl_backend->ValueObject::GetSyntheticChildAtOffset(offset, type, can_create);
#else
    m_impl_backend->UpdateValueIfNeeded(false);
    
    if (NeedsDerefOnTarget())
        return DerefOnTarget()->GetSyntheticChildAtOffset(offset, type, can_create);
    else
        return m_impl_backend->ValueObject::GetSyntheticChildAtOffset(offset, type, can_create);
#endif
}

lldb::ValueObjectSP
ValueObjectConstResultImpl::AddressOf (Error &error)
{
    if (m_address_of_backend.get() != NULL)
        return m_address_of_backend;
    
    if (m_impl_backend == NULL)
        return lldb::ValueObjectSP();
    if (m_live_address != LLDB_INVALID_ADDRESS)
    {
        ClangASTType type(m_impl_backend->GetClangAST(), m_impl_backend->GetClangType());
        
        lldb::DataBufferSP buffer(new lldb_private::DataBufferHeap(&m_live_address,sizeof(lldb::addr_t)));
        
        std::string new_name("&");
        new_name.append(m_impl_backend->GetName().AsCString(""));
        ExecutionContext exe_ctx (m_impl_backend->GetExecutionContextRef());
        m_address_of_backend = ValueObjectConstResult::Create (exe_ctx.GetBestExecutionContextScope(),
                                                               type.GetASTContext(),
                                                               type.GetPointerType(),
                                                               ConstString(new_name.c_str()),
                                                               buffer,
                                                               lldb::endian::InlHostByteOrder(), 
                                                               exe_ctx.GetAddressByteSize());
        
        m_address_of_backend->GetValue().SetValueType(Value::eValueTypeScalar);
        m_address_of_backend->GetValue().GetScalar() = m_live_address;
        
        return m_address_of_backend;
    }
    else
        return lldb::ValueObjectSP();
}

lldb::addr_t
ValueObjectConstResultImpl::GetAddressOf (bool scalar_is_load_address,
                                          AddressType *address_type)
{
    
    if (m_impl_backend == NULL)
        return 0;
    
    if (m_live_address == LLDB_INVALID_ADDRESS)
    {
        return m_impl_backend->ValueObject::GetAddressOf (scalar_is_load_address,
                                                          address_type);
    }
    
    if (address_type)
        *address_type = m_live_address_type;
        
    return m_live_address;
}

size_t
ValueObjectConstResultImpl::GetPointeeData (DataExtractor& data,
                                            uint32_t item_idx,
                                            uint32_t item_count)
{
    if (m_impl_backend == NULL)
        return 0;
#if defined (TRIVIAL_IMPL) && TRIVIAL_IMPL == 1
    return m_impl_backend->ValueObject::GetPointeeData(data, item_idx, item_count);
#else
    m_impl_backend->UpdateValueIfNeeded(false);
    
    if (NeedsDerefOnTarget() && m_impl_backend->IsPointerType())
        return DerefOnTarget()->GetPointeeData(data, item_idx, item_count);
    else
        return m_impl_backend->ValueObject::GetPointeeData(data, item_idx, item_count);
#endif
}
