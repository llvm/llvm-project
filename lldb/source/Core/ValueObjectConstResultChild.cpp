//===-- ValueObjectConstResultChild.cpp ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/ValueObjectConstResultChild.h"

#include "lldb/Core/ValueObjectConstResult.h"
#include "lldb/Core/ValueObjectList.h"

#include "lldb/Symbol/ClangASTContext.h"

using namespace lldb_private;

ValueObjectConstResultChild::ValueObjectConstResultChild
(
    ValueObject &parent,
    clang::ASTContext *clang_ast,
    void *clang_type,
    const ConstString &name,
    uint32_t byte_size,
    int32_t byte_offset,
    uint32_t bitfield_bit_size,
    uint32_t bitfield_bit_offset,
    bool is_base_class,
    bool is_deref_of_parent
) :
    ValueObjectChild (parent,
                      clang_ast,
                      clang_type,
                      name,
                      byte_size,
                      byte_offset,
                      bitfield_bit_size,
                      bitfield_bit_offset,
                      is_base_class,
                      is_deref_of_parent,
                      eAddressTypeLoad),
    m_impl(this)
{
    m_name = name;
}

ValueObjectConstResultChild::~ValueObjectConstResultChild()
{
}

lldb::ValueObjectSP
ValueObjectConstResultChild::Dereference (Error &error)
{
    return m_impl.Dereference(error);
}

lldb::ValueObjectSP
ValueObjectConstResultChild::GetSyntheticChildAtOffset(uint32_t offset, const ClangASTType& type, bool can_create)
{
    return m_impl.GetSyntheticChildAtOffset(offset, type, can_create);
}

lldb::ValueObjectSP
ValueObjectConstResultChild::AddressOf (Error &error)
{
    return m_impl.AddressOf(error);
}

ValueObject *
ValueObjectConstResultChild::CreateChildAtIndex (uint32_t idx, bool synthetic_array_member, int32_t synthetic_index)
{
    return m_impl.CreateChildAtIndex(idx, synthetic_array_member, synthetic_index);
}

size_t
ValueObjectConstResultChild::GetPointeeData (DataExtractor& data,
                                             uint32_t item_idx,
                                             uint32_t item_count)
{
    return m_impl.GetPointeeData(data, item_idx, item_count);
}
