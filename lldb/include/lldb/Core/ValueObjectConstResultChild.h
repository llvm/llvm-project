//===-- ValueObjectConstResultChild.h -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ValueObjectConstResultChild_h_
#define liblldb_ValueObjectConstResultChild_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/ValueObjectChild.h"
#include "lldb/Core/ValueObjectConstResultImpl.h"

namespace lldb_private {

//----------------------------------------------------------------------
// A child of a ValueObjectConstResult.
//----------------------------------------------------------------------
class ValueObjectConstResultChild : public ValueObjectChild
{
public:
    
    ValueObjectConstResultChild (ValueObject &parent,
                                 clang::ASTContext *clang_ast,
                                 void *clang_type,
                                 const ConstString &name,
                                 uint32_t byte_size,
                                 int32_t byte_offset,
                                 uint32_t bitfield_bit_size,
                                 uint32_t bitfield_bit_offset,
                                 bool is_base_class,
                                 bool is_deref_of_parent);
    
    virtual ~ValueObjectConstResultChild();
    
    virtual lldb::ValueObjectSP
    Dereference (Error &error);
    
    virtual ValueObject *
    CreateChildAtIndex (uint32_t idx, bool synthetic_array_member, int32_t synthetic_index);

    virtual lldb::clang_type_t
    GetClangType ()
    {
        return ValueObjectChild::GetClangType();
    }
    
    virtual lldb::ValueObjectSP
    GetSyntheticChildAtOffset(uint32_t offset, const ClangASTType& type, bool can_create);
    
    virtual lldb::ValueObjectSP
    AddressOf (Error &error);
    
    virtual size_t
    GetPointeeData (DataExtractor& data,
                    uint32_t item_idx = 0,
					uint32_t item_count = 1);
    
protected:
    ValueObjectConstResultImpl m_impl;
    
private:
    friend class ValueObject;
    friend class ValueObjectConstResult;
    friend class ValueObjectConstResultImpl;

    DISALLOW_COPY_AND_ASSIGN (ValueObjectConstResultChild);
};

} // namespace lldb_private

#endif  // liblldb_ValueObjectConstResultChild_h_
