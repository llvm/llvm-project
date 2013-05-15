//===-- ValueObjectChild.h --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ValueObjectChild_h_
#define liblldb_ValueObjectChild_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/ValueObject.h"

namespace lldb_private {

//----------------------------------------------------------------------
// A child of another ValueObject.
//----------------------------------------------------------------------
class ValueObjectChild : public ValueObject
{
public:
    virtual ~ValueObjectChild();

    virtual uint64_t
    GetByteSize()
    {
        return m_byte_size;
    }

    virtual off_t
    GetByteOffset()
    {
        return m_byte_offset;
    }

    virtual uint32_t
    GetBitfieldBitSize()
    {
        return m_bitfield_bit_size;
    }

    virtual uint32_t
    GetBitfieldBitOffset()
    {
        return m_bitfield_bit_offset;
    }

    virtual lldb::ValueType
    GetValueType() const;

    virtual size_t
    CalculateNumChildren();

    virtual ConstString
    GetTypeName();

    virtual ConstString
    GetQualifiedTypeName();
    
    virtual bool
    IsInScope ();

    virtual bool
    IsBaseClass ()
    {
        return m_is_base_class;
    }

    virtual bool
    IsDereferenceOfParent ()
    {
        return m_is_deref_of_parent;
    }

protected:
    virtual bool
    UpdateValue ();

    virtual clang::ASTContext *
    GetClangASTImpl ()
    {
        return m_clang_ast;
    }
    
    virtual lldb::clang_type_t
    GetClangTypeImpl ()
    {
        return m_clang_type;
    }
    
    clang::ASTContext *m_clang_ast; // The clang AST that the clang type comes from
    void *m_clang_type; // The type of the child in question within the parent (m_parent_sp)
    ConstString m_type_name;
    uint64_t m_byte_size;
    int32_t m_byte_offset;
    uint8_t m_bitfield_bit_size;
    uint8_t m_bitfield_bit_offset;
    bool m_is_base_class;
    bool m_is_deref_of_parent;

//
//  void
//  ReadValueFromMemory (ValueObject* parent, lldb::addr_t address);

protected:
    friend class ValueObject;
    friend class ValueObjectConstResult;
    ValueObjectChild (ValueObject &parent,
                      clang::ASTContext *clang_ast,
                      void *clang_type,
                      const ConstString &name,
                      uint64_t byte_size,
                      int32_t byte_offset,
                      uint32_t bitfield_bit_size,
                      uint32_t bitfield_bit_offset,
                      bool is_base_class,
                      bool is_deref_of_parent,
                      AddressType child_ptr_or_ref_addr_type);

    DISALLOW_COPY_AND_ASSIGN (ValueObjectChild);
};

} // namespace lldb_private

#endif  // liblldb_ValueObjectChild_h_
