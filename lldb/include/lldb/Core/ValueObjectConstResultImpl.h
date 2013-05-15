//===-- ValueObjectConstResultImpl.h -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ValueObjectConstResultImpl_h_
#define liblldb_ValueObjectConstResultImpl_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/ValueObject.h"

namespace lldb_private {

//----------------------------------------------------------------------
// A class wrapping common implementation details for operations in
// ValueObjectConstResult ( & Child ) that may need to jump from the host
// memory space into the target's memory space
//----------------------------------------------------------------------
class ValueObjectConstResultImpl
{
public:
    
    ValueObjectConstResultImpl (ValueObject* valobj,
                                lldb::addr_t live_address = LLDB_INVALID_ADDRESS);
    
    virtual
    ~ValueObjectConstResultImpl()
    {
    }
    
    lldb::ValueObjectSP
    Dereference (Error &error);
    
    ValueObject *
    CreateChildAtIndex (size_t idx, bool synthetic_array_member, int32_t synthetic_index);
    
    lldb::ValueObjectSP
    GetSyntheticChildAtOffset (uint32_t offset, const ClangASTType& type, bool can_create);
    
    lldb::ValueObjectSP
    AddressOf (Error &error);
    
    bool
    NeedsDerefOnTarget()
    {
        m_impl_backend->UpdateValueIfNeeded(false);
        return (m_impl_backend->GetValue().GetValueType() == Value::eValueTypeHostAddress);
    }
    
    lldb::addr_t
    GetLiveAddress()
    {
        return m_live_address;
    }
    
    void
    SetLiveAddress(lldb::addr_t addr = LLDB_INVALID_ADDRESS,
                   AddressType address_type = eAddressTypeLoad)
    {
        m_live_address = addr;
        m_live_address_type = address_type;
    }
    
    lldb::ValueObjectSP
    DerefOnTarget();
    
    virtual lldb::addr_t
    GetAddressOf (bool scalar_is_load_address = true,
                  AddressType *address_type = NULL);
    
    virtual size_t
    GetPointeeData (DataExtractor& data,
                    uint32_t item_idx = 0,
					uint32_t item_count = 1);
    
private:
    
    ValueObject *m_impl_backend;
    lldb::addr_t m_live_address;
    AddressType m_live_address_type;
    lldb::ValueObjectSP m_load_addr_backend;
    lldb::ValueObjectSP m_address_of_backend;
    
    DISALLOW_COPY_AND_ASSIGN (ValueObjectConstResultImpl);
};

} // namespace lldb_private

#endif  // liblldb_ValueObjectConstResultImpl_h_
