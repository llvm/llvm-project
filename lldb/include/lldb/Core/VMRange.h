//===-- VMRange.h -----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_VMRange_h_
#define liblldb_VMRange_h_

#include "lldb/lldb-private.h"
#include <vector>

namespace lldb_private {

//----------------------------------------------------------------------
// A vm address range. These can represent offsets ranges or actual
// addresses.
//----------------------------------------------------------------------
class VMRange
{
public:

    typedef std::vector<VMRange> collection;
    typedef collection::iterator iterator;
    typedef collection::const_iterator const_iterator;

    VMRange() :
        m_base_addr(0),
        m_byte_size(0)
    {
    }

    VMRange(lldb::addr_t start_addr, lldb::addr_t end_addr) :
        m_base_addr(start_addr),
        m_byte_size(end_addr > start_addr ? end_addr - start_addr : 0)
    {
    }

    ~VMRange()
    {
    }

    void
    Clear ()
    {
        m_base_addr = 0;
        m_byte_size = 0;  
    }

    // Set the start and end values
    void 
    Reset (lldb::addr_t start_addr, lldb::addr_t end_addr)
    {
        SetBaseAddress (start_addr);
        SetEndAddress (end_addr);
    }

    // Set the start value for the range, and keep the same size
    void
    SetBaseAddress (lldb::addr_t base_addr)
    {
        m_base_addr = base_addr;
    }

    void
    SetEndAddress (lldb::addr_t end_addr)
    {
        const lldb::addr_t base_addr = GetBaseAddress();
        if (end_addr > base_addr)
            m_byte_size = end_addr - base_addr;
        else
            m_byte_size = 0;
    }

    lldb::addr_t
    GetByteSize () const
    {
        return m_byte_size;
    }

    void
    SetByteSize (lldb::addr_t byte_size)
    {
        m_byte_size = byte_size;
    }

    lldb::addr_t
    GetBaseAddress () const
    {
        return m_base_addr;
    }

    lldb::addr_t
    GetEndAddress () const
    {
        return GetBaseAddress() + m_byte_size;
    }

    bool
    IsValid() const
    {
        return m_byte_size > 0;
    }

    bool
    Contains (lldb::addr_t addr) const
    {
        return (GetBaseAddress() <= addr) && (addr < GetEndAddress());
    }

    bool 
    Contains (const VMRange& range) const
    {
        if (Contains(range.GetBaseAddress()))
        {
            lldb::addr_t range_end = range.GetEndAddress();
            return (GetBaseAddress() <= range_end) && (range_end <= GetEndAddress());
        }
        return false;
    }

    void
    Dump (Stream *s, lldb::addr_t base_addr = 0, uint32_t addr_width = 8) const;

    class ValueInRangeUnaryPredicate
    {
    public:
        ValueInRangeUnaryPredicate(lldb::addr_t value) :
            _value(value)
        {
        }
        bool operator()(const VMRange& range) const
        {
            return range.Contains(_value);
        }
        lldb::addr_t _value;
    };

    class RangeInRangeUnaryPredicate
    {
    public:
        RangeInRangeUnaryPredicate(VMRange range) :
            _range(range)
        {
        }
        bool operator()(const VMRange& range) const
        {
            return range.Contains(_range);
        }
        const VMRange& _range;
    };

    static bool
    ContainsValue(const VMRange::collection& coll, lldb::addr_t value);

    static bool
    ContainsRange(const VMRange::collection& coll, const VMRange& range);

    // Returns a valid index into coll when a match is found, else UINT32_MAX
    // is returned
    static uint32_t
    FindRangeIndexThatContainsValue (const VMRange::collection& coll, lldb::addr_t value);

protected:
    lldb::addr_t m_base_addr;
    lldb::addr_t m_byte_size;
};

bool operator== (const VMRange& lhs, const VMRange& rhs);
bool operator!= (const VMRange& lhs, const VMRange& rhs);
bool operator<  (const VMRange& lhs, const VMRange& rhs);
bool operator<= (const VMRange& lhs, const VMRange& rhs);
bool operator>  (const VMRange& lhs, const VMRange& rhs);
bool operator>= (const VMRange& lhs, const VMRange& rhs);

} // namespace lldb_private

#endif  // liblldb_VMRange_h_
