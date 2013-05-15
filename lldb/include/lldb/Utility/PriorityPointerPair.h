//===-- PriorityPointerPair.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_PriorityPointerPair_h_
#define liblldb_PriorityPointerPair_h_

#include "lldb/lldb-public.h"
#include "lldb/Utility/SharingPtr.h"

namespace lldb_utility {

//----------------------------------------------------------------------
// A prioritized pair of SharedPtr<T>. One of the two pointers is high
// priority, the other is low priority.
// The Get() method always returns high, if *high != NULL,
// otherwise, low is returned (even if *low == NULL)
//----------------------------------------------------------------------

template<typename T>
class PriorityPointerPair
{
public:
    
    typedef T& reference_type;
    typedef T* pointer_type;
    
    typedef typename std::shared_ptr<T> T_SP;
    
    PriorityPointerPair() : 
    m_high(),
    m_low()
    {}
    
    PriorityPointerPair(pointer_type high,
                        pointer_type low) : 
    m_high(high),
    m_low(low)
    {}

    PriorityPointerPair(pointer_type low) : 
    m_high(),
    m_low(low)
    {}

    PriorityPointerPair(T_SP& high,
                        T_SP& low) : 
    m_high(high),
    m_low(low)
    {}
    
    PriorityPointerPair(T_SP& low) : 
    m_high(),
    m_low(low)
    {}
    
    void
    SwapLow(pointer_type l)
    {
        m_low.swap(l);
    }
    
    void
    SwapHigh(pointer_type h)
    {
        m_high.swap(h);
    }
    
    void
    SwapLow(T_SP l)
    {
        m_low.swap(l);
    }

    void
    SwapHigh(T_SP h)
    {
        m_high.swap(h);
    }
    
    T_SP
    GetLow()
    {
        return m_low;
    }
    
    T_SP
    GetHigh()
    {
        return m_high;
    }
    
    T_SP
    Get()
    {
        if (m_high.get())
            return m_high;
        return m_low;
    }
    
    void
    ResetHigh()
    {
        m_high.reset();
    }
    
    void
    ResetLow()
    {
        m_low.reset();
    }
    
    void
    Reset()
    {
        ResetLow();
        ResetHigh();
    }
    
    reference_type
    operator*() const
    {
        return Get().operator*();
    }
    
    pointer_type
    operator->() const
    {
        return Get().operator->();
    }
    
    ~PriorityPointerPair();
    
private:

    T_SP m_high;
    T_SP m_low;
    
    DISALLOW_COPY_AND_ASSIGN (PriorityPointerPair);
        
};

} // namespace lldb_utility

#endif // #ifndef liblldb_PriorityPointerPair_h_
