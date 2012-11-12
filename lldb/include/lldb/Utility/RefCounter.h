//===-- RefCounter.h --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_RefCounter_h_
#define liblldb_RefCounter_h_

#include "lldb/lldb-public.h"

namespace lldb_utility {

//----------------------------------------------------------------------
// A simple reference counter object. You need an uint32_t* to use it
// Once that is in place, everyone who needs to ref-count, can say
// RefCounter ref(ptr);
// (of course, the pointer is a shared resource, and must be accessible to
// everyone who needs it). Synchronization is handled by RefCounter itself
// The counter is decreased each time a RefCounter to it goes out of scope
//----------------------------------------------------------------------
class RefCounter
{
public:
    typedef uint32_t value_type;
    
    RefCounter(value_type* ctr);
    
    ~RefCounter();
    
private:
    value_type* m_counter;
    DISALLOW_COPY_AND_ASSIGN (RefCounter);
    
    template <class T>
    inline T
    increment(T* t)
    {
        return __sync_fetch_and_add(t, 1);
    }
    
    template <class T>
    inline T
    decrement(T* t)
    {
        return __sync_fetch_and_add(t, -1);
    }
    
};

} // namespace lldb_utility

#endif // #ifndef liblldb_RefCounter_h_
