//===-- History.h -----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_History_h_
#define lldb_History_h_

// C Includes
#include <stdint.h>

// C++ Includes
#include <mutex>
#include <stack>
#include <string>

// Other libraries and framework includes
// Project includes
#include "lldb/lldb-public.h"

namespace lldb_private {

//----------------------------------------------------------------------
/// @class HistorySource History.h "lldb/Core/History.h"
/// @brief A class that defines history events.
//----------------------------------------------------------------------
    
class HistorySource
{
public:
    typedef const void * HistoryEvent;

    HistorySource() : m_mutex(), m_events() {}

    virtual 
    ~HistorySource()
    {
    }

    // Create a new history event. Subclasses should use any data or members
    // in the subclass of this class to produce a history event and push it
    // onto the end of the history stack.

    virtual HistoryEvent
    CreateHistoryEvent () = 0;

    virtual void
    DeleteHistoryEvent (HistoryEvent event) = 0;

    virtual void
    DumpHistoryEvent (Stream &strm, HistoryEvent event) = 0;

    virtual size_t
    GetHistoryEventCount() = 0;

    virtual HistoryEvent
    GetHistoryEventAtIndex (uint32_t idx) = 0;

    virtual HistoryEvent
    GetCurrentHistoryEvent () = 0;

    // Return 0 when lhs == rhs, 1 if lhs > rhs, or -1 if lhs < rhs.
    virtual int
    CompareHistoryEvents (const HistoryEvent lhs, 
                          const HistoryEvent rhs) = 0;

    virtual bool
    IsCurrentHistoryEvent (const HistoryEvent event) = 0;

private:
    typedef std::stack<HistoryEvent> collection;

    std::recursive_mutex m_mutex;
    collection m_events;

    DISALLOW_COPY_AND_ASSIGN (HistorySource);
};
    
//----------------------------------------------------------------------
/// @class HistorySourceUInt History.h "lldb/Core/History.h"
/// @brief A class that defines history events that are represented by
/// unsigned integers.
///
/// Any history event that is defined by a unique monotonically 
/// increasing unsigned integer
//----------------------------------------------------------------------

class HistorySourceUInt : public HistorySource
{
    HistorySourceUInt (const char *id_name, uintptr_t start_value = 0u) :
        HistorySource(),
        m_name (id_name),
        m_curr_id (start_value)
    {
    }
    
    ~HistorySourceUInt() override
    {
    }
    
    // Create a new history event. Subclasses should use any data or members
    // in the subclass of this class to produce a history event and push it
    // onto the end of the history stack.
    
    HistoryEvent
    CreateHistoryEvent () override
    {
        ++m_curr_id;
        return (HistoryEvent)m_curr_id;
    }
    
    void
    DeleteHistoryEvent (HistoryEvent event) override
    {
        // Nothing to delete, the event contains the integer
    }
    
    void
    DumpHistoryEvent (Stream &strm, HistoryEvent event) override;
    
    size_t
    GetHistoryEventCount() override
    {
        return m_curr_id;
    }
    
    HistoryEvent
    GetHistoryEventAtIndex (uint32_t idx) override
    {
        return (HistoryEvent)((uintptr_t)idx);
    }
    
    HistoryEvent
    GetCurrentHistoryEvent () override
    {
        return (HistoryEvent)m_curr_id;
    }
    
    // Return 0 when lhs == rhs, 1 if lhs > rhs, or -1 if lhs < rhs.
    int
    CompareHistoryEvents (const HistoryEvent lhs, 
                          const HistoryEvent rhs) override
    {
        uintptr_t lhs_uint = (uintptr_t)lhs;
        uintptr_t rhs_uint = (uintptr_t)rhs;
        if (lhs_uint < rhs_uint)
            return -1;
        if (lhs_uint > rhs_uint)
            return +1;
        return 0;
    }
    
    bool
    IsCurrentHistoryEvent (const HistoryEvent event) override
    {
        return (uintptr_t)event == m_curr_id;
    }

protected:
    std::string m_name; // The name of the history unsigned integer
    uintptr_t m_curr_id; // The current value of the history unsigned unteger
};

} // namespace lldb_private

#endif // lldb_History_h_
