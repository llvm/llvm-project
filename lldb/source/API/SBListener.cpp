//===-- SBListener.cpp ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/lldb-python.h"

#include "lldb/API/SBListener.h"
#include "lldb/API/SBBroadcaster.h"
#include "lldb/API/SBDebugger.h"
#include "lldb/API/SBEvent.h"
#include "lldb/API/SBStream.h"
#include "lldb/Core/Broadcaster.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Listener.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Host/TimeValue.h"


using namespace lldb;
using namespace lldb_private;


SBListener::SBListener () :
    m_opaque_sp (),
    m_opaque_ptr (NULL)
{
}

SBListener::SBListener (const char *name) :
    m_opaque_sp (new Listener (name)),
    m_opaque_ptr (NULL)
{
    m_opaque_ptr = m_opaque_sp.get();

    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    if (log)
        log->Printf ("SBListener::SBListener (name=\"%s\") => SBListener(%p)",
                     name, m_opaque_ptr);
}


SBListener::SBListener (const SBListener &rhs) :
    m_opaque_sp (rhs.m_opaque_sp),
    m_opaque_ptr (rhs.m_opaque_ptr)
{
}

const lldb::SBListener &
SBListener::operator = (const lldb::SBListener &rhs)
{
    if (this != &rhs)
    {
        m_opaque_sp = rhs.m_opaque_sp;
        m_opaque_ptr = rhs.m_opaque_ptr;
    }
    return *this;
}

SBListener::SBListener (Listener &listener) :
    m_opaque_sp (),
    m_opaque_ptr (&listener)
{
}

SBListener::~SBListener ()
{
}

bool
SBListener::IsValid() const
{
    return m_opaque_ptr != NULL;
}

void
SBListener::AddEvent (const SBEvent &event)
{
    EventSP &event_sp = event.GetSP ();
    if (event_sp)
        m_opaque_ptr->AddEvent (event_sp);
}

void
SBListener::Clear ()
{
    if (m_opaque_ptr)
        m_opaque_ptr->Clear ();
}

uint32_t
SBListener::StartListeningForEventClass (SBDebugger &debugger,
                             const char *broadcaster_class, 
                             uint32_t event_mask)
{
    if (m_opaque_ptr)
    {
        Debugger *lldb_debugger = debugger.get();
        if (!lldb_debugger)
            return 0;
        BroadcastEventSpec event_spec (ConstString (broadcaster_class), event_mask);
        return m_opaque_ptr->StartListeningForEventSpec (*lldb_debugger, event_spec);
    }
    else
        return 0;
}
                             
bool
SBListener::StopListeningForEventClass (SBDebugger &debugger,
                            const char *broadcaster_class,
                            uint32_t event_mask)
{
    if (m_opaque_ptr)
    {
        Debugger *lldb_debugger = debugger.get();
        if (!lldb_debugger)
            return false;
        BroadcastEventSpec event_spec (ConstString (broadcaster_class), event_mask);
        return m_opaque_ptr->StopListeningForEventSpec (*lldb_debugger, event_spec);
    }
    else
        return false;
}
    
uint32_t
SBListener::StartListeningForEvents (const SBBroadcaster& broadcaster, uint32_t event_mask)
{
    uint32_t acquired_event_mask = 0;
    if (m_opaque_ptr && broadcaster.IsValid())
    {
        acquired_event_mask = m_opaque_ptr->StartListeningForEvents (broadcaster.get(), event_mask);
    }
    
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);
    if (log)
    {
        StreamString sstr_requested;
        StreamString sstr_acquired;
        
        Broadcaster *lldb_broadcaster = broadcaster.get();
        if (lldb_broadcaster)
        {
            const bool got_requested_names = lldb_broadcaster->GetEventNames (sstr_requested, event_mask, false);
            const bool got_acquired_names = lldb_broadcaster->GetEventNames (sstr_acquired, acquired_event_mask, false);
            log->Printf ("SBListener(%p)::StartListeneingForEvents (SBBroadcaster(%p): %s, event_mask=0x%8.8x%s%s%s) => 0x%8.8x%s%s%s",
                         m_opaque_ptr, 
                         lldb_broadcaster, 
                         lldb_broadcaster->GetBroadcasterName().GetCString(), 
                         event_mask, 
                         got_requested_names ? " (" : "",
                         sstr_requested.GetData(),
                         got_requested_names ? ")" : "",
                         acquired_event_mask,
                         got_acquired_names ? " (" : "",
                         sstr_acquired.GetData(),
                         got_acquired_names ? ")" : "");
        }
        else
        {
            log->Printf ("SBListener(%p)::StartListeneingForEvents (SBBroadcaster(%p), event_mask=0x%8.8x) => 0x%8.8x", 
                         m_opaque_ptr, 
                         lldb_broadcaster, 
                         event_mask, 
                         acquired_event_mask);
            
        }
    }

    return acquired_event_mask;
}

bool
SBListener::StopListeningForEvents (const SBBroadcaster& broadcaster, uint32_t event_mask)
{
    if (m_opaque_ptr && broadcaster.IsValid())
    {
        return m_opaque_ptr->StopListeningForEvents (broadcaster.get(), event_mask);
    }
    return false;
}

bool
SBListener::WaitForEvent (uint32_t timeout_secs, SBEvent &event)
{
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
    {
        if (timeout_secs == UINT32_MAX)
        {
            log->Printf ("SBListener(%p)::WaitForEvent (timeout_secs=INFINITE, SBEvent(%p))...",
                         m_opaque_ptr, event.get());
        }
        else
        {
            log->Printf ("SBListener(%p)::WaitForEvent (timeout_secs=%d, SBEvent(%p))...",
                         m_opaque_ptr, timeout_secs, event.get());
        }
    }
    bool success = false;

    if (m_opaque_ptr)
    {
        TimeValue time_value;
        if (timeout_secs != UINT32_MAX)
        {
            assert (timeout_secs != 0); // Take this out after all calls with timeout set to zero have been removed....
            time_value = TimeValue::Now();
            time_value.OffsetWithSeconds (timeout_secs);
        }
        EventSP event_sp;
        if (m_opaque_ptr->WaitForEvent (time_value.IsValid() ? &time_value : NULL, event_sp))
        {
            event.reset (event_sp);
            success = true;
        }
    }

    if (log)
    {
        if (timeout_secs == UINT32_MAX)
        {
            log->Printf ("SBListener(%p)::WaitForEvent (timeout_secs=INFINITE, SBEvent(%p)) => %i",
                         m_opaque_ptr, event.get(), success);
        }
        else
        {
            log->Printf ("SBListener(%p)::WaitForEvent (timeout_secs=%d, SBEvent(%p)) => %i",
                         m_opaque_ptr, timeout_secs, event.get(), success);
        }
    }
    if (!success)
        event.reset (NULL);
    return success;
}

bool
SBListener::WaitForEventForBroadcaster
(
    uint32_t num_seconds,
    const SBBroadcaster &broadcaster,
    SBEvent &event
)
{
    if (m_opaque_ptr && broadcaster.IsValid())
    {
        TimeValue time_value;
        if (num_seconds != UINT32_MAX)
        {
            time_value = TimeValue::Now();
            time_value.OffsetWithSeconds (num_seconds);
        }
        EventSP event_sp;
        if (m_opaque_ptr->WaitForEventForBroadcaster (time_value.IsValid() ? &time_value : NULL,
                                                         broadcaster.get(),
                                                         event_sp))
        {
            event.reset (event_sp);
            return true;
        }

    }
    event.reset (NULL);
    return false;
}

bool
SBListener::WaitForEventForBroadcasterWithType
(
    uint32_t num_seconds,
    const SBBroadcaster &broadcaster,
    uint32_t event_type_mask,
    SBEvent &event
)
{
    if (m_opaque_ptr && broadcaster.IsValid())
    {
        TimeValue time_value;
        if (num_seconds != UINT32_MAX)
        {
            time_value = TimeValue::Now();
            time_value.OffsetWithSeconds (num_seconds);
        }
        EventSP event_sp;
        if (m_opaque_ptr->WaitForEventForBroadcasterWithType (time_value.IsValid() ? &time_value : NULL,
                                                              broadcaster.get(),
                                                              event_type_mask,
                                                              event_sp))
        {
            event.reset (event_sp);
            return true;
        }
    }
    event.reset (NULL);
    return false;
}

bool
SBListener::PeekAtNextEvent (SBEvent &event)
{
    if (m_opaque_ptr)
    {
        event.reset (m_opaque_ptr->PeekAtNextEvent ());
        return event.IsValid();
    }
    event.reset (NULL);
    return false;
}

bool
SBListener::PeekAtNextEventForBroadcaster (const SBBroadcaster &broadcaster, SBEvent &event)
{
    if (m_opaque_ptr && broadcaster.IsValid())
    {
        event.reset (m_opaque_ptr->PeekAtNextEventForBroadcaster (broadcaster.get()));
        return event.IsValid();
    }
    event.reset (NULL);
    return false;
}

bool
SBListener::PeekAtNextEventForBroadcasterWithType (const SBBroadcaster &broadcaster, uint32_t event_type_mask,
                                                   SBEvent &event)
{
    if (m_opaque_ptr && broadcaster.IsValid())
    {
        event.reset(m_opaque_ptr->PeekAtNextEventForBroadcasterWithType (broadcaster.get(), event_type_mask));
        return event.IsValid();
    }
    event.reset (NULL);
    return false;
}

bool
SBListener::GetNextEvent (SBEvent &event)
{
    if (m_opaque_ptr)
    {
        EventSP event_sp;
        if (m_opaque_ptr->GetNextEvent (event_sp))
        {
            event.reset (event_sp);
            return true;
        }
    }
    event.reset (NULL);
    return false;
}

bool
SBListener::GetNextEventForBroadcaster (const SBBroadcaster &broadcaster, SBEvent &event)
{
    if (m_opaque_ptr && broadcaster.IsValid())
    {
        EventSP event_sp;
        if (m_opaque_ptr->GetNextEventForBroadcaster (broadcaster.get(), event_sp))
        {
            event.reset (event_sp);
            return true;
        }
    }
    event.reset (NULL);
    return false;
}

bool
SBListener::GetNextEventForBroadcasterWithType
(
    const SBBroadcaster &broadcaster,
    uint32_t event_type_mask,
    SBEvent &event
)
{
    if (m_opaque_ptr && broadcaster.IsValid())
    {
        EventSP event_sp;
        if (m_opaque_ptr->GetNextEventForBroadcasterWithType (broadcaster.get(),
                                                              event_type_mask,
                                                              event_sp))
        {
            event.reset (event_sp);
            return true;
        }
    }
    event.reset (NULL);
    return false;
}

bool
SBListener::HandleBroadcastEvent (const SBEvent &event)
{
    if (m_opaque_ptr)
        return m_opaque_ptr->HandleBroadcastEvent (event.GetSP());
    return false;
}

Listener *
SBListener::operator->() const
{
    return m_opaque_ptr;
}

Listener *
SBListener::get() const
{
    return m_opaque_ptr;
}

void
SBListener::reset(Listener *listener, bool owns)
{
    if (owns)
        m_opaque_sp.reset (listener);
    else
        m_opaque_sp.reset ();
    m_opaque_ptr = listener;
}

Listener &
SBListener::ref() const
{
    return *m_opaque_ptr;    
}

Listener &
SBListener::operator *()
{
    return *m_opaque_ptr;
}

const Listener &
SBListener::operator *() const
{
    return *m_opaque_ptr;
}


