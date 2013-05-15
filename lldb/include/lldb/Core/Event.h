//===-- Event.h -------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Event_h_
#define liblldb_Event_h_

// C Includes
// C++ Includes
#include <list>
#include <string>

// Other libraries and framework includes
// Project includes
#include "lldb/lldb-private.h"
#include "lldb/Core/ConstString.h"
#include "lldb/Host/Predicate.h"

namespace lldb_private {

//----------------------------------------------------------------------
// lldb::EventData
//----------------------------------------------------------------------
class EventData
{
    friend class Event;

public:
    EventData ();

    virtual
    ~EventData();

    virtual const ConstString &
    GetFlavor () const = 0;

    virtual void
    Dump (Stream *s) const;

private:
    virtual void
    DoOnRemoval (Event *event_ptr)
    {
    }

    DISALLOW_COPY_AND_ASSIGN (EventData);

};

//----------------------------------------------------------------------
// lldb::EventDataBytes
//----------------------------------------------------------------------
class EventDataBytes : public EventData
{
public:
    //------------------------------------------------------------------
    // Constructors
    //------------------------------------------------------------------
    EventDataBytes ();

    EventDataBytes (const char *cstr);

    EventDataBytes (const void *src, size_t src_len);

    virtual
    ~EventDataBytes();

    //------------------------------------------------------------------
    // Member functions
    //------------------------------------------------------------------
    virtual const ConstString &
    GetFlavor () const;

    virtual void
    Dump (Stream *s) const;

    const void *
    GetBytes() const;

    size_t
    GetByteSize() const;

    void
    SetBytes (const void *src, size_t src_len);
    
    void
    SwapBytes (std::string &new_bytes);

    void
    SetBytesFromCString (const char *cstr);

    //------------------------------------------------------------------
    // Static functions
    //------------------------------------------------------------------
    static const EventDataBytes *
    GetEventDataFromEvent (const Event *event_ptr);

    static const void *
    GetBytesFromEvent (const Event *event_ptr);

    static size_t
    GetByteSizeFromEvent (const Event *event_ptr);

    static const ConstString &
    GetFlavorString ();

private:
    std::string m_bytes;

    DISALLOW_COPY_AND_ASSIGN (EventDataBytes);

};

//----------------------------------------------------------------------
// lldb::Event
//----------------------------------------------------------------------
class Event
{
    friend class Broadcaster;
    friend class Listener;
    friend class EventData;

public:

    Event (Broadcaster *broadcaster, uint32_t event_type, EventData *data = NULL);

    Event (uint32_t event_type, EventData *data = NULL);

    ~Event ();

    void
    Dump (Stream *s) const;

    EventData *
    GetData ()
    {
        return m_data_ap.get();
    }

    const EventData *
    GetData () const
    {
        return m_data_ap.get();
    }
    
    void
    SetData (EventData *new_data)
    {
        m_data_ap.reset (new_data);
    }

    uint32_t
    GetType () const
    {
        return m_type;
    }
    
    void
    SetType (uint32_t new_type)
    {
        m_type = new_type;
    }

    Broadcaster *
    GetBroadcaster () const
    {
        return m_broadcaster;
    }
    
    bool
    BroadcasterIs (Broadcaster *broadcaster)
    {
        return broadcaster == m_broadcaster;
    }

    void
    Clear()
    {
        m_data_ap.reset();
    }


private:
    // This is only called by Listener when it pops an event off the queue for
    // the listener.  It calls the Event Data's DoOnRemoval() method, which is
    // virtual and can be overridden by the specific data classes.

    void
    DoOnRemoval ();

    // Called by Broadcaster::BroadcastEvent prior to letting all the listeners
    // know about it update the contained broadcaster so that events can be
    // popped off one queue and re-broadcast to others.
    void
    SetBroadcaster (Broadcaster *broadcaster)
    {
        m_broadcaster = broadcaster;
    }


    Broadcaster *   m_broadcaster;  // The broadcaster that sent this event
    uint32_t        m_type;         // The bit describing this event
    std::unique_ptr<EventData> m_data_ap;         // User specific data for this event


    DISALLOW_COPY_AND_ASSIGN (Event);
    Event();    // Disallow default constructor
};

} // namespace lldb_private

#endif  // liblldb_Event_h_
