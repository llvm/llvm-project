//===-- Event.cpp -----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/Event.h"
#include "lldb/Core/Broadcaster.h"
#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/State.h"
#include "lldb/Core/Stream.h"
#include "lldb/Host/Endian.h"
#include "lldb/Target/Process.h"
#include <algorithm>

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// Event constructor
//----------------------------------------------------------------------
Event::Event (Broadcaster *broadcaster, uint32_t event_type, EventData *data) :
    m_broadcaster (broadcaster),
    m_type (event_type),
    m_data_ap (data)
{
}

Event::Event(uint32_t event_type, EventData *data) :
    m_broadcaster (NULL),   // Set by the broadcaster when this event gets broadcast
    m_type (event_type),
    m_data_ap (data)
{
}


//----------------------------------------------------------------------
// Event destructor
//----------------------------------------------------------------------
Event::~Event ()
{
}

void
Event::Dump (Stream *s) const
{
    if (m_broadcaster)
    {
        StreamString event_name;
        if (m_broadcaster->GetEventNames (event_name, m_type, false))
            s->Printf("%p Event: broadcaster = %p (%s), type = 0x%8.8x (%s), data = ",
                      this,
                      m_broadcaster,
                      m_broadcaster->GetBroadcasterName().GetCString(),
                      m_type,
                      event_name.GetString().c_str());
        else
            s->Printf("%p Event: broadcaster = %p (%s), type = 0x%8.8x, data = ",
                      this,
                      m_broadcaster,
                      m_broadcaster->GetBroadcasterName().GetCString(),
                      m_type);
    }
    else
        s->Printf("%p Event: broadcaster = NULL, type = 0x%8.8x, data = ", this, m_type);

    if (m_data_ap.get() == NULL)
        s->Printf ("<NULL>");
    else
    {
        s->PutChar('{');
        m_data_ap->Dump (s);
        s->PutChar('}');
    }
}

void
Event::DoOnRemoval ()
{
    if (m_data_ap.get())
        m_data_ap->DoOnRemoval (this);
}

EventData::EventData()
{
}

EventData::~EventData()
{
}

void
EventData::Dump (Stream *s) const
{
    s->PutCString ("Generic Event Data");
}

EventDataBytes::EventDataBytes () :
    m_bytes()
{
}

EventDataBytes::EventDataBytes (const char *cstr) :
    m_bytes()
{
    SetBytesFromCString (cstr);
}

EventDataBytes::EventDataBytes (const void *src, size_t src_len) :
    m_bytes()
{
    SetBytes (src, src_len);
}

EventDataBytes::~EventDataBytes()
{
}

const ConstString &
EventDataBytes::GetFlavorString ()
{
    static ConstString g_flavor ("EventDataBytes");
    return g_flavor;
}

const ConstString &
EventDataBytes::GetFlavor () const
{
    return EventDataBytes::GetFlavorString ();
}

void
EventDataBytes::Dump (Stream *s) const
{
    size_t num_printable_chars = std::count_if (m_bytes.begin(), m_bytes.end(), isprint);
    if (num_printable_chars == m_bytes.size())
    {
        s->Printf("\"%s\"", m_bytes.c_str());
    }
    else if (m_bytes.size() > 0)
    {
        DataExtractor data;
        data.SetData(&m_bytes[0], m_bytes.size(), lldb::endian::InlHostByteOrder());
        data.Dump(s, 0, eFormatBytes, 1, m_bytes.size(), 32, LLDB_INVALID_ADDRESS, 0, 0);
    }
}

const void *
EventDataBytes::GetBytes() const
{
    if (m_bytes.empty())
        return NULL;
    return &m_bytes[0];
}

size_t
EventDataBytes::GetByteSize() const
{
    return m_bytes.size ();
}

void
EventDataBytes::SetBytes (const void *src, size_t src_len)
{
    if (src && src_len > 0)
        m_bytes.assign ((const char *)src, src_len);
    else
        m_bytes.clear();
}

void
EventDataBytes::SetBytesFromCString (const char *cstr)
{
    if (cstr && cstr[0])
        m_bytes.assign (cstr);
    else
        m_bytes.clear();
}


const void *
EventDataBytes::GetBytesFromEvent (const Event *event_ptr)
{
    const EventDataBytes *e = GetEventDataFromEvent (event_ptr);
    if (e)
        return e->GetBytes();
    return NULL;
}

size_t
EventDataBytes::GetByteSizeFromEvent (const Event *event_ptr)
{
    const EventDataBytes *e = GetEventDataFromEvent (event_ptr);
    if (e)
        return e->GetByteSize();
    return 0;
}

const EventDataBytes *
EventDataBytes::GetEventDataFromEvent (const Event *event_ptr)
{
    if (event_ptr)
    {
        const EventData *event_data = event_ptr->GetData();
        if (event_data && event_data->GetFlavor() == EventDataBytes::GetFlavorString())
            return static_cast <const EventDataBytes *> (event_data);
    }
    return NULL;
}

void
EventDataBytes::SwapBytes (std::string &new_bytes)
{
    m_bytes.swap (new_bytes);
}


