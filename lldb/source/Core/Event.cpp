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
#include <algorithm>

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

using namespace lldb;
using namespace lldb_private;

Event::Event (Broadcaster *broadcaster, uint32_t event_type, EventData *data) :
    m_broadcaster_wp(broadcaster->GetBroadcasterImpl()),
    m_type(event_type),
    m_data_sp(data)
{
}

Event::Event (Broadcaster *broadcaster, uint32_t event_type, const EventDataSP &event_data_sp) :
    m_broadcaster_wp(broadcaster->GetBroadcasterImpl()),
    m_type(event_type),
    m_data_sp(event_data_sp)
{
}

Event::Event(uint32_t event_type, EventData *data) :
    m_broadcaster_wp(),
    m_type(event_type),
    m_data_sp(data)
{
}

Event::Event(uint32_t event_type, const EventDataSP &event_data_sp) :
    m_broadcaster_wp(),
    m_type(event_type),
    m_data_sp(event_data_sp)
{
}

Event::~Event() = default;

void
Event::Dump (Stream *s) const
{
    Broadcaster *broadcaster;
    Broadcaster::BroadcasterImplSP broadcaster_impl_sp(m_broadcaster_wp.lock());
    if (broadcaster_impl_sp)
        broadcaster = broadcaster_impl_sp->GetBroadcaster();
    else
        broadcaster = nullptr;
    
    if (broadcaster)
    {
        StreamString event_name;
        if (broadcaster->GetEventNames (event_name, m_type, false))
            s->Printf("%p Event: broadcaster = %p (%s), type = 0x%8.8x (%s), data = ",
                      static_cast<const void*>(this),
                      static_cast<void*>(broadcaster),
                      broadcaster->GetBroadcasterName().GetCString(),
                      m_type, event_name.GetString().c_str());
        else
            s->Printf("%p Event: broadcaster = %p (%s), type = 0x%8.8x, data = ",
                      static_cast<const void*>(this),
                      static_cast<void*>(broadcaster),
                      broadcaster->GetBroadcasterName().GetCString(), m_type);
    }
    else
        s->Printf("%p Event: broadcaster = NULL, type = 0x%8.8x, data = ",
                  static_cast<const void*>(this), m_type);

    if (m_data_sp)
    {
        s->PutChar('{');
        m_data_sp->Dump (s);
        s->PutChar('}');
    }
    else
        s->Printf ("<NULL>");
}

void
Event::DoOnRemoval ()
{
    if (m_data_sp)
        m_data_sp->DoOnRemoval (this);
}

EventData::EventData() = default;

EventData::~EventData() = default;

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

EventDataBytes::~EventDataBytes() = default;

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
    else if (!m_bytes.empty())
    {
        DataExtractor data;
        data.SetData(m_bytes.data(), m_bytes.size(), endian::InlHostByteOrder());
        data.Dump(s, 0, eFormatBytes, 1, m_bytes.size(), 32, LLDB_INVALID_ADDRESS, 0, 0);
    }
}

const void *
EventDataBytes::GetBytes() const
{
    return (m_bytes.empty() ? nullptr : m_bytes.data());
}

size_t
EventDataBytes::GetByteSize() const
{
    return m_bytes.size ();
}

void
EventDataBytes::SetBytes (const void *src, size_t src_len)
{
    if (src != nullptr && src_len > 0)
        m_bytes.assign ((const char *)src, src_len);
    else
        m_bytes.clear();
}

void
EventDataBytes::SetBytesFromCString (const char *cstr)
{
    if (cstr != nullptr && cstr[0])
        m_bytes.assign (cstr);
    else
        m_bytes.clear();
}

const void *
EventDataBytes::GetBytesFromEvent (const Event *event_ptr)
{
    const EventDataBytes *e = GetEventDataFromEvent (event_ptr);
    if (e != nullptr)
        return e->GetBytes();
    return nullptr;
}

size_t
EventDataBytes::GetByteSizeFromEvent (const Event *event_ptr)
{
    const EventDataBytes *e = GetEventDataFromEvent (event_ptr);
    if (e != nullptr)
        return e->GetByteSize();
    return 0;
}

const EventDataBytes *
EventDataBytes::GetEventDataFromEvent (const Event *event_ptr)
{
    if (event_ptr != nullptr)
    {
        const EventData *event_data = event_ptr->GetData();
        if (event_data && event_data->GetFlavor() == EventDataBytes::GetFlavorString())
            return static_cast <const EventDataBytes *> (event_data);
    }
    return nullptr;
}

void
EventDataBytes::SwapBytes (std::string &new_bytes)
{
    m_bytes.swap (new_bytes);
}
