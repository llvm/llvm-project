//===-- Broadcaster.cpp -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/Broadcaster.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/Log.h"
#include "lldb/Core/Event.h"
#include "lldb/Core/StreamString.h"
#include "lldb/lldb-private-log.h"

using namespace lldb;
using namespace lldb_private;

Broadcaster::Broadcaster (BroadcasterManager *manager, const char *name) :
    m_broadcaster_name (name),
    m_listeners (),
    m_listeners_mutex (Mutex::eMutexTypeRecursive),
    m_hijacking_listeners(),
    m_hijacking_masks(),
    m_manager (manager)
{
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_OBJECT));
    if (log)
        log->Printf ("%p Broadcaster::Broadcaster(\"%s\")", this, m_broadcaster_name.AsCString());

}

Broadcaster::~Broadcaster()
{
    Log *log (lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_OBJECT));
    if (log)
        log->Printf ("%p Broadcaster::~Broadcaster(\"%s\")", this, m_broadcaster_name.AsCString());

    Clear();
}

void
Broadcaster::CheckInWithManager ()
{
    if (m_manager != NULL)
    {
        m_manager->SignUpListenersForBroadcaster(*this);
    }
}

void
Broadcaster::Clear()
{
    Mutex::Locker listeners_locker(m_listeners_mutex);
    
    // Make sure the listener forgets about this broadcaster. We do
    // this in the broadcaster in case the broadcaster object initiates
    // the removal.
    
    collection::iterator pos, end = m_listeners.end();
    for (pos = m_listeners.begin(); pos != end; ++pos)
        pos->first->BroadcasterWillDestruct (this);
    
    m_listeners.clear();
}
const ConstString &
Broadcaster::GetBroadcasterName ()
{
    return m_broadcaster_name;
}

bool
Broadcaster::GetEventNames (Stream &s, uint32_t event_mask, bool prefix_with_broadcaster_name) const
{
    uint32_t num_names_added = 0;
    if (event_mask && !m_event_names.empty())
    {
        event_names_map::const_iterator end = m_event_names.end();
        for (uint32_t bit=1u, mask=event_mask; mask != 0 && bit != 0; bit <<= 1, mask >>= 1)
        {
            if (mask & 1)
            {
                event_names_map::const_iterator pos = m_event_names.find(bit);
                if (pos != end)
                {
                    if (num_names_added > 0)
                        s.PutCString(", ");

                    if (prefix_with_broadcaster_name)
                    {
                        s.PutCString (m_broadcaster_name.GetCString());
                        s.PutChar('.');
                    }
                    s.PutCString(pos->second.c_str());
                    ++num_names_added;
                }
            }
        }
    }
    return num_names_added > 0;
}

void
Broadcaster::AddInitialEventsToListener (Listener *listener, uint32_t requested_events)
{

}

uint32_t
Broadcaster::AddListener (Listener* listener, uint32_t event_mask)
{
    if (listener == NULL)
        return 0;

    Mutex::Locker locker(m_listeners_mutex);
    collection::iterator pos, end = m_listeners.end();

    collection::iterator existing_pos = end;
    // See if we already have this listener, and if so, update its mask
    uint32_t taken_event_types = 0;
    for (pos = m_listeners.begin(); pos != end; ++pos)
    {
        if (pos->first == listener)
            existing_pos = pos;
    // For now don't descriminate on who gets what
    // FIXME: Implement "unique listener for this bit" mask
    //  taken_event_types |= pos->second;
    }

    // Each event bit in a Broadcaster object can only be used
    // by one listener
    uint32_t available_event_types = ~taken_event_types & event_mask;

    if (available_event_types)
    {
        // If we didn't find our listener, add it
        if (existing_pos == end)
        {
            // Grant a new listener the available event bits
            m_listeners.push_back(std::make_pair(listener, available_event_types));
        }
        else
        {
            // Grant the existing listener the available event bits
            existing_pos->second |= available_event_types;
        }

        // Individual broadcasters decide whether they have outstanding data when a
        // listener attaches, and insert it into the listener with this method.

        AddInitialEventsToListener (listener, available_event_types);
    }

    // Return the event bits that were granted to the listener
    return available_event_types;
}

bool
Broadcaster::EventTypeHasListeners (uint32_t event_type)
{
    Mutex::Locker locker (m_listeners_mutex);
    
    if (m_hijacking_listeners.size() > 0 && event_type & m_hijacking_masks.back())
        return true;
        
    if (m_listeners.empty())
        return false;

    collection::iterator pos, end = m_listeners.end();
    for (pos = m_listeners.begin(); pos != end; ++pos)
    {
        if (pos->second & event_type)
            return true;
    }
    return false;
}

bool
Broadcaster::RemoveListener (Listener* listener, uint32_t event_mask)
{
    Mutex::Locker locker(m_listeners_mutex);
    collection::iterator pos, end = m_listeners.end();
    // See if we already have this listener, and if so, update its mask
    for (pos = m_listeners.begin(); pos != end; ++pos)
    {
        if (pos->first == listener)
        {
            // Relinquish all event bits in "event_mask"
            pos->second &= ~event_mask;
            // If all bits have been relinquished then remove this listener
            if (pos->second == 0)
                m_listeners.erase (pos);
            return true;
        }
    }
    return false;
}

void
Broadcaster::BroadcastEvent (EventSP &event_sp)
{
    return PrivateBroadcastEvent (event_sp, false);
}

void
Broadcaster::BroadcastEventIfUnique (EventSP &event_sp)
{
    return PrivateBroadcastEvent (event_sp, true);
}

void
Broadcaster::PrivateBroadcastEvent (EventSP &event_sp, bool unique)
{
    // Can't add a NULL event...
    if (event_sp.get() == NULL)
        return;

    // Update the broadcaster on this event
    event_sp->SetBroadcaster (this);

    const uint32_t event_type = event_sp->GetType();

    Mutex::Locker event_types_locker(m_listeners_mutex);
    
    Listener *hijacking_listener = NULL;
    if (!m_hijacking_listeners.empty())
    {
        assert (!m_hijacking_masks.empty());
        hijacking_listener = m_hijacking_listeners.back();
        if ((event_type & m_hijacking_masks.back()) == 0)
            hijacking_listener = NULL;
    }

    Log *log(lldb_private::GetLogIfAnyCategoriesSet (LIBLLDB_LOG_EVENTS));
    if (log)
    {
        StreamString event_description;
        event_sp->Dump  (&event_description);
        log->Printf ("%p Broadcaster(\"%s\")::BroadcastEvent (event_sp = {%s}, unique =%i) hijack = %p",
                     this,
                     m_broadcaster_name.AsCString(""),
                     event_description.GetData(),
                     unique,
                     hijacking_listener);
    }

    if (hijacking_listener)
    {
        if (unique && hijacking_listener->PeekAtNextEventForBroadcasterWithType (this, event_type))
            return;
        hijacking_listener->AddEvent (event_sp);
    }
    else
    {
        collection::iterator pos, end = m_listeners.end();


        // Iterate through all listener/mask pairs
        for (pos = m_listeners.begin(); pos != end; ++pos)
        {
            // If the listener's mask matches any bits that we just set, then
            // put the new event on its event queue.
            if (event_type & pos->second)
            {
                if (unique && pos->first->PeekAtNextEventForBroadcasterWithType (this, event_type))
                    continue;
                pos->first->AddEvent (event_sp);
            }
        }
    }
}

void
Broadcaster::BroadcastEvent (uint32_t event_type, EventData *event_data)
{
    EventSP event_sp (new Event (event_type, event_data));
    PrivateBroadcastEvent (event_sp, false);
}

void
Broadcaster::BroadcastEventIfUnique (uint32_t event_type, EventData *event_data)
{
    EventSP event_sp (new Event (event_type, event_data));
    PrivateBroadcastEvent (event_sp, true);
}

bool
Broadcaster::HijackBroadcaster (Listener *listener, uint32_t event_mask)
{
    Mutex::Locker event_types_locker(m_listeners_mutex);
    
    Log *log(lldb_private::GetLogIfAnyCategoriesSet (LIBLLDB_LOG_EVENTS));
    if (log)
    {
        log->Printf ("%p Broadcaster(\"%s\")::HijackBroadcaster (listener(\"%s\")=%p)",
                     this,
                     m_broadcaster_name.AsCString(""),
                     listener->m_name.c_str(),
                     listener);
    }
    m_hijacking_listeners.push_back(listener);
    m_hijacking_masks.push_back(event_mask);
    return true;
}

void
Broadcaster::RestoreBroadcaster ()
{
    Mutex::Locker event_types_locker(m_listeners_mutex);

    Log *log(lldb_private::GetLogIfAnyCategoriesSet (LIBLLDB_LOG_EVENTS));
    if (log)
    {
        Listener *listener = m_hijacking_listeners.back();
        log->Printf ("%p Broadcaster(\"%s\")::RestoreBroadcaster (about to pop listener(\"%s\")=%p)",
                     this,
                     m_broadcaster_name.AsCString(""),
                     listener->m_name.c_str(),
                     listener);
    }
    m_hijacking_listeners.pop_back();
    m_hijacking_masks.pop_back();
}

ConstString &
Broadcaster::GetBroadcasterClass() const
{
    static ConstString class_name ("lldb.anonymous");
    return class_name;
}

BroadcastEventSpec::BroadcastEventSpec (const BroadcastEventSpec &rhs) :
    m_broadcaster_class (rhs.m_broadcaster_class), 
    m_event_bits (rhs.m_event_bits)
{
}

bool 
BroadcastEventSpec::operator< (const BroadcastEventSpec &rhs) const
{
    if (GetBroadcasterClass() == rhs.GetBroadcasterClass())
    {
        return GetEventBits() < rhs.GetEventBits();
    }
    else
    {
        return GetBroadcasterClass() < rhs.GetBroadcasterClass();
    }
}

const BroadcastEventSpec &
BroadcastEventSpec::operator= (const BroadcastEventSpec &rhs)
{
    m_broadcaster_class = rhs.m_broadcaster_class;
    m_event_bits = rhs.m_event_bits;
    return *this;
}

BroadcasterManager::BroadcasterManager() :
    m_manager_mutex(Mutex::eMutexTypeRecursive)
{
    
}

uint32_t
BroadcasterManager::RegisterListenerForEvents (Listener &listener, BroadcastEventSpec event_spec)
{
    Mutex::Locker locker(m_manager_mutex);
    
    collection::iterator iter = m_event_map.begin(), end_iter = m_event_map.end();
    uint32_t available_bits = event_spec.GetEventBits();
    
    while (iter != end_iter 
           && (iter = find_if (iter, end_iter, BroadcasterClassMatches(event_spec.GetBroadcasterClass()))) != end_iter)
    {
        available_bits &= ~((*iter).first.GetEventBits());
        iter++;  
    }
    
    if (available_bits != 0)
    {
        m_event_map.insert (event_listener_key (BroadcastEventSpec (event_spec.GetBroadcasterClass(), available_bits), &listener));
        m_listeners.insert(&listener);
    }
    
    return available_bits;
}

bool
BroadcasterManager::UnregisterListenerForEvents (Listener &listener, BroadcastEventSpec event_spec)
{
    Mutex::Locker locker(m_manager_mutex);
    bool removed_some = false;
    
    if (m_listeners.erase(&listener) == 0)
        return false;
    
    ListenerMatchesAndSharedBits predicate (event_spec, listener);
    std::vector<BroadcastEventSpec> to_be_readded;
    uint32_t event_bits_to_remove = event_spec.GetEventBits();
    
    // Go through the map and delete the exact matches, and build a list of matches that weren't exact to re-add:
    while (1)
    {
        collection::iterator iter, end_iter = m_event_map.end();
        iter = find_if (m_event_map.begin(), end_iter, predicate);
        if (iter == end_iter)
        {
            break;
        }  
        else 
        {
            uint32_t iter_event_bits = (*iter).first.GetEventBits();
            removed_some = true;

            if (event_bits_to_remove != iter_event_bits)
            {
                uint32_t new_event_bits = iter_event_bits & ~event_bits_to_remove;
                to_be_readded.push_back(BroadcastEventSpec (event_spec.GetBroadcasterClass(), new_event_bits));
            } 
            m_event_map.erase (iter);
        }
    }
    
    // Okay now add back the bits that weren't completely removed:
    for (size_t i = 0; i < to_be_readded.size(); i++)
    {
        m_event_map.insert (event_listener_key (to_be_readded[i], &listener));
    }
    
    return removed_some;
}
    
Listener *
BroadcasterManager::GetListenerForEventSpec (BroadcastEventSpec event_spec) const
{
    Mutex::Locker locker(*(const_cast<Mutex *> (&m_manager_mutex)));
    
    collection::const_iterator iter, end_iter = m_event_map.end();
    iter = find_if (m_event_map.begin(), end_iter, BroadcastEventSpecMatches (event_spec));
    if (iter != end_iter)
        return (*iter).second;
    else
        return NULL;
}

void
BroadcasterManager::RemoveListener (Listener &listener)
{
    Mutex::Locker locker(m_manager_mutex);
    ListenerMatches predicate (listener);


    if (m_listeners.erase (&listener) == 0)
        return;
        
    while (1)
    {
        collection::iterator iter, end_iter = m_event_map.end();
        iter = find_if (m_event_map.begin(), end_iter, predicate);
        if (iter == end_iter)
            break;
        else
            m_event_map.erase(iter);
    }
}
    
void
BroadcasterManager::SignUpListenersForBroadcaster (Broadcaster &broadcaster)
{
    Mutex::Locker locker(m_manager_mutex);
    
    collection::iterator iter = m_event_map.begin(), end_iter = m_event_map.end();
    
    while (iter != end_iter 
           && (iter = find_if (iter, end_iter, BroadcasterClassMatches(broadcaster.GetBroadcasterClass()))) != end_iter)
    {
        (*iter).second->StartListeningForEvents (&broadcaster, (*iter).first.GetEventBits());
        iter++;
    }
}

void
BroadcasterManager::Clear ()
{
    Mutex::Locker locker(m_manager_mutex);
    listener_collection::iterator end_iter = m_listeners.end();
    
    for (listener_collection::iterator iter = m_listeners.begin(); iter != end_iter; iter++)
        (*iter)->BroadcasterManagerWillDestruct(this);
    m_listeners.clear();
    m_event_map.clear();
    
}
