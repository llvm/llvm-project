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
#include "lldb/Core/Listener.h"
#include "lldb/Core/StreamString.h"

using namespace lldb;
using namespace lldb_private;

Broadcaster::Broadcaster(BroadcasterManagerSP manager_sp, const char *name) :
    m_broadcaster_sp(new BroadcasterImpl(*this)),
    m_manager_sp(manager_sp),
    m_broadcaster_name(name)
{
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_OBJECT));
    if (log)
        log->Printf ("%p Broadcaster::Broadcaster(\"%s\")",
                     static_cast<void*>(this), GetBroadcasterName().AsCString());
}

Broadcaster::BroadcasterImpl::BroadcasterImpl (Broadcaster &broadcaster) :
    m_broadcaster(broadcaster),
    m_listeners (),
    m_listeners_mutex (Mutex::eMutexTypeRecursive),
    m_hijacking_listeners(),
    m_hijacking_masks()
{
}

Broadcaster::~Broadcaster()
{
    Log *log (lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_OBJECT));
    if (log)
        log->Printf ("%p Broadcaster::~Broadcaster(\"%s\")",
                     static_cast<void*>(this), m_broadcaster_name.AsCString());

    Clear();
}

void
Broadcaster::CheckInWithManager ()
{
    if (m_manager_sp)
    {
        m_manager_sp->SignUpListenersForBroadcaster(*this);
    }
}

void
Broadcaster::BroadcasterImpl::Clear()
{
    Mutex::Locker listeners_locker(m_listeners_mutex);
    
    // Make sure the listener forgets about this broadcaster. We do
    // this in the broadcaster in case the broadcaster object initiates
    // the removal.
    
    collection::iterator pos, end = m_listeners.end();
    for (pos = m_listeners.begin(); pos != end; ++pos)
        pos->first->BroadcasterWillDestruct (&m_broadcaster);
    
    m_listeners.clear();
}

Broadcaster *
Broadcaster::BroadcasterImpl::GetBroadcaster()
{
    return &m_broadcaster;
}

bool
Broadcaster::BroadcasterImpl::GetEventNames (Stream &s, uint32_t event_mask, bool prefix_with_broadcaster_name) const
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
                        s.PutCString (GetBroadcasterName());
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
Broadcaster::AddInitialEventsToListener (lldb::ListenerSP listener_sp, uint32_t requested_events)
{
}

uint32_t
Broadcaster::BroadcasterImpl::AddListener (lldb::ListenerSP listener_sp, uint32_t event_mask)
{
    if (!listener_sp)
        return 0;

    Mutex::Locker locker(m_listeners_mutex);
    collection::iterator pos, end = m_listeners.end();

    collection::iterator existing_pos = end;
    // See if we already have this listener, and if so, update its mask
    uint32_t taken_event_types = 0;
    for (pos = m_listeners.begin(); pos != end; ++pos)
    {
        if (pos->first == listener_sp)
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
            m_listeners.push_back(std::make_pair(listener_sp, available_event_types));
        }
        else
        {
            // Grant the existing listener the available event bits
            existing_pos->second |= available_event_types;
        }

        // Individual broadcasters decide whether they have outstanding data when a
        // listener attaches, and insert it into the listener with this method.

        m_broadcaster.AddInitialEventsToListener (listener_sp, available_event_types);
    }

    // Return the event bits that were granted to the listener
    return available_event_types;
}

bool
Broadcaster::BroadcasterImpl::EventTypeHasListeners (uint32_t event_type)
{
    Mutex::Locker locker (m_listeners_mutex);
    
    if (!m_hijacking_listeners.empty() && event_type & m_hijacking_masks.back())
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
Broadcaster::BroadcasterImpl::RemoveListener (lldb::ListenerSP listener_sp, uint32_t event_mask)
{
    Mutex::Locker locker(m_listeners_mutex);
    collection::iterator pos, end = m_listeners.end();
    // See if we already have this listener, and if so, update its mask
    for (pos = m_listeners.begin(); pos != end; ++pos)
    {
        if (pos->first == listener_sp)
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
Broadcaster::BroadcasterImpl::BroadcastEvent (EventSP &event_sp)
{
    return PrivateBroadcastEvent (event_sp, false);
}

void
Broadcaster::BroadcasterImpl::BroadcastEventIfUnique (EventSP &event_sp)
{
    return PrivateBroadcastEvent (event_sp, true);
}

void
Broadcaster::BroadcasterImpl::PrivateBroadcastEvent (EventSP &event_sp, bool unique)
{
    // Can't add a nullptr event...
    if (!event_sp)
        return;

    // Update the broadcaster on this event
    event_sp->SetBroadcaster (&m_broadcaster);

    const uint32_t event_type = event_sp->GetType();

    Mutex::Locker event_types_locker(m_listeners_mutex);

    ListenerSP hijacking_listener_sp;

    if (!m_hijacking_listeners.empty())
    {
        assert (!m_hijacking_masks.empty());
        hijacking_listener_sp = m_hijacking_listeners.back();
        if ((event_type & m_hijacking_masks.back()) == 0)
            hijacking_listener_sp.reset();
    }

    Log *log(lldb_private::GetLogIfAnyCategoriesSet (LIBLLDB_LOG_EVENTS));
    if (log)
    {
        StreamString event_description;
        event_sp->Dump  (&event_description);
        log->Printf ("%p Broadcaster(\"%s\")::BroadcastEvent (event_sp = {%s}, unique =%i) hijack = %p",
                     static_cast<void*>(this), GetBroadcasterName(),
                     event_description.GetData(), unique,
                     static_cast<void*>(hijacking_listener_sp.get()));
    }

    if (hijacking_listener_sp)
    {
        if (unique && hijacking_listener_sp->PeekAtNextEventForBroadcasterWithType (&m_broadcaster, event_type))
            return;
        hijacking_listener_sp->AddEvent (event_sp);
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
                if (unique && pos->first->PeekAtNextEventForBroadcasterWithType (&m_broadcaster, event_type))
                    continue;
                pos->first->AddEvent (event_sp);
            }
        }
    }
}

void
Broadcaster::BroadcasterImpl::BroadcastEvent (uint32_t event_type, EventData *event_data)
{
    EventSP event_sp (new Event (event_type, event_data));
    PrivateBroadcastEvent (event_sp, false);
}

void
Broadcaster::BroadcasterImpl::BroadcastEventIfUnique (uint32_t event_type, EventData *event_data)
{
    EventSP event_sp (new Event (event_type, event_data));
    PrivateBroadcastEvent (event_sp, true);
}

bool
Broadcaster::BroadcasterImpl::HijackBroadcaster (ListenerSP listener_sp, uint32_t event_mask)
{
    Mutex::Locker event_types_locker(m_listeners_mutex);

    Log *log(lldb_private::GetLogIfAnyCategoriesSet (LIBLLDB_LOG_EVENTS));
    if (log)
        log->Printf ("%p Broadcaster(\"%s\")::HijackBroadcaster (listener(\"%s\")=%p)",
                     static_cast<void*>(this), GetBroadcasterName(),
                     listener_sp->m_name.c_str(), static_cast<void*>(listener_sp.get()));
    m_hijacking_listeners.push_back(listener_sp);
    m_hijacking_masks.push_back(event_mask);
    return true;
}

bool
Broadcaster::BroadcasterImpl::IsHijackedForEvent (uint32_t event_mask)
{
    Mutex::Locker event_types_locker(m_listeners_mutex);

    if (!m_hijacking_listeners.empty())
        return (event_mask & m_hijacking_masks.back()) != 0;
    return false;
}

const char *
Broadcaster::BroadcasterImpl::GetHijackingListenerName()
{
    if (m_hijacking_listeners.size())
    {
        return m_hijacking_listeners.back()->GetName();
    }
    else
    {
        return nullptr;
    }
}

void
Broadcaster::BroadcasterImpl::RestoreBroadcaster ()
{
    Mutex::Locker event_types_locker(m_listeners_mutex);

    if (!m_hijacking_listeners.empty())
    {
        Log *log(lldb_private::GetLogIfAnyCategoriesSet (LIBLLDB_LOG_EVENTS));
        if (log)
        {
            ListenerSP listener_sp = m_hijacking_listeners.back();
            log->Printf ("%p Broadcaster(\"%s\")::RestoreBroadcaster (about to pop listener(\"%s\")=%p)",
                         static_cast<void*>(this),
                         GetBroadcasterName(),
                         listener_sp->m_name.c_str(), static_cast<void*>(listener_sp.get()));
        }
        m_hijacking_listeners.pop_back();
    }
    if (!m_hijacking_masks.empty())
        m_hijacking_masks.pop_back();
}

ConstString &
Broadcaster::GetBroadcasterClass() const
{
    static ConstString class_name ("lldb.anonymous");
    return class_name;
}

BroadcastEventSpec::BroadcastEventSpec(const BroadcastEventSpec &rhs) = default;

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

BroadcastEventSpec &
BroadcastEventSpec::operator=(const BroadcastEventSpec &rhs) = default;

BroadcasterManager::BroadcasterManager() :
    m_manager_mutex(Mutex::eMutexTypeRecursive)
{
}

lldb::BroadcasterManagerSP
BroadcasterManager::MakeBroadcasterManager()
{
    return BroadcasterManagerSP(new BroadcasterManager());
}

uint32_t
BroadcasterManager::RegisterListenerForEvents (ListenerSP listener_sp, BroadcastEventSpec event_spec)
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
        m_event_map.insert (event_listener_key (BroadcastEventSpec (event_spec.GetBroadcasterClass(), available_bits), listener_sp));
        m_listeners.insert(listener_sp);
    }
    
    return available_bits;
}

bool
BroadcasterManager::UnregisterListenerForEvents (ListenerSP listener_sp, BroadcastEventSpec event_spec)
{
    Mutex::Locker locker(m_manager_mutex);
    bool removed_some = false;
    
    if (m_listeners.erase(listener_sp) == 0)
        return false;
    
    ListenerMatchesAndSharedBits predicate (event_spec, listener_sp);
    std::vector<BroadcastEventSpec> to_be_readded;
    uint32_t event_bits_to_remove = event_spec.GetEventBits();
    
    // Go through the map and delete the exact matches, and build a list of matches that weren't exact to re-add:
    while (true)
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
        m_event_map.insert (event_listener_key (to_be_readded[i], listener_sp));
    }
    
    return removed_some;
}
    
ListenerSP
BroadcasterManager::GetListenerForEventSpec (BroadcastEventSpec event_spec) const
{
    Mutex::Locker locker(*(const_cast<Mutex *> (&m_manager_mutex)));
    
    collection::const_iterator iter, end_iter = m_event_map.end();
    iter = find_if (m_event_map.begin(), end_iter, BroadcastEventSpecMatches (event_spec));
    if (iter != end_iter)
        return (*iter).second;
    else
        return nullptr;
}

void
BroadcasterManager::RemoveListener(Listener *listener)
{
    Mutex::Locker locker(m_manager_mutex);
    ListenerMatchesPointer predicate (listener);
    listener_collection::iterator iter = m_listeners.begin(), end_iter = m_listeners.end();
    
    std::find_if (iter, end_iter, predicate);
    if (iter != end_iter)
        m_listeners.erase(iter);
    
    while (true)
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
BroadcasterManager::RemoveListener (ListenerSP listener_sp)
{
    Mutex::Locker locker(m_manager_mutex);
    ListenerMatches predicate (listener_sp);

    if (m_listeners.erase (listener_sp) == 0)
        return;
        
    while (true)
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
        (*iter)->BroadcasterManagerWillDestruct(this->shared_from_this());
    m_listeners.clear();
    m_event_map.clear();
}
