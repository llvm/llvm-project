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

Broadcaster::BroadcasterImpl::BroadcasterImpl(Broadcaster &broadcaster)
    : m_broadcaster(broadcaster), m_listeners(), m_listeners_mutex(), m_hijacking_listeners(), m_hijacking_masks()
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
Broadcaster::BroadcasterImpl::ListenerIterator (std::function <bool (const lldb::ListenerSP &listener_sp, uint32_t &event_mask)> const &callback)
{
    // Private iterator that should be used by everyone except BroadcasterImpl::RemoveListener().
    // We have weak pointers to our listeners which means that at any point the listener can
    // expire which means we will need to take it out of our list. To take care of this, we
    // iterate and check that the weak pointer can be made into a valid shared pointer before
    // we call the callback. If the weak pointer has expired, we remove it from our list.
    collection::iterator pos = m_listeners.begin();
    while (pos != m_listeners.end())
    {
        lldb::ListenerSP curr_listener_sp(pos->first.lock());
        if (curr_listener_sp)
        {
            if (callback(curr_listener_sp, pos->second))
                ++pos;  // Keep iterating
            else
                return; // Done iterating
        }
        else
        {
            // The listener has been expired. Remove this entry.
            pos = m_listeners.erase(pos);
        }
    }
}

void
Broadcaster::BroadcasterImpl::Clear()
{
    std::lock_guard<std::recursive_mutex> guard(m_listeners_mutex);

    // Make sure the listener forgets about this broadcaster. We do
    // this in the broadcaster in case the broadcaster object initiates
    // the removal.

    ListenerIterator([this](const lldb::ListenerSP &curr_listener_sp, uint32_t &curr_event_mask) -> bool {
        curr_listener_sp->BroadcasterWillDestruct (&m_broadcaster);
        return true; // Keep iterating
    });
    
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
Broadcaster::AddInitialEventsToListener (const lldb::ListenerSP &listener_sp, uint32_t requested_events)
{
}

uint32_t
Broadcaster::BroadcasterImpl::AddListener (const lldb::ListenerSP &listener_sp, uint32_t event_mask)
{
    if (!listener_sp)
        return 0;

    std::lock_guard<std::recursive_mutex> guard(m_listeners_mutex);

    // See if we already have this listener, and if so, update its mask

    bool handled = false;

    ListenerIterator([this, &listener_sp, &handled, event_mask](const lldb::ListenerSP &curr_listener_sp, uint32_t &curr_event_mask) -> bool {
        if (curr_listener_sp == listener_sp)
        {
            handled = true;
            curr_event_mask |= event_mask;
            m_broadcaster.AddInitialEventsToListener (listener_sp, event_mask);
            return false; // Stop iterating
        }
        return true; // Keep iterating
    });

    if (!handled)
    {
        // Grant a new listener the available event bits
        m_listeners.push_back(std::make_pair(lldb::ListenerWP(listener_sp), event_mask));

        // Individual broadcasters decide whether they have outstanding data when a
        // listener attaches, and insert it into the listener with this method.
        m_broadcaster.AddInitialEventsToListener (listener_sp, event_mask);
    }

    // Return the event bits that were granted to the listener
    return event_mask;
}

bool
Broadcaster::BroadcasterImpl::EventTypeHasListeners (uint32_t event_type)
{
    std::lock_guard<std::recursive_mutex> guard(m_listeners_mutex);

    if (!m_hijacking_listeners.empty() && event_type & m_hijacking_masks.back())
        return true;
        
    if (m_listeners.empty())
        return false;

    bool result = false;
    ListenerIterator([this, event_type, &result](const lldb::ListenerSP &curr_listener_sp, uint32_t &curr_event_mask) -> bool {

        if (curr_event_mask & event_type)
        {
            result = true;
            return false; // Stop iterating
        }
        else
        {
            return true; // Keep iterating
        }
    });
    return result;
}

bool
Broadcaster::BroadcasterImpl::RemoveListener (lldb_private::Listener *listener, uint32_t event_mask)
{
    if (listener)
    {
        std::lock_guard<std::recursive_mutex> guard(m_listeners_mutex);
        collection::iterator pos = m_listeners.begin();
        // See if we already have this listener, and if so, update its mask
        while (pos != m_listeners.end())
        {
            lldb::ListenerSP curr_listener_sp(pos->first.lock());
            if (curr_listener_sp)
            {
                if (curr_listener_sp.get() == listener)
                {
                    // Relinquish all event bits in "event_mask"
                    pos->second &= ~event_mask;
                    // If all bits have been relinquished then remove this listener
                    if (pos->second == 0)
                        m_listeners.erase (pos);
                    return true;
                }
                ++pos;
            }
            else
            {
                // The listener has been destroyed since we couldn't turn the std::weak_ptr
                // into a valid shared pointer, so we can remove it.
                pos = m_listeners.erase (pos);
            }
        }
    }
    return false;
}

bool
Broadcaster::BroadcasterImpl::RemoveListener (const lldb::ListenerSP &listener_sp, uint32_t event_mask)
{
    return RemoveListener (listener_sp.get(), event_mask);
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

    std::lock_guard<std::recursive_mutex> guard(m_listeners_mutex);

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

        ListenerIterator([this, unique, event_type, &event_sp](const lldb::ListenerSP &curr_listener_sp, uint32_t &curr_event_mask) -> bool {

            if (event_type & curr_event_mask)
            {
                if (!unique || curr_listener_sp->PeekAtNextEventForBroadcasterWithType (&m_broadcaster, event_type) == nullptr)
                    curr_listener_sp->AddEvent (event_sp);
            }
            return true; // Keep iterating
        });
    }
}

void
Broadcaster::BroadcasterImpl::BroadcastEvent (uint32_t event_type, EventData *event_data)
{
    EventSP event_sp (new Event (event_type, event_data));
    PrivateBroadcastEvent (event_sp, false);
}

void
Broadcaster::BroadcasterImpl::BroadcastEvent (uint32_t event_type, const lldb::EventDataSP &event_data_sp)
{
    EventSP event_sp (new Event (event_type, event_data_sp));
    PrivateBroadcastEvent (event_sp, false);
}

void
Broadcaster::BroadcasterImpl::BroadcastEventIfUnique (uint32_t event_type, EventData *event_data)
{
    EventSP event_sp (new Event (event_type, event_data));
    PrivateBroadcastEvent (event_sp, true);
}

bool
Broadcaster::BroadcasterImpl::HijackBroadcaster (const lldb::ListenerSP &listener_sp, uint32_t event_mask)
{
    std::lock_guard<std::recursive_mutex> guard(m_listeners_mutex);

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
    std::lock_guard<std::recursive_mutex> guard(m_listeners_mutex);

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
    std::lock_guard<std::recursive_mutex> guard(m_listeners_mutex);

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

BroadcasterManager::BroadcasterManager() : m_manager_mutex()
{
}

lldb::BroadcasterManagerSP
BroadcasterManager::MakeBroadcasterManager()
{
    return BroadcasterManagerSP(new BroadcasterManager());
}

uint32_t
BroadcasterManager::RegisterListenerForEvents (const lldb::ListenerSP &listener_sp, BroadcastEventSpec event_spec)
{
    std::lock_guard<std::recursive_mutex> guard(m_manager_mutex);

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
BroadcasterManager::UnregisterListenerForEvents (const lldb::ListenerSP &listener_sp, BroadcastEventSpec event_spec)
{
    std::lock_guard<std::recursive_mutex> guard(m_manager_mutex);
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
    std::lock_guard<std::recursive_mutex> guard(m_manager_mutex);

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
    std::lock_guard<std::recursive_mutex> guard(m_manager_mutex);
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
BroadcasterManager::RemoveListener (const lldb::ListenerSP &listener_sp)
{
    std::lock_guard<std::recursive_mutex> guard(m_manager_mutex);
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
    std::lock_guard<std::recursive_mutex> guard(m_manager_mutex);

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
    std::lock_guard<std::recursive_mutex> guard(m_manager_mutex);
    listener_collection::iterator end_iter = m_listeners.end();
    
    for (listener_collection::iterator iter = m_listeners.begin(); iter != end_iter; iter++)
        (*iter)->BroadcasterManagerWillDestruct(this->shared_from_this());
    m_listeners.clear();
    m_event_map.clear();
}
