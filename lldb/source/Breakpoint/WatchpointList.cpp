//===-- WatchpointList.cpp --------------------------------------*- C++ -*-===//
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
#include "lldb/Breakpoint/WatchpointList.h"
#include "lldb/Breakpoint/Watchpoint.h"

using namespace lldb;
using namespace lldb_private;

WatchpointList::WatchpointList() :
    m_watchpoints (),
    m_mutex (Mutex::eMutexTypeRecursive),
    m_next_wp_id (0)
{
}

WatchpointList::~WatchpointList()
{
}

// Add a watchpoint to the list.
lldb::watch_id_t
WatchpointList::Add (const WatchpointSP &wp_sp, bool notify)
{
    Mutex::Locker locker (m_mutex);
    wp_sp->SetID(++m_next_wp_id);
    m_watchpoints.push_back(wp_sp);
    if (notify)
    {
        if (wp_sp->GetTarget().EventTypeHasListeners(Target::eBroadcastBitWatchpointChanged))
            wp_sp->GetTarget().BroadcastEvent (Target::eBroadcastBitWatchpointChanged,
                                               new Watchpoint::WatchpointEventData (eWatchpointEventTypeAdded, wp_sp));
    }
    return wp_sp->GetID();
}

void
WatchpointList::Dump (Stream *s) const
{
    DumpWithLevel(s, lldb::eDescriptionLevelBrief);
}

void
WatchpointList::DumpWithLevel (Stream *s, lldb::DescriptionLevel description_level) const
{
    Mutex::Locker locker (m_mutex);
    s->Printf("%p: ", this);
    //s->Indent();
    s->Printf("WatchpointList with %" PRIu64 " Watchpoints:\n",
              (uint64_t)m_watchpoints.size());
    s->IndentMore();
    wp_collection::const_iterator pos, end = m_watchpoints.end();
    for (pos = m_watchpoints.begin(); pos != end; ++pos)
        (*pos)->DumpWithLevel(s, description_level);
    s->IndentLess();
}

const WatchpointSP
WatchpointList::FindByAddress (lldb::addr_t addr) const
{
    WatchpointSP wp_sp;
    Mutex::Locker locker (m_mutex);
    if (!m_watchpoints.empty())
    {
        wp_collection::const_iterator pos, end = m_watchpoints.end();
        for (pos = m_watchpoints.begin(); pos != end; ++pos)
            if ((*pos)->GetLoadAddress() == addr) {
                wp_sp = *pos;
                break;
            }
    }

    return wp_sp;
}

const WatchpointSP
WatchpointList::FindBySpec (std::string spec) const
{
    WatchpointSP wp_sp;
    Mutex::Locker locker (m_mutex);
    if (!m_watchpoints.empty())
    {
        wp_collection::const_iterator pos, end = m_watchpoints.end();
        for (pos = m_watchpoints.begin(); pos != end; ++pos)
            if ((*pos)->GetWatchSpec() == spec) {
                wp_sp = *pos;
                break;
            }
    }

    return wp_sp;
}

class WatchpointIDMatches
{
public:
    WatchpointIDMatches (lldb::watch_id_t watch_id) :
        m_watch_id(watch_id)
    {
    }

    bool operator() (const WatchpointSP &wp) const
    {
        return m_watch_id == wp->GetID();
    }

private:
   const lldb::watch_id_t m_watch_id;
};

WatchpointList::wp_collection::iterator
WatchpointList::GetIDIterator (lldb::watch_id_t watch_id)
{
    return std::find_if(m_watchpoints.begin(), m_watchpoints.end(), // Search full range
                        WatchpointIDMatches(watch_id));             // Predicate
}

WatchpointList::wp_collection::const_iterator
WatchpointList::GetIDConstIterator (lldb::watch_id_t watch_id) const
{
    return std::find_if(m_watchpoints.begin(), m_watchpoints.end(), // Search full range
                        WatchpointIDMatches(watch_id));             // Predicate
}

WatchpointSP
WatchpointList::FindByID (lldb::watch_id_t watch_id) const
{
    WatchpointSP wp_sp;
    Mutex::Locker locker (m_mutex);
    wp_collection::const_iterator pos = GetIDConstIterator(watch_id);
    if (pos != m_watchpoints.end())
        wp_sp = *pos;

    return wp_sp;
}

lldb::watch_id_t
WatchpointList::FindIDByAddress (lldb::addr_t addr)
{
    WatchpointSP wp_sp = FindByAddress (addr);
    if (wp_sp)
    {
        return wp_sp->GetID();
    }
    return LLDB_INVALID_WATCH_ID;
}

lldb::watch_id_t
WatchpointList::FindIDBySpec (std::string spec)
{
    WatchpointSP wp_sp = FindBySpec (spec);
    if (wp_sp)
    {
        return wp_sp->GetID();
    }
    return LLDB_INVALID_WATCH_ID;
}

WatchpointSP
WatchpointList::GetByIndex (uint32_t i)
{
    Mutex::Locker locker (m_mutex);
    WatchpointSP wp_sp;
    if (i < m_watchpoints.size())
    {
        wp_collection::const_iterator pos = m_watchpoints.begin();
        std::advance(pos, i);
        wp_sp = *pos;
    }
    return wp_sp;
}

const WatchpointSP
WatchpointList::GetByIndex (uint32_t i) const
{
    Mutex::Locker locker (m_mutex);
    WatchpointSP wp_sp;
    if (i < m_watchpoints.size())
    {
        wp_collection::const_iterator pos = m_watchpoints.begin();
        std::advance(pos, i);
        wp_sp = *pos;
    }
    return wp_sp;
}

std::vector<lldb::watch_id_t>
WatchpointList::GetWatchpointIDs() const
{
    std::vector<lldb::watch_id_t> IDs;
    wp_collection::const_iterator pos, end = m_watchpoints.end();
    for (pos = m_watchpoints.begin(); pos != end; ++pos)
        IDs.push_back((*pos)->GetID());
    return IDs;
}

bool
WatchpointList::Remove (lldb::watch_id_t watch_id, bool notify)
{
    Mutex::Locker locker (m_mutex);
    wp_collection::iterator pos = GetIDIterator(watch_id);
    if (pos != m_watchpoints.end())
    {
        WatchpointSP wp_sp = *pos;
        if (notify)
        {
            if (wp_sp->GetTarget().EventTypeHasListeners(Target::eBroadcastBitWatchpointChanged))
                wp_sp->GetTarget().BroadcastEvent (Target::eBroadcastBitWatchpointChanged,
                                                   new Watchpoint::WatchpointEventData (eWatchpointEventTypeRemoved, wp_sp));
        }
        m_watchpoints.erase(pos);
        return true;
    }
    return false;
}

uint32_t
WatchpointList::GetHitCount () const
{
    uint32_t hit_count = 0;
    Mutex::Locker locker (m_mutex);
    wp_collection::const_iterator pos, end = m_watchpoints.end();
    for (pos = m_watchpoints.begin(); pos != end; ++pos)
        hit_count += (*pos)->GetHitCount();
    return hit_count;
}

bool
WatchpointList::ShouldStop (StoppointCallbackContext *context, lldb::watch_id_t watch_id)
{

    WatchpointSP wp_sp = FindByID (watch_id);
    if (wp_sp)
    {
        // Let the Watchpoint decide if it should stop here (could not have
        // reached it's target hit count yet, or it could have a callback
        // that decided it shouldn't stop.
        return wp_sp->ShouldStop (context);
    }
    // We should stop here since this Watchpoint isn't valid anymore or it
    // doesn't exist.
    return true;
}

void
WatchpointList::GetDescription (Stream *s, lldb::DescriptionLevel level)
{
    Mutex::Locker locker (m_mutex);
    wp_collection::iterator pos, end = m_watchpoints.end();

    for (pos = m_watchpoints.begin(); pos != end; ++pos)
    {
        s->Printf(" ");
        (*pos)->Dump(s);
    }
}

void
WatchpointList::SetEnabledAll (bool enabled)
{
    Mutex::Locker locker(m_mutex);

    wp_collection::iterator pos, end = m_watchpoints.end();
    for (pos = m_watchpoints.begin(); pos != end; ++pos)
        (*pos)->SetEnabled (enabled);
}

void
WatchpointList::RemoveAll (bool notify)
{
    Mutex::Locker locker(m_mutex);
    if (notify)
    {
        
        {
            wp_collection::iterator pos, end = m_watchpoints.end();
            for (pos = m_watchpoints.begin(); pos != end; ++pos)
            {
                if ((*pos)->GetTarget().EventTypeHasListeners(Target::eBroadcastBitBreakpointChanged))
                {
                    (*pos)->GetTarget().BroadcastEvent (Target::eBroadcastBitWatchpointChanged,
                                                        new Watchpoint::WatchpointEventData (eWatchpointEventTypeRemoved,
                                                                                             *pos));
                }
            }
        }
    }
    m_watchpoints.clear();
}

void
WatchpointList::GetListMutex (Mutex::Locker &locker)
{
    return locker.Lock (m_mutex);
}
