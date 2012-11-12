//===-- BreakpointLocationList.cpp ------------------------------*- C++ -*-===//
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
#include "lldb/Breakpoint/BreakpointLocationList.h"
#include "lldb/Breakpoint/BreakpointLocation.h"
#include "lldb/Core/Section.h"

using namespace lldb;
using namespace lldb_private;

BreakpointLocationList::BreakpointLocationList(Breakpoint &owner) :
    m_owner (owner),
    m_locations(),
    m_address_to_location (),
    m_mutex (Mutex::eMutexTypeRecursive),
    m_next_id (0),
    m_new_location_recorder (NULL)
{
}

BreakpointLocationList::~BreakpointLocationList()
{
}

BreakpointLocationSP
BreakpointLocationList::Create (const Address &addr)
{
    Mutex::Locker locker (m_mutex);
    // The location ID is just the size of the location list + 1
    lldb::break_id_t bp_loc_id = ++m_next_id;
    BreakpointLocationSP bp_loc_sp (new BreakpointLocation (bp_loc_id, m_owner, addr));
    m_locations.push_back (bp_loc_sp);
    m_address_to_location[addr] = bp_loc_sp;
    return bp_loc_sp;
}

bool
BreakpointLocationList::ShouldStop (StoppointCallbackContext *context, lldb::break_id_t break_id)
{
    BreakpointLocationSP bp = FindByID (break_id);
    if (bp)
    {
        // Let the BreakpointLocation decide if it should stop here (could not have
        // reached it's target hit count yet, or it could have a callback
        // that decided it shouldn't stop (shared library loads/unloads).
        return bp->ShouldStop (context);
    }
    // We should stop here since this BreakpointLocation isn't valid anymore or it
    // doesn't exist.
    return true;
}

lldb::break_id_t
BreakpointLocationList::FindIDByAddress (const Address &addr)
{
    BreakpointLocationSP bp_loc_sp = FindByAddress (addr);
    if (bp_loc_sp)
    {
        return bp_loc_sp->GetID();
    }
    return LLDB_INVALID_BREAK_ID;
}

BreakpointLocationSP
BreakpointLocationList::FindByID (lldb::break_id_t break_id) const
{
    BreakpointLocationSP bp_loc_sp;
    Mutex::Locker locker (m_mutex);
    // We never remove a breakpoint locations, so the ID can be translated into
    // the location index by subtracting 1
    uint32_t idx = break_id - 1;
    if (idx <= m_locations.size())
    {
        bp_loc_sp = m_locations[idx];
    }
    return bp_loc_sp;
}

size_t
BreakpointLocationList::FindInModule (Module *module,
                                      BreakpointLocationCollection& bp_loc_list)
{
    Mutex::Locker locker (m_mutex);
    const size_t orig_size = bp_loc_list.GetSize();
    collection::iterator pos, end = m_locations.end();

    for (pos = m_locations.begin(); pos != end; ++pos)
    {
        BreakpointLocationSP break_loc = (*pos);
        SectionSP section_sp (break_loc->GetAddress().GetSection());
        if (section_sp && section_sp->GetModule().get() == module)
        {
            bp_loc_list.Add (break_loc);
        }
    }
    return bp_loc_list.GetSize() - orig_size;
}

const BreakpointLocationSP
BreakpointLocationList::FindByAddress (const Address &addr) const
{
    Mutex::Locker locker (m_mutex);
    BreakpointLocationSP bp_loc_sp;
    if (!m_locations.empty())
    {
        addr_map::const_iterator pos = m_address_to_location.find (addr);
        if (pos != m_address_to_location.end())
            bp_loc_sp = pos->second;
    }

    return bp_loc_sp;
}

void
BreakpointLocationList::Dump (Stream *s) const
{
    s->Printf("%p: ", this);
    //s->Indent();
    Mutex::Locker locker (m_mutex);
    s->Printf("BreakpointLocationList with %llu BreakpointLocations:\n", (uint64_t)m_locations.size());
    s->IndentMore();
    collection::const_iterator pos, end = m_locations.end();
    for (pos = m_locations.begin(); pos != end; ++pos)
        (*pos).get()->Dump(s);
    s->IndentLess();
}


BreakpointLocationSP
BreakpointLocationList::GetByIndex (uint32_t i)
{
    Mutex::Locker locker (m_mutex);
    BreakpointLocationSP bp_loc_sp;
    if (i < m_locations.size())
        bp_loc_sp = m_locations[i];

    return bp_loc_sp;
}

const BreakpointLocationSP
BreakpointLocationList::GetByIndex (uint32_t i) const
{
    Mutex::Locker locker (m_mutex);
    BreakpointLocationSP bp_loc_sp;
    if (i < m_locations.size())
        bp_loc_sp = m_locations[i];

    return bp_loc_sp;
}

void
BreakpointLocationList::ClearAllBreakpointSites ()
{
    Mutex::Locker locker (m_mutex);
    collection::iterator pos, end = m_locations.end();
    for (pos = m_locations.begin(); pos != end; ++pos)
        (*pos)->ClearBreakpointSite();
}

void
BreakpointLocationList::ResolveAllBreakpointSites ()
{
    Mutex::Locker locker (m_mutex);
    collection::iterator pos, end = m_locations.end();

    for (pos = m_locations.begin(); pos != end; ++pos)
    {
        if ((*pos)->IsEnabled())
            (*pos)->ResolveBreakpointSite();
    }
}

uint32_t
BreakpointLocationList::GetHitCount () const
{
    uint32_t hit_count = 0;
    Mutex::Locker locker (m_mutex);
    collection::const_iterator pos, end = m_locations.end();
    for (pos = m_locations.begin(); pos != end; ++pos)
        hit_count += (*pos)->GetHitCount();
    return hit_count;
}

size_t
BreakpointLocationList::GetNumResolvedLocations() const
{
    Mutex::Locker locker (m_mutex);
    size_t resolve_count = 0;
    collection::const_iterator pos, end = m_locations.end();
    for (pos = m_locations.begin(); pos != end; ++pos)
    {
        if ((*pos)->IsResolved())
            ++resolve_count;
    }
    return resolve_count;
}

void
BreakpointLocationList::GetDescription (Stream *s, lldb::DescriptionLevel level)
{
    Mutex::Locker locker (m_mutex);
    collection::iterator pos, end = m_locations.end();

    for (pos = m_locations.begin(); pos != end; ++pos)
    {
        s->Printf(" ");
        (*pos)->GetDescription(s, level);
    }
}

BreakpointLocationSP
BreakpointLocationList::AddLocation (const Address &addr, bool *new_location)
{
    Mutex::Locker locker (m_mutex);

    if (new_location)
        *new_location = false;
    BreakpointLocationSP bp_loc_sp (FindByAddress(addr));
    if (!bp_loc_sp)
	{
		bp_loc_sp = Create (addr);
		if (bp_loc_sp)
		{
	    	bp_loc_sp->ResolveBreakpointSite();

		    if (new_location)
	    	    *new_location = true;
            if(m_new_location_recorder)
            {
                m_new_location_recorder->Add(bp_loc_sp);
            }
		}
	}
    return bp_loc_sp;
}

bool
BreakpointLocationList::RemoveLocation (const lldb::BreakpointLocationSP &bp_loc_sp)
{
    if (bp_loc_sp)
    {
        Mutex::Locker locker (m_mutex);
        
        m_address_to_location.erase (bp_loc_sp->GetAddress());

        collection::iterator pos, end = m_locations.end();
        for (pos = m_locations.begin(); pos != end; ++pos)
        {
            if ((*pos).get() == bp_loc_sp.get())
            {
                m_locations.erase (pos);
                return true;
            }
        }
	}
    return false;
}



void
BreakpointLocationList::StartRecordingNewLocations (BreakpointLocationCollection &new_locations)
{
    Mutex::Locker locker (m_mutex);
    assert (m_new_location_recorder == NULL);
    m_new_location_recorder = &new_locations;
}

void
BreakpointLocationList::StopRecordingNewLocations ()
{
    Mutex::Locker locker (m_mutex);
    m_new_location_recorder = NULL;
}

