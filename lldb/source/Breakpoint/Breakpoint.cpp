//===-- Breakpoint.cpp ------------------------------------------*- C++ -*-===//
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

#include "lldb/Core/Address.h"
#include "lldb/Breakpoint/Breakpoint.h"
#include "lldb/Breakpoint/BreakpointLocation.h"
#include "lldb/Breakpoint/BreakpointLocationCollection.h"
#include "lldb/Breakpoint/BreakpointResolver.h"
#include "lldb/Breakpoint/BreakpointResolverFileLine.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/ModuleList.h"
#include "lldb/Core/SearchFilter.h"
#include "lldb/Core/Section.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/ThreadSpec.h"
#include "lldb/lldb-private-log.h"
#include "llvm/Support/Casting.h"

using namespace lldb;
using namespace lldb_private;
using namespace llvm;

const ConstString &
Breakpoint::GetEventIdentifier ()
{
    static ConstString g_identifier("event-identifier.breakpoint.changed");
    return g_identifier;
}

//----------------------------------------------------------------------
// Breakpoint constructor
//----------------------------------------------------------------------
Breakpoint::Breakpoint(Target &target, SearchFilterSP &filter_sp, BreakpointResolverSP &resolver_sp) :
    m_being_created(true),
    m_target (target),
    m_filter_sp (filter_sp),
    m_resolver_sp (resolver_sp),
    m_options (),
    m_locations (*this)
{
    m_being_created = false;
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
Breakpoint::~Breakpoint()
{
}

bool
Breakpoint::IsInternal () const
{
    return LLDB_BREAK_ID_IS_INTERNAL(m_bid);
}



Target&
Breakpoint::GetTarget ()
{
    return m_target;
}

const Target&
Breakpoint::GetTarget () const
{
    return m_target;
}

BreakpointLocationSP
Breakpoint::AddLocation (const Address &addr, bool *new_location)
{
    return m_locations.AddLocation (addr, new_location);
}

BreakpointLocationSP
Breakpoint::FindLocationByAddress (const Address &addr)
{
    return m_locations.FindByAddress(addr);
}

break_id_t
Breakpoint::FindLocationIDByAddress (const Address &addr)
{
    return m_locations.FindIDByAddress(addr);
}

BreakpointLocationSP
Breakpoint::FindLocationByID (break_id_t bp_loc_id)
{
    return m_locations.FindByID(bp_loc_id);
}

BreakpointLocationSP
Breakpoint::GetLocationAtIndex (size_t index)
{
    return m_locations.GetByIndex(index);
}

// For each of the overall options we need to decide how they propagate to
// the location options.  This will determine the precedence of options on
// the breakpoint vs. its locations.

// Disable at the breakpoint level should override the location settings.
// That way you can conveniently turn off a whole breakpoint without messing
// up the individual settings.

void
Breakpoint::SetEnabled (bool enable)
{
    if (enable == m_options.IsEnabled())
        return;

    m_options.SetEnabled(enable);
    if (enable)
        m_locations.ResolveAllBreakpointSites();
    else
        m_locations.ClearAllBreakpointSites();
        
    SendBreakpointChangedEvent (enable ? eBreakpointEventTypeEnabled : eBreakpointEventTypeDisabled);

}

bool
Breakpoint::IsEnabled ()
{
    return m_options.IsEnabled();
}

void
Breakpoint::SetIgnoreCount (uint32_t n)
{
    if (m_options.GetIgnoreCount() == n)
        return;
        
    m_options.SetIgnoreCount(n);
    SendBreakpointChangedEvent (eBreakpointEventTypeIgnoreChanged);
}

void
Breakpoint::DecrementIgnoreCount ()
{
    uint32_t ignore = m_options.GetIgnoreCount();
    if (ignore != 0)
        m_options.SetIgnoreCount(ignore - 1);
}

uint32_t
Breakpoint::GetIgnoreCount () const
{
    return m_options.GetIgnoreCount();
}

bool
Breakpoint::IgnoreCountShouldStop ()
{
    uint32_t ignore = GetIgnoreCount();
    if (ignore != 0)
    {
        // When we get here we know the location that caused the stop doesn't have an ignore count,
        // since by contract we call it first...  So we don't have to find & decrement it, we only have
        // to decrement our own ignore count.
        DecrementIgnoreCount();
        return false;
    }
    else
        return true;
}

uint32_t
Breakpoint::GetHitCount () const
{
    return m_locations.GetHitCount();
}

bool
Breakpoint::IsOneShot () const
{
    return m_options.IsOneShot();
}

void
Breakpoint::SetOneShot (bool one_shot)
{
    m_options.SetOneShot (one_shot);
}

void
Breakpoint::SetThreadID (lldb::tid_t thread_id)
{
    if (m_options.GetThreadSpec()->GetTID() == thread_id)
        return;
        
    m_options.GetThreadSpec()->SetTID(thread_id);
    SendBreakpointChangedEvent (eBreakpointEventTypeThreadChanged);
}

lldb::tid_t
Breakpoint::GetThreadID () const
{
    if (m_options.GetThreadSpecNoCreate() == NULL)
        return LLDB_INVALID_THREAD_ID;
    else
        return m_options.GetThreadSpecNoCreate()->GetTID();
}

void
Breakpoint::SetThreadIndex (uint32_t index)
{
    if (m_options.GetThreadSpec()->GetIndex() == index)
        return;
        
    m_options.GetThreadSpec()->SetIndex(index);
    SendBreakpointChangedEvent (eBreakpointEventTypeThreadChanged);
}

uint32_t
Breakpoint::GetThreadIndex() const
{
    if (m_options.GetThreadSpecNoCreate() == NULL)
        return 0;
    else
        return m_options.GetThreadSpecNoCreate()->GetIndex();
}

void
Breakpoint::SetThreadName (const char *thread_name)
{
    if (m_options.GetThreadSpec()->GetName() != NULL
        && ::strcmp (m_options.GetThreadSpec()->GetName(), thread_name) == 0)
        return;
        
    m_options.GetThreadSpec()->SetName (thread_name);
    SendBreakpointChangedEvent (eBreakpointEventTypeThreadChanged);
}

const char *
Breakpoint::GetThreadName () const
{
    if (m_options.GetThreadSpecNoCreate() == NULL)
        return NULL;
    else
        return m_options.GetThreadSpecNoCreate()->GetName();
}

void 
Breakpoint::SetQueueName (const char *queue_name)
{
    if (m_options.GetThreadSpec()->GetQueueName() != NULL
        && ::strcmp (m_options.GetThreadSpec()->GetQueueName(), queue_name) == 0)
        return;
        
    m_options.GetThreadSpec()->SetQueueName (queue_name);
    SendBreakpointChangedEvent (eBreakpointEventTypeThreadChanged);
}

const char *
Breakpoint::GetQueueName () const
{
    if (m_options.GetThreadSpecNoCreate() == NULL)
        return NULL;
    else
        return m_options.GetThreadSpecNoCreate()->GetQueueName();
}

void 
Breakpoint::SetCondition (const char *condition)
{
    m_options.SetCondition (condition);
    SendBreakpointChangedEvent (eBreakpointEventTypeConditionChanged);
}

const char *
Breakpoint::GetConditionText () const
{
    return m_options.GetConditionText();
}

// This function is used when "baton" doesn't need to be freed
void
Breakpoint::SetCallback (BreakpointHitCallback callback, void *baton, bool is_synchronous)
{
    // The default "Baton" class will keep a copy of "baton" and won't free
    // or delete it when it goes goes out of scope.
    m_options.SetCallback(callback, BatonSP (new Baton(baton)), is_synchronous);
    
    SendBreakpointChangedEvent (eBreakpointEventTypeCommandChanged);
}

// This function is used when a baton needs to be freed and therefore is 
// contained in a "Baton" subclass.
void
Breakpoint::SetCallback (BreakpointHitCallback callback, const BatonSP &callback_baton_sp, bool is_synchronous)
{
    m_options.SetCallback(callback, callback_baton_sp, is_synchronous);
}

void
Breakpoint::ClearCallback ()
{
    m_options.ClearCallback ();
}

bool
Breakpoint::InvokeCallback (StoppointCallbackContext *context, break_id_t bp_loc_id)
{
    return m_options.InvokeCallback (context, GetID(), bp_loc_id);
}

BreakpointOptions *
Breakpoint::GetOptions ()
{
    return &m_options;
}

void
Breakpoint::ResolveBreakpoint ()
{
    if (m_resolver_sp)
        m_resolver_sp->ResolveBreakpoint(*m_filter_sp);
}

void
Breakpoint::ResolveBreakpointInModules (ModuleList &module_list)
{
    if (m_resolver_sp)
        m_resolver_sp->ResolveBreakpointInModules(*m_filter_sp, module_list);
}

void
Breakpoint::ClearAllBreakpointSites ()
{
    m_locations.ClearAllBreakpointSites();
}

//----------------------------------------------------------------------
// ModulesChanged: Pass in a list of new modules, and
//----------------------------------------------------------------------

void
Breakpoint::ModulesChanged (ModuleList &module_list, bool load, bool delete_locations)
{
    Mutex::Locker modules_mutex(module_list.GetMutex());
    if (load)
    {
        // The logic for handling new modules is:
        // 1) If the filter rejects this module, then skip it.
        // 2) Run through the current location list and if there are any locations
        //    for that module, we mark the module as "seen" and we don't try to re-resolve
        //    breakpoint locations for that module.
        //    However, we do add breakpoint sites to these locations if needed.
        // 3) If we don't see this module in our breakpoint location list, call ResolveInModules.

        ModuleList new_modules;  // We'll stuff the "unseen" modules in this list, and then resolve
                                 // them after the locations pass.  Have to do it this way because
                                 // resolving breakpoints will add new locations potentially.

        const size_t num_locs = m_locations.GetSize();
        size_t num_modules = module_list.GetSize();
        for (size_t i = 0; i < num_modules; i++)
        {
            bool seen = false;
            ModuleSP module_sp (module_list.GetModuleAtIndexUnlocked (i));
            if (!m_filter_sp->ModulePasses (module_sp))
                continue;

            for (size_t loc_idx = 0; loc_idx < num_locs; loc_idx++)
            {
                BreakpointLocationSP break_loc = m_locations.GetByIndex(loc_idx);
                if (!break_loc->IsEnabled())
                    continue;
                SectionSP section_sp (break_loc->GetAddress().GetSection());
                if (!section_sp || section_sp->GetModule() == module_sp)
                {
                    if (!seen)
                        seen = true;

                    if (!break_loc->ResolveBreakpointSite())
                    {
                        Log *log (lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_BREAKPOINTS));
                        if (log)
                            log->Printf ("Warning: could not set breakpoint site for breakpoint location %d of breakpoint %d.\n",
                                         break_loc->GetID(), GetID());
                    }
                }
            }

            if (!seen)
                new_modules.AppendIfNeeded (module_sp);

        }
        
        if (new_modules.GetSize() > 0)
        {
            // If this is not an internal breakpoint, set up to record the new locations, then dispatch
            // an event with the new locations.
            if (!IsInternal())
            {
                BreakpointEventData *new_locations_event = new BreakpointEventData (eBreakpointEventTypeLocationsAdded, 
                                                                                    shared_from_this());
                
                m_locations.StartRecordingNewLocations(new_locations_event->GetBreakpointLocationCollection());
                
                ResolveBreakpointInModules(new_modules);

                m_locations.StopRecordingNewLocations();
                if (new_locations_event->GetBreakpointLocationCollection().GetSize() != 0)
                {
                    SendBreakpointChangedEvent (new_locations_event);
                }
                else
                    delete new_locations_event;
            }
            else
                ResolveBreakpointInModules(new_modules);
            
        }
    }
    else
    {
        // Go through the currently set locations and if any have breakpoints in
        // the module list, then remove their breakpoint sites, and their locations if asked to.

        BreakpointEventData *removed_locations_event;
        if (!IsInternal())
            removed_locations_event = new BreakpointEventData (eBreakpointEventTypeLocationsRemoved, 
                                                               shared_from_this());
        else
            removed_locations_event = NULL;
        
        size_t num_modules = module_list.GetSize();
        for (size_t i = 0; i < num_modules; i++)
        {
            ModuleSP module_sp (module_list.GetModuleAtIndexUnlocked (i));
            if (m_filter_sp->ModulePasses (module_sp))
            {
                size_t loc_idx = 0;
                size_t num_locations = m_locations.GetSize();
                BreakpointLocationCollection locations_to_remove;
                for (loc_idx = 0; loc_idx < num_locations; loc_idx++)
                {
                    BreakpointLocationSP break_loc_sp (m_locations.GetByIndex(loc_idx));
                    SectionSP section_sp (break_loc_sp->GetAddress().GetSection());
                    if (section_sp && section_sp->GetModule() == module_sp)
                    {
                        // Remove this breakpoint since the shared library is 
                        // unloaded, but keep the breakpoint location around
                        // so we always get complete hit count and breakpoint
                        // lifetime info
                        break_loc_sp->ClearBreakpointSite();
                        if (removed_locations_event)
                        {
                            removed_locations_event->GetBreakpointLocationCollection().Add(break_loc_sp);
                        }
                        if (delete_locations)
                            locations_to_remove.Add (break_loc_sp);
                            
                    }
                }
                
                if (delete_locations)
                {
                    size_t num_locations_to_remove = locations_to_remove.GetSize();
                    for (loc_idx = 0; loc_idx < num_locations_to_remove; loc_idx++)
                        m_locations.RemoveLocation  (locations_to_remove.GetByIndex(loc_idx));
                }
            }
        }
        SendBreakpointChangedEvent (removed_locations_event);
    }
}

void
Breakpoint::ModuleReplaced (ModuleSP old_module_sp, ModuleSP new_module_sp)
{
    ModuleList temp_list;
    temp_list.Append (new_module_sp);
    ModulesChanged (temp_list, true);

    // TO DO: For now I'm just adding locations for the new module and removing the
    // breakpoint locations that were in the old module.
    // We should really go find the ones that are in the new module & if we can determine that they are "equivalent"
    // carry over the options from the old location to the new.

    temp_list.Clear();
    temp_list.Append (old_module_sp);
    ModulesChanged (temp_list, false, true);
}

void
Breakpoint::Dump (Stream *)
{
}

size_t
Breakpoint::GetNumResolvedLocations() const
{
    // Return the number of breakpoints that are actually resolved and set
    // down in the inferior process.
    return m_locations.GetNumResolvedLocations();
}

size_t
Breakpoint::GetNumLocations() const
{
    return m_locations.GetSize();
}

void
Breakpoint::GetDescription (Stream *s, lldb::DescriptionLevel level, bool show_locations)
{
    assert (s != NULL);
    
    if (!m_kind_description.empty())
    {
        if (eDescriptionLevelBrief)
        {
            s->PutCString (GetBreakpointKind());
            return;
        }
        else
            s->Printf("Kind: %s\n", GetBreakpointKind ());
    }
    
    const size_t num_locations = GetNumLocations ();
    const size_t num_resolved_locations = GetNumResolvedLocations ();
    
    // They just made the breakpoint, they don't need to be told HOW they made it...
    // Also, we'll print the breakpoint number differently depending on whether there is 1 or more locations.
    if (level != eDescriptionLevelInitial)
    {
        s->Printf("%i: ", GetID());
        GetResolverDescription (s);
        GetFilterDescription (s);
    }
    
    switch (level)
    {
    case lldb::eDescriptionLevelBrief:
    case lldb::eDescriptionLevelFull:
        if (num_locations > 0)
        {
            s->Printf(", locations = %" PRIu64, (uint64_t)num_locations);
            if (num_resolved_locations > 0)
                s->Printf(", resolved = %" PRIu64, (uint64_t)num_resolved_locations);
        }
        else
        {
            // Don't print the pending notification for exception resolvers since we don't generally
            // know how to set them until the target is run.
            if (m_resolver_sp->getResolverID() != BreakpointResolver::ExceptionResolver)
                s->Printf(", locations = 0 (pending)");
        }

        GetOptions()->GetDescription(s, level);
        
        if (level == lldb::eDescriptionLevelFull)
        {
            s->IndentLess();
            s->EOL();
        }
        break;
        
    case lldb::eDescriptionLevelInitial:
        s->Printf ("Breakpoint %i: ", GetID());
        if (num_locations == 0)
        {
            s->Printf ("no locations (pending).");
        }
        else if (num_locations == 1)
        {
            // If there is one location only, we'll just print that location information.  But don't do this if
            // show locations is true, then that will be handled below.
            if (show_locations == false)
            {
                GetLocationAtIndex(0)->GetDescription(s, level);
            }
            else
            {
                s->Printf ("%zd locations.", num_locations);
            }
        }
        else
        {
            s->Printf ("%zd locations.", num_locations);
        }
        s->EOL();
        break;
    case lldb::eDescriptionLevelVerbose:
        // Verbose mode does a debug dump of the breakpoint
        Dump (s);
        s->EOL ();
            //s->Indent();
        GetOptions()->GetDescription(s, level);
        break;

    default: 
        break;
    }

    // The brief description is just the location name (1.2 or whatever).  That's pointless to
    // show in the breakpoint's description, so suppress it.
    if (show_locations && level != lldb::eDescriptionLevelBrief)
    {
        s->IndentMore();
        for (size_t i = 0; i < num_locations; ++i)
        {
            BreakpointLocation *loc = GetLocationAtIndex(i).get();
            loc->GetDescription(s, level);
            s->EOL();
        }
        s->IndentLess();
    }
}

void
Breakpoint::GetResolverDescription (Stream *s)
{
    if (m_resolver_sp)
        m_resolver_sp->GetDescription (s);
}


bool
Breakpoint::GetMatchingFileLine (const ConstString &filename, uint32_t line_number, BreakpointLocationCollection &loc_coll)
{
    // TODO: To be correct, this method needs to fill the breakpoint location collection
    //       with the location IDs which match the filename and line_number.
    //

    if (m_resolver_sp)
    {
        BreakpointResolverFileLine *resolverFileLine = dyn_cast<BreakpointResolverFileLine>(m_resolver_sp.get());
        if (resolverFileLine &&
            resolverFileLine->m_file_spec.GetFilename() == filename &&
            resolverFileLine->m_line_number == line_number)
        {
            return true;
        }
    }
    return false;
}

void
Breakpoint::GetFilterDescription (Stream *s)
{
    m_filter_sp->GetDescription (s);
}

void
Breakpoint::SendBreakpointChangedEvent (lldb::BreakpointEventType eventKind)
{
    if (!m_being_created
        && !IsInternal() 
        && GetTarget().EventTypeHasListeners(Target::eBroadcastBitBreakpointChanged))
    {
        BreakpointEventData *data = new Breakpoint::BreakpointEventData (eventKind, shared_from_this());
            
        GetTarget().BroadcastEvent (Target::eBroadcastBitBreakpointChanged, data);
    }
}

void
Breakpoint::SendBreakpointChangedEvent (BreakpointEventData *data)
{

    if (data == NULL)
        return;
        
    if (!m_being_created
        && !IsInternal() 
        && GetTarget().EventTypeHasListeners(Target::eBroadcastBitBreakpointChanged))
        GetTarget().BroadcastEvent (Target::eBroadcastBitBreakpointChanged, data);
    else
        delete data;
}

Breakpoint::BreakpointEventData::BreakpointEventData (BreakpointEventType sub_type, 
                                                      const BreakpointSP &new_breakpoint_sp) :
    EventData (),
    m_breakpoint_event (sub_type),
    m_new_breakpoint_sp (new_breakpoint_sp)
{
}

Breakpoint::BreakpointEventData::~BreakpointEventData ()
{
}

const ConstString &
Breakpoint::BreakpointEventData::GetFlavorString ()
{
    static ConstString g_flavor ("Breakpoint::BreakpointEventData");
    return g_flavor;
}

const ConstString &
Breakpoint::BreakpointEventData::GetFlavor () const
{
    return BreakpointEventData::GetFlavorString ();
}


BreakpointSP &
Breakpoint::BreakpointEventData::GetBreakpoint ()
{
    return m_new_breakpoint_sp;
}

BreakpointEventType
Breakpoint::BreakpointEventData::GetBreakpointEventType () const
{
    return m_breakpoint_event;
}

void
Breakpoint::BreakpointEventData::Dump (Stream *s) const
{
}

const Breakpoint::BreakpointEventData *
Breakpoint::BreakpointEventData::GetEventDataFromEvent (const Event *event)
{
    if (event)
    {
        const EventData *event_data = event->GetData();
        if (event_data && event_data->GetFlavor() == BreakpointEventData::GetFlavorString())
            return static_cast <const BreakpointEventData *> (event->GetData());
    }
    return NULL;
}

BreakpointEventType
Breakpoint::BreakpointEventData::GetBreakpointEventTypeFromEvent (const EventSP &event_sp)
{
    const BreakpointEventData *data = GetEventDataFromEvent (event_sp.get());

    if (data == NULL)
        return eBreakpointEventTypeInvalidType;
    else
        return data->GetBreakpointEventType();
}

BreakpointSP
Breakpoint::BreakpointEventData::GetBreakpointFromEvent (const EventSP &event_sp)
{
    BreakpointSP bp_sp;

    const BreakpointEventData *data = GetEventDataFromEvent (event_sp.get());
    if (data)
        bp_sp = data->m_new_breakpoint_sp;

    return bp_sp;
}

size_t
Breakpoint::BreakpointEventData::GetNumBreakpointLocationsFromEvent (const EventSP &event_sp)
{
    const BreakpointEventData *data = GetEventDataFromEvent (event_sp.get());
    if (data)
        return data->m_locations.GetSize();

    return 0;
}

lldb::BreakpointLocationSP
Breakpoint::BreakpointEventData::GetBreakpointLocationAtIndexFromEvent (const lldb::EventSP &event_sp, uint32_t bp_loc_idx)
{
    lldb::BreakpointLocationSP bp_loc_sp;

    const BreakpointEventData *data = GetEventDataFromEvent (event_sp.get());
    if (data)
    {
        bp_loc_sp = data->m_locations.GetByIndex(bp_loc_idx);
    }

    return bp_loc_sp;
}
