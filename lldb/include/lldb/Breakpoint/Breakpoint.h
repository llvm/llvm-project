//===-- Breakpoint.h --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Breakpoint_h_
#define liblldb_Breakpoint_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Breakpoint/BreakpointLocationList.h"
#include "lldb/Breakpoint/BreakpointOptions.h"
#include "lldb/Breakpoint/BreakpointLocationCollection.h"
#include "lldb/Breakpoint/Stoppoint.h"
#include "lldb/Core/SearchFilter.h"
#include "lldb/Core/Event.h"
#include "lldb/Core/StringList.h"

namespace lldb_private {

//----------------------------------------------------------------------
/// @class Breakpoint Breakpoint.h "lldb/Breakpoint/Breakpoint.h"
/// @brief Class that manages logical breakpoint setting.
//----------------------------------------------------------------------

//----------------------------------------------------------------------
/// General Outline:
/// A breakpoint has four main parts, a filter, a resolver, the list of breakpoint
/// locations that have been determined for the filter/resolver pair, and finally
/// a set of options for the breakpoint.
///
/// \b Filter:
/// This is an object derived from SearchFilter.  It manages the search
/// for breakpoint location matches through the symbols in the module list of the target
/// that owns it.  It also filters out locations based on whatever logic it wants.
///
/// \b Resolver:
/// This is an object derived from BreakpointResolver.  It provides a
/// callback to the filter that will find breakpoint locations.  How it does this is
/// determined by what kind of resolver it is.
///
/// The Breakpoint class also provides constructors for the common breakpoint cases
/// which make the appropriate filter and resolver for you.
///
/// \b Location List:
/// This stores the breakpoint locations that have been determined
/// to date.  For a given breakpoint, there will be only one location with a given
/// address.  Adding a location at an already taken address will just return the location
/// already at that address.  Locations can be looked up by ID, or by address.
///
/// \b Options:
/// This includes:
///    \b Enabled/Disabled
///    \b Ignore Count
///    \b Callback
///    \b Condition
/// Note, these options can be set on the breakpoint, and they can also be set on the
/// individual locations.  The options set on the breakpoint take precedence over the
/// options set on the individual location.
/// So for instance disabling the breakpoint will cause NONE of the locations to get hit.
/// But if the breakpoint is enabled, then the location's enabled state will be checked
/// to determine whether to insert that breakpoint location.
/// Similarly, if the breakpoint condition says "stop", we won't check the location's condition.
/// But if the breakpoint condition says "continue", then we will check the location for whether
/// to actually stop or not.
/// One subtle point worth observing here is that you don't actually stop at a Breakpoint, you
/// always stop at one of its locations.  So the "should stop" tests are done by the location,
/// not by the breakpoint.
//----------------------------------------------------------------------
class Breakpoint:
    public STD_ENABLE_SHARED_FROM_THIS(Breakpoint),
    public Stoppoint
{
public:

    static const ConstString &
    GetEventIdentifier ();


    //------------------------------------------------------------------
    /// An enum specifying the match style for breakpoint settings.  At
    /// present only used for function name style breakpoints.
    //------------------------------------------------------------------
    typedef enum
    {
        Exact,
        Regexp,
        Glob
    } MatchType;

    class BreakpointEventData :
        public EventData
    {
    public:

        static const ConstString &
        GetFlavorString ();

        virtual const ConstString &
        GetFlavor () const;

        BreakpointEventData (lldb::BreakpointEventType sub_type,
                             const lldb::BreakpointSP &new_breakpoint_sp);

        virtual
        ~BreakpointEventData();

        lldb::BreakpointEventType
        GetBreakpointEventType () const;

        lldb::BreakpointSP &
        GetBreakpoint ();
        
        BreakpointLocationCollection &
        GetBreakpointLocationCollection()
        {
            return m_locations;
        }


        virtual void
        Dump (Stream *s) const;

        static lldb::BreakpointEventType
        GetBreakpointEventTypeFromEvent (const lldb::EventSP &event_sp);

        static lldb::BreakpointSP
        GetBreakpointFromEvent (const lldb::EventSP &event_sp);

        static lldb::BreakpointLocationSP
        GetBreakpointLocationAtIndexFromEvent (const lldb::EventSP &event_sp, uint32_t loc_idx);
        
        static uint32_t
        GetNumBreakpointLocationsFromEvent (const lldb::EventSP &event_sp);

        static const BreakpointEventData *
        GetEventDataFromEvent (const Event *event_sp);

    private:

        lldb::BreakpointEventType m_breakpoint_event;
        lldb::BreakpointSP m_new_breakpoint_sp;
        BreakpointLocationCollection m_locations;

        DISALLOW_COPY_AND_ASSIGN (BreakpointEventData);
    };


    //------------------------------------------------------------------
    /// Destructor.
    ///
    /// The destructor is not virtual since there should be no reason to subclass
    /// breakpoints.  The varieties of breakpoints are specified instead by
    /// providing different resolvers & filters.
    //------------------------------------------------------------------
    ~Breakpoint();

    //------------------------------------------------------------------
    // Methods
    //------------------------------------------------------------------

    //------------------------------------------------------------------
    /// Tell whether this breakpoint is an "internal" breakpoint.
    /// @return
    ///     Returns \b true if this is an internal breakpoint, \b false otherwise.
    //------------------------------------------------------------------
    bool
    IsInternal () const;

    //------------------------------------------------------------------
    /// Standard "Dump" method.  At present it does nothing.
    //------------------------------------------------------------------
    void
    Dump (Stream *s);

    //------------------------------------------------------------------
    // The next set of methods provide ways to tell the breakpoint to update
    // it's location list - usually done when modules appear or disappear.
    //------------------------------------------------------------------


    //------------------------------------------------------------------
    /// Tell this breakpoint to clear all its breakpoint sites.  Done
    /// when the process holding the breakpoint sites is destroyed.
    //------------------------------------------------------------------
    void
    ClearAllBreakpointSites ();

    //------------------------------------------------------------------
    /// Tell this breakpoint to scan it's target's module list and resolve any
    /// new locations that match the breakpoint's specifications.
    //------------------------------------------------------------------
    void
    ResolveBreakpoint ();

    //------------------------------------------------------------------
    /// Tell this breakpoint to scan a given module list and resolve any
    /// new locations that match the breakpoint's specifications.
    ///
    /// @param[in] changed_modules
    ///    The list of modules to look in for new locations.
    //------------------------------------------------------------------
    void
    ResolveBreakpointInModules (ModuleList &changed_modules);


    //------------------------------------------------------------------
    /// Like ResolveBreakpointInModules, but allows for "unload" events, in
    /// which case we will remove any locations that are in modules that got
    /// unloaded.
    ///
    /// @param[in] changedModules
    ///    The list of modules to look in for new locations.
    /// @param[in] load_event
    ///    If \b true then the modules were loaded, if \b false, unloaded.
    /// @param[in] delete_locations
    ///    If \b true then the modules were unloaded delete any locations in the changed modules.
    //------------------------------------------------------------------
    void
    ModulesChanged (ModuleList &changed_modules,
                    bool load_event,
                    bool delete_locations = false);


    //------------------------------------------------------------------
    /// Tells the breakpoint the old module \a old_module_sp has been
    /// replaced by new_module_sp (usually because the underlying file has been
    /// rebuilt, and the old version is gone.)
    ///
    /// @param[in] old_module_sp
    ///    The old module that is going away.
    /// @param[in] new_module_sp
    ///    The new module that is replacing it.
    //------------------------------------------------------------------
    void
    ModuleReplaced (lldb::ModuleSP old_module_sp, lldb::ModuleSP new_module_sp);
    
    //------------------------------------------------------------------
    // The next set of methods provide access to the breakpoint locations
    // for this breakpoint.
    //------------------------------------------------------------------

    //------------------------------------------------------------------
    /// Add a location to the breakpoint's location list.  This is only meant
    /// to be called by the breakpoint's resolver.  FIXME: how do I ensure that?
    ///
    /// @param[in] addr
    ///    The Address specifying the new location.
    /// @param[out] new_location
    ///    Set to \b true if a new location was created, to \b false if there
    ///    already was a location at this Address.
    /// @return
    ///    Returns a pointer to the new location.
    //------------------------------------------------------------------
    lldb::BreakpointLocationSP
    AddLocation (const Address &addr,
                 bool *new_location = NULL);

    //------------------------------------------------------------------
    /// Find a breakpoint location by Address.
    ///
    /// @param[in] addr
    ///    The Address specifying the location.
    /// @return
    ///    Returns a shared pointer to the location at \a addr.  The pointer
    ///    in the shared pointer will be NULL if there is no location at that address.
    //------------------------------------------------------------------
    lldb::BreakpointLocationSP
    FindLocationByAddress (const Address &addr);

    //------------------------------------------------------------------
    /// Find a breakpoint location ID by Address.
    ///
    /// @param[in] addr
    ///    The Address specifying the location.
    /// @return
    ///    Returns the UID of the location at \a addr, or \b LLDB_INVALID_ID if
    ///    there is no breakpoint location at that address.
    //------------------------------------------------------------------
    lldb::break_id_t
    FindLocationIDByAddress (const Address &addr);

    //------------------------------------------------------------------
    /// Find a breakpoint location for a given breakpoint location ID.
    ///
    /// @param[in] bp_loc_id
    ///    The ID specifying the location.
    /// @return
    ///    Returns a shared pointer to the location with ID \a bp_loc_id.  The pointer
    ///    in the shared pointer will be NULL if there is no location with that ID.
    //------------------------------------------------------------------
    lldb::BreakpointLocationSP
    FindLocationByID (lldb::break_id_t bp_loc_id);

    //------------------------------------------------------------------
    /// Get breakpoint locations by index.
    ///
    /// @param[in] index
    ///    The location index.
    ///
    /// @return
    ///     Returns a shared pointer to the location with index \a 
    ///     index. The shared pointer might contain NULL if \a index is
    ///     greater than then number of actual locations.
    //------------------------------------------------------------------
    lldb::BreakpointLocationSP
    GetLocationAtIndex (uint32_t index);

    //------------------------------------------------------------------
    // The next section deals with various breakpoint options.
    //------------------------------------------------------------------

    //------------------------------------------------------------------
    /// If \a enable is \b true, enable the breakpoint, if \b false disable it.
    //------------------------------------------------------------------
    void
    SetEnabled (bool enable);

    //------------------------------------------------------------------
    /// Check the Enable/Disable state.
    /// @return
    ///     \b true if the breakpoint is enabled, \b false if disabled.
    //------------------------------------------------------------------
    bool
    IsEnabled ();

    //------------------------------------------------------------------
    /// Set the breakpoint to ignore the next \a count breakpoint hits.
    /// @param[in] count
    ///    The number of breakpoint hits to ignore.
    //------------------------------------------------------------------
    void
    SetIgnoreCount (uint32_t count);

    //------------------------------------------------------------------
    /// Return the current ignore count/
    /// @return
    ///     The number of breakpoint hits to be ignored.
    //------------------------------------------------------------------
    uint32_t
    GetIgnoreCount () const;
    
    //------------------------------------------------------------------
    /// Return the current hit count for all locations.
    /// @return
    ///     The current hit count for all locations.
    //------------------------------------------------------------------
    uint32_t
    GetHitCount () const;


    //------------------------------------------------------------------
    /// If \a one_shot is \b true, breakpoint will be deleted on first hit.
    //------------------------------------------------------------------
    void
    SetOneShot (bool one_shot);

    //------------------------------------------------------------------
    /// Check the OneShot state.
    /// @return
    ///     \b true if the breakpoint is one shot, \b false otherwise.
    //------------------------------------------------------------------
    bool
    IsOneShot () const;

    //------------------------------------------------------------------
    /// Set the valid thread to be checked when the breakpoint is hit.
    /// @param[in] thread_id
    ///    If this thread hits the breakpoint, we stop, otherwise not.
    //------------------------------------------------------------------
    void
    SetThreadID (lldb::tid_t thread_id);

    //------------------------------------------------------------------
    /// Return the current stop thread value.
    /// @return
    ///     The thread id for which the breakpoint hit will stop, LLDB_INVALID_THREAD_ID for all threads.
    //------------------------------------------------------------------
    lldb::tid_t
    GetThreadID () const;

    void
    SetThreadIndex (uint32_t index);
    
    uint32_t
    GetThreadIndex() const;
    
    void
    SetThreadName (const char *thread_name);
    
    const char *
    GetThreadName () const;
    
    void 
    SetQueueName (const char *queue_name);
    
    const char *
    GetQueueName () const;

    //------------------------------------------------------------------
    /// Set the callback action invoked when the breakpoint is hit.  
    /// 
    /// @param[in] callback
    ///    The method that will get called when the breakpoint is hit.
    /// @param[in] baton
    ///    A void * pointer that will get passed back to the callback function.
    /// @param[in] is_synchronous
    ///    If \b true the callback will be run on the private event thread
    ///    before the stop event gets reported.  If false, the callback will get
    ///    handled on the public event thead after the stop has been posted.
    ///
    /// @return
    ///    \b true if the process should stop when you hit the breakpoint.
    ///    \b false if it should continue.
    //------------------------------------------------------------------
    void
    SetCallback (BreakpointHitCallback callback, 
                 void *baton,
                 bool is_synchronous = false);

    void
    SetCallback (BreakpointHitCallback callback, 
                 const lldb::BatonSP &callback_baton_sp,
                 bool is_synchronous = false);

    void
    ClearCallback ();

    //------------------------------------------------------------------
    /// Set the breakpoint's condition.
    ///
    /// @param[in] condition
    ///    The condition expression to evaluate when the breakpoint is hit.
    ///    Pass in NULL to clear the condition.
    //------------------------------------------------------------------
    void SetCondition (const char *condition);
    
    //------------------------------------------------------------------
    /// Return a pointer to the text of the condition expression.
    ///
    /// @return
    ///    A pointer to the condition expression text, or NULL if no
    //     condition has been set.
    //------------------------------------------------------------------
    const char *GetConditionText () const;

    //------------------------------------------------------------------
    // The next section are various utility functions.
    //------------------------------------------------------------------

    //------------------------------------------------------------------
    /// Return the number of breakpoint locations that have resolved to
    /// actual breakpoint sites.
    ///
    /// @return
    ///     The number locations resolved breakpoint sites.
    //------------------------------------------------------------------
    size_t
    GetNumResolvedLocations() const;

    //------------------------------------------------------------------
    /// Return the number of breakpoint locations.
    ///
    /// @return
    ///     The number breakpoint locations.
    //------------------------------------------------------------------
    size_t
    GetNumLocations() const;

    //------------------------------------------------------------------
    /// Put a description of this breakpoint into the stream \a s.
    ///
    /// @param[in] s
    ///     Stream into which to dump the description.
    ///
    /// @param[in] level
    ///     The description level that indicates the detail level to
    ///     provide.
    ///
    /// @see lldb::DescriptionLevel
    //------------------------------------------------------------------
    void
    GetDescription (Stream *s, lldb::DescriptionLevel level, bool show_locations = false);

    //------------------------------------------------------------------
    /// Accessor for the breakpoint Target.
    /// @return
    ///     This breakpoint's Target.
    //------------------------------------------------------------------
    Target &
    GetTarget ();

    const Target &
    GetTarget () const;

    void
    GetResolverDescription (Stream *s);

    //------------------------------------------------------------------
    /// Find breakpoint locations which match the (filename, line_number) description.
    /// The breakpoint location collection is to be filled with the matching locations.
    /// It should be initialized with 0 size by the API client.
    ///
    /// @return
    ///     True if there is a match
    ///
    ///     The locations which match the filename and line_number in loc_coll.  If its
    ///     size is 0 and true is returned, it means the breakpoint fully matches the
    ///     description.
    //------------------------------------------------------------------
    bool GetMatchingFileLine(const ConstString &filename, uint32_t line_number,
                             BreakpointLocationCollection &loc_coll);

    void
    GetFilterDescription (Stream *s);

    //------------------------------------------------------------------
    /// Returns the BreakpointOptions structure set at the breakpoint level.
    ///
    /// Meant to be used by the BreakpointLocation class.
    ///
    /// @return
    ///     A pointer to this breakpoint's BreakpointOptions.
    //------------------------------------------------------------------
    BreakpointOptions *
    GetOptions ();


    //------------------------------------------------------------------
    /// Invoke the callback action when the breakpoint is hit.
    ///
    /// Meant to be used by the BreakpointLocation class.
    ///
    /// @param[in] context
    ///     Described the breakpoint event.
    ///
    /// @param[in] bp_loc_id
    ///     Which breakpoint location hit this breakpoint.
    ///
    /// @return
    ///     \b true if the target should stop at this breakpoint and \b false not.
    //------------------------------------------------------------------
    bool
    InvokeCallback (StoppointCallbackContext *context,
                    lldb::break_id_t bp_loc_id);

protected:
    friend class Target;
    //------------------------------------------------------------------
    // Protected Methods
    //------------------------------------------------------------------


    //------------------------------------------------------------------
    /// Constructors and Destructors
    /// Only the Target can make a breakpoint, and it owns the breakpoint lifespans.
    /// The constructor takes a filter and a resolver.  Up in Target there are convenience
    /// variants that make breakpoints for some common cases.
    //------------------------------------------------------------------
    // This is the generic constructor
    Breakpoint(Target &target, lldb::SearchFilterSP &filter_sp, lldb::BreakpointResolverSP &resolver_sp);
    
    friend class BreakpointLocation;  // To call the following two when determining whether to stop.

    void
    DecrementIgnoreCount();

    // BreakpointLocation::IgnoreCountShouldStop & Breakpoint::IgnoreCountShouldStop can only be called once per stop, 
    // and BreakpointLocation::IgnoreCountShouldStop should be tested first, and if it returns false we should
    // continue, otherwise we should test Breakpoint::IgnoreCountShouldStop.
    
    bool
    IgnoreCountShouldStop ();

private:
    //------------------------------------------------------------------
    // For Breakpoint only
    //------------------------------------------------------------------
    bool m_being_created;
    Target &m_target;                         // The target that holds this breakpoint.
    lldb::SearchFilterSP m_filter_sp;         // The filter that constrains the breakpoint's domain.
    lldb::BreakpointResolverSP m_resolver_sp; // The resolver that defines this breakpoint.
    BreakpointOptions m_options;              // Settable breakpoint options
    BreakpointLocationList m_locations;       // The list of locations currently found for this breakpoint.
    
    void
    SendBreakpointChangedEvent (lldb::BreakpointEventType eventKind);
    
    void
    SendBreakpointChangedEvent (BreakpointEventData *data);

    DISALLOW_COPY_AND_ASSIGN(Breakpoint);
};

} // namespace lldb_private

#endif  // liblldb_Breakpoint_h_
