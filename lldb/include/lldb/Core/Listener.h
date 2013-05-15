//===-- Listener.h ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Select_h_
#define liblldb_Select_h_

// C Includes
// C++ Includes
#include <list>
#include <map>
#include <set>
#include <string>
#include <vector>


// Other libraries and framework includes
// Project includes
#include "lldb/lldb-private.h"
#include "lldb/Host/Predicate.h"
#include "lldb/Core/Event.h"

namespace lldb_private {

class Listener
{
public:
    typedef bool (*HandleBroadcastCallback) (lldb::EventSP &event_sp, void *baton);

    friend class Broadcaster;
    friend class BroadcasterManager;

    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
    Listener (const char *name);

    ~Listener ();

    void
    AddEvent (lldb::EventSP &event);

    void
    Clear ();

    const char *
    GetName ()
    {
        return m_name.c_str();
    }

    uint32_t
    StartListeningForEventSpec (BroadcasterManager &manager, 
                                 const BroadcastEventSpec &event_spec);
    
    bool
    StopListeningForEventSpec (BroadcasterManager &manager, 
                                 const BroadcastEventSpec &event_spec);
    
    uint32_t
    StartListeningForEvents (Broadcaster* broadcaster, 
                             uint32_t event_mask);
    
    uint32_t
    StartListeningForEvents (Broadcaster* broadcaster,
                             uint32_t event_mask,
                             HandleBroadcastCallback callback,
                             void *callback_user_data);

    bool
    StopListeningForEvents (Broadcaster* broadcaster,
                            uint32_t event_mask);

    // Returns true if an event was recieved, false if we timed out.
    bool
    WaitForEvent (const TimeValue *timeout,
                  lldb::EventSP &event_sp);

    bool
    WaitForEventForBroadcaster (const TimeValue *timeout,
                                Broadcaster *broadcaster,
                                lldb::EventSP &event_sp);

    bool
    WaitForEventForBroadcasterWithType (const TimeValue *timeout,
                                        Broadcaster *broadcaster,
                                        uint32_t event_type_mask,
                                        lldb::EventSP &event_sp);

    Event *
    PeekAtNextEvent ();

    Event *
    PeekAtNextEventForBroadcaster (Broadcaster *broadcaster);

    Event *
    PeekAtNextEventForBroadcasterWithType (Broadcaster *broadcaster,
                                           uint32_t event_type_mask);

    bool
    GetNextEvent (lldb::EventSP &event_sp);

    bool
    GetNextEventForBroadcaster (Broadcaster *broadcaster,
                                lldb::EventSP &event_sp);

    bool
    GetNextEventForBroadcasterWithType (Broadcaster *broadcaster,
                                        uint32_t event_type_mask,
                                        lldb::EventSP &event_sp);

    size_t
    HandleBroadcastEvent (lldb::EventSP &event_sp);

private:

    //------------------------------------------------------------------
    // Classes that inherit from Listener can see and modify these
    //------------------------------------------------------------------
    struct BroadcasterInfo
    {
        BroadcasterInfo(uint32_t mask, HandleBroadcastCallback cb = NULL, void *ud = NULL) :
            event_mask (mask),
            callback (cb),
            callback_user_data (ud)
        {
        }

        uint32_t event_mask;
        HandleBroadcastCallback callback;
        void *callback_user_data;
    };

    typedef std::multimap<Broadcaster*, BroadcasterInfo> broadcaster_collection;
    typedef std::list<lldb::EventSP> event_collection;
    typedef std::vector<BroadcasterManager *> broadcaster_manager_collection;

    bool
    FindNextEventInternal (Broadcaster *broadcaster,   // NULL for any broadcaster
                           const ConstString *sources, // NULL for any event
                           uint32_t num_sources,
                           uint32_t event_type_mask,
                           lldb::EventSP &event_sp,
                           bool remove);

    bool
    GetNextEventInternal (Broadcaster *broadcaster,   // NULL for any broadcaster
                          const ConstString *sources, // NULL for any event
                          uint32_t num_sources,
                          uint32_t event_type_mask,
                          lldb::EventSP &event_sp);

    bool
    WaitForEventsInternal (const TimeValue *timeout,
                           Broadcaster *broadcaster,   // NULL for any broadcaster
                           const ConstString *sources, // NULL for any event
                           uint32_t num_sources,
                           uint32_t event_type_mask,
                           lldb::EventSP &event_sp);

    std::string m_name;
    broadcaster_collection m_broadcasters;
    Mutex m_broadcasters_mutex; // Protects m_broadcasters
    event_collection m_events;
    Mutex m_events_mutex; // Protects m_broadcasters and m_events
    Predicate<bool> m_cond_wait;
    broadcaster_manager_collection m_broadcaster_managers;

    void
    BroadcasterWillDestruct (Broadcaster *);
    
    void
    BroadcasterManagerWillDestruct (BroadcasterManager *manager);
    

//    broadcaster_collection::iterator
//    FindBroadcasterWithMask (Broadcaster *broadcaster,
//                             uint32_t event_mask,
//                             bool exact);

    //------------------------------------------------------------------
    // For Listener only
    //------------------------------------------------------------------
    DISALLOW_COPY_AND_ASSIGN (Listener);
};

} // namespace lldb_private

#endif  // liblldb_Select_h_
