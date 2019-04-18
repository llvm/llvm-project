//===-- SWIG Interface for SBBroadcaster ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

namespace lldb {

%feature("docstring",
"Represents an entity which can broadcast events. A default broadcaster is
associated with an SBCommandInterpreter, SBProcess, and SBTarget.  For
example, use

    broadcaster = process.GetBroadcaster()

to retrieve the process's broadcaster.

See also SBEvent for example usage of interacting with a broadcaster."
) SBBroadcaster;
class SBBroadcaster
{
public:
    SBBroadcaster ();

    SBBroadcaster (const char *name);

    SBBroadcaster (const SBBroadcaster &rhs);

    ~SBBroadcaster();

    bool
    IsValid () const;

    explicit operator bool() const;

    void
    Clear ();

    void
    BroadcastEventByType (uint32_t event_type, bool unique = false);

    void
    BroadcastEvent (const lldb::SBEvent &event, bool unique = false);

    void
    AddInitialEventsToListener (const lldb::SBListener &listener, uint32_t requested_events);

    uint32_t
    AddListener (const lldb::SBListener &listener, uint32_t event_mask);

    const char *
    GetName () const;

    bool
    EventTypeHasListeners (uint32_t event_type);

    bool
    RemoveListener (const lldb::SBListener &listener, uint32_t event_mask = UINT32_MAX);

    bool
    operator == (const lldb::SBBroadcaster &rhs) const;

    bool
    operator != (const lldb::SBBroadcaster &rhs) const;
};

} // namespace lldb
