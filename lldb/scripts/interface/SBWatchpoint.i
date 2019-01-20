//===-- SWIG Interface for SBWatchpoint -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

namespace lldb {

%feature("docstring",
"Represents an instance of watchpoint for a specific target program.

A watchpoint is determined by the address and the byte size that resulted in
this particular instantiation.  Each watchpoint has its settable options.

See also SBTarget.watchpoint_iter() for example usage of iterating through the
watchpoints of the target."
) SBWatchpoint;
class SBWatchpoint
{
public:

    SBWatchpoint ();

    SBWatchpoint (const lldb::SBWatchpoint &rhs);

    ~SBWatchpoint ();

    bool
    IsValid();

    SBError
    GetError();

    watch_id_t
    GetID ();

    %feature("docstring", "
    //------------------------------------------------------------------
    /// With -1 representing an invalid hardware index.
    //------------------------------------------------------------------
    ") GetHardwareIndex;
    int32_t
    GetHardwareIndex ();

    lldb::addr_t
    GetWatchAddress ();

    size_t
    GetWatchSize();

    void
    SetEnabled(bool enabled);

    bool
    IsEnabled ();

    uint32_t
    GetHitCount ();

    uint32_t
    GetIgnoreCount ();

    void
    SetIgnoreCount (uint32_t n);

    %feature("docstring", "
    //------------------------------------------------------------------
    /// Get the condition expression for the watchpoint.
    //------------------------------------------------------------------
    ") GetCondition;
    const char *
    GetCondition ();

    %feature("docstring", "
    //--------------------------------------------------------------------------
    /// The watchpoint stops only if the condition expression evaluates to true.
    //--------------------------------------------------------------------------
    ") SetCondition;
    void 
    SetCondition (const char *condition);
    
    bool
    GetDescription (lldb::SBStream &description, DescriptionLevel level);

    static bool
    EventIsWatchpointEvent (const lldb::SBEvent &event);
    
    static lldb::WatchpointEventType
    GetWatchpointEventTypeFromEvent (const lldb::SBEvent& event);

    static lldb::SBWatchpoint
    GetWatchpointFromEvent (const lldb::SBEvent& event);

};

} // namespace lldb
