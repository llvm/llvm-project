//===-- SWIG Interface for SBBreakpointLocation -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

namespace lldb {

%feature("docstring",
"Represents one unique instance (by address) of a logical breakpoint.

A breakpoint location is defined by the breakpoint that produces it,
and the address that resulted in this particular instantiation.
Each breakpoint location has its settable options.

SBBreakpoint contains SBBreakpointLocation(s). See docstring of SBBreakpoint
for retrieval of an SBBreakpointLocation from an SBBreakpoint."
) SBBreakpointLocation;
class SBBreakpointLocation
{
public:

    SBBreakpointLocation ();

    SBBreakpointLocation (const lldb::SBBreakpointLocation &rhs);

    ~SBBreakpointLocation ();

    break_id_t
    GetID ();
    
    bool
    IsValid() const;

    lldb::SBAddress
    GetAddress();
    
    lldb::addr_t
    GetLoadAddress ();

    void
    SetEnabled(bool enabled);

    bool
    IsEnabled ();

    uint32_t
    GetIgnoreCount ();

    void
    SetIgnoreCount (uint32_t n);

    %feature("docstring", "
    //--------------------------------------------------------------------------
    /// The breakpoint location stops only if the condition expression evaluates
    /// to true.
    //--------------------------------------------------------------------------
    ") SetCondition;
    void 
    SetCondition (const char *condition);
    
    %feature("docstring", "
    //------------------------------------------------------------------
    /// Get the condition expression for the breakpoint location.
    //------------------------------------------------------------------
    ") GetCondition;
    const char *
    GetCondition ();

    void
    SetThreadID (lldb::tid_t sb_thread_id);

    lldb::tid_t
    GetThreadID ();
    
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

    bool
    IsResolved ();

    bool
    GetDescription (lldb::SBStream &description, DescriptionLevel level);

    SBBreakpoint
    GetBreakpoint ();
};

} // namespace lldb
