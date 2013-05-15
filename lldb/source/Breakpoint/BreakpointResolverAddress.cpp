//===-- BreakpointResolverAddress.cpp ---------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Breakpoint/BreakpointResolverAddress.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/lldb-private-log.h"

#include "lldb/Breakpoint/BreakpointLocation.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Target/Target.h"

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// BreakpointResolverAddress:
//----------------------------------------------------------------------
BreakpointResolverAddress::BreakpointResolverAddress
(
    Breakpoint *bkpt,
    const Address &addr
) :
    BreakpointResolver (bkpt, BreakpointResolver::AddressResolver),
    m_addr (addr)
{
}

BreakpointResolverAddress::~BreakpointResolverAddress ()
{

}

void
BreakpointResolverAddress::ResolveBreakpoint (SearchFilter &filter)
{
    // The address breakpoint only takes once, so if we've already set it we're done.
    if (m_breakpoint->GetNumLocations() > 0)
        return;
    else
        BreakpointResolver::ResolveBreakpoint(filter);
}

void
BreakpointResolverAddress::ResolveBreakpointInModules
(
    SearchFilter &filter,
    ModuleList &modules
)
{
    // The address breakpoint only takes once, so if we've already set it we're done.
    if (m_breakpoint->GetNumLocations() > 0)
        return;
    else
        BreakpointResolver::ResolveBreakpointInModules (filter, modules);
}

Searcher::CallbackReturn
BreakpointResolverAddress::SearchCallback
(
    SearchFilter &filter,
    SymbolContext &context,
    Address *addr,
    bool containing
)
{
    assert (m_breakpoint != NULL);

    if (filter.AddressPasses (m_addr))
    {
        BreakpointLocationSP bp_loc_sp(m_breakpoint->AddLocation(m_addr));
        if (bp_loc_sp && !m_breakpoint->IsInternal())
        {
            StreamString s;
            bp_loc_sp->GetDescription(&s, lldb::eDescriptionLevelVerbose);
            Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_BREAKPOINTS));
            if (log)
                log->Printf ("Added location: %s\n", s.GetData());
        }
    }
    return Searcher::eCallbackReturnStop;
}

Searcher::Depth
BreakpointResolverAddress::GetDepth()
{
    return Searcher::eDepthTarget;
}

void
BreakpointResolverAddress::GetDescription (Stream *s)
{
    s->PutCString ("address = ");
    m_addr.Dump(s, m_breakpoint->GetTarget().GetProcessSP().get(), Address::DumpStyleLoadAddress, Address::DumpStyleModuleWithFileAddress);
}

void
BreakpointResolverAddress::Dump (Stream *s) const
{

}
