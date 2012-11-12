//===-- BreakpointResolver.cpp ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Breakpoint/BreakpointResolver.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/Address.h"
#include "lldb/Breakpoint/Breakpoint.h"
#include "lldb/Breakpoint/BreakpointLocation.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/ModuleList.h"
#include "lldb/Core/SearchFilter.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Target/Target.h"
#include "lldb/lldb-private-log.h"

using namespace lldb_private;

//----------------------------------------------------------------------
// BreakpointResolver:
//----------------------------------------------------------------------
BreakpointResolver::BreakpointResolver (Breakpoint *bkpt, const unsigned char resolverTy) :
    m_breakpoint (bkpt),
    SubclassID (resolverTy)
{
}

BreakpointResolver::~BreakpointResolver ()
{

}

void
BreakpointResolver::SetBreakpoint (Breakpoint *bkpt)
{
    m_breakpoint = bkpt;
}

void
BreakpointResolver::ResolveBreakpointInModules (SearchFilter &filter, ModuleList &modules)
{
    filter.SearchInModuleList(*this, modules);
}

void
BreakpointResolver::ResolveBreakpoint (SearchFilter &filter)
{
    filter.Search (*this);
}

