//===-- SBBreakpointLocation.cpp --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBBreakpointLocation.h"
#include "lldb/API/SBDefines.h"
#include "lldb/API/SBAddress.h"
#include "lldb/API/SBDebugger.h"
#include "lldb/API/SBStream.h"

#include "lldb/lldb-types.h"
#include "lldb/lldb-defines.h"
#include "lldb/Breakpoint/Breakpoint.h"
#include "lldb/Breakpoint/BreakpointLocation.h"
#include "lldb/Target/ThreadSpec.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/ThreadSpec.h"

using namespace lldb;
using namespace lldb_private;


SBBreakpointLocation::SBBreakpointLocation () :
    m_opaque_sp ()
{
}

SBBreakpointLocation::SBBreakpointLocation (const lldb::BreakpointLocationSP &break_loc_sp) :
    m_opaque_sp (break_loc_sp)
{
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    if (log)
    {
        SBStream sstr;
        GetDescription (sstr, lldb::eDescriptionLevelBrief);
        log->Printf ("SBBreakpointLocation::SBBreakpointLocaiton (const lldb::BreakpointLocationsSP &break_loc_sp"
                     "=%p)  => this.sp = %p (%s)", break_loc_sp.get(), m_opaque_sp.get(), sstr.GetData());
    }
}

SBBreakpointLocation::SBBreakpointLocation(const SBBreakpointLocation &rhs) :
    m_opaque_sp (rhs.m_opaque_sp)
{
}

const SBBreakpointLocation &
SBBreakpointLocation::operator = (const SBBreakpointLocation &rhs)
{
    if (this != &rhs)
        m_opaque_sp = rhs.m_opaque_sp;
    return *this;
}


SBBreakpointLocation::~SBBreakpointLocation ()
{
}

bool
SBBreakpointLocation::IsValid() const
{
    return m_opaque_sp.get() != NULL;
}

SBAddress
SBBreakpointLocation::GetAddress ()
{
    if (m_opaque_sp)
        return SBAddress(&m_opaque_sp->GetAddress());
    else
        return SBAddress();
}

addr_t
SBBreakpointLocation::GetLoadAddress ()
{
    addr_t ret_addr = LLDB_INVALID_ADDRESS;

    if (m_opaque_sp)
    {
        Mutex::Locker api_locker (m_opaque_sp->GetBreakpoint().GetTarget().GetAPIMutex());
        ret_addr = m_opaque_sp->GetLoadAddress();
    }

    return ret_addr;
}

void
SBBreakpointLocation::SetEnabled (bool enabled)
{
    if (m_opaque_sp)
    {
        Mutex::Locker api_locker (m_opaque_sp->GetBreakpoint().GetTarget().GetAPIMutex());
        m_opaque_sp->SetEnabled (enabled);
    }
}

bool
SBBreakpointLocation::IsEnabled ()
{
    if (m_opaque_sp)
    {
        Mutex::Locker api_locker (m_opaque_sp->GetBreakpoint().GetTarget().GetAPIMutex());
        return m_opaque_sp->IsEnabled();
    }
    else
        return false;
}

uint32_t
SBBreakpointLocation::GetIgnoreCount ()
{
    if (m_opaque_sp)
    {
        Mutex::Locker api_locker (m_opaque_sp->GetBreakpoint().GetTarget().GetAPIMutex());
        return m_opaque_sp->GetIgnoreCount();
    }
    else
        return 0;
}

void
SBBreakpointLocation::SetIgnoreCount (uint32_t n)
{
    if (m_opaque_sp)
    {
        Mutex::Locker api_locker (m_opaque_sp->GetBreakpoint().GetTarget().GetAPIMutex());
        m_opaque_sp->SetIgnoreCount (n);
    }
}

void
SBBreakpointLocation::SetCondition (const char *condition)
{
    if (m_opaque_sp)
    {
        Mutex::Locker api_locker (m_opaque_sp->GetBreakpoint().GetTarget().GetAPIMutex());
        m_opaque_sp->SetCondition (condition);
    }
}

const char *
SBBreakpointLocation::GetCondition ()
{
    if (m_opaque_sp)
    {
        Mutex::Locker api_locker (m_opaque_sp->GetBreakpoint().GetTarget().GetAPIMutex());
        return m_opaque_sp->GetConditionText ();
    }
    return NULL;
}

void
SBBreakpointLocation::SetThreadID (tid_t thread_id)
{
    if (m_opaque_sp)
    {
        Mutex::Locker api_locker (m_opaque_sp->GetBreakpoint().GetTarget().GetAPIMutex());
        m_opaque_sp->SetThreadID (thread_id);
    }
}

tid_t
SBBreakpointLocation::GetThreadID ()
{
    tid_t tid = LLDB_INVALID_THREAD_ID;
    if (m_opaque_sp)
    {
        Mutex::Locker api_locker (m_opaque_sp->GetBreakpoint().GetTarget().GetAPIMutex());
        return m_opaque_sp->GetThreadID();
    }
    return tid;
}

void
SBBreakpointLocation::SetThreadIndex (uint32_t index)
{
    if (m_opaque_sp)
    {
        Mutex::Locker api_locker (m_opaque_sp->GetBreakpoint().GetTarget().GetAPIMutex());
        m_opaque_sp->SetThreadIndex (index);
    }
}

uint32_t
SBBreakpointLocation::GetThreadIndex() const
{
    uint32_t thread_idx = UINT32_MAX;
    if (m_opaque_sp)
    {
        Mutex::Locker api_locker (m_opaque_sp->GetBreakpoint().GetTarget().GetAPIMutex());
        return m_opaque_sp->GetThreadIndex();
    }
    return thread_idx;
}
    

void
SBBreakpointLocation::SetThreadName (const char *thread_name)
{
    if (m_opaque_sp)
    {
        Mutex::Locker api_locker (m_opaque_sp->GetBreakpoint().GetTarget().GetAPIMutex());
        m_opaque_sp->SetThreadName (thread_name);
    }
}

const char *
SBBreakpointLocation::GetThreadName () const
{
    if (m_opaque_sp)
    {
        Mutex::Locker api_locker (m_opaque_sp->GetBreakpoint().GetTarget().GetAPIMutex());
        return m_opaque_sp->GetThreadName();
    }
    return NULL;
}

void
SBBreakpointLocation::SetQueueName (const char *queue_name)
{
    if (m_opaque_sp)
    {
        Mutex::Locker api_locker (m_opaque_sp->GetBreakpoint().GetTarget().GetAPIMutex());
        m_opaque_sp->SetQueueName (queue_name);
    }
}

const char *
SBBreakpointLocation::GetQueueName () const
{
    if (m_opaque_sp)
    {
        Mutex::Locker api_locker (m_opaque_sp->GetBreakpoint().GetTarget().GetAPIMutex());
        m_opaque_sp->GetQueueName ();
    }
    return NULL;
}

bool
SBBreakpointLocation::IsResolved ()
{
    if (m_opaque_sp)
    {
        Mutex::Locker api_locker (m_opaque_sp->GetBreakpoint().GetTarget().GetAPIMutex());
        return m_opaque_sp->IsResolved();
    }
    return false;
}

void
SBBreakpointLocation::SetLocation (const lldb::BreakpointLocationSP &break_loc_sp)
{
    // Uninstall the callbacks?
    m_opaque_sp = break_loc_sp;
}

bool
SBBreakpointLocation::GetDescription (SBStream &description, DescriptionLevel level)
{
    Stream &strm = description.ref();

    if (m_opaque_sp)
    {
        Mutex::Locker api_locker (m_opaque_sp->GetBreakpoint().GetTarget().GetAPIMutex());
        m_opaque_sp->GetDescription (&strm, level);
        strm.EOL();
    }
    else
        strm.PutCString ("No value");

    return true;
}

break_id_t
SBBreakpointLocation::GetID ()
{
    if (m_opaque_sp)
    {
        Mutex::Locker api_locker (m_opaque_sp->GetBreakpoint().GetTarget().GetAPIMutex());
        return m_opaque_sp->GetID ();
    }
    else
        return LLDB_INVALID_BREAK_ID;
}

SBBreakpoint
SBBreakpointLocation::GetBreakpoint ()
{
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    //if (log)
    //    log->Printf ("SBBreakpointLocation::GetBreakpoint ()");

    SBBreakpoint sb_bp;
    if (m_opaque_sp)
    {
        Mutex::Locker api_locker (m_opaque_sp->GetBreakpoint().GetTarget().GetAPIMutex());
        *sb_bp = m_opaque_sp->GetBreakpoint ().shared_from_this();
    }

    if (log)
    {
        SBStream sstr;
        sb_bp.GetDescription (sstr);
        log->Printf ("SBBreakpointLocation(%p)::GetBreakpoint () => SBBreakpoint(%p) %s", 
                     m_opaque_sp.get(), sb_bp.get(), sstr.GetData());
    }
    return sb_bp;
}

