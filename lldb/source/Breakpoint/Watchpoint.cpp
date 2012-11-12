//===-- Watchpoint.cpp ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Breakpoint/Watchpoint.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Breakpoint/StoppointCallbackContext.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/Value.h"
#include "lldb/Core/ValueObject.h"
#include "lldb/Core/ValueObjectMemory.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/ThreadSpec.h"
#include "lldb/Expression/ClangUserExpression.h"

using namespace lldb;
using namespace lldb_private;

Watchpoint::Watchpoint (Target& target, lldb::addr_t addr, size_t size, const ClangASTType *type, bool hardware) :
    StoppointLocation (0, addr, size, hardware),
    m_target(target),
    m_enabled(false),
    m_is_hardware(hardware),
    m_is_watch_variable(false),
    m_is_ephemeral(false),
    m_disabled_count(0),
    m_watch_read(0),
    m_watch_write(0),
    m_watch_was_read(0),
    m_watch_was_written(0),
    m_ignore_count(0),
    m_false_alarms(0),
    m_decl_str(),
    m_watch_spec_str(),
    m_type(),
    m_error(),
    m_options ()
{
    if (type && type->IsValid())
        m_type = *type;
    else
    {
        // If we don't have a known type, then we force it to unsigned int of the right size.
        ClangASTContext *ast_context = target.GetScratchClangASTContext();
        clang_type_t clang_type = ast_context->GetBuiltinTypeForEncodingAndBitSize(eEncodingUint, 8 * size);
        m_type.SetClangType(ast_context->getASTContext(), clang_type);
    }
    
    // Set the initial value of the watched variable:
    if (m_target.GetProcessSP())
    {
        ExecutionContext exe_ctx;
        m_target.GetProcessSP()->CalculateExecutionContext(exe_ctx);
        CaptureWatchedValue (exe_ctx);
    }
}

Watchpoint::~Watchpoint()
{
}

// This function is used when "baton" doesn't need to be freed
void
Watchpoint::SetCallback (WatchpointHitCallback callback, void *baton, bool is_synchronous)
{
    // The default "Baton" class will keep a copy of "baton" and won't free
    // or delete it when it goes goes out of scope.
    m_options.SetCallback(callback, BatonSP (new Baton(baton)), is_synchronous);
    
    //SendWatchpointChangedEvent (eWatchpointEventTypeCommandChanged);
}

// This function is used when a baton needs to be freed and therefore is 
// contained in a "Baton" subclass.
void
Watchpoint::SetCallback (WatchpointHitCallback callback, const BatonSP &callback_baton_sp, bool is_synchronous)
{
    m_options.SetCallback(callback, callback_baton_sp, is_synchronous);
}

void
Watchpoint::ClearCallback ()
{
    m_options.ClearCallback ();
}

void
Watchpoint::SetDeclInfo (const std::string &str)
{
    m_decl_str = str;
    return;
}

std::string
Watchpoint::GetWatchSpec()
{
    return m_watch_spec_str;
}

void
Watchpoint::SetWatchSpec (const std::string &str)
{
    m_watch_spec_str = str;
    return;
}

// Override default impl of StoppointLocation::IsHardware() since m_is_hardware
// member field is more accurate.
bool
Watchpoint::IsHardware () const
{
    return m_is_hardware;
}

bool
Watchpoint::IsWatchVariable() const
{
    return m_is_watch_variable;
}

void
Watchpoint::SetWatchVariable(bool val)
{
    m_is_watch_variable = val;
}

bool
Watchpoint::CaptureWatchedValue (const ExecutionContext &exe_ctx)
{
    ConstString watch_name("$__lldb__watch_value");
    m_old_value_sp = m_new_value_sp;
    Address watch_address(GetLoadAddress());
    if (!m_type.IsValid())
    {
        // Don't know how to report new & old values, since we couldn't make a scalar type for this watchpoint.
        // This works around an assert in ValueObjectMemory::Create.
        // FIXME: This should not happen, but if it does in some case we care about,
        // we can go grab the value raw and print it as unsigned.
        return false;
    }
    m_new_value_sp = ValueObjectMemory::Create (exe_ctx.GetBestExecutionContextScope(), watch_name.AsCString(), watch_address, m_type);
    m_new_value_sp = m_new_value_sp->CreateConstantValue(watch_name);
    if (m_new_value_sp && m_new_value_sp->GetError().Success())
        return true;
    else
        return false;
}

void
Watchpoint::IncrementFalseAlarmsAndReviseHitCount()
{
    ++m_false_alarms;
    if (m_false_alarms)
    {
        if (m_hit_count >= m_false_alarms)
        {
            m_hit_count -= m_false_alarms;
            m_false_alarms = 0;
        }
        else
        {
            m_false_alarms -= m_hit_count;
            m_hit_count = 0;
        }
    }
}

// RETURNS - true if we should stop at this breakpoint, false if we
// should continue.

bool
Watchpoint::ShouldStop (StoppointCallbackContext *context)
{
    IncrementHitCount();

    if (!IsEnabled())
        return false;

    if (GetHitCount() <= GetIgnoreCount())
        return false;

    return true;
}

void
Watchpoint::GetDescription (Stream *s, lldb::DescriptionLevel level)
{
    DumpWithLevel(s, level);
    return;
}

void
Watchpoint::Dump(Stream *s) const
{
    DumpWithLevel(s, lldb::eDescriptionLevelBrief);
}

// If prefix is NULL, we display the watch id and ignore the prefix altogether.
void
Watchpoint::DumpSnapshots(Stream *s, const char *prefix) const
{
    if (!prefix)
    {
        s->Printf("\nWatchpoint %u hit:", GetID());
        prefix = "";
    }
    
    if (m_old_value_sp)
    {
        s->Printf("\n%sold value: %s", prefix, m_old_value_sp->GetValueAsCString());
    }
    if (m_new_value_sp)
    {
        s->Printf("\n%snew value: %s", prefix, m_new_value_sp->GetValueAsCString());
    }
}

void
Watchpoint::DumpWithLevel(Stream *s, lldb::DescriptionLevel description_level) const
{
    if (s == NULL)
        return;

    assert(description_level >= lldb::eDescriptionLevelBrief &&
           description_level <= lldb::eDescriptionLevelVerbose);

    s->Printf("Watchpoint %u: addr = 0x%8.8llx size = %u state = %s type = %s%s",
              GetID(),
              GetLoadAddress(),
              m_byte_size,
              IsEnabled() ? "enabled" : "disabled",
              m_watch_read ? "r" : "",
              m_watch_write ? "w" : "");

    if (description_level >= lldb::eDescriptionLevelFull) {
        if (!m_decl_str.empty())
            s->Printf("\n    declare @ '%s'", m_decl_str.c_str());
        if (!m_watch_spec_str.empty())
            s->Printf("\n    watchpoint spec = '%s'", m_watch_spec_str.c_str());

        // Dump the snapshots we have taken.
        DumpSnapshots(s, "    ");

        if (GetConditionText())
            s->Printf("\n    condition = '%s'", GetConditionText());
        m_options.GetCallbackDescription(s, description_level);
    }

    if (description_level >= lldb::eDescriptionLevelVerbose)
    {
        s->Printf("\n    hw_index = %i  hit_count = %-4u  ignore_count = %-4u",
                  GetHardwareIndex(),
                  GetHitCount(),
                  GetIgnoreCount());
    }
}

bool
Watchpoint::IsEnabled() const
{
    return m_enabled;
}

// Within StopInfo.cpp, we purposely turn on the ephemeral mode right before temporarily disable the watchpoint
// in order to perform possible watchpoint actions without triggering further watchpoint events.
// After the temporary disabled watchpoint is enabled, we then turn off the ephemeral mode.

void
Watchpoint::TurnOnEphemeralMode()
{
    m_is_ephemeral = true;
}

void
Watchpoint::TurnOffEphemeralMode()
{
    m_is_ephemeral = false;
    // Leaving ephemeral mode, reset the m_disabled_count!
    m_disabled_count = 0;
}

bool
Watchpoint::IsDisabledDuringEphemeralMode()
{
    return m_disabled_count > 1;
}

void
Watchpoint::SetEnabled(bool enabled)
{
    if (!enabled)
    {
        if (!m_is_ephemeral)
            SetHardwareIndex(LLDB_INVALID_INDEX32);
        else
            ++m_disabled_count;

        // Don't clear the snapshots for now.
        // Within StopInfo.cpp, we purposely do disable/enable watchpoint while performing watchpoint actions.
    }
    m_enabled = enabled;
}

void
Watchpoint::SetWatchpointType (uint32_t type)
{
    m_watch_read = (type & LLDB_WATCH_TYPE_READ) != 0;
    m_watch_write = (type & LLDB_WATCH_TYPE_WRITE) != 0;
}

bool
Watchpoint::WatchpointRead () const
{
    return m_watch_read != 0;
}
bool
Watchpoint::WatchpointWrite () const
{
    return m_watch_write != 0;
}
uint32_t
Watchpoint::GetIgnoreCount () const
{
    return m_ignore_count;
}

void
Watchpoint::SetIgnoreCount (uint32_t n)
{
    m_ignore_count = n;
}

bool
Watchpoint::InvokeCallback (StoppointCallbackContext *context)
{
    return m_options.InvokeCallback (context, GetID());
}

void 
Watchpoint::SetCondition (const char *condition)
{
    if (condition == NULL || condition[0] == '\0')
    {
        if (m_condition_ap.get())
            m_condition_ap.reset();
    }
    else
    {
        // Pass NULL for expr_prefix (no translation-unit level definitions).
        m_condition_ap.reset(new ClangUserExpression (condition, NULL, lldb::eLanguageTypeUnknown, ClangUserExpression::eResultTypeAny));
    }
}

const char *
Watchpoint::GetConditionText () const
{
    if (m_condition_ap.get())
        return m_condition_ap->GetUserText();
    else
        return NULL;
}

