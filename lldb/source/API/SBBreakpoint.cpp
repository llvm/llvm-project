//===-- SBBreakpoint.cpp ----------------------------------------*- C++ -*-===//
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
#include "lldb/API/SBBreakpoint.h"
#include "lldb/API/SBBreakpointLocation.h"
#include "lldb/API/SBDebugger.h"
#include "lldb/API/SBEvent.h"
#include "lldb/API/SBProcess.h"
#include "lldb/API/SBStream.h"
#include "lldb/API/SBStringList.h"
#include "lldb/API/SBThread.h"

#include "lldb/Breakpoint/Breakpoint.h"
#include "lldb/Breakpoint/BreakpointLocation.h"
#include "lldb/Breakpoint/StoppointCallbackContext.h"
#include "lldb/Core/Address.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/ScriptInterpreter.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/SectionLoadList.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/ThreadSpec.h"

#include "lldb/lldb-enumerations.h"

using namespace lldb;
using namespace lldb_private;

struct CallbackData {
  SBBreakpoint::BreakpointHitCallback callback;
  void *callback_baton;
};

class SBBreakpointCallbackBaton : public Baton {
public:
  SBBreakpointCallbackBaton(SBBreakpoint::BreakpointHitCallback callback,
                            void *baton)
      : Baton(new CallbackData) {
    CallbackData *data = (CallbackData *)m_data;
    data->callback = callback;
    data->callback_baton = baton;
  }

  ~SBBreakpointCallbackBaton() override {
    CallbackData *data = (CallbackData *)m_data;

    if (data) {
      delete data;
      m_data = nullptr;
    }
  }
};

SBBreakpoint::SBBreakpoint() : m_opaque_sp() {}

SBBreakpoint::SBBreakpoint(const SBBreakpoint &rhs)
    : m_opaque_sp(rhs.m_opaque_sp) {}

SBBreakpoint::SBBreakpoint(const lldb::BreakpointSP &bp_sp)
    : m_opaque_sp(bp_sp) {}

SBBreakpoint::~SBBreakpoint() = default;

const SBBreakpoint &SBBreakpoint::operator=(const SBBreakpoint &rhs) {
  if (this != &rhs)
    m_opaque_sp = rhs.m_opaque_sp;
  return *this;
}

bool SBBreakpoint::operator==(const lldb::SBBreakpoint &rhs) {
  if (m_opaque_sp && rhs.m_opaque_sp)
    return m_opaque_sp.get() == rhs.m_opaque_sp.get();
  return false;
}

bool SBBreakpoint::operator!=(const lldb::SBBreakpoint &rhs) {
  if (m_opaque_sp && rhs.m_opaque_sp)
    return m_opaque_sp.get() != rhs.m_opaque_sp.get();
  return (m_opaque_sp && !rhs.m_opaque_sp) || (rhs.m_opaque_sp && !m_opaque_sp);
}

break_id_t SBBreakpoint::GetID() const {
  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_API));

  break_id_t break_id = LLDB_INVALID_BREAK_ID;
  if (m_opaque_sp)
    break_id = m_opaque_sp->GetID();

  if (log) {
    if (break_id == LLDB_INVALID_BREAK_ID)
      log->Printf("SBBreakpoint(%p)::GetID () => LLDB_INVALID_BREAK_ID",
                  static_cast<void *>(m_opaque_sp.get()));
    else
      log->Printf("SBBreakpoint(%p)::GetID () => %u",
                  static_cast<void *>(m_opaque_sp.get()), break_id);
  }

  return break_id;
}

bool SBBreakpoint::IsValid() const {
  if (!m_opaque_sp)
    return false;
  else if (m_opaque_sp->GetTarget().GetBreakpointByID(m_opaque_sp->GetID()))
    return true;
  else
    return false;
}

void SBBreakpoint::ClearAllBreakpointSites() {
  if (m_opaque_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        m_opaque_sp->GetTarget().GetAPIMutex());
    m_opaque_sp->ClearAllBreakpointSites();
  }
}

SBBreakpointLocation SBBreakpoint::FindLocationByAddress(addr_t vm_addr) {
  SBBreakpointLocation sb_bp_location;

  if (m_opaque_sp) {
    if (vm_addr != LLDB_INVALID_ADDRESS) {
      std::lock_guard<std::recursive_mutex> guard(
          m_opaque_sp->GetTarget().GetAPIMutex());
      Address address;
      Target &target = m_opaque_sp->GetTarget();
      if (!target.GetSectionLoadList().ResolveLoadAddress(vm_addr, address)) {
        address.SetRawAddress(vm_addr);
      }
      sb_bp_location.SetLocation(m_opaque_sp->FindLocationByAddress(address));
    }
  }
  return sb_bp_location;
}

break_id_t SBBreakpoint::FindLocationIDByAddress(addr_t vm_addr) {
  break_id_t break_id = LLDB_INVALID_BREAK_ID;

  if (m_opaque_sp && vm_addr != LLDB_INVALID_ADDRESS) {
    std::lock_guard<std::recursive_mutex> guard(
        m_opaque_sp->GetTarget().GetAPIMutex());
    Address address;
    Target &target = m_opaque_sp->GetTarget();
    if (!target.GetSectionLoadList().ResolveLoadAddress(vm_addr, address)) {
      address.SetRawAddress(vm_addr);
    }
    break_id = m_opaque_sp->FindLocationIDByAddress(address);
  }

  return break_id;
}

SBBreakpointLocation SBBreakpoint::FindLocationByID(break_id_t bp_loc_id) {
  SBBreakpointLocation sb_bp_location;

  if (m_opaque_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        m_opaque_sp->GetTarget().GetAPIMutex());
    sb_bp_location.SetLocation(m_opaque_sp->FindLocationByID(bp_loc_id));
  }

  return sb_bp_location;
}

SBBreakpointLocation SBBreakpoint::GetLocationAtIndex(uint32_t index) {
  SBBreakpointLocation sb_bp_location;

  if (m_opaque_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        m_opaque_sp->GetTarget().GetAPIMutex());
    sb_bp_location.SetLocation(m_opaque_sp->GetLocationAtIndex(index));
  }

  return sb_bp_location;
}

void SBBreakpoint::SetEnabled(bool enable) {
  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_API));

  if (log)
    log->Printf("SBBreakpoint(%p)::SetEnabled (enabled=%i)",
                static_cast<void *>(m_opaque_sp.get()), enable);

  if (m_opaque_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        m_opaque_sp->GetTarget().GetAPIMutex());
    m_opaque_sp->SetEnabled(enable);
  }
}

bool SBBreakpoint::IsEnabled() {
  if (m_opaque_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        m_opaque_sp->GetTarget().GetAPIMutex());
    return m_opaque_sp->IsEnabled();
  } else
    return false;
}

void SBBreakpoint::SetOneShot(bool one_shot) {
  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_API));

  if (log)
    log->Printf("SBBreakpoint(%p)::SetOneShot (one_shot=%i)",
                static_cast<void *>(m_opaque_sp.get()), one_shot);

  if (m_opaque_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        m_opaque_sp->GetTarget().GetAPIMutex());
    m_opaque_sp->SetOneShot(one_shot);
  }
}

bool SBBreakpoint::IsOneShot() const {
  if (m_opaque_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        m_opaque_sp->GetTarget().GetAPIMutex());
    return m_opaque_sp->IsOneShot();
  } else
    return false;
}

bool SBBreakpoint::IsInternal() {
  if (m_opaque_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        m_opaque_sp->GetTarget().GetAPIMutex());
    return m_opaque_sp->IsInternal();
  } else
    return false;
}

void SBBreakpoint::SetIgnoreCount(uint32_t count) {
  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_API));

  if (log)
    log->Printf("SBBreakpoint(%p)::SetIgnoreCount (count=%u)",
                static_cast<void *>(m_opaque_sp.get()), count);

  if (m_opaque_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        m_opaque_sp->GetTarget().GetAPIMutex());
    m_opaque_sp->SetIgnoreCount(count);
  }
}

void SBBreakpoint::SetCondition(const char *condition) {
  if (m_opaque_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        m_opaque_sp->GetTarget().GetAPIMutex());
    m_opaque_sp->SetCondition(condition);
  }
}

const char *SBBreakpoint::GetCondition() {
  if (m_opaque_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        m_opaque_sp->GetTarget().GetAPIMutex());
    return m_opaque_sp->GetConditionText();
  }
  return nullptr;
}

uint32_t SBBreakpoint::GetHitCount() const {
  uint32_t count = 0;
  if (m_opaque_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        m_opaque_sp->GetTarget().GetAPIMutex());
    count = m_opaque_sp->GetHitCount();
  }

  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_API));
  if (log)
    log->Printf("SBBreakpoint(%p)::GetHitCount () => %u",
                static_cast<void *>(m_opaque_sp.get()), count);

  return count;
}

uint32_t SBBreakpoint::GetIgnoreCount() const {
  uint32_t count = 0;
  if (m_opaque_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        m_opaque_sp->GetTarget().GetAPIMutex());
    count = m_opaque_sp->GetIgnoreCount();
  }

  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_API));
  if (log)
    log->Printf("SBBreakpoint(%p)::GetIgnoreCount () => %u",
                static_cast<void *>(m_opaque_sp.get()), count);

  return count;
}

void SBBreakpoint::SetThreadID(tid_t tid) {
  if (m_opaque_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        m_opaque_sp->GetTarget().GetAPIMutex());
    m_opaque_sp->SetThreadID(tid);
  }
  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_API));
  if (log)
    log->Printf("SBBreakpoint(%p)::SetThreadID (tid=0x%4.4" PRIx64 ")",
                static_cast<void *>(m_opaque_sp.get()), tid);
}

tid_t SBBreakpoint::GetThreadID() {
  tid_t tid = LLDB_INVALID_THREAD_ID;
  if (m_opaque_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        m_opaque_sp->GetTarget().GetAPIMutex());
    tid = m_opaque_sp->GetThreadID();
  }

  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_API));
  if (log)
    log->Printf("SBBreakpoint(%p)::GetThreadID () => 0x%4.4" PRIx64,
                static_cast<void *>(m_opaque_sp.get()), tid);
  return tid;
}

void SBBreakpoint::SetThreadIndex(uint32_t index) {
  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_API));
  if (log)
    log->Printf("SBBreakpoint(%p)::SetThreadIndex (%u)",
                static_cast<void *>(m_opaque_sp.get()), index);
  if (m_opaque_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        m_opaque_sp->GetTarget().GetAPIMutex());
    m_opaque_sp->GetOptions()->GetThreadSpec()->SetIndex(index);
  }
}

uint32_t SBBreakpoint::GetThreadIndex() const {
  uint32_t thread_idx = UINT32_MAX;
  if (m_opaque_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        m_opaque_sp->GetTarget().GetAPIMutex());
    const ThreadSpec *thread_spec =
        m_opaque_sp->GetOptions()->GetThreadSpecNoCreate();
    if (thread_spec != nullptr)
      thread_idx = thread_spec->GetIndex();
  }
  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_API));
  if (log)
    log->Printf("SBBreakpoint(%p)::GetThreadIndex () => %u",
                static_cast<void *>(m_opaque_sp.get()), thread_idx);

  return thread_idx;
}

void SBBreakpoint::SetThreadName(const char *thread_name) {
  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_API));
  if (log)
    log->Printf("SBBreakpoint(%p)::SetThreadName (%s)",
                static_cast<void *>(m_opaque_sp.get()), thread_name);

  if (m_opaque_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        m_opaque_sp->GetTarget().GetAPIMutex());
    m_opaque_sp->GetOptions()->GetThreadSpec()->SetName(thread_name);
  }
}

const char *SBBreakpoint::GetThreadName() const {
  const char *name = nullptr;
  if (m_opaque_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        m_opaque_sp->GetTarget().GetAPIMutex());
    const ThreadSpec *thread_spec =
        m_opaque_sp->GetOptions()->GetThreadSpecNoCreate();
    if (thread_spec != nullptr)
      name = thread_spec->GetName();
  }
  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_API));
  if (log)
    log->Printf("SBBreakpoint(%p)::GetThreadName () => %s",
                static_cast<void *>(m_opaque_sp.get()), name);

  return name;
}

void SBBreakpoint::SetQueueName(const char *queue_name) {
  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_API));
  if (log)
    log->Printf("SBBreakpoint(%p)::SetQueueName (%s)",
                static_cast<void *>(m_opaque_sp.get()), queue_name);
  if (m_opaque_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        m_opaque_sp->GetTarget().GetAPIMutex());
    m_opaque_sp->GetOptions()->GetThreadSpec()->SetQueueName(queue_name);
  }
}

const char *SBBreakpoint::GetQueueName() const {
  const char *name = nullptr;
  if (m_opaque_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        m_opaque_sp->GetTarget().GetAPIMutex());
    const ThreadSpec *thread_spec =
        m_opaque_sp->GetOptions()->GetThreadSpecNoCreate();
    if (thread_spec)
      name = thread_spec->GetQueueName();
  }
  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_API));
  if (log)
    log->Printf("SBBreakpoint(%p)::GetQueueName () => %s",
                static_cast<void *>(m_opaque_sp.get()), name);

  return name;
}

size_t SBBreakpoint::GetNumResolvedLocations() const {
  size_t num_resolved = 0;
  if (m_opaque_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        m_opaque_sp->GetTarget().GetAPIMutex());
    num_resolved = m_opaque_sp->GetNumResolvedLocations();
  }
  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_API));
  if (log)
    log->Printf("SBBreakpoint(%p)::GetNumResolvedLocations () => %" PRIu64,
                static_cast<void *>(m_opaque_sp.get()),
                static_cast<uint64_t>(num_resolved));
  return num_resolved;
}

size_t SBBreakpoint::GetNumLocations() const {
  size_t num_locs = 0;
  if (m_opaque_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        m_opaque_sp->GetTarget().GetAPIMutex());
    num_locs = m_opaque_sp->GetNumLocations();
  }
  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_API));
  if (log)
    log->Printf("SBBreakpoint(%p)::GetNumLocations () => %" PRIu64,
                static_cast<void *>(m_opaque_sp.get()),
                static_cast<uint64_t>(num_locs));
  return num_locs;
}

bool SBBreakpoint::GetDescription(SBStream &s) {
  if (m_opaque_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        m_opaque_sp->GetTarget().GetAPIMutex());
    s.Printf("SBBreakpoint: id = %i, ", m_opaque_sp->GetID());
    m_opaque_sp->GetResolverDescription(s.get());
    m_opaque_sp->GetFilterDescription(s.get());
    const size_t num_locations = m_opaque_sp->GetNumLocations();
    s.Printf(", locations = %" PRIu64, (uint64_t)num_locations);
    return true;
  }
  s.Printf("No value");
  return false;
}

bool SBBreakpoint::PrivateBreakpointHitCallback(void *baton,
                                                StoppointCallbackContext *ctx,
                                                lldb::user_id_t break_id,
                                                lldb::user_id_t break_loc_id) {
  ExecutionContext exe_ctx(ctx->exe_ctx_ref);
  BreakpointSP bp_sp(
      exe_ctx.GetTargetRef().GetBreakpointList().FindBreakpointByID(break_id));
  if (baton && bp_sp) {
    CallbackData *data = (CallbackData *)baton;
    lldb_private::Breakpoint *bp = bp_sp.get();
    if (bp && data->callback) {
      Process *process = exe_ctx.GetProcessPtr();
      if (process) {
        SBProcess sb_process(process->shared_from_this());
        SBThread sb_thread;
        SBBreakpointLocation sb_location;
        assert(bp_sp);
        sb_location.SetLocation(bp_sp->FindLocationByID(break_loc_id));
        Thread *thread = exe_ctx.GetThreadPtr();
        if (thread)
          sb_thread.SetThread(thread->shared_from_this());

        return data->callback(data->callback_baton, sb_process, sb_thread,
                              sb_location);
      }
    }
  }
  return true; // Return true if we should stop at this breakpoint
}

void SBBreakpoint::SetCallback(BreakpointHitCallback callback, void *baton) {
  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_API));

  if (log) {
    void *pointer = &callback;
    log->Printf("SBBreakpoint(%p)::SetCallback (callback=%p, baton=%p)",
                static_cast<void *>(m_opaque_sp.get()),
                *static_cast<void **>(&pointer), static_cast<void *>(baton));
  }

  if (m_opaque_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        m_opaque_sp->GetTarget().GetAPIMutex());
    BatonSP baton_sp(new SBBreakpointCallbackBaton(callback, baton));
    m_opaque_sp->SetCallback(SBBreakpoint::PrivateBreakpointHitCallback,
                             baton_sp, false);
  }
}

void SBBreakpoint::SetScriptCallbackFunction(
    const char *callback_function_name) {
  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_API));

  if (log)
    log->Printf("SBBreakpoint(%p)::SetScriptCallbackFunction (callback=%s)",
                static_cast<void *>(m_opaque_sp.get()), callback_function_name);

  if (m_opaque_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        m_opaque_sp->GetTarget().GetAPIMutex());
    BreakpointOptions *bp_options = m_opaque_sp->GetOptions();
    m_opaque_sp->GetTarget()
        .GetDebugger()
        .GetCommandInterpreter()
        .GetScriptInterpreter()
        ->SetBreakpointCommandCallbackFunction(bp_options,
                                               callback_function_name);
  }
}

SBError SBBreakpoint::SetScriptCallbackBody(const char *callback_body_text) {
  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_API));

  if (log)
    log->Printf("SBBreakpoint(%p)::SetScriptCallbackBody: callback body:\n%s)",
                static_cast<void *>(m_opaque_sp.get()), callback_body_text);

  SBError sb_error;
  if (m_opaque_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        m_opaque_sp->GetTarget().GetAPIMutex());
    BreakpointOptions *bp_options = m_opaque_sp->GetOptions();
    Error error =
        m_opaque_sp->GetTarget()
            .GetDebugger()
            .GetCommandInterpreter()
            .GetScriptInterpreter()
            ->SetBreakpointCommandCallback(bp_options, callback_body_text);
    sb_error.SetError(error);
  } else
    sb_error.SetErrorString("invalid breakpoint");

  return sb_error;
}

bool SBBreakpoint::AddName(const char *new_name) {
  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_API));

  if (log)
    log->Printf("SBBreakpoint(%p)::AddName (name=%s)",
                static_cast<void *>(m_opaque_sp.get()), new_name);

  if (m_opaque_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        m_opaque_sp->GetTarget().GetAPIMutex());
    Error error; // Think I'm just going to swallow the error here, it's
                 // probably more annoying to have to provide it.
    return m_opaque_sp->AddName(new_name, error);
  }

  return false;
}

void SBBreakpoint::RemoveName(const char *name_to_remove) {
  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_API));

  if (log)
    log->Printf("SBBreakpoint(%p)::RemoveName (name=%s)",
                static_cast<void *>(m_opaque_sp.get()), name_to_remove);

  if (m_opaque_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        m_opaque_sp->GetTarget().GetAPIMutex());
    m_opaque_sp->RemoveName(name_to_remove);
  }
}

bool SBBreakpoint::MatchesName(const char *name) {
  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_API));

  if (log)
    log->Printf("SBBreakpoint(%p)::MatchesName (name=%s)",
                static_cast<void *>(m_opaque_sp.get()), name);

  if (m_opaque_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        m_opaque_sp->GetTarget().GetAPIMutex());
    return m_opaque_sp->MatchesName(name);
  }

  return false;
}

void SBBreakpoint::GetNames(SBStringList &names) {
  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_API));

  if (log)
    log->Printf("SBBreakpoint(%p)::GetNames ()",
                static_cast<void *>(m_opaque_sp.get()));

  if (m_opaque_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        m_opaque_sp->GetTarget().GetAPIMutex());
    std::vector<std::string> names_vec;
    m_opaque_sp->GetNames(names_vec);
    for (std::string name : names_vec) {
      names.AppendString(name.c_str());
    }
  }
}

lldb_private::Breakpoint *SBBreakpoint::operator->() const {
  return m_opaque_sp.get();
}

lldb_private::Breakpoint *SBBreakpoint::get() const {
  return m_opaque_sp.get();
}

lldb::BreakpointSP &SBBreakpoint::operator*() { return m_opaque_sp; }

const lldb::BreakpointSP &SBBreakpoint::operator*() const {
  return m_opaque_sp;
}

bool SBBreakpoint::EventIsBreakpointEvent(const lldb::SBEvent &event) {
  return Breakpoint::BreakpointEventData::GetEventDataFromEvent(event.get()) !=
         nullptr;
}

BreakpointEventType
SBBreakpoint::GetBreakpointEventTypeFromEvent(const SBEvent &event) {
  if (event.IsValid())
    return Breakpoint::BreakpointEventData::GetBreakpointEventTypeFromEvent(
        event.GetSP());
  return eBreakpointEventTypeInvalidType;
}

SBBreakpoint SBBreakpoint::GetBreakpointFromEvent(const lldb::SBEvent &event) {
  SBBreakpoint sb_breakpoint;
  if (event.IsValid())
    sb_breakpoint.m_opaque_sp =
        Breakpoint::BreakpointEventData::GetBreakpointFromEvent(event.GetSP());
  return sb_breakpoint;
}

SBBreakpointLocation
SBBreakpoint::GetBreakpointLocationAtIndexFromEvent(const lldb::SBEvent &event,
                                                    uint32_t loc_idx) {
  SBBreakpointLocation sb_breakpoint_loc;
  if (event.IsValid())
    sb_breakpoint_loc.SetLocation(
        Breakpoint::BreakpointEventData::GetBreakpointLocationAtIndexFromEvent(
            event.GetSP(), loc_idx));
  return sb_breakpoint_loc;
}

uint32_t
SBBreakpoint::GetNumBreakpointLocationsFromEvent(const lldb::SBEvent &event) {
  uint32_t num_locations = 0;
  if (event.IsValid())
    num_locations =
        (Breakpoint::BreakpointEventData::GetNumBreakpointLocationsFromEvent(
            event.GetSP()));
  return num_locations;
}
