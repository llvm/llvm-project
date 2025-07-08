#include "rsan_report.hpp"
#include "rsan_defs.hpp"
#include "sanitizer_common/sanitizer_stacktrace_printer.h"
namespace __tsan {

void GetLineOfCode(InternalScopedString& res, const ReportStack *ent) {
  if (ent == 0 || ent->frames == 0) {
    res.Append("[failed to locate source]");
    return;
  }
  SymbolizedStack *frame = ent->frames;
  for (int i = 0; frame && frame->info.address; frame = frame->next, i++) {
	const char *formatString = "%S";
	// FIXME: Need to extract this...
	StackTracePrinter::GetOrInit()->RenderFrame(
			&res, formatString, i, frame->info.address,
			&frame->info, common_flags()->symbolize_vs_style,
			common_flags()->strip_path_prefix);
  }
}

ReportStack SymbolizeStack(StackTrace trace) {
  if (trace.size == 0)
    return ReportStack();
  SymbolizedStack *top = nullptr;
  for (uptr si = 0; si < trace.size; si++) {
    const uptr pc = trace.trace[si];
    uptr pc1 = pc;
    // We obtain the return address, but we're interested in the previous
    // instruction.
    if ((pc & kExternalPCBit) == 0)
      pc1 = StackTrace::GetPreviousInstructionPc(pc);
    SymbolizedStack *ent = SymbolizeCode(pc1);
    CHECK_NE(ent, 0);
    SymbolizedStack *last = ent;
    while (last->next) {
      last->info.address = pc;  // restore original pc for report
      last = last->next;
    }
    last->info.address = pc;  // restore original pc for report
    last->next = top;
    top = ent;
  }
  //StackStripMain(top);

  ReportStack stack;
  stack.frames = top;
  return stack;
}

void getCurrentLine(InternalScopedString &ss, ThreadState *thr, uptr pc) {
	//CheckedMutex::CheckNoLocks();
	ScopedIgnoreInterceptors ignore;
	//
  // We need to lock the slot during RestoreStack because it protects
  // the slot journal.
  //Lock slot_lock(&ctx->slots[static_cast<uptr>(s[1].sid())].mtx);
  //ThreadRegistryLock l0(&ctx->thread_registry);
  //Lock slots_lock(&ctx->slot_mtx);

  VarSizeStackTrace trace;
  ObtainCurrentLine(thr, pc, &trace);
  auto stack = SymbolizeStack(trace);
  GetLineOfCode(ss, &stack);
}


} //namespace __tsan

namespace Robustness {
}// namespace Robustness

