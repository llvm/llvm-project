#pragma once
#include "tsan_defs.h"
#include "tsan_rtl.h"
#include "tsan_symbolize.h"
#include "tsan_flags.h"
#include "rsan_defs.hpp"
#include "tsan_stack_trace.h"
#include "rsan_stacktrace.hpp"
namespace __tsan {
	class ThreadState;

void GetLineOfCode(InternalScopedString& res, const ReportStack *ent);
ReportStack SymbolizeStack(StackTrace trace);

template<typename StackTraceTy>
void ObtainCurrentLine(ThreadState *thr, uptr toppc, StackTraceTy *stack,
                        uptr *tag = nullptr) {
  uptr size = thr->shadow_stack_pos - thr->shadow_stack;
  uptr start = 0;
  const auto kStackTraceMax = Robustness::kStackTraceMax;
  if (size + !!toppc > kStackTraceMax) {
    start = size + !!toppc - kStackTraceMax;
    size = kStackTraceMax - !!toppc;
  }
  stack->Init(&thr->shadow_stack[start], size, toppc);
  ExtractTagFromStack(stack, tag);
}
void getCurrentLine(InternalScopedString &s, ThreadState *thr, uptr pc);

} //namespace __tsan

namespace Robustness {
	using __tsan::ObtainCurrentLine;
	using __tsan::getCurrentLine;



template<Robustness::ViolationType V> void reportViolation(Robustness::ThreadId t, Robustness::Address a, Robustness::LocationId lid, const DebugInfo &dbg, const LittleStackTrace &prev = {}){
	InternalScopedString ss;
	getCurrentLine(ss, dbg.thr, dbg.pc);
	InternalScopedString prevs;

	auto oldstack = __tsan::SymbolizeStack(prev);
	__tsan::GetLineOfCode(prevs, &oldstack);

	// TODO: Make the color codes easier to use
	// TODO: Update print functions
	if constexpr (V == Robustness::ViolationType::read ||
			V == Robustness::ViolationType::write){
		//++violationsCount;
		const char *fmtString;
#define PRESTRING "\033[1;31mRobustness Violation: Tid: %u, Address: %llx (%d), Type: "
#define POSTSTRING " %d, Violation: %s, PrevAccess: %s\033[0m\n"
		if constexpr (V == Robustness::ViolationType::read)
			fmtString = PRESTRING "rd" POSTSTRING;
		else
			fmtString = PRESTRING "st" POSTSTRING;
#undef PRESTRING
#undef POSTRING
		Printf(fmtString, t, a, lid, (int)V, ss.data(), prevs.data());
	} else
		static_assert(Robustness::always_false_v<decltype(V)>, "Unknwon error type");
}

template<Robustness::ViolationType V> void reportViolation(Robustness::ThreadId t, Robustness::Address a, const DebugInfo& dbg, const LittleStackTrace &prev, uint64_t val){
	InternalScopedString ss;
	getCurrentLine(ss, dbg.thr, dbg.pc);
	InternalScopedString prevs;

	auto oldstack = __tsan::SymbolizeStack(prev);
	__tsan::GetLineOfCode(prevs, &oldstack);

	if constexpr (V == Robustness::ViolationType::read ||
			V == Robustness::ViolationType::write){
		const char *fmtString;
		if constexpr (V == Robustness::ViolationType::read)
			fmtString = "\033[1;31mRobustness Violation: Tid: %u, Address: %llx, Type: rd %d, Val: %llu Violation: %s, PrevAccess: %s\033[0m\n";
		else
			fmtString = "\033[1;31mRobustness Violation: Tid: %u, Address: %llx, Type: st %d, Val: %llu Violation: %s, PrevAccess: %s\033[0m\n";
		Printf(fmtString, t, a, (int)V, val, ss.data(), prevs.data());
	} else
		static_assert(Robustness::always_false_v<decltype(V)>, "Unknwon error type");
}
} //namespace Robustness

