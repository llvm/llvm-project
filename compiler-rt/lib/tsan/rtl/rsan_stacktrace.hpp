#pragma once

#include "sanitizer_common/sanitizer_stacktrace.h"
#include "sanitizer_common/sanitizer_internal_defs.h"
#include "rsan_defs.hpp"

namespace Robustness {
	using __sanitizer::uhwptr;

static const u32 kStackTraceMax = 4;

// StackTrace that owns the buffer used to store the addresses.
struct LittleStackTrace : public __sanitizer::StackTrace {
  uptr trace_buffer[kStackTraceMax] = {};
  uptr top_frame_bp;  // Optional bp of a top frame.

  LittleStackTrace() : __sanitizer::StackTrace(trace_buffer, 0), top_frame_bp(0) {}

  void Init(const uptr *pcs, uptr cnt, uptr extra_top_pc = 0);

  // Get the stack trace with the given pc and bp.
  // The pc will be in the position 0 of the resulting stack trace.
  // The bp may refer to the current frame or to the caller's frame.
  void Unwind(uptr pc, uptr bp, void *context, bool request_fast,
              u32 max_depth = kStackTraceMax) {
    top_frame_bp = (max_depth > 0) ? bp : 0;
    // Small max_depth optimization
    if (max_depth <= 1) {
      if (max_depth == 1)
        trace_buffer[0] = pc;
      size = max_depth;
      return;
    }
    UnwindImpl(pc, bp, context, request_fast, max_depth);
  }

  void Unwind(u32 max_depth, uptr pc, uptr bp, void *context, uptr stack_top,
              uptr stack_bottom, bool request_fast_unwind);

  void Reset() {
    *static_cast<StackTrace *>(this) = StackTrace(trace_buffer, 0);
    top_frame_bp = 0;
  }

  LittleStackTrace(const LittleStackTrace &rhs) : StackTrace(trace, 0) {
	  trace = trace_buffer;
	  size = rhs.size;
	  for (auto i = 0u; i < kStackTraceMax; ++i)
		  trace_buffer[i] = rhs.trace_buffer[i];
	  top_frame_bp = rhs.top_frame_bp;
  }
  //void operator=(const LittleStackTrace &rhs) : StackTrace(trace, 0) {
  //    trace = trace_buffer;
  //    size = rhs.size;
  //    for (auto i = 0u; i < kStackTraceMax; ++i)
  //  	  trace_buffer[i] = rhs.trace_buffer[i];
  //    top_frame_bp = rhs.top_frame_bp;
  //}

 private:
  // Every runtime defines its own implementation of this method
  void UnwindImpl(uptr pc, uptr bp, void *context, bool request_fast,
                  u32 max_depth);

  // UnwindFast/Slow have platform-specific implementations
  void UnwindFast(uptr pc, uptr bp, uptr stack_top, uptr stack_bottom,
                  u32 max_depth);
  void UnwindSlow(uptr pc, u32 max_depth);
  void UnwindSlow(uptr pc, void *context, u32 max_depth);

  void PopStackFrames(uptr count);
  uptr LocatePcInTrace(uptr pc);


  friend class FastUnwindTest;
};

#if defined(__s390x__)
static const uptr kFrameSize = 160;
#elif defined(__s390__)
static const uptr kFrameSize = 96;
#else
static const uptr kFrameSize = 2 * sizeof(uhwptr);
#endif

// Check if given pointer points into allocated stack area.
static inline bool IsValidFrame(uptr frame, uptr stack_top, uptr stack_bottom) {
  return frame > stack_bottom && frame < stack_top - kFrameSize;
}

}  // namespace Robustness

