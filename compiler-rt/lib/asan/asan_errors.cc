//===-- asan_errors.cc ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of AddressSanitizer, an address sanity checker.
//
// ASan implementation for error structures.
//===----------------------------------------------------------------------===//

#include "asan_errors.h"
#include "asan_descriptions.h"
#include "asan_stack.h"

namespace __asan {

void ErrorStackOverflow::Print() {
  Decorator d;
  Printf("%s", d.Warning());
  Report(
      "ERROR: AddressSanitizer: stack-overflow on address %p"
      " (pc %p bp %p sp %p T%d)\n",
      (void *)addr, (void *)pc, (void *)bp, (void *)sp, tid);
  Printf("%s", d.EndWarning());
  scariness.Print();
  BufferedStackTrace stack;
  GetStackTraceWithPcBpAndContext(&stack, kStackTraceMax, pc, bp, context,
                                  common_flags()->fast_unwind_on_fatal);
  stack.Print();
  ReportErrorSummary("stack-overflow", &stack);
}

void ErrorDoubleFree::Print() {
  Decorator d;
  Printf("%s", d.Warning());
  char tname[128];
  Report(
      "ERROR: AddressSanitizer: attempting double-free on %p in "
      "thread T%d%s:\n",
      addr_description.addr, tid,
      ThreadNameWithParenthesis(tid, tname, sizeof(tname)));
  Printf("%s", d.EndWarning());
  scariness.Print();
  GET_STACK_TRACE_FATAL(second_free_stack->trace[0],
                        second_free_stack->top_frame_bp);
  stack.Print();
  addr_description.Print();
  ReportErrorSummary("double-free", &stack);
}

void ErrorNewDeleteSizeMismatch::Print() {
  Decorator d;
  Printf("%s", d.Warning());
  char tname[128];
  Report(
      "ERROR: AddressSanitizer: new-delete-type-mismatch on %p in thread "
      "T%d%s:\n",
      addr_description.addr, tid,
      ThreadNameWithParenthesis(tid, tname, sizeof(tname)));
  Printf("%s  object passed to delete has wrong type:\n", d.EndWarning());
  Printf(
      "  size of the allocated type:   %zd bytes;\n"
      "  size of the deallocated type: %zd bytes.\n",
      addr_description.chunk_access.chunk_size, delete_size);
  CHECK_GT(free_stack->size, 0);
  scariness.Print();
  GET_STACK_TRACE_FATAL(free_stack->trace[0], free_stack->top_frame_bp);
  stack.Print();
  addr_description.Print();
  ReportErrorSummary("new-delete-type-mismatch", &stack);
  Report(
      "HINT: if you don't care about these errors you may set "
      "ASAN_OPTIONS=new_delete_type_mismatch=0\n");
}

}  // namespace __asan
