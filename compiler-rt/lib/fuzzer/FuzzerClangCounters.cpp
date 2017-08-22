//===- FuzzerExtraCounters.cpp - Extra coverage counters ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// Coverage counters from Clang's SourceBasedCodeCoverage.
//===----------------------------------------------------------------------===//

// Support for SourceBasedCodeCoverage is experimental:
// * Works only for the main binary, not DSOs yet.
// * Works only on Linux.
// * Does not implement print_pcs/print_coverage yet.
// * Is not fully evaluated for performance and sensitivity.
//   We expect large performance drop due to 64-bit counters,
//   and *maybe* better sensitivity due to more fine-grained counters.
//   Preliminary comparison on a single benchmark (RE2) shows
//   a bit worse sensitivity though.

#include "FuzzerDefs.h"

#if LIBFUZZER_LINUX
__attribute__((weak)) extern uint64_t __start___llvm_prf_cnts;
__attribute__((weak)) extern uint64_t __stop___llvm_prf_cnts;
namespace fuzzer {
uint64_t *ClangCountersBegin() { return &__start___llvm_prf_cnts; }
uint64_t *ClangCountersEnd() { return &__stop___llvm_prf_cnts; }
}  // namespace fuzzer
#else
// TODO: Implement on Mac (if the data shows it's worth it).
//__attribute__((visibility("hidden")))
//extern uint64_t CountersStart __asm("section$start$__DATA$__llvm_prf_cnts");
//__attribute__((visibility("hidden")))
//extern uint64_t CountersEnd __asm("section$end$__DATA$__llvm_prf_cnts");
namespace fuzzer {
uint64_t *ClangCountersBegin() { return nullptr; }
uint64_t *ClangCountersEnd() { return  nullptr; }
}  // namespace fuzzer
#endif

namespace fuzzer {
ATTRIBUTE_NO_SANITIZE_ALL
void ClearClangCounters() {  // hand-written memset, don't asan-ify.
  for (auto P = ClangCountersBegin(); P < ClangCountersEnd(); P++)
    *P = 0;
}
}
