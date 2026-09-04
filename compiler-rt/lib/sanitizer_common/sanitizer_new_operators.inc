//===-- sanitizer_new_operators.inc -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Shared OPERATOR_NEW_BODY* macros for sanitizer operator new overrides.
//
// All eight variants compose the same per-call setup (a sanitizer-specific
// stack-trace acquisition) with one of the two chain-handling templates from
// sanitizer_new_handler.h and two sanitizer-supplied lambdas: an Alloc that
// invokes the sanitizer's internal allocator (returning nullptr on failure)
// and an OnExhausted that calls the sanitizer's "abort with diagnostic"
// handler.
//
// This file is included after the consuming TU has defined the prerequisite
// macros listed below. Names reference symbols (e.g. `stack`, `size`) that
// exist only in the macro's expansion context, so this is an .inc, not a
// freestanding header.
//
// The consuming TU must define before including:
//
//   SANITIZER_NEW_STACK_TRACE
//       A statement that, when expanded inside an operator new body, puts a
//       BufferedStackTrace named `stack` in scope. e.g. for asan:
//         #define SANITIZER_NEW_STACK_TRACE GET_STACK_TRACE_MALLOC
//
//   SANITIZER_NEW_REPORT_OOM(size)
//       The sanitizer's noreturn diagnostic path for chain-exhausted OOM.
//       Must call a function that does not return (the framework will run
//       UNREACHABLE() and abort if the macro's expansion ever returns).
//       e.g. for asan:
//         #define SANITIZER_NEW_REPORT_OOM(size) \
//           ReportOutOfMemory(size, &stack)
//
//   SANITIZER_NEW(size)
//   SANITIZER_NEW_ARRAY(size)
//   SANITIZER_NEW_ALIGNED(size, align)
//   SANITIZER_NEW_ARRAY_ALIGNED(size, align)
//       The sanitizer's four internal alloc helpers. Each must return nullptr
//       on OOM (so the chain-handling templates can drive the
//       std::get_new_handler() loop).  Other failure modes (e.g. invalid
//       alignment) should cause the helper to abort with a diagnostic.
//
//===----------------------------------------------------------------------===//

#ifndef SANITIZER_NEW_STACK_TRACE
#  error "SANITIZER_NEW_STACK_TRACE must be defined before including"
#endif
#ifndef SANITIZER_NEW_REPORT_OOM
#  error "SANITIZER_NEW_REPORT_OOM must be defined before including"
#endif
#ifndef SANITIZER_NEW
#  error "SANITIZER_NEW must be defined before including"
#endif
#ifndef SANITIZER_NEW_ARRAY
#  error "SANITIZER_NEW_ARRAY must be defined before including"
#endif
#ifndef SANITIZER_NEW_ALIGNED
#  error "SANITIZER_NEW_ALIGNED must be defined before including"
#endif
#ifndef SANITIZER_NEW_ARRAY_ALIGNED
#  error "SANITIZER_NEW_ARRAY_ALIGNED must be defined before including"
#endif

#define OPERATOR_NEW_BODY                    \
  SANITIZER_NEW_STACK_TRACE;                 \
  return __sanitizer::NewImplThrowing(       \
      [&]() { return SANITIZER_NEW(size); }, \
      [&]() { SANITIZER_NEW_REPORT_OOM(size); })

#define OPERATOR_NEW_BODY_NOTHROW            \
  SANITIZER_NEW_STACK_TRACE;                 \
  return __sanitizer::NewImplNothrow(        \
      [&]() { return SANITIZER_NEW(size); }, \
      [&]() { SANITIZER_NEW_REPORT_OOM(size); })

#define OPERATOR_NEW_BODY_ARRAY                    \
  SANITIZER_NEW_STACK_TRACE;                       \
  return __sanitizer::NewImplThrowing(             \
      [&]() { return SANITIZER_NEW_ARRAY(size); }, \
      [&]() { SANITIZER_NEW_REPORT_OOM(size); })

#define OPERATOR_NEW_BODY_ARRAY_NOTHROW            \
  SANITIZER_NEW_STACK_TRACE;                       \
  return __sanitizer::NewImplNothrow(              \
      [&]() { return SANITIZER_NEW_ARRAY(size); }, \
      [&]() { SANITIZER_NEW_REPORT_OOM(size); })

#define OPERATOR_NEW_BODY_ALIGN                             \
  SANITIZER_NEW_STACK_TRACE;                                \
  return __sanitizer::NewImplThrowing(                      \
      [&]() { return SANITIZER_NEW_ALIGNED(size, align); }, \
      [&]() { SANITIZER_NEW_REPORT_OOM(size); })

#define OPERATOR_NEW_BODY_ALIGN_NOTHROW                     \
  SANITIZER_NEW_STACK_TRACE;                                \
  return __sanitizer::NewImplNothrow(                       \
      [&]() { return SANITIZER_NEW_ALIGNED(size, align); }, \
      [&]() { SANITIZER_NEW_REPORT_OOM(size); })

#define OPERATOR_NEW_BODY_ALIGN_ARRAY                             \
  SANITIZER_NEW_STACK_TRACE;                                      \
  return __sanitizer::NewImplThrowing(                            \
      [&]() { return SANITIZER_NEW_ARRAY_ALIGNED(size, align); }, \
      [&]() { SANITIZER_NEW_REPORT_OOM(size); })

#define OPERATOR_NEW_BODY_ALIGN_ARRAY_NOTHROW                     \
  SANITIZER_NEW_STACK_TRACE;                                      \
  return __sanitizer::NewImplNothrow(                             \
      [&]() { return SANITIZER_NEW_ARRAY_ALIGNED(size, align); }, \
      [&]() { SANITIZER_NEW_REPORT_OOM(size); })
