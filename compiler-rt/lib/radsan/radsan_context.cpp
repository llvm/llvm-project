//===--- radsan_context.cpp - Realtime Sanitizer --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include <radsan/radsan_context.h>

#include <radsan/radsan_stack.h>

#include <sanitizer_common/sanitizer_allocator_internal.h>
#include <sanitizer_common/sanitizer_stacktrace.h>

#include <new>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

static pthread_key_t context_key;
static pthread_once_t key_once = PTHREAD_ONCE_INIT;

static void internalFree(void *ptr) { __sanitizer::InternalFree(ptr); }

static __radsan::Context &GetContextForThisThreadImpl() {
  auto make_thread_local_context_key = []() {
    CHECK_EQ(pthread_key_create(&context_key, internalFree), 0);
  };

  pthread_once(&key_once, make_thread_local_context_key);
  __radsan::Context *current_thread_context =
      static_cast<__radsan::Context *>(pthread_getspecific(context_key));
  if (current_thread_context == nullptr) {
    current_thread_context = static_cast<__radsan::Context *>(
        __sanitizer::InternalAlloc(sizeof(__radsan::Context)));
    new (current_thread_context) __radsan::Context();
    pthread_setspecific(context_key, current_thread_context);
  }

  return *current_thread_context;
}

/*
    This is a placeholder stub for a future feature that will allow
    a user to configure RADSan's behaviour when a real-time safety
    violation is detected. The RADSan developers intend for the
    following choices to be made available, via a RADSAN_OPTIONS
    environment variable, in a future PR:

        i) exit,
       ii) continue, or
      iii) wait for user input from stdin.

    Until then, and to keep the first PRs small, only the exit mode
    is available.
*/
static void InvokeViolationDetectedAction() { exit(EXIT_FAILURE); }

namespace __radsan {

Context::Context() = default;

void Context::RealtimePush() { realtime_depth++; }

void Context::RealtimePop() { realtime_depth--; }

void Context::BypassPush() { bypass_depth++; }

void Context::BypassPop() { bypass_depth--; }

void Context::ExpectNotRealtime(const char *intercepted_function_name) {
  if (InRealtimeContext() && !IsBypassed()) {
    BypassPush();
    PrintDiagnostics(intercepted_function_name);
    InvokeViolationDetectedAction();
    BypassPop();
  }
}

bool Context::InRealtimeContext() const { return realtime_depth > 0; }

bool Context::IsBypassed() const { return bypass_depth > 0; }

void Context::PrintDiagnostics(const char *intercepted_function_name) {
  fprintf(stderr,
          "Real-time violation: intercepted call to real-time unsafe function "
          "`%s` in real-time context! Stack trace:\n",
          intercepted_function_name);
  __radsan::PrintStackTrace();
}

Context &GetContextForThisThread() { return GetContextForThisThreadImpl(); }

} // namespace __radsan
