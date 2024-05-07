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

using namespace __sanitizer;

namespace detail {

static pthread_key_t Key;
static pthread_once_t KeyOnce = PTHREAD_ONCE_INIT;
void internalFree(void *Ptr) { __sanitizer::InternalFree(Ptr); }

} // namespace detail

namespace radsan {

Context::Context() = default;

void Context::RealtimePush() { RealtimeDepth++; }

void Context::RealtimePop() { RealtimeDepth--; }

void Context::BypassPush() { BypassDepth++; }

void Context::BypassPop() { BypassDepth--; }

void Context::ExpectNotRealtime(const char *InterceptedFunctionName) {
  if (InRealtimeContext() && !IsBypassed()) {
    BypassPush();
    PrintDiagnostics(InterceptedFunctionName);
    exit(EXIT_FAILURE);
    BypassPop();
  }
}

bool Context::InRealtimeContext() const { return RealtimeDepth > 0; }

bool Context::IsBypassed() const { return BypassDepth > 0; }

void Context::PrintDiagnostics(const char *InterceptedFunctionName) {
  fprintf(stderr,
          "Real-time violation: intercepted call to real-time unsafe function "
          "`%s` in real-time context! Stack trace:\n",
          InterceptedFunctionName);
  radsan::printStackTrace();
}

Context &getContextForThisThread() {
  auto MakeTlsKey = []() {
    CHECK_EQ(pthread_key_create(&detail::Key, detail::internalFree), 0);
  };

  pthread_once(&detail::KeyOnce, MakeTlsKey);
  Context *CurrentThreadContext = static_cast<Context *>(pthread_getspecific(detail::Key));
  if (CurrentThreadContext == nullptr) {
    CurrentThreadContext = static_cast<Context *>(InternalAlloc(sizeof(Context)));
    new(CurrentThreadContext) Context();
    pthread_setspecific(detail::Key, CurrentThreadContext);
  }

  return *CurrentThreadContext;
}

} // namespace radsan
