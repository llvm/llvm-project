//===--- radsan_context.h - Realtime Sanitizer --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#pragma once

namespace radsan {

class Context {
public:
  Context();

  void RealtimePush();
  void RealtimePop();

  void BypassPush();
  void BypassPop();

  void ExpectNotRealtime(const char *interpreted_function_name);

private:
  bool InRealtimeContext() const;
  bool IsBypassed() const;
  void PrintDiagnostics(const char *InterceptedFunctionName);

  int RealtimeDepth{0};
  int BypassDepth{0};
};

Context &getContextForThisThread();

} // namespace radsan
