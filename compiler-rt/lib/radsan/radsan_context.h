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

  void realtimePush();
  void realtimePop();

  void bypassPush();
  void bypassPop();

  void expectNotRealtime(const char *interpreted_function_name);

private:
  bool inRealtimeContext() const;
  bool isBypassed() const;
  void printDiagnostics(const char *intercepted_function_name);

  int realtime_depth_{0};
  int bypass_depth_{0};
};

Context &getContextForThisThread();

} // namespace radsan
