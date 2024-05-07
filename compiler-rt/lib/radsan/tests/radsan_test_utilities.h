//===--- radsan_test_utilities.h - Realtime Sanitizer --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "gmock/gmock.h"
#include "radsan.h"
#include <string>

namespace radsan_testing {

template <typename Function>
void RealtimeInvoke(Function &&Func) 
{
  radsan_realtime_enter();
  std::forward<Function>(Func)();
  radsan_realtime_exit();
}

template <typename Function>
void ExpectRealtimeDeath(Function &&Func,
                         const char *intercepted_method_name = nullptr) {

  using namespace testing;

  auto ExpectedErrorSubstring = [&]() -> std::string {
    return intercepted_method_name != nullptr
               ? "Real-time violation: intercepted call to real-time unsafe "
                 "function `" +
                     std::string(intercepted_method_name) + "`"
               : "";
  };

  EXPECT_EXIT(RealtimeInvoke(std::forward<Function>(Func)),
              ExitedWithCode(EXIT_FAILURE), ExpectedErrorSubstring());
}

template <typename Function> void ExpectNonRealtimeSurvival(Function &&Func) {
  std::forward<Function>(Func)();
}

} // namespace radsan_testing
