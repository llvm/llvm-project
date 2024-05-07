//===--- radsan.cpp - Realtime Sanitizer --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include <radsan/radsan.h>
#include <radsan/radsan_context.h>
#include <radsan/radsan_interceptors.h>
#include <unistd.h>

extern "C" {
RADSAN_EXPORT void radsan_init() { radsan::initialiseInterceptors(); }

RADSAN_EXPORT void radsan_realtime_enter() {
  radsan::getContextForThisThread().realtimePush();
}

RADSAN_EXPORT void radsan_realtime_exit() {
  radsan::getContextForThisThread().realtimePop();
}

RADSAN_EXPORT void radsan_off() {
  radsan::getContextForThisThread().bypassPush();
}

RADSAN_EXPORT void radsan_on() {
  radsan::getContextForThisThread().bypassPop();
}
}
