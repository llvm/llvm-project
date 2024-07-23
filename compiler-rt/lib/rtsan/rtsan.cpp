//===--- rtsan.cpp - Realtime Sanitizer -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include <rtsan/rtsan.h>
#include <rtsan/rtsan_context.h>
#include <rtsan/rtsan_interceptors.h>

extern "C" {

SANITIZER_INTERFACE_ATTRIBUTE void __rtsan_init() {
  __rtsan::InitializeInterceptors();
}

SANITIZER_INTERFACE_ATTRIBUTE void __rtsan_realtime_enter() {
  __rtsan::GetContextForThisThread().RealtimePush();
}

SANITIZER_INTERFACE_ATTRIBUTE void __rtsan_realtime_exit() {
  __rtsan::GetContextForThisThread().RealtimePop();
}

SANITIZER_INTERFACE_ATTRIBUTE void __rtsan_off() {
  __rtsan::GetContextForThisThread().BypassPush();
}

SANITIZER_INTERFACE_ATTRIBUTE void __rtsan_on() {
  __rtsan::GetContextForThisThread().BypassPop();
}

} // extern "C"
