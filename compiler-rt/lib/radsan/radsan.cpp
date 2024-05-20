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

extern "C" {

SANITIZER_INTERFACE_ATTRIBUTE void __radsan_init() {
  __radsan::InitializeInterceptors();
}

SANITIZER_INTERFACE_ATTRIBUTE void __radsan_realtime_enter() {
  __radsan::GetContextForThisThread().RealtimePush();
}

SANITIZER_INTERFACE_ATTRIBUTE void __radsan_realtime_exit() {
  __radsan::GetContextForThisThread().RealtimePop();
}

SANITIZER_INTERFACE_ATTRIBUTE void __radsan_off() {
  __radsan::GetContextForThisThread().BypassPush();
}

SANITIZER_INTERFACE_ATTRIBUTE void __radsan_on() {
  __radsan::GetContextForThisThread().BypassPop();
}

} // extern "C"
