//===--- Definitions of common thread items ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "thread.h"
#include "mutex.h"

#include "src/__support/fixedvector.h"

namespace __llvm_libc {

thread_local Thread self;

namespace {

using AtExitCallback = void(void *);

struct AtExitUnit {
  AtExitCallback *callback = nullptr;
  void *obj = nullptr;
  constexpr AtExitUnit() = default;
  constexpr AtExitUnit(AtExitCallback *cb, void *o) : callback(cb), obj(o) {}
};

} // anonymous namespace

class ThreadAtExitCallbackMgr {
  Mutex mtx;
  // TODO: Use a BlockStore when compiled for production.
  FixedVector<AtExitUnit, 1024> callback_list;

public:
  constexpr ThreadAtExitCallbackMgr() : mtx(false, false, false) {}

  int add_callback(AtExitCallback *callback, void *obj) {
    MutexLock lock(&mtx);
    return callback_list.push_back({callback, obj});
  }

  void call() {
    mtx.lock();
    while (!callback_list.empty()) {
      auto atexit_unit = callback_list.back();
      callback_list.pop_back();
      mtx.unlock();
      atexit_unit.callback(atexit_unit.obj);
      mtx.lock();
    }
  }
};

static thread_local ThreadAtExitCallbackMgr atexit_callback_mgr;

// The function __cxa_thread_atexit is provided by C++ runtimes like libcxxabi.
// It is used by thread local object runtime to register destructor calls. To
// actually register destructor call with the threading library, it calls
// __cxa_thread_atexit_impl, which is to be provided by the threading library.
// The semantics are very similar to the __cxa_atexit function except for the
// fact that the registered callback is thread specific.
extern "C" int __cxa_thread_atexit_impl(AtExitCallback *callback, void *obj,
                                        void *) {
  return atexit_callback_mgr.add_callback(callback, obj);
}

namespace internal {

ThreadAtExitCallbackMgr *get_thread_atexit_callback_mgr() {
  return &atexit_callback_mgr;
}

void call_atexit_callbacks(ThreadAttributes *attrib) {
  attrib->atexit_callback_mgr->call();
}

} // namespace internal

} // namespace __llvm_libc
