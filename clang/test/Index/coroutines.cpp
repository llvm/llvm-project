// RUN: c-index-test -test-load-source all -c %s -fsyntax-only -target x86_64-apple-darwin9 -fcoroutines-ts -std=c++1z -I%S/../SemaCXX/Inputs | FileCheck %s
#include "std-coroutine.h"

using std::experimental::suspend_always;
using std::experimental::suspend_never;

struct promise_void {
  void get_return_object();
  suspend_always initial_suspend();
  suspend_always final_suspend();
  void return_void();
  void unhandled_exception();
};

template <>
struct std::experimental::coroutine_traits<void> { using promise_type = promise_void; };

void CoroutineTestRet() {
  co_return;
}
// CHECK: [[@LINE-3]]:25: UnexposedStmt=
// CHECK-SAME: [[@LINE-4]]:25 - [[@LINE-2]]:2]
// CHECK: [[@LINE-4]]:3: UnexposedStmt=
// CHECK-SAME: [[@LINE-5]]:3 - [[@LINE-5]]:12]
