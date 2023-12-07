// Verifies lifetime of __gro local variable
// Verify that coroutine promise and allocated memory are freed up on exception.
// RUN: %clang_cc1 -std=c++20 -triple=x86_64-unknown-linux-gnu -emit-llvm -o - %s -disable-llvm-passes | FileCheck %s

#include "Inputs/coroutine.h"

using namespace std;

struct GroType {
  ~GroType();
  operator int() noexcept;
};

template <> struct std::coroutine_traits<int> {
  struct promise_type {
    GroType get_return_object() noexcept;
    suspend_always initial_suspend() noexcept;
    suspend_always final_suspend() noexcept;
    void return_void() noexcept;
    promise_type();
    ~promise_type();
    void unhandled_exception() noexcept;
  };
};

struct Cleanup { ~Cleanup(); };
void doSomething() noexcept;

// CHECK: define{{.*}} i32 @_Z1fv(
int f() {
  // CHECK: %[[RetVal:.+]] = alloca i32
  // CHECK: %[[GroActive:.+]] = alloca i1

  // CHECK: %[[Size:.+]] = call i64 @llvm.coro.size.i64()
  // CHECK: call noalias noundef nonnull ptr @_Znwm(i64 noundef %[[Size]])
  // CHECK: store i1 false, ptr %[[GroActive]]
  // CHECK: call void @_ZNSt16coroutine_traitsIiJEE12promise_typeC1Ev(
  // CHECK: call void @_ZNSt16coroutine_traitsIiJEE12promise_type17get_return_objectEv(
  // CHECK: store i1 true, ptr %[[GroActive]]

  Cleanup cleanup;
  doSomething();
  co_return;

  // CHECK: call void @_Z11doSomethingv(
  // CHECK: call void @_ZNSt16coroutine_traitsIiJEE12promise_type11return_voidEv(
  // CHECK: call void @_ZN7CleanupD1Ev(

  // Destroy promise and free the memory.

  // CHECK: call void @_ZNSt16coroutine_traitsIiJEE12promise_typeD1Ev(
  // CHECK: %[[Mem:.+]] = call ptr @llvm.coro.free(
  // CHECK: call void @_ZdlPv(ptr noundef %[[Mem]])

  // Initialize retval from Gro and destroy Gro
  // Note this also tests delaying initialization when Gro and function return
  // types mismatch (see cwg2563).

  // CHECK: %[[Conv:.+]] = call noundef i32 @_ZN7GroTypecviEv(
  // CHECK: store i32 %[[Conv]], ptr %[[RetVal]]
  // CHECK: %[[IsActive:.+]] = load i1, ptr %[[GroActive]]
  // CHECK: br i1 %[[IsActive]], label %[[CleanupGro:.+]], label %[[Done:.+]]

  // CHECK: [[CleanupGro]]:
  // CHECK:   call void @_ZN7GroTypeD1Ev(
  // CHECK:   br label %[[Done]]

  // CHECK: [[Done]]:
  // CHECK:   %[[LoadRet:.+]] = load i32, ptr %[[RetVal]]
  // CHECK:   ret i32 %[[LoadRet]]
}

class invoker {
public:
  class invoker_promise {
  public:
    invoker get_return_object() { return invoker{}; }
    auto initial_suspend() { return suspend_always{}; }
    auto final_suspend() noexcept { return suspend_always{}; }
    void return_void() {}
    void unhandled_exception() {}
  };
  using promise_type = invoker_promise;
  invoker() {}
  invoker(const invoker &) = delete;
  invoker &operator=(const invoker &) = delete;
  invoker(invoker &&) = delete;
  invoker &operator=(invoker &&) = delete;
};

// According to cwg2563, matching GRO and function return type must allow
// for eager initialization and RVO.
// CHECK: define{{.*}} void @_Z1gv({{.*}} %[[AggRes:.+]])
invoker g() {
  // CHECK: %[[ResultPtr:.+]] = alloca ptr
  // CHECK-NEXT: %[[Promise:.+]] = alloca %"class.invoker::invoker_promise"

  // CHECK: store ptr %[[AggRes]], ptr %[[ResultPtr]]
  // CHECK: coro.init:
  // CHECK: = call ptr @llvm.coro.begin

  // delayed GRO pattern stores a GRO active flag, make sure to not emit it.
  // CHECK-NOT: store i1 false, ptr
  // CHECK: call void @_ZN7invoker15invoker_promise17get_return_objectEv({{.*}} %[[AggRes]]
  co_return;
}
