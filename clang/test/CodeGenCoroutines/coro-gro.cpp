// Verifies lifetime of __gro local variable
// Verify that coroutine promise and allocated memory are freed up on exception.
// RUN: %clang_cc1 -std=c++20 -triple=x86_64-unknown-linux-gnu -fexceptions -fcxx-exceptions -emit-llvm -o - %s -disable-llvm-passes | FileCheck %s

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
    promise_type() noexcept;
    ~promise_type();
    void unhandled_exception() noexcept;
  };
};

struct Cleanup { ~Cleanup(); };
void doSomething() noexcept;

// CHECK: define{{.*}} i32 @_Z1fv(
int f() {
  // CHECK: %[[RetVal:.+]] = alloca i32
  // CHECK-NEXT: %[[GroActive:.+]] = alloca i1
  // CHECK-NEXT: %[[Promise:.+]] = alloca %"struct.std::coroutine_traits<int>::promise_type", align 1
  // CHECK-NEXT: %[[CoroGro:.+]] = alloca %struct.GroType, {{.*}} !coro.outside.frame ![[OutFrameMetadata:.+]]

  // CHECK: %[[Size:.+]] = call i64 @llvm.coro.size.i64()
  // CHECK-NEXT: call noalias noundef nonnull ptr @_Znwm(i64 noundef %[[Size]])

  // CHECK: store i1 false, ptr %[[GroActive]]
  // CHECK-NEXT: call void @llvm.lifetime.start.p0(ptr %[[Promise]])
  // CHECK-NEXT: call void @_ZNSt16coroutine_traitsIiJEE12promise_typeC1Ev(
  // CHECK-NEXT: call void @llvm.lifetime.start.p0(ptr %[[CoroGro]])
  // CHECK-NEXT: call void @_ZNSt16coroutine_traitsIiJEE12promise_type17get_return_objectEv({{.*}} %[[CoroGro]]
  // CHECK-NEXT: store i1 true, ptr %[[GroActive]]

  Cleanup cleanup;
  doSomething();
  co_return;

  // CHECK: call void @_Z11doSomethingv(
  // CHECK-NEXT: call void @_ZNSt16coroutine_traitsIiJEE12promise_type11return_voidEv(
  // CHECK-NEXT: call void @_ZN7CleanupD1Ev(

  // Initialize retval from Gro and destroy Gro
  // Note this also tests delaying initialization when Gro and function return
  // types mismatch (see cwg2563).

  // CHECK: pre.gvo.conv:
  // CHECK-NEXT: %[[IsFinalExit:.+]] = phi i1 [ true, %cleanup8 ], [ false, %final.suspend ], [ false, %init.suspend ]
  // CHECK-NEXT: %InRamp = call i1 @llvm.coro.is_in_ramp()
  // CHECK-NEXT: br i1 %InRamp, label %[[GroConv:.+]], label %[[AfterGroConv:.+]]

  // CHECK: [[GroConv]]:
  // CHECK-NEXT: %[[Conv:.+]] = call noundef i32 @_ZN7GroTypecviEv(
  // CHECK-NEXT: store i32 %[[Conv]], ptr %[[RetVal]]
  // CHECK-NEXT: %[[IsActive:.+]] = load i1, ptr %[[GroActive]]
  // CHECK-NEXT: br i1 %[[IsActive]], label %[[CleanupGro:.+]], label %{{.*}}

  // CHECK: [[CleanupGro]]:
  // CHECK-NEXT: call void @_ZN7GroTypeD1Ev(
  // CHECK-NEXT: br label %cleanup.done

  // CHECK: after.gro.conv:
  // CHECK-NEXT: br i1  %[[IsFinalExit]], label %cleanup.cont10, label %[[CoroRet:.+]]

  // CHECK: cleanup.cont10:
  // CHECK-NEXT: br label %coro.cleanup

  // Destroy promise and free the memory.

  // CHECK: coro.cleanup:
  // CHECK-NEXT: call void @llvm.lifetime.end.p0(ptr %[[CoroGro]])
  // CHECK-NEXT: call void @_ZNSt16coroutine_traitsIiJEE12promise_typeD1Ev(
  // CHECK-NEXT: call void @llvm.lifetime.end.p0(ptr %[[Promise]])
  // CHECK-NEXT: %[[Mem:.+]] = call ptr @llvm.coro.free(

  // CHECK: %[[SIZE:.+]] = call i64 @llvm.coro.size.i64()
  // CHECK-NEXT: call void @_ZdlPvm(ptr noundef %[[Mem]], i64 noundef %[[SIZE]])

  // CHECK: [[CoroRet]]:
  // CHECK-NEXT: call void @llvm.coro.end(
  // CHECK-NEXT: %[[LoadRet:.+]] = load i32, ptr %[[RetVal]]
  // CHECK-NEXT: ret i32 %[[LoadRet]]
}

class invoker {
public:
  class invoker_promise {
  public:
    invoker get_return_object() noexcept { return invoker{}; }
    auto initial_suspend()  noexcept { return suspend_always{}; }
    auto final_suspend() noexcept { return suspend_always{}; }
    void return_void() noexcept {}
    void unhandled_exception() noexcept {}
  };
  using promise_type = invoker_promise;
  invoker() noexcept {}
  ~invoker() {}
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
  // CHECK-NEXT: %[[RESULT_ACTIVE:.+]] = alloca i1

  // CHECK: store ptr %[[AggRes]], ptr %[[ResultPtr]]
  // CHECK: coro.init:
  // CHECK: = call ptr @llvm.coro.begin

  // CHECK: store i1 false, ptr %[[RESULT_ACTIVE]]
  // CHECK-NEXT: call void @_ZN7invoker15invoker_promise17get_return_objectEv({{.*}} %[[AggRes]]
  // CHECK-NEXT: store i1 true, ptr %[[RESULT_ACTIVE]]
  // CHECK-NEXT: call void @llvm.lifetime.start
  // CHECK-NEXT: call void @_ZN7invoker15invoker_promise15initial_suspendEv({{.*}} %[[Promise]]

  throw 0;
  // Test that GRO is destructed in EHCleanup
  // CHECK: call void @llvm.coro.end(ptr null, i1 true, token none)
  // CHECK-NEXT: call i1 @llvm.coro.is_in_ramp()
  // CHECK-NEXT: br i1 {{.*}}, label %[[CLEANUP_CONT:.+]], label %eh.resume

  // CHECK: [[CLEANUP_CONT]]
  // CHECK-NEXT: %[[IS_ACTIVE:.+]] = load i1, ptr %[[RESULT_ACTIVE]]
  // CHECK-NEXT: br i1 %[[IS_ACTIVE]], label %[[CLEANUP_ACTION:.+]], label %{{.*}}

  // CHECK: [[CLEANUP_ACTION]]:
  // CHECK-NEXT: call void @_ZN7invokerD1Ev({{.*}} %[[AggRes]]
  co_return;
}

namespace gh148953 {

struct Task {
  struct promise_type {
    Task get_return_object() noexcept;
    std::suspend_always initial_suspend() noexcept { return {}; }
    std::suspend_always final_suspend() noexcept { return {}; }
    void return_void() noexcept {}
    void unhandled_exception() noexcept {}
  };
  Task() noexcept {}
  // Different from `invoker`, this Task is copy constructible.
  Task(const Task&) noexcept {};
};

// NRVO on const qualified return type should work.
// CHECK: define{{.*}} void @_ZN8gh1489537exampleEv({{.*}} sret(%"struct.gh148953::Task") align 1 %[[NrvoRes:.+]])
const Task example() {
  // CHECK: %[[ResultPtr:.+]] = alloca ptr
  // CHECK: store ptr %[[NrvoRes]], ptr %[[ResultPtr]]
  // CHECK: coro.init:
  // CHECK: call void @_ZN8gh1489534Task12promise_type17get_return_objectEv({{.*}} %[[NrvoRes:.+]], {{.*}})
  co_return;
}

} // namespace gh148953
// CHECK: ![[OutFrameMetadata]] = !{}
