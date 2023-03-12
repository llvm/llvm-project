// Verifies lifetime of __gro local variable
// Verify that coroutine promise and allocated memory are freed up on exception.
// RUN: %clang_cc1 -std=c++20 -triple=x86_64-unknown-linux-gnu -emit-llvm -o - %s -disable-llvm-passes | FileCheck %s

namespace std {
template <typename... T> struct coroutine_traits;

template <class Promise = void> struct coroutine_handle {
  coroutine_handle() = default;
  static coroutine_handle from_address(void *) noexcept;
};
template <> struct coroutine_handle<void> {
  static coroutine_handle from_address(void *) noexcept;
  coroutine_handle() = default;
  template <class PromiseType>
  coroutine_handle(coroutine_handle<PromiseType>) noexcept;
};
} // namespace std

struct suspend_always {
  bool await_ready() noexcept;
  void await_suspend(std::coroutine_handle<>) noexcept;
  void await_resume() noexcept;
};

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
  // CHECK: call void @_ZNSt16coroutine_traitsIJiEE12promise_typeC1Ev(
  // CHECK: call void @_ZNSt16coroutine_traitsIJiEE12promise_type17get_return_objectEv(
  // CHECK: store i1 true, ptr %[[GroActive]]

  Cleanup cleanup;
  doSomething();
  co_return;

  // CHECK: call void @_Z11doSomethingv(
  // CHECK: call void @_ZNSt16coroutine_traitsIJiEE12promise_type11return_voidEv(
  // CHECK: call void @_ZN7CleanupD1Ev(

  // Destroy promise and free the memory.

  // CHECK: call void @_ZNSt16coroutine_traitsIJiEE12promise_typeD1Ev(
  // CHECK: %[[Mem:.+]] = call ptr @llvm.coro.free(
  // CHECK: call void @_ZdlPv(ptr noundef %[[Mem]])

  // Initialize retval from Gro and destroy Gro

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
