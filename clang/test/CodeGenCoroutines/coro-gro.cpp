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

  // CHECK: %[[Size:.+]] = call i64 @llvm.coro.size.i64()
  // CHECK: call noalias noundef nonnull i8* @_Znwm(i64 noundef %[[Size]])
  // CHECK: call void @_ZNSt16coroutine_traitsIJiEE12promise_typeC1Ev(
  // CHECK: call void @_ZNSt16coroutine_traitsIJiEE12promise_type17get_return_objectEv(%struct.GroType* sret(%struct.GroType) align {{[0-9]+}} %[[GRO:.+]],
  // CHECK: %[[Conv:.+]] = call noundef i32 @_ZN7GroTypecviEv({{.*}}[[GRO]]
  // CHECK: store i32 %[[Conv]], i32* %[[RetVal]]

  Cleanup cleanup;
  doSomething();
  co_return;

  // CHECK: call void @_Z11doSomethingv(
  // CHECK: call void @_ZNSt16coroutine_traitsIJiEE12promise_type11return_voidEv(
  // CHECK: call void @_ZN7CleanupD1Ev(

  // Destroy promise and free the memory.

  // CHECK: call void @_ZNSt16coroutine_traitsIJiEE12promise_typeD1Ev(
  // CHECK: %[[Mem:.+]] = call i8* @llvm.coro.free(
  // CHECK: call void @_ZdlPv(i8* noundef %[[Mem]])

  // CHECK: coro.ret:
  // CHECK:   %[[LoadRet:.+]] = load i32, i32* %[[RetVal]]
  // CHECK:   ret i32 %[[LoadRet]]
}
