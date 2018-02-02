// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fcoroutines-ts -std=c++14 \
// RUN:    -Wno-coroutine-missing-unhandled-exception -emit-llvm %s -o - -disable-llvm-passes \
// RUN:   | FileCheck %s

namespace std {
namespace experimental {
template <typename... T>
struct coroutine_traits; // expected-note {{declared here}}

template <class Promise = void>
struct coroutine_handle {
  coroutine_handle() = default;
  static coroutine_handle from_address(void *) { return {}; }
};

template <>
struct coroutine_handle<void> {
  static coroutine_handle from_address(void *) { return {}; }
  coroutine_handle() = default;
  template <class PromiseType>
  coroutine_handle(coroutine_handle<PromiseType>) {}
};

} // end namespace experimental

struct nothrow_t {};
constexpr nothrow_t nothrow = {};

} // end namespace std

// Required when get_return_object_on_allocation_failure() is defined by
// the promise.
using SizeT = decltype(sizeof(int));
void* operator new(SizeT __sz, const std::nothrow_t&) noexcept;
void  operator delete(void* __p, const std::nothrow_t&) noexcept;


struct suspend_always {
  bool await_ready() { return false; }
  void await_suspend(std::experimental::coroutine_handle<>) {}
  void await_resume() {}
};

struct global_new_delete_tag {};

template<>
struct std::experimental::coroutine_traits<void, global_new_delete_tag> {
  struct promise_type {
    void get_return_object() {}
    suspend_always initial_suspend() { return {}; }
    suspend_always final_suspend() { return {}; }
    void return_void() {}
  };
};

// CHECK-LABEL: f0(
extern "C" void f0(global_new_delete_tag) {
  // CHECK: %[[ID:.+]] = call token @llvm.coro.id(i32 16
  // CHECK: %[[NeedAlloc:.+]] = call i1 @llvm.coro.alloc(token %[[ID]])
  // CHECK: br i1 %[[NeedAlloc]], label %[[AllocBB:.+]], label %[[InitBB:.+]]

  // CHECK: [[AllocBB]]:
  // CHECK: %[[SIZE:.+]] = call i64 @llvm.coro.size.i64()
  // CHECK: %[[MEM:.+]] = call i8* @_Znwm(i64 %[[SIZE]])
  // CHECK: br label %[[InitBB]]

  // CHECK: [[InitBB]]:
  // CHECK: %[[PHI:.+]] = phi i8* [ null, %{{.+}} ], [ %call, %[[AllocBB]] ]
  // CHECK: %[[FRAME:.+]] = call i8* @llvm.coro.begin(token %[[ID]], i8* %[[PHI]])

  // CHECK: %[[MEM:.+]] = call i8* @llvm.coro.free(token %[[ID]], i8* %[[FRAME]])
  // CHECK: %[[NeedDealloc:.+]] = icmp ne i8* %[[MEM]], null
  // CHECK: br i1 %[[NeedDealloc]], label %[[FreeBB:.+]], label %[[Afterwards:.+]]

  // CHECK: [[FreeBB]]:
  // CHECK: call void @_ZdlPv(i8* %[[MEM]])
  // CHECK: br label %[[Afterwards]]

  // CHECK: [[Afterwards]]:
  // CHECK: ret void
  co_return;
}

struct promise_new_tag {};

template<>
struct std::experimental::coroutine_traits<void, promise_new_tag> {
  struct promise_type {
    void *operator new(unsigned long);
    void get_return_object() {}
    suspend_always initial_suspend() { return {}; }
    suspend_always final_suspend() { return {}; }
    void return_void() {}
  };
};

// CHECK-LABEL: f1(
extern "C" void f1(promise_new_tag ) {
  // CHECK: %[[ID:.+]] = call token @llvm.coro.id(i32 16
  // CHECK: %[[SIZE:.+]] = call i64 @llvm.coro.size.i64()
  // CHECK: call i8* @_ZNSt12experimental16coroutine_traitsIJv15promise_new_tagEE12promise_typenwEm(i64 %[[SIZE]])

  // CHECK: %[[FRAME:.+]] = call i8* @llvm.coro.begin(
  // CHECK: %[[MEM:.+]] = call i8* @llvm.coro.free(token %[[ID]], i8* %[[FRAME]])
  // CHECK: call void @_ZdlPv(i8* %[[MEM]])
  co_return;
}

struct promise_delete_tag {};

template<>
struct std::experimental::coroutine_traits<void, promise_delete_tag> {
  struct promise_type {
    void operator delete(void*);
    void get_return_object() {}
    suspend_always initial_suspend() { return {}; }
    suspend_always final_suspend() { return {}; }
    void return_void() {}
  };
};

// CHECK-LABEL: f2(
extern "C" void f2(promise_delete_tag) {
  // CHECK: %[[ID:.+]] = call token @llvm.coro.id(i32 16
  // CHECK: %[[SIZE:.+]] = call i64 @llvm.coro.size.i64()
  // CHECK: call i8* @_Znwm(i64 %[[SIZE]])

  // CHECK: %[[FRAME:.+]] = call i8* @llvm.coro.begin(
  // CHECK: %[[MEM:.+]] = call i8* @llvm.coro.free(token %[[ID]], i8* %[[FRAME]])
  // CHECK: call void @_ZNSt12experimental16coroutine_traitsIJv18promise_delete_tagEE12promise_typedlEPv(i8* %[[MEM]])
  co_return;
}

struct promise_sized_delete_tag {};

template<>
struct std::experimental::coroutine_traits<void, promise_sized_delete_tag> {
  struct promise_type {
    void operator delete(void*, unsigned long);
    void get_return_object() {}
    suspend_always initial_suspend() { return {}; }
    suspend_always final_suspend() { return {}; }
    void return_void() {}
  };
};

// CHECK-LABEL: f3(
extern "C" void f3(promise_sized_delete_tag) {
  // CHECK: %[[ID:.+]] = call token @llvm.coro.id(i32 16
  // CHECK: %[[SIZE:.+]] = call i64 @llvm.coro.size.i64()
  // CHECK: call i8* @_Znwm(i64 %[[SIZE]])

  // CHECK: %[[FRAME:.+]] = call i8* @llvm.coro.begin(
  // CHECK: %[[MEM:.+]] = call i8* @llvm.coro.free(token %[[ID]], i8* %[[FRAME]])
  // CHECK: %[[SIZE2:.+]] = call i64 @llvm.coro.size.i64()
  // CHECK: call void @_ZNSt12experimental16coroutine_traitsIJv24promise_sized_delete_tagEE12promise_typedlEPvm(i8* %[[MEM]], i64 %[[SIZE2]])
  co_return;
}

struct promise_on_alloc_failure_tag {};

template<>
struct std::experimental::coroutine_traits<int, promise_on_alloc_failure_tag> {
  struct promise_type {
    int get_return_object() { return 0; }
    suspend_always initial_suspend() { return {}; }
    suspend_always final_suspend() { return {}; }
    void return_void() {}
    static int get_return_object_on_allocation_failure() { return -1; }
  };
};

// CHECK-LABEL: f4(
extern "C" int f4(promise_on_alloc_failure_tag) {
  // CHECK: %[[RetVal:.+]] = alloca i32
  // CHECK: %[[Gro:.+]] = alloca i32
  // CHECK: %[[ID:.+]] = call token @llvm.coro.id(i32 16
  // CHECK: %[[SIZE:.+]] = call i64 @llvm.coro.size.i64()
  // CHECK: %[[MEM:.+]] = call i8* @_ZnwmRKSt9nothrow_t(i64 %[[SIZE]], %"struct.std::nothrow_t"* dereferenceable(1) @_ZStL7nothrow)
  // CHECK: %[[OK:.+]] = icmp ne i8* %[[MEM]], null
  // CHECK: br i1 %[[OK]], label %[[OKBB:.+]], label %[[ERRBB:.+]]

  // CHECK: [[ERRBB]]:
  // CHECK:   %[[FailRet:.+]] = call i32 @_ZNSt12experimental16coroutine_traitsIJi28promise_on_alloc_failure_tagEE12promise_type39get_return_object_on_allocation_failureEv(
  // CHECK:   store i32 %[[FailRet]], i32* %[[RetVal]]
  // CHECK:   br label %[[RetBB:.+]]

  // CHECK: [[OKBB]]:
  // CHECK:   %[[OkRet:.+]] = call i32 @_ZNSt12experimental16coroutine_traitsIJi28promise_on_alloc_failure_tagEE12promise_type17get_return_objectEv(
  // CHECK:   store i32 %[[OkRet]], i32* %[[Gro]]

  // CHECK: %[[Tmp1:.*]] = load i32, i32* %[[Gro]]
  // CHECK-NEXT: store i32 %[[Tmp1]], i32* %[[RetVal]]
  // CHECK-NEXT: br label %[[RetBB]]

  // CHECK: [[RetBB]]:
  // CHECK:   %[[LoadRet:.+]] = load i32, i32* %[[RetVal]], align 4
  // CHECK:   ret i32 %[[LoadRet]]
  co_return;
}
