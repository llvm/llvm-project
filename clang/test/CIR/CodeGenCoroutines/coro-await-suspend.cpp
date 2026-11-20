// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu \
// RUN:   -fclangir -Wno-coroutine-missing-unhandled-exception \
// RUN:   -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: cir-opt --cir-flatten-cfg %t.cir -o %t.flat.cir
// RUN: FileCheck --input-file=%t.flat.cir %s --check-prefix=CIR-FLAT \
// RUN:   --implicit-check-not=cir.coroutine --implicit-check-not=cir.await \
// RUN:   --implicit-check-not=cir.co_return --implicit-check-not=cir.coro.exit \
// RUN:   --implicit-check-not=cir.coro.suspend_point
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu \
// RUN:   -fclangir -emit-llvm -disable-llvm-passes \
// RUN:   -no-enable-noundef-analysis \
// RUN:   -Wno-coroutine-missing-unhandled-exception %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=LLVM

#include "Inputs/coroutine.h"

struct Task {
  struct promise_type {
    Task get_return_object() { return {}; }
    std::suspend_never initial_suspend() noexcept { return {}; }
    std::suspend_never final_suspend() noexcept { return {}; }
    void return_void() noexcept {}
    void unhandled_exception() {}
  };
};

struct HandleAwaiter {
  bool await_ready() noexcept;
  std::coroutine_handle<>
  await_suspend(std::coroutine_handle<>) noexcept;
  void await_resume() noexcept;
};

struct VoidAwaiter {
  bool await_ready() noexcept;
  void await_suspend(std::coroutine_handle<>) noexcept;
  void await_resume() noexcept;
};

struct BoolAwaiter {
  bool await_ready() noexcept;
  bool await_suspend(std::coroutine_handle<>) noexcept;
  void await_resume() noexcept;
};

// CIR-LABEL: cir.func coroutine {{.*}} @_Z13await_suspendv

// CIR: cir.call @_ZN13HandleAwaiter13await_suspendESt16coroutine_handleIvE
// CIR: %[[HANDLE_RET:[0-9]+]] = cir.call @_ZNKSt16coroutine_handleIvE7addressEv
// CIR-NEXT: cir.coro.suspend_point(%[[HANDLE_RET]] : !cir.ptr<!void>)

// CIR: cir.call @_ZN11VoidAwaiter13await_suspendESt16coroutine_handleIvE
// CIR-NEXT: cir.coro.suspend_point

// CIR: %[[SUSPEND_RET:[0-9]+]] = cir.call @_ZN11BoolAwaiter13await_suspendESt16coroutine_handleIvE
// CIR-NEXT: cir.coro.suspend_point(%[[SUSPEND_RET]] : !cir.bool)

// CIR-NOT: cir.func {{.*}}__await_suspend_wrapper__

// CIR-FLAT-LABEL: cir.func flattened_coroutine {{.*}} @_Z13await_suspendv

// CIR-FLAT: %[[FLAT_HANDLE_WRAPPER:[0-9]+]] = cir.get_global @_Z13await_suspendv.__await_suspend_wrapper__await :
// CIR-FLAT-NEXT: cir.coro.intrinsic.await.suspend.handle({{.*}}, {{.*}}, %[[FLAT_HANDLE_WRAPPER]])

// CIR-FLAT: %[[FLAT_VOID_WRAPPER:[0-9]+]] = cir.get_global @_Z13await_suspendv.__await_suspend_wrapper__await2 :
// CIR-FLAT-NEXT: cir.coro.intrinsic.await.suspend.void({{.*}}, {{.*}}, %[[FLAT_VOID_WRAPPER]]){{.*}}nothrow

// CIR-FLAT: %[[FLAT_BOOL_WRAPPER:[0-9]+]] = cir.get_global @_Z13await_suspendv.__await_suspend_wrapper__await3 :
// CIR-FLAT-NEXT: %[[FLAT_SUSPEND_RET:[0-9]+]] = cir.coro.intrinsic.await.suspend.bool({{.*}}, {{.*}}, %[[FLAT_BOOL_WRAPPER]]){{.*}}nothrow
// CIR-FLAT-NEXT: cir.brcond %[[FLAT_SUSPEND_RET]]

// LLVM-LABEL: define {{.*}} @_Z13await_suspendv(
// LLVM: call void @llvm.coro.await.suspend.handle(ptr %{{[0-9]+}}, ptr %[[LLVM_FRAME:[0-9]+]], ptr @_Z13await_suspendv.__await_suspend_wrapper__await)
// LLVM: call void @llvm.coro.await.suspend.void(ptr %{{[0-9]+}}, ptr %[[LLVM_FRAME]], ptr @_Z13await_suspendv.__await_suspend_wrapper__await2)
// LLVM: %[[LLVM_SUSPEND_RET:[0-9]+]] = call i1 @llvm.coro.await.suspend.bool(ptr %{{[0-9]+}}, ptr %[[LLVM_FRAME]], ptr @_Z13await_suspendv.__await_suspend_wrapper__await3)
// LLVM-NEXT: br i1 %[[LLVM_SUSPEND_RET]]

Task await_suspend() {
  co_await HandleAwaiter{};
  co_await VoidAwaiter{};
  co_await BoolAwaiter{};
}

// CIR-FLAT-LABEL: cir.func flattened_coroutine {{.*}} @_Z12target_awaitv
// CIR-FLAT-SAME: "cir.target-features" = "{{.*}}+sse4.2{{.*}}"

// LLVM-LABEL: define {{.*}} @_Z12target_awaitv()
// LLVM-SAME: #[[TARGET_FUNC_ATTRS:[0-9]+]] {

__attribute__((target("sse4.2"))) Task target_await() {
  co_await VoidAwaiter{};
}

// CIR-FLAT-LABEL: cir.func always_inline internal private @_Z13await_suspendv.__await_suspend_wrapper__await(
// CIR-FLAT-SAME: %{{.*}}: !cir.ptr<!void> {llvm.nonnull, llvm.noundef}
// CIR-FLAT-SAME: %{{.*}}: !cir.ptr<!void> {llvm.noundef}
// CIR-FLAT-SAME: -> (!cir.ptr<!void> {llvm.noundef})
// CIR-FLAT-SAME: attributes {{.*}}mustprogress
// CIR-FLAT: cir.call @_ZN13HandleAwaiter13await_suspendESt16coroutine_handleIvE
// CIR-FLAT: cir.call @_ZNKSt16coroutine_handleIvE7addressEv
// CIR-FLAT: cir.return

// CIR-FLAT-LABEL: cir.func always_inline internal private @_Z13await_suspendv.__await_suspend_wrapper__await2(
// CIR-FLAT-SAME: %{{.*}}: !cir.ptr<!void> {llvm.nonnull, llvm.noundef}
// CIR-FLAT-SAME: %{{.*}}: !cir.ptr<!void> {llvm.noundef}
// CIR-FLAT-SAME: attributes {{.*}}mustprogress
// CIR-FLAT: cir.call @_ZN11VoidAwaiter13await_suspendESt16coroutine_handleIvE
// CIR-FLAT: cir.return

// CIR-FLAT-LABEL: cir.func always_inline internal private @_Z13await_suspendv.__await_suspend_wrapper__await3(
// CIR-FLAT-SAME: %{{.*}}: !cir.ptr<!void> {llvm.nonnull, llvm.noundef}
// CIR-FLAT-SAME: %{{.*}}: !cir.ptr<!void> {llvm.noundef}
// CIR-FLAT-SAME: -> (!cir.bool {llvm.noundef, llvm.zeroext})
// CIR-FLAT-SAME: attributes {{.*}}mustprogress
// CIR-FLAT: cir.call @_ZN11BoolAwaiter13await_suspendESt16coroutine_handleIvE
// CIR-FLAT: cir.return

// LLVM-LABEL: define internal noundef ptr @_Z13await_suspendv.__await_suspend_wrapper__await(
// LLVM-SAME: ptr noundef nonnull %{{.*}}, ptr noundef %{{.*}}) #[[WRAPPER_ATTRS:[0-9]+]] {
// LLVM-LABEL: define internal void @_Z13await_suspendv.__await_suspend_wrapper__await2(
// LLVM-SAME: ptr noundef nonnull %{{.*}}, ptr noundef %{{.*}}) #[[WRAPPER_ATTRS]] {
// LLVM-LABEL: define internal noundef zeroext i1 @_Z13await_suspendv.__await_suspend_wrapper__await3(
// LLVM-SAME: ptr noundef nonnull %{{.*}}, ptr noundef %{{.*}}) #[[WRAPPER_ATTRS]] {

// CIR-FLAT-LABEL: cir.func always_inline internal private @_Z12target_awaitv.__await_suspend_wrapper__await(
// CIR-FLAT-SAME: "cir.target-features" = "{{.*}}+sse4.2{{.*}}"

// LLVM-LABEL: define internal void @_Z12target_awaitv.__await_suspend_wrapper__await(
// LLVM-SAME: ptr noundef nonnull %{{.*}}, ptr noundef %{{.*}}) #[[TARGET_WRAPPER_ATTRS:[0-9]+]] {

// LLVM-DAG: attributes #[[WRAPPER_ATTRS]] = { alwaysinline mustprogress
// LLVM-DAG: attributes #[[TARGET_FUNC_ATTRS]] = { {{.*}}"target-features"="{{.*}}+sse4.2{{.*}}"
// LLVM-DAG: attributes #[[TARGET_WRAPPER_ATTRS]] = { alwaysinline mustprogress {{.*}}"target-features"="{{.*}}+sse4.2{{.*}}"
