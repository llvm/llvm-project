// RUN: %clang_cc1 -triple spirv64-unknown-unknown -fsycl-is-device -std=c++20
// -O2 \
// RUN:   -Wno-coroutine-missing-unhandled-exception -emit-llvm %s -o -
// -disable-llvm-passes \ RUN:   | FileCheck %s --check-prefix=SPIRV
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -std=c++20 -O2 \
// RUN:   -Wno-coroutine-missing-unhandled-exception -emit-llvm %s -o -
// -disable-llvm-passes \ RUN:   | FileCheck %s --check-prefix=AMDGCN

namespace std {
template <typename... T> struct coroutine_traits;

template <class Promise = void> struct coroutine_handle {
  coroutine_handle() = default;
  static coroutine_handle from_address(void *) noexcept { return {}; }
};

template <> struct coroutine_handle<void> {
  static coroutine_handle from_address(void *) { return {}; }
  coroutine_handle() = default;
  template <class PromiseType>
  coroutine_handle(coroutine_handle<PromiseType>) noexcept {}
};
} // namespace std

struct suspend_always {
  bool await_ready() noexcept { return false; }
  void await_suspend(std::coroutine_handle<>) noexcept {}
  void await_resume() noexcept {}
};

template <> struct std::coroutine_traits<void> {
  struct promise_type {
    void get_return_object() {}
    suspend_always initial_suspend() { return {}; }
    suspend_always final_suspend() noexcept { return {}; }
    void return_void() {}
    static void *operator new(decltype(sizeof(int)) sz) noexcept;
    static void operator delete(void *p) noexcept;
  };
};

// SPIRV-LABEL: define spir_func void @_Z1fv(
// SPIRV: %[[ID:.+]] = call token @llvm.coro.id.p0(i32 8, ptr %__promise, ptr
// null, ptr null) SPIRV: %[[PHI:.+]] = phi ptr addrspace(4) [ null, %entry ], [
// %{{.+}}, %coro.alloc ] SPIRV: %[[FRAME:.+]] = call ptr addrspace(4)
// @llvm.coro.begin.p4(token %[[ID]], ptr addrspace(4) %[[PHI]]) SPIRV: call
// token @llvm.coro.save.p4(ptr addrspace(4) null) SPIRV: call spir_func void
// @llvm.coro.await.suspend.void.p4.p4(ptr addrspace(4) %{{.+}}, ptr
// addrspace(4) %[[FRAME]], ptr @_Z1fv.__await_suspend_wrapper__init) SPIRV:
// %[[MEM:.+]] = call ptr addrspace(4) @llvm.coro.free.p4(token %[[ID]], ptr
// addrspace(4) %[[FRAME]]) SPIRV: call void @llvm.coro.dead.p4(ptr addrspace(4)
// %[[FRAME]]) SPIRV: call void @llvm.coro.end.p4(ptr addrspace(4) null, i1
// false, token none)

// AMDGCN-LABEL: define dso_local void @_Z1fv(
// AMDGCN: %[[ID:.+]] = call token @llvm.coro.id.p5(i32 8, ptr addrspace(5)
// %__promise, ptr null, ptr null) AMDGCN: %[[PHI:.+]] = phi ptr [ null, %entry
// ], [ %{{.+}}, %coro.alloc ] AMDGCN: %[[FRAME:.+]] = call ptr
// @llvm.coro.begin.p0(token %[[ID]], ptr %[[PHI]]) AMDGCN: call token
// @llvm.coro.save.p0(ptr null) AMDGCN: %[[MEM:.+]] = call ptr
// @llvm.coro.free.p0(token %[[ID]], ptr %[[FRAME]]) AMDGCN: call void
// @llvm.coro.dead.p0(ptr %[[FRAME]]) AMDGCN: call void @llvm.coro.end.p0(ptr
// null, i1 false, token none)
#ifdef __SYCL_DEVICE_ONLY__
[[clang::sycl_external]]
#endif
void f() {
  co_return;
}
