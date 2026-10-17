// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -Wno-coroutine-missing-unhandled-exception -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -emit-llvm -disable-llvm-passes -Wno-coroutine-missing-unhandled-exception %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

namespace std {
template <class Ret, typename... T>
struct coroutine_traits { using promise_type = typename Ret::promise_type; };

template <class Promise = void> struct coroutine_handle {
  coroutine_handle() = default;
  static coroutine_handle from_address(void *) noexcept;
  void *address() const noexcept;
};

template <> struct coroutine_handle<void> {
  coroutine_handle() = default;
  template <class PromiseType>
  coroutine_handle(coroutine_handle<PromiseType>) noexcept;
  static coroutine_handle from_address(void *) noexcept;
  void *address() const noexcept;
};

struct suspend_always {
  bool await_ready() noexcept { return false; }
  void await_suspend(coroutine_handle<>) noexcept {}
  void await_resume() noexcept {}
};

struct suspend_never {
  bool await_ready() noexcept { return true; }
  void await_suspend(coroutine_handle<>) noexcept {}
  void await_resume() noexcept {}
};
} // namespace std

struct Task {
  struct promise_type {
    Task get_return_object() { return {}; }
    std::suspend_never initial_suspend() noexcept { return {}; }
    std::suspend_never final_suspend() noexcept { return {}; }
    void return_void() {}
    void unhandled_exception() {}
  };
};

struct Awaiter {
  bool await_ready() { return false; }
  void await_suspend(std::coroutine_handle<>) {}
  int &await_resume() { return value; }
  int value;
};

// CIR-LABEL: cir.func coroutine {{.*}} @_Z14test_assign_lvv
// OGCG-LABEL: define dso_local void @_Z14test_assign_lvv
Task test_assign_lv() {
  // CIR: %[[VAL:.*]] = cir.const #cir.int<42> : !s32i
  // CIR: cir.await(user, ready : {
  // CIR:   %[[READY:.*]] = cir.call @_ZN7Awaiter11await_readyEv(%{{.*}}) : (!cir.ptr<!rec_Awaiter>{{.*}}) -> (!cir.bool{{.*}})
  // CIR:   cir.condition(%[[READY]])
  // CIR: }, suspend : {
  // CIR:   cir.call @_ZN7Awaiter13await_suspendESt16coroutine_handleIvE(%{{.*}})
  // CIR:   cir.coro.suspend_point
  // CIR: }, resume : {
  // CIR:   %[[RES:.*]] = cir.call @_ZN7Awaiter12await_resumeEv(%{{.*}}) : (!cir.ptr<!rec_Awaiter>{{.*}}) -> (!cir.ptr<!s32i>{{.*}})
  // CIR:   cir.store %[[RES]], %[[TMP_ADDR:.*]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
  // CIR:   cir.yield
  // CIR: },)
  // CIR: %[[LOADED_PTR:.*]] = cir.load %[[TMP_ADDR]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
  // CIR: cir.store align(4) %[[VAL]], %[[LOADED_PTR]] : !s32i, !cir.ptr<!s32i>

  // OGCG: %[[RES_OGCG:.*]] = call noundef nonnull align 4 dereferenceable(4) ptr @_ZN7Awaiter12await_resumeEv(ptr {{.*}})
  // OGCG: store i32 42, ptr %[[RES_OGCG]], align 4
  Awaiter a;
  co_await a = 42;
}

// CIR-LABEL: cir.func coroutine {{.*}} @_Z11test_ref_lvv
// OGCG-LABEL: define dso_local void @_Z11test_ref_lvv
Task test_ref_lv() {
  // CIR: %[[R_ADDR:.*]] = cir.alloca "r" align(8) init const : !cir.ptr<!cir.ptr<!s32i>>
  // CIR: cir.await(user, ready : {
  // CIR:   %[[READY:.*]] = cir.call @_ZN7Awaiter11await_readyEv(%{{.*}}) : (!cir.ptr<!rec_Awaiter>{{.*}}) -> (!cir.bool{{.*}})
  // CIR:   cir.condition(%[[READY]])
  // CIR: }, suspend : {
  // CIR:   cir.call @_ZN7Awaiter13await_suspendESt16coroutine_handleIvE(%{{.*}})
  // CIR:   cir.coro.suspend_point
  // CIR: }, resume : {
  // CIR:   %[[RES:.*]] = cir.call @_ZN7Awaiter12await_resumeEv(%{{.*}}) : (!cir.ptr<!rec_Awaiter>{{.*}}) -> (!cir.ptr<!s32i>{{.*}})
  // CIR:   cir.store %[[RES]], %[[TMP_ADDR:.*]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
  // CIR:   cir.yield
  // CIR: },)
  // CIR: %[[LOADED_PTR:.*]] = cir.load %[[TMP_ADDR]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
  // CIR: cir.store align(8) %[[LOADED_PTR]], %[[R_ADDR]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>

  // OGCG: %[[R_ADDR_OGCG:.*]] = alloca ptr, align 8
  // OGCG: %[[RES_OGCG:.*]] = call noundef nonnull align 4 dereferenceable(4) ptr @_ZN7Awaiter12await_resumeEv(ptr {{.*}})
  // OGCG: store ptr %[[RES_OGCG]], ptr %[[R_ADDR_OGCG]], align 8
  Awaiter a;
  int &r = co_await a;
}
