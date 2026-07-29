// RUN: %clang_cc1 -std=c++20 -triple=i686-pc-windows-msvc -emit-llvm -o - %s -disable-llvm-passes | FileCheck %s

namespace std {
template <typename R, typename... Args>
struct coroutine_traits {
  using promise_type = typename R::promise_type;
};

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
  bool await_ready() noexcept { return false; }
  void await_suspend(std::coroutine_handle<>) noexcept {}
  void await_resume() noexcept {}
};

struct suspend_never {
  bool await_ready() noexcept { return true; }
  void await_suspend(std::coroutine_handle<>) noexcept {}
  void await_resume() noexcept {}
};

struct task {
  struct promise_type {
    task get_return_object() { return {}; }
    suspend_never initial_suspend() { return {}; }
    suspend_never final_suspend() noexcept { return {}; }
    void return_void() {}
    void unhandled_exception() {}
  };
};

struct Noisy {
  int val;
  Noisy(int v);
  Noisy(const Noisy&) = delete;
  Noisy(Noisy&& o) noexcept;
  ~Noisy();
};

struct Awaiter {
  bool await_ready() noexcept { return false; }
  void await_suspend(std::coroutine_handle<>) noexcept {}
  Noisy await_resume() noexcept;
};

void consume_two(Noisy x, Noisy y);

// CHECK-LABEL: define dso_local void @"?my_task@@YA?AUtask@@XZ"(
task my_task() {
  // CHECK: %[[MTE_Y:.+]] = alloca %struct.Noisy,
  // CHECK: %[[MTE_X:.+]] = alloca %struct.Noisy,
  
  // Evaluate Noisy(42) before suspend:
  // CHECK: call x86_thiscallcc noundef ptr @"??0Noisy@@QAE@H@Z"(ptr {{[^,]*}} %[[MTE_Y]], i32 noundef 42)
  
  // Suspend for co_await Awaiter{}:
  // CHECK: call void @llvm.coro.await.suspend.void(
  // CHECK: call i8 @llvm.coro.suspend(

  // After resume:
  // CHECK: call x86_thiscallcc void @"?await_resume@Awaiter@@QAE?AUNoisy@@XZ"(ptr {{[^,]*}} %{{.*}}, ptr dead_on_unwind writable sret(%struct.Noisy) align 4 %[[MTE_X]])
  
  // Allocate inalloca:
  // CHECK: %[[STACKSAVE:.+]] = call ptr @llvm.stacksave.p0()
  // CHECK: %[[ARGMEM:.+]] = alloca inalloca <{ %struct.Noisy, %struct.Noisy }>, align 4, !coro.outside.frame ![[METADATA_NUM:[0-9]+]]
  
  // Move y (pre-evaluated Noisy(42)) to inalloca:
  // CHECK: %[[GEP_Y:.+]] = getelementptr inbounds nuw <{ %struct.Noisy, %struct.Noisy }>, ptr %[[ARGMEM]], i32 0, i32 1
  // CHECK: call x86_thiscallcc noundef ptr @"??0Noisy@@QAE@$$QAU0@@Z"(ptr {{[^,]*}} %[[GEP_Y]], ptr noundef nonnull align 4 dereferenceable(4) %[[MTE_Y]])
  
  // Move x (co_await Awaiter{} result) to inalloca:
  // CHECK: %[[GEP_X:.+]] = getelementptr inbounds nuw <{ %struct.Noisy, %struct.Noisy }>, ptr %[[ARGMEM]], i32 0, i32 0
  // CHECK: call x86_thiscallcc noundef ptr @"??0Noisy@@QAE@$$QAU0@@Z"(ptr {{[^,]*}} %[[GEP_X]], ptr noundef nonnull align 4 dereferenceable(4) %[[MTE_X]])

  // Lifetime start and call:
  // CHECK: call void @llvm.lifetime.start.p0(ptr %[[ARGMEM]])
  // CHECK: call void @"?consume_two@@YAXUNoisy@@0@Z"(ptr inalloca(<{ %struct.Noisy, %struct.Noisy }>) %[[ARGMEM]])
  
  consume_two(co_await Awaiter{}, Noisy(42));
}

// CHECK: ![[METADATA_NUM]] = !{}
