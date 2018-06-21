// Verifies that parameters are copied with move constructors
// Verifies that parameter copies are destroyed
// Vefifies that parameter copies are used in the body of the coroutine
// RUN: %clang_cc1 -std=c++1z -fcoroutines-ts -triple=x86_64-unknown-linux-gnu -emit-llvm -o - %s -disable-llvm-passes -fexceptions | FileCheck %s

namespace std::experimental {
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
}

struct suspend_always {
  bool await_ready() noexcept;
  void await_suspend(std::experimental::coroutine_handle<>) noexcept;
  void await_resume() noexcept;
};

template <typename... Args> struct std::experimental::coroutine_traits<void, Args...> {
  struct promise_type {
    void get_return_object() noexcept;
    suspend_always initial_suspend() noexcept;
    suspend_always final_suspend() noexcept;
    void return_void() noexcept;
    promise_type();
    ~promise_type() noexcept;
    void unhandled_exception() noexcept;
  };
};

// TODO: Not supported yet
struct CopyOnly {
  int val;
  CopyOnly(const CopyOnly&) noexcept;
  CopyOnly(CopyOnly&&) = delete;
  ~CopyOnly();
};

struct MoveOnly {
  int val;
  MoveOnly(const MoveOnly&) = delete;
  MoveOnly(MoveOnly&&) noexcept;
  ~MoveOnly();
};

struct MoveAndCopy {
  int val;
  MoveAndCopy(const MoveAndCopy&)noexcept;
  MoveAndCopy(MoveAndCopy&&) noexcept;
  ~MoveAndCopy();
};

void consume(int,int,int) noexcept;

// TODO: Add support for CopyOnly params
// CHECK: define void @_Z1fi8MoveOnly11MoveAndCopy(i32 %val, %struct.MoveOnly* %[[MoParam:.+]], %struct.MoveAndCopy* %[[McParam:.+]]) #0 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*
void f(int val, MoveOnly moParam, MoveAndCopy mcParam) {
  // CHECK: %[[MoCopy:.+]] = alloca %struct.MoveOnly
  // CHECK: %[[McCopy:.+]] = alloca %struct.MoveAndCopy
  // CHECK: store i32 %val, i32* %[[ValAddr:.+]]

  // CHECK: call i8* @llvm.coro.begin(
  // CHECK-NEXT: call void @_ZN8MoveOnlyC1EOS_(%struct.MoveOnly* %[[MoCopy]], %struct.MoveOnly* dereferenceable(4) %[[MoParam]])
  // CHECK-NEXT: call void @_ZN11MoveAndCopyC1EOS_(%struct.MoveAndCopy* %[[McCopy]], %struct.MoveAndCopy* dereferenceable(4) %[[McParam]]) #
  // CHECK-NEXT: invoke void @_ZNSt12experimental16coroutine_traitsIJvi8MoveOnly11MoveAndCopyEE12promise_typeC1Ev(

  // CHECK: call void @_ZN14suspend_always12await_resumeEv(
  // CHECK: %[[IntParam:.+]] = load i32, i32* %val.addr
  // CHECK: %[[MoGep:.+]] = getelementptr inbounds %struct.MoveOnly, %struct.MoveOnly* %[[MoCopy]], i32 0, i32 0
  // CHECK: %[[MoVal:.+]] = load i32, i32* %[[MoGep]]
  // CHECK: %[[McGep:.+]] =  getelementptr inbounds %struct.MoveAndCopy, %struct.MoveAndCopy* %[[McCopy]], i32 0, i32 0
  // CHECK: %[[McVal:.+]] = load i32, i32* %[[McGep]]
  // CHECK: call void @_Z7consumeiii(i32 %[[IntParam]], i32 %[[MoVal]], i32 %[[McVal]])

  consume(val, moParam.val, mcParam.val);
  co_return;

  // Skip to final suspend:
  // CHECK: call void @_ZNSt12experimental16coroutine_traitsIJvi8MoveOnly11MoveAndCopyEE12promise_type13final_suspendEv(
  // CHECK: call void @_ZN14suspend_always12await_resumeEv(

  // Destroy promise, then parameter copies:
  // CHECK: call void @_ZNSt12experimental16coroutine_traitsIJvi8MoveOnly11MoveAndCopyEE12promise_typeD1Ev(%"struct.std::experimental::coroutine_traits<void, int, MoveOnly, MoveAndCopy>::promise_type"* %__promise) #2
  // CHECK-NEXT: call void @_ZN11MoveAndCopyD1Ev(%struct.MoveAndCopy* %[[McCopy]])
  // CHECK-NEXT: call void @_ZN8MoveOnlyD1Ev(%struct.MoveOnly* %[[MoCopy]]
  // CHECK-NEXT: call i8* @llvm.coro.free(
}

// CHECK-LABEL: void @_Z16dependent_paramsI1A1BEvT_T0_S3_(%struct.A* %x, %struct.B*, %struct.B* %y)
template <typename T, typename U>
void dependent_params(T x, U, U y) {
  // CHECK: %[[x_copy:.+]] = alloca %struct.A
  // CHECK-NEXT: %[[unnamed_copy:.+]] = alloca %struct.B
  // CHECK-NEXT: %[[y_copy:.+]] = alloca %struct.B

  // CHECK: call i8* @llvm.coro.begin
  // CHECK-NEXT: call void @_ZN1AC1EOS_(%struct.A* %[[x_copy]], %struct.A* dereferenceable(512) %x)
  // CHECK-NEXT: call void @_ZN1BC1EOS_(%struct.B* %[[unnamed_copy]], %struct.B* dereferenceable(512) %0)
  // CHECK-NEXT: call void @_ZN1BC1EOS_(%struct.B* %[[y_copy]], %struct.B* dereferenceable(512) %y)
  // CHECK-NEXT: invoke void @_ZNSt12experimental16coroutine_traitsIJv1A1BS2_EE12promise_typeC1Ev(

  co_return;
}

struct A {
  int WontFitIntoRegisterForSure[128];
  A();
  A(A&&) noexcept;
  ~A();
};

struct B {
  int WontFitIntoRegisterForSure[128];
  B();
  B(B&&) noexcept;
  ~B();
};

void call_dependent_params() {
  dependent_params(A{}, B{}, B{});
}
