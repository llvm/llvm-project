// Verifies that parameters are copied with move constructors
// Verifies that parameter copies are destroyed
// Vefifies that parameter copies are used in the body of the coroutine
// Verifies that parameter copies are used to construct the promise type, if that type has a matching constructor
// RUN: %clang_cc1 -std=c++20 -triple=x86_64-unknown-linux-gnu -emit-llvm -o - %s -disable-llvm-passes -fexceptions | FileCheck %s
// RUN: %clang_cc1 -std=c++20 -triple=x86_64-pc-win32          -emit-llvm -o - %s -disable-llvm-passes -fexceptions | FileCheck %s --check-prefix=MSABI

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

template <typename... Args> struct std::coroutine_traits<void, Args...> {
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

struct [[clang::trivial_abi]] TrivialABI {
  int val;
  TrivialABI(TrivialABI&&) noexcept;
  ~TrivialABI();
};

void consume(int,int,int,int) noexcept;

// TODO: Add support for CopyOnly params
// CHECK: define{{.*}} void @_Z1fi8MoveOnly11MoveAndCopy10TrivialABI(i32 noundef %val, ptr noundef %[[MoParam:.+]], ptr noundef %[[McParam:.+]], i32 %[[TrivialParam:.+]]) #0 personality ptr @__gxx_personality_v0
void f(int val, MoveOnly moParam, MoveAndCopy mcParam, TrivialABI trivialParam) {
  // CHECK: %[[TrivialAlloca:.+]] = alloca %struct.TrivialABI,
  // CHECK-SAME: !coro.outside.frame
  // CHECK: %[[MoCopy:.+]] = alloca %struct.MoveOnly,
  // CHECK: %[[McCopy:.+]] = alloca %struct.MoveAndCopy,
  // CHECK: %[[TrivialCopy:.+]] = alloca %struct.TrivialABI,
  // CHECK: store i32 %val, ptr %[[ValAddr:.+]]

  // CHECK: call ptr @llvm.coro.begin(
  // CHECK: call void @_ZN8MoveOnlyC1EOS_(ptr {{[^,]*}} %[[MoCopy]], ptr noundef nonnull align 4 dereferenceable(4) %[[MoParam]])
  // CHECK-NEXT: call void @llvm.lifetime.start.p0(
  // CHECK-NEXT: call void @_ZN11MoveAndCopyC1EOS_(ptr {{[^,]*}} %[[McCopy]], ptr noundef nonnull align 4 dereferenceable(4) %[[McParam]]) #
  // CHECK-NEXT: call void @llvm.lifetime.start.p0(
  // CHECK-NEXT: call void @_ZN10TrivialABIC1EOS_(ptr {{[^,]*}} %[[TrivialCopy]], ptr {{[^,]*}} %[[TrivialAlloca]])
  // CHECK-NEXT: call void @llvm.lifetime.start.p0(
  // CHECK-NEXT: invoke void @_ZNSt16coroutine_traitsIJvi8MoveOnly11MoveAndCopy10TrivialABIEE12promise_typeC1Ev(

  // CHECK: call void @_ZN14suspend_always12await_resumeEv(
  // CHECK: %[[IntParam:.+]] = load i32, ptr %{{.*}}
  // CHECK: %[[MoGep:.+]] = getelementptr inbounds nuw %struct.MoveOnly, ptr %[[MoCopy]], i32 0, i32 0
  // CHECK: %[[MoVal:.+]] = load i32, ptr %[[MoGep]]
  // CHECK: %[[McGep:.+]] = getelementptr inbounds nuw %struct.MoveAndCopy, ptr %[[McCopy]], i32 0, i32 0
  // CHECK: %[[McVal:.+]] = load i32, ptr %[[McGep]]
  // CHECK: %[[TrivialGep:.+]] = getelementptr inbounds nuw %struct.TrivialABI, ptr %[[TrivialCopy]], i32 0, i32 0
  // CHECK: %[[TrivialVal:.+]] = load i32, ptr %[[TrivialGep]]
  // CHECK: call void @_Z7consumeiiii(i32 noundef %[[IntParam]], i32 noundef %[[MoVal]], i32 noundef %[[McVal]], i32 noundef %[[TrivialVal]])

  consume(val, moParam.val, mcParam.val, trivialParam.val);
  co_return;

  // Skip to final suspend:
  // CHECK: call void @_ZNSt16coroutine_traitsIJvi8MoveOnly11MoveAndCopy10TrivialABIEE12promise_type13final_suspendEv(
  // CHECK: call void @_ZN14suspend_always12await_resumeEv(

  // Destroy promise, then parameter copies:
  // CHECK: call void @_ZNSt16coroutine_traitsIJvi8MoveOnly11MoveAndCopy10TrivialABIEE12promise_typeD1Ev(ptr {{[^,]*}} %__promise)
  // CHECK-NEXT: call void @llvm.lifetime.end.p0(
  // CHECK-NEXT: call void @_ZN10TrivialABID1Ev(ptr {{[^,]*}} %[[TrivialCopy]])
  // CHECK-NEXT: call void @llvm.lifetime.end.p0(
  // CHECK-NEXT: call void @_ZN11MoveAndCopyD1Ev(ptr {{[^,]*}} %[[McCopy]])
  // CHECK-NEXT: call void @llvm.lifetime.end.p0(
  // CHECK-NEXT: call void @_ZN8MoveOnlyD1Ev(ptr {{[^,]*}} %[[MoCopy]]
  // CHECK-NEXT: call void @llvm.lifetime.end.p0(
  // CHECK-NEXT: call void @llvm.lifetime.end.p0(
  // CHECK-NEXT: call ptr @llvm.coro.free(

  // The original trivial_abi parameter is destroyed when returning from the ramp.
  // CHECK: call i1 @llvm.coro.end
  // CHECK: call void @_ZN10TrivialABID1Ev(ptr {{[^,]*}} %[[TrivialAlloca]])
}

// CHECK-LABEL: void @_Z16dependent_paramsI1A1BEvT_T0_S3_(ptr noundef %x, ptr noundef %0, ptr noundef %y)
template <typename T, typename U>
void dependent_params(T x, U, U y) {
  // CHECK: %[[x_copy:.+]] = alloca %struct.A,
  // CHECK-NEXT: %[[unnamed_copy:.+]] = alloca %struct.B
  // CHECK-NEXT: %[[y_copy:.+]] = alloca %struct.B

  // CHECK: call ptr @llvm.coro.begin
  // CHECK-NEXT: call void @llvm.lifetime.start.p0(
  // CHECK-NEXT: call void @_ZN1AC1EOS_(ptr {{[^,]*}} %[[x_copy]], ptr noundef nonnull align 4 dereferenceable(512) %x)
  // CHECK-NEXT: call void @llvm.lifetime.start.p0(
  // CHECK-NEXT: call void @_ZN1BC1EOS_(ptr {{[^,]*}} %[[unnamed_copy]], ptr noundef nonnull align 4 dereferenceable(512) %0)
  // CHECK-NEXT: call void @llvm.lifetime.start.p0(
  // CHECK-NEXT: call void @_ZN1BC1EOS_(ptr {{[^,]*}} %[[y_copy]], ptr noundef nonnull align 4 dereferenceable(512) %y)
  // CHECK-NEXT: call void @llvm.lifetime.start.p0(
  // CHECK-NEXT: invoke void @_ZNSt16coroutine_traitsIJv1A1BS1_EE12promise_typeC1Ev(

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

// Test that, when the promise type has a constructor whose signature matches
// that of the coroutine function, that constructor is used.

struct promise_matching_constructor {};

template <>
struct std::coroutine_traits<void, promise_matching_constructor, int, float, double> {
  struct promise_type {
    promise_type(promise_matching_constructor, int, float, double) {}
    promise_type() = delete;
    void get_return_object() {}
    suspend_always initial_suspend() { return {}; }
    suspend_always final_suspend() noexcept { return {}; }
    void return_void() {}
    void unhandled_exception() {}
  };
};

// CHECK-LABEL: void @_Z38coroutine_matching_promise_constructor28promise_matching_constructorifd(i32 noundef %0, float noundef %1, double noundef %2)
void coroutine_matching_promise_constructor(promise_matching_constructor, int, float, double) {
  // CHECK: %[[INT:.+]] = load i32, ptr %5, align 4
  // CHECK: %[[FLOAT:.+]] = load float, ptr %6, align 4
  // CHECK: %[[DOUBLE:.+]] = load double, ptr %7, align 8
  // CHECK: invoke void @_ZNSt16coroutine_traitsIJv28promise_matching_constructorifdEE12promise_typeC1ES0_ifd(ptr {{[^,]*}} %__promise, i32 noundef %[[INT]], float noundef %[[FLOAT]], double noundef %[[DOUBLE]])
  co_return;
}

struct some_class;

struct method {};

template <typename... Args> struct std::coroutine_traits<method, Args...> {
  struct promise_type {
    promise_type(some_class&, float);
    method get_return_object();
    suspend_always initial_suspend();
    suspend_always final_suspend() noexcept;
    void return_void();
    void unhandled_exception();
  };
};

struct some_class {
  method good_coroutine_calls_custom_constructor(float);
};

// CHECK-LABEL: define{{.*}} void @_ZN10some_class39good_coroutine_calls_custom_constructorEf(ptr
method some_class::good_coroutine_calls_custom_constructor(float) {
  // CHECK: invoke void @_ZNSt16coroutine_traitsIJ6methodR10some_classfEE12promise_typeC1ES2_f(ptr {{[^,]*}} %__promise, ptr noundef nonnull align 1 dereferenceable(1) %{{.+}}, float
  co_return;
}


struct MSParm {
  int val;
  ~MSParm();
};

void consume(int) noexcept;

// Similarly to the [[clang::trivial_abi]] parameters, with the MSVC ABI
// parameters are also destroyed by the callee, and on x86-64 such parameters
// may get passed in registers. In that case it's again important that the
// parameter's local alloca does not become part of the coro frame since that
// may be destroyed before the destructor call.
void msabi(MSParm p) {
  // MSABI: define{{.*}} void @"?msabi@@YAXUMSParm@@@Z"(i32 %[[Param:.+]])

  // The parameter's local alloca is marked not part of the frame.
  // MSABI: %[[ParamAlloca:.+]] = alloca %struct.MSParm
  // MSABI-SAME: !coro.outside.frame

  // MSABI: %[[ParamCopy:.+]] = alloca %struct.MSParm

  consume(p.val);
  // The parameter's copy is used by the coroutine.
  // MSABI: %[[ValPtr:.+]] = getelementptr inbounds nuw %struct.MSParm, ptr %[[ParamCopy]], i32 0, i32 0
  // MSABI: %[[Val:.+]] = load i32, ptr %[[ValPtr]]
  // MSABI: call void @"?consume@@YAXH@Z"(i32{{.*}} %[[Val]])

  co_return;

  // The local alloca is used for the destructor call at the end of the ramp.
  // MSABI: call i1 @llvm.coro.end
  // MSABI: call void @"??1MSParm@@QEAA@XZ"(ptr{{.*}} %[[ParamAlloca]])
}
