// RUN: %clang_cc1 -no-enable-noundef-analysis -triple x86_64-unknown-linux-gnu -std=c++20 \
// RUN:   -emit-llvm %s -o - -disable-llvm-passes -Wno-coroutine -Wno-unused | FileCheck %s

namespace std {
template <typename... T>
struct coroutine_traits;

template <typename Promise = void> struct coroutine_handle;

template <>
struct coroutine_handle<void> {
  void *ptr;
  static coroutine_handle from_address(void *);
  void *address();
};

template <typename Promise>
struct coroutine_handle : coroutine_handle<> {
  static coroutine_handle from_address(void *) noexcept;
};

} // namespace std

struct init_susp {
  bool await_ready();
  void await_suspend(std::coroutine_handle<>);
  void await_resume();
};
struct final_susp {
  bool await_ready() noexcept;
  void await_suspend(std::coroutine_handle<>) noexcept;
  void await_resume() noexcept;
};

struct suspend_always {
  int stuff;
  bool await_ready();
  void await_suspend(std::coroutine_handle<>);
  void await_resume();
};

template <>
struct std::coroutine_traits<void> {
  struct promise_type {
    void get_return_object();
    init_susp initial_suspend();
    final_susp final_suspend() noexcept;
    void return_void();
  };
};

// CHECK-LABEL: f0(
extern "C" void f0() {
  // CHECK: %[[FRAME:.+]] = call ptr @llvm.coro.begin(

  // See if initial_suspend was issued:
  // ----------------------------------
  // CHECK: call void @_ZNSt16coroutine_traitsIJvEE12promise_type15initial_suspendEv(
  // CHECK-NEXT: call zeroext i1 @_ZN9init_susp11await_readyEv(ptr
  // CHECK: %[[INITSP_ID:.+]] = call token @llvm.coro.save(
  // CHECK: call i8 @llvm.coro.suspend(token %[[INITSP_ID]], i1 false)

  co_await suspend_always{};
  // See if we need to suspend:
  // --------------------------
  // CHECK: %[[READY:.+]] = call zeroext i1 @_ZN14suspend_always11await_readyEv(ptr {{[^,]*}} %[[AWAITABLE:.+]])
  // CHECK: br i1 %[[READY]], label %[[READY_BB:.+]], label %[[SUSPEND_BB:.+]]

  // If we are suspending:
  // ---------------------
  // CHECK: [[SUSPEND_BB]]:
  // CHECK: %[[SUSPEND_ID:.+]] = call token @llvm.coro.save(
  // ---------------------------
  // Call coro.await.suspend
  // ---------------------------
  // CHECK-NEXT: call void @llvm.coro.await.suspend.void(ptr %[[AWAITABLE]], ptr %[[FRAME]], ptr @f0.__await_suspend_wrapper__await)
  // -------------------------
  // Generate a suspend point:
  // -------------------------
  // CHECK-NEXT: %[[OUTCOME:.+]] = call i8 @llvm.coro.suspend(token %[[SUSPEND_ID]], i1 false)
  // CHECK: switch i8 %[[OUTCOME]], label %[[RET_BB:.+]] [
  // CHECK:   i8 0, label %[[READY_BB]]
  // CHECK:   i8 1, label %[[CLEANUP_BB:.+]]
  // CHECK: ]

  // Cleanup code goes here:
  // -----------------------
  // CHECK: [[CLEANUP_BB]]:

  // When coroutine is resumed, call await_resume
  // --------------------------
  // CHECK: [[READY_BB]]:
  // CHECK:  call void @_ZN14suspend_always12await_resumeEv(ptr {{[^,]*}} %[[AWAITABLE]])

  // See if final_suspend was issued:
  // ----------------------------------
  // CHECK: call void @_ZNSt16coroutine_traitsIJvEE12promise_type13final_suspendEv(
  // CHECK-NEXT: call zeroext i1 @_ZN10final_susp11await_readyEv(ptr
  // CHECK: %[[FINALSP_ID:.+]] = call token @llvm.coro.save(
  // CHECK: call i8 @llvm.coro.suspend(token %[[FINALSP_ID]], i1 true)

  // Await suspend wrapper
  // CHECK: define{{.*}} @f0.__await_suspend_wrapper__await(ptr {{[^,]*}} %[[AWAITABLE_ARG:.+]], ptr {{[^,]*}} %[[FRAME_ARG:.+]])
  // CHECK: store ptr %[[AWAITABLE_ARG]], ptr %[[AWAITABLE_TMP:.+]],
  // CHECK: store ptr %[[FRAME_ARG]], ptr %[[FRAME_TMP:.+]],
  // CHECK: %[[AWAITABLE:.+]] = load ptr, ptr %[[AWAITABLE_TMP]]
  // CHECK: %[[FRAME:.+]] = load ptr, ptr %[[FRAME_TMP]]
  // CHECK: call ptr @_ZNSt16coroutine_handleINSt16coroutine_traitsIJvEE12promise_typeEE12from_addressEPv(ptr %[[FRAME]])
  //   ... many lines of code to coerce coroutine_handle into an ptr scalar
  // CHECK: %[[CH:.+]] = load ptr, ptr %{{.+}}
  // CHECK: call void @_ZN14suspend_always13await_suspendESt16coroutine_handleIvE(ptr {{[^,]*}} %[[AWAITABLE]], ptr %[[CH]])
}

struct suspend_maybe {
  float stuff;
  ~suspend_maybe();
  bool await_ready();
  bool await_suspend(std::coroutine_handle<>);
  void await_resume();
};

template <>
struct std::coroutine_traits<void, int> {
  struct promise_type {
    void get_return_object();
    init_susp initial_suspend();
    final_susp final_suspend() noexcept;
    void return_void();
    suspend_maybe yield_value(int);
  };
};

// CHECK-LABEL: f1(
extern "C" void f1(int) {
  // CHECK: %[[PROMISE:.+]] = alloca %"struct.std::coroutine_traits<void, int>::promise_type"
  // CHECK: %[[FRAME:.+]] = call ptr @llvm.coro.begin(
  co_yield 42;
  // CHECK: call void @_ZNSt16coroutine_traitsIJviEE12promise_type11yield_valueEi(ptr dead_on_unwind writable sret(%struct.suspend_maybe) align 4 %[[AWAITER:.+]], ptr {{[^,]*}} %[[PROMISE]], i32 42)

  // See if we need to suspend:
  // --------------------------
  // CHECK: %[[READY:.+]] = call zeroext i1 @_ZN13suspend_maybe11await_readyEv(ptr {{[^,]*}} %[[AWAITABLE:.+]])
  // CHECK: br i1 %[[READY]], label %[[READY_BB:.+]], label %[[SUSPEND_BB:.+]]

  // If we are suspending:
  // ---------------------
  // CHECK: [[SUSPEND_BB]]:
  // CHECK: %[[SUSPEND_ID:.+]] = call token @llvm.coro.save(
  // ---------------------------
  // Call coro.await.suspend
  // ---------------------------
  // CHECK-NEXT: %[[YES:.+]] = call i1 @llvm.coro.await.suspend.bool(ptr %[[AWAITABLE]], ptr %[[FRAME]], ptr @f1.__await_suspend_wrapper__yield)
  // -------------------------------------------
  // See if await_suspend decided not to suspend
  // -------------------------------------------
  // CHECK: br i1 %[[YES]], label %[[SUSPEND_PLEASE:.+]], label %[[READY_BB]]

  // CHECK: [[SUSPEND_PLEASE]]:
  // CHECK:    call i8 @llvm.coro.suspend(token %[[SUSPEND_ID]], i1 false)

  // CHECK: [[READY_BB]]:
  // CHECK:     call void @_ZN13suspend_maybe12await_resumeEv(ptr {{[^,]*}} %[[AWAITABLE]])

  // Await suspend wrapper
  // CHECK: define {{.*}} i1 @f1.__await_suspend_wrapper__yield(ptr {{[^,]*}} %[[AWAITABLE_ARG:.+]], ptr {{[^,]*}} %[[FRAME_ARG:.+]])
  // CHECK: store ptr %[[AWAITABLE_ARG]], ptr %[[AWAITABLE_TMP:.+]],
  // CHECK: store ptr %[[FRAME_ARG]], ptr %[[FRAME_TMP:.+]],
  // CHECK: %[[AWAITABLE:.+]] = load ptr, ptr %[[AWAITABLE_TMP]]
  // CHECK: %[[FRAME:.+]] = load ptr, ptr %[[FRAME_TMP]]
  // CHECK: call ptr @_ZNSt16coroutine_handleINSt16coroutine_traitsIJviEE12promise_typeEE12from_addressEPv(ptr %[[FRAME]])
  //   ... many lines of code to coerce coroutine_handle into an ptr scalar
  // CHECK: %[[CH:.+]] = load ptr, ptr %{{.+}}
  // CHECK: %[[YES:.+]] = call zeroext i1 @_ZN13suspend_maybe13await_suspendESt16coroutine_handleIvE(ptr {{[^,]*}} %[[AWAITABLE]], ptr %[[CH]]) 
  // CHECK-NEXT: ret i1 %[[YES]]
}

struct ComplexAwaiter {
  template <typename F> void await_suspend(F);
  bool await_ready();
  _Complex float await_resume();
};
extern "C" void UseComplex(_Complex float);

// CHECK-LABEL: @TestComplex(
extern "C" void TestComplex() {
  UseComplex(co_await ComplexAwaiter{});
  // CHECK: call <2 x float> @_ZN14ComplexAwaiter12await_resumeEv(ptr
  // CHECK: call void @UseComplex(<2 x float> %{{.+}})

  co_await ComplexAwaiter{};
  // CHECK: call <2 x float> @_ZN14ComplexAwaiter12await_resumeEv(ptr

  _Complex float Val = co_await ComplexAwaiter{};
  // CHECK: call <2 x float> @_ZN14ComplexAwaiter12await_resumeEv(ptr
}

struct Aggr { int X, Y, Z; ~Aggr(); };
struct AggrAwaiter {
  template <typename F> void await_suspend(F);
  bool await_ready();
  Aggr await_resume();
};

extern "C" void Whatever();
extern "C" void UseAggr(Aggr&&);

// FIXME: Once the cleanup code is in, add testing that destructors for Aggr
// are invoked properly on the cleanup branches.

// CHECK-LABEL: @TestAggr(
extern "C" void TestAggr() {
  UseAggr(co_await AggrAwaiter{});
  Whatever();
  // CHECK: call void @_ZN11AggrAwaiter12await_resumeEv(ptr dead_on_unwind writable sret(%struct.Aggr) align 4 %[[AwaitResume:.+]],
  // CHECK: call void @UseAggr(ptr nonnull align 4 dereferenceable(12) %[[AwaitResume]])
  // CHECK: call void @_ZN4AggrD1Ev(ptr {{[^,]*}} %[[AwaitResume]])
  // CHECK: call void @Whatever()

  co_await AggrAwaiter{};
  Whatever();
  // CHECK: call void @_ZN11AggrAwaiter12await_resumeEv(ptr dead_on_unwind writable sret(%struct.Aggr) align 4 %[[AwaitResume2:.+]],
  // CHECK: call void @_ZN4AggrD1Ev(ptr {{[^,]*}} %[[AwaitResume2]])
  // CHECK: call void @Whatever()

  Aggr Val = co_await AggrAwaiter{};
  Whatever();
  // CHECK: call void @_ZN11AggrAwaiter12await_resumeEv(ptr dead_on_unwind writable sret(%struct.Aggr) align 4 %[[AwaitResume3:.+]],
  // CHECK: call void @Whatever()
  // CHECK: call void @_ZN4AggrD1Ev(ptr {{[^,]*}} %[[AwaitResume3]])
}

struct ScalarAwaiter {
  template <typename F> void await_suspend(F);
  bool await_ready();
  int await_resume();
};

extern "C" void UseScalar(int);

// CHECK-LABEL: @TestScalar(
extern "C" void TestScalar() {
  UseScalar(co_await ScalarAwaiter{});
  // CHECK: %[[Result:.+]] = call i32 @_ZN13ScalarAwaiter12await_resumeEv(ptr
  // CHECK: call void @UseScalar(i32 %[[Result]])

  int Val = co_await ScalarAwaiter{};
  // CHECK: %[[Result2:.+]] = call i32 @_ZN13ScalarAwaiter12await_resumeEv(ptr
  // CHECK: store i32 %[[Result2]], ptr %[[TMP_EXPRCLEANUP:.+]],
  // CHECK: %[[TMP:.+]] = load i32, ptr %[[TMP_EXPRCLEANUP]],
  // CHECK: store i32 %[[TMP]], ptr %Val,

  co_await ScalarAwaiter{};
  // CHECK: call i32 @_ZN13ScalarAwaiter12await_resumeEv(ptr
}

// Test operator co_await codegen.
enum class MyInt: int {};
ScalarAwaiter operator co_await(MyInt);

struct MyAgg {
  AggrAwaiter operator co_await();
};

// CHECK-LABEL: @TestOpAwait(
extern "C" void TestOpAwait() {
  co_await MyInt(42);
  // CHECK: call void @_Zaw5MyInt(i32 42)
  // CHECK: call i32 @_ZN13ScalarAwaiter12await_resumeEv(ptr {{[^,]*}} %

  co_await MyAgg{};
  // CHECK: call void @_ZN5MyAggawEv(ptr {{[^,]*}} %
  // CHECK: call void @_ZN11AggrAwaiter12await_resumeEv(ptr dead_on_unwind writable sret(%struct.Aggr) align 4 %
}

// CHECK-LABEL: EndlessLoop(
extern "C" void EndlessLoop() {
  // CHECK: %[[FRAME:.+]] = call ptr @llvm.coro.begin(

  // See if initial_suspend was issued:
  // ----------------------------------
  // CHECK: call void @_ZNSt16coroutine_traitsIJvEE12promise_type15initial_suspendEv(
  // CHECK-NEXT: call zeroext i1 @_ZN9init_susp11await_readyEv(ptr

  for (;;)
    co_await suspend_always{};

  // Verify that final_suspend was NOT issued:
  // ----------------------------------
  // CHECK-NOT: call void @_ZNSt16coroutine_traitsIJvEE12promise_type13final_suspendEv(
  // CHECK-NOT: call zeroext i1 @_ZN10final_susp11await_readyEv(ptr
}

// Verifies that we don't crash when awaiting on an lvalue.
// CHECK-LABEL: @_Z11AwaitLValuev(
void AwaitLValue() {
  suspend_always lval;
  co_await lval;
}

struct RefTag { };

struct AwaitResumeReturnsLValue {
  bool await_ready();
  void await_suspend(std::coroutine_handle<>);
  RefTag& await_resume();
};

template <>
struct std::coroutine_traits<void, double> {
  struct promise_type {
    void get_return_object();
    init_susp initial_suspend();
    final_susp final_suspend() noexcept;
    void return_void();
    AwaitResumeReturnsLValue yield_value(int);
  };
};

// Verifies that we don't crash when returning an lvalue from an await_resume()
// expression.
// CHECK-LABEL:  define{{.*}} void @_Z18AwaitReturnsLValued(double %0)
void AwaitReturnsLValue(double) {
  AwaitResumeReturnsLValue a;
  // CHECK: %[[AVAR:.+]] = alloca %struct.AwaitResumeReturnsLValue,
  // CHECK: %[[XVAR:.+]] = alloca ptr,

  // CHECK: %[[YVAR:.+]] = alloca ptr,
  // CHECK-NEXT: %[[TMP1:.+]] = alloca %struct.AwaitResumeReturnsLValue,

  // CHECK: %[[TMP_EXPRCLEANUP1:.+]] = alloca ptr,
  // CHECK: %[[ZVAR:.+]] = alloca ptr,
  // CHECK-NEXT: %[[TMP2:.+]] = alloca %struct.AwaitResumeReturnsLValue,
  // CHECK: %[[TMP_EXPRCLEANUP2:.+]] = alloca ptr,

  // CHECK: %[[RES1:.+]] = call nonnull align 1 dereferenceable({{.*}}) ptr @_ZN24AwaitResumeReturnsLValue12await_resumeEv(ptr {{[^,]*}} %[[AVAR]])
  // CHECK-NEXT: store ptr %[[RES1]], ptr %[[XVAR]],
  RefTag& x = co_await a;

  // CHECK: %[[RES2:.+]] = call nonnull align 1 dereferenceable({{.*}}) ptr @_ZN24AwaitResumeReturnsLValue12await_resumeEv(ptr {{[^,]*}} %[[TMP1]])
  // CHECK-NEXT: store ptr %[[RES2]], ptr %[[TMP_EXPRCLEANUP1]],
  // CHECK: %[[LOAD_TMP1:.+]] = load ptr, ptr %[[TMP_EXPRCLEANUP1]],
  // CHECK: store ptr %[[LOAD_TMP1]], ptr %[[YVAR]],

  RefTag& y = co_await AwaitResumeReturnsLValue{};
  // CHECK: %[[RES3:.+]] = call nonnull align 1 dereferenceable({{.*}}) ptr @_ZN24AwaitResumeReturnsLValue12await_resumeEv(ptr {{[^,]*}} %[[TMP2]])
  // CHECK-NEXT: store ptr %[[RES3]], ptr %[[TMP_EXPRCLEANUP2]],
  // CHECK: %[[LOAD_TMP2:.+]] = load ptr, ptr %[[TMP_EXPRCLEANUP2]],
  // CHECK: store ptr %[[LOAD_TMP2]], ptr %[[ZVAR]],
  RefTag& z = co_yield 42;
}

struct TailCallAwait {
  bool await_ready();
  std::coroutine_handle<> await_suspend(std::coroutine_handle<>);
  void await_resume();
};

// CHECK-LABEL: @TestTailcall(
extern "C" void TestTailcall() {
  // CHECK: %[[PROMISE:.+]] = alloca %"struct.std::coroutine_traits<void>::promise_type"
  // CHECK: %[[FRAME:.+]] = call ptr @llvm.coro.begin(
  co_await TailCallAwait{};
  // CHECK: %[[READY:.+]] = call zeroext i1 @_ZN13TailCallAwait11await_readyEv(ptr {{[^,]*}} %[[AWAITABLE:.+]])
  // CHECK: br i1 %[[READY]], label %[[READY_BB:.+]], label %[[SUSPEND_BB:.+]]

  // If we are suspending:
  // ---------------------
  // CHECK: [[SUSPEND_BB]]:
  // CHECK: %[[SUSPEND_ID:.+]] = call token @llvm.coro.save(
  // ---------------------------
  // Call coro.await.suspend
  // ---------------------------
  // Note: The call must not be nounwind since the resumed function could throw.
  // CHECK-NEXT: call void @llvm.coro.await.suspend.handle(ptr %[[AWAITABLE]], ptr %[[FRAME]], ptr @TestTailcall.__await_suspend_wrapper__await){{$}}
  // CHECK-NEXT: %[[OUTCOME:.+]] = call i8 @llvm.coro.suspend(token %[[SUSPEND_ID]], i1 false)
  // CHECK-NEXT: switch i8 %[[OUTCOME]], label %[[RET_BB:.+]] [
  // CHECK-NEXT:   i8 0, label %[[READY_BB]]
  // CHECK-NEXT:   i8 1, label %{{.+}}
  // CHECK-NEXT: ]

  // Await suspend wrapper
  // CHECK: define {{.*}} ptr @TestTailcall.__await_suspend_wrapper__await(ptr {{[^,]*}} %[[AWAITABLE_ARG:.+]], ptr {{[^,]*}} %[[FRAME_ARG:.+]])
  // CHECK: store ptr %[[AWAITABLE_ARG]], ptr %[[AWAITABLE_TMP:.+]],
  // CHECK: store ptr %[[FRAME_ARG]], ptr %[[FRAME_TMP:.+]],
  // CHECK: %[[AWAITABLE:.+]] = load ptr, ptr %[[AWAITABLE_TMP]]
  // CHECK: %[[FRAME:.+]] = load ptr, ptr %[[FRAME_TMP]]
  // CHECK: call ptr  @_ZNSt16coroutine_handleINSt16coroutine_traitsIJvEE12promise_typeEE12from_addressEPv(ptr %[[FRAME]])
  //   ... many lines of code to coerce coroutine_handle into an ptr scalar
  // CHECK: %[[CH:.+]] = load ptr, ptr %{{.+}}
  // CHECK-NEXT: %[[RESULT:.+]] = call ptr @_ZN13TailCallAwait13await_suspendESt16coroutine_handleIvE(ptr {{[^,]*}} %[[AWAITABLE]], ptr %[[CH]]) 
  // CHECK-NEXT: %[[COERCE:.+]] = getelementptr inbounds nuw %"struct.std::coroutine_handle", ptr %[[TMP:.+]], i32 0, i32 0
  // CHECK-NEXT: store ptr %[[RESULT]], ptr %[[COERCE]]
  // CHECK-NEXT: %[[ADDR:.+]] = call ptr @_ZNSt16coroutine_handleIvE7addressEv(ptr {{[^,]*}} %[[TMP]])
  // CHECK-NEXT: ret ptr %[[ADDR]]
}
