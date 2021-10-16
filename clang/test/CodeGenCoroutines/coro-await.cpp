// RUN: %clang_cc1 -disable-noundef-analysis -triple x86_64-unknown-linux-gnu -fcoroutines-ts -std=c++14 \
// RUN:   -emit-llvm %s -o - -disable-llvm-passes -Wno-coroutine -Wno-unused | FileCheck %s

namespace std {
namespace experimental {
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

}
}

struct init_susp {
  bool await_ready();
  void await_suspend(std::experimental::coroutine_handle<>);
  void await_resume();
};
struct final_susp {
  bool await_ready() noexcept;
  void await_suspend(std::experimental::coroutine_handle<>) noexcept;
  void await_resume() noexcept;
};

struct suspend_always {
  int stuff;
  bool await_ready();
  void await_suspend(std::experimental::coroutine_handle<>);
  void await_resume();
};

template<>
struct std::experimental::coroutine_traits<void> {
  struct promise_type {
    void get_return_object();
    init_susp initial_suspend();
    final_susp final_suspend() noexcept;
    void return_void();
  };
};

// CHECK-LABEL: f0(
extern "C" void f0() {
  // CHECK: %[[FRAME:.+]] = call i8* @llvm.coro.begin(

  // See if initial_suspend was issued:
  // ----------------------------------
  // CHECK: call void @_ZNSt12experimental16coroutine_traitsIJvEE12promise_type15initial_suspendEv(
  // CHECK-NEXT: call zeroext i1 @_ZN9init_susp11await_readyEv(%struct.init_susp*
  // CHECK: %[[INITSP_ID:.+]] = call token @llvm.coro.save(
  // CHECK: call i8 @llvm.coro.suspend(token %[[INITSP_ID]], i1 false)

  co_await suspend_always{};
  // See if we need to suspend:
  // --------------------------
  // CHECK: %[[READY:.+]] = call zeroext i1 @_ZN14suspend_always11await_readyEv(%struct.suspend_always* {{[^,]*}} %[[AWAITABLE:.+]])
  // CHECK: br i1 %[[READY]], label %[[READY_BB:.+]], label %[[SUSPEND_BB:.+]]

  // If we are suspending:
  // ---------------------
  // CHECK: [[SUSPEND_BB]]:
  // CHECK: %[[SUSPEND_ID:.+]] = call token @llvm.coro.save(
  // ---------------------------
  // Build the coroutine handle and pass it to await_suspend
  // ---------------------------
  // CHECK: call i8* @_ZNSt12experimental16coroutine_handleINS_16coroutine_traitsIJvEE12promise_typeEE12from_addressEPv(i8* %[[FRAME]])
  //   ... many lines of code to coerce coroutine_handle into an i8* scalar
  // CHECK: %[[CH:.+]] = load i8*, i8** %{{.+}}
  // CHECK: call void @_ZN14suspend_always13await_suspendENSt12experimental16coroutine_handleIvEE(%struct.suspend_always* {{[^,]*}} %[[AWAITABLE]], i8* %[[CH]])
  // -------------------------
  // Generate a suspend point:
  // -------------------------
  // CHECK: %[[OUTCOME:.+]] = call i8 @llvm.coro.suspend(token %[[SUSPEND_ID]], i1 false)
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
  // CHECK:  call void @_ZN14suspend_always12await_resumeEv(%struct.suspend_always* {{[^,]*}} %[[AWAITABLE]])

  // See if final_suspend was issued:
  // ----------------------------------
  // CHECK: call void @_ZNSt12experimental16coroutine_traitsIJvEE12promise_type13final_suspendEv(
  // CHECK-NEXT: call zeroext i1 @_ZN10final_susp11await_readyEv(%struct.final_susp*
  // CHECK: %[[FINALSP_ID:.+]] = call token @llvm.coro.save(
  // CHECK: call i8 @llvm.coro.suspend(token %[[FINALSP_ID]], i1 true)
}

struct suspend_maybe {
  float stuff;
  ~suspend_maybe();
  bool await_ready();
  bool await_suspend(std::experimental::coroutine_handle<>);
  void await_resume();
};


template<>
struct std::experimental::coroutine_traits<void,int> {
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
  // CHECK: %[[PROMISE:.+]] = alloca %"struct.std::experimental::coroutine_traits<void, int>::promise_type"
  // CHECK: %[[FRAME:.+]] = call i8* @llvm.coro.begin(
  co_yield 42;
  // CHECK: call void @_ZNSt12experimental16coroutine_traitsIJviEE12promise_type11yield_valueEi(%struct.suspend_maybe* sret(%struct.suspend_maybe) align 4 %[[AWAITER:.+]], %"struct.std::experimental::coroutine_traits<void, int>::promise_type"* {{[^,]*}} %[[PROMISE]], i32 42)

  // See if we need to suspend:
  // --------------------------
  // CHECK: %[[READY:.+]] = call zeroext i1 @_ZN13suspend_maybe11await_readyEv(%struct.suspend_maybe* {{[^,]*}} %[[AWAITABLE]])
  // CHECK: br i1 %[[READY]], label %[[READY_BB:.+]], label %[[SUSPEND_BB:.+]]

  // If we are suspending:
  // ---------------------
  // CHECK: [[SUSPEND_BB]]:
  // CHECK: %[[SUSPEND_ID:.+]] = call token @llvm.coro.save(
  // ---------------------------
  // Build the coroutine handle and pass it to await_suspend
  // ---------------------------
  // CHECK: call i8* @_ZNSt12experimental16coroutine_handleINS_16coroutine_traitsIJviEE12promise_typeEE12from_addressEPv(i8* %[[FRAME]])
  //   ... many lines of code to coerce coroutine_handle into an i8* scalar
  // CHECK: %[[CH:.+]] = load i8*, i8** %{{.+}}
  // CHECK: %[[YES:.+]] = call zeroext i1 @_ZN13suspend_maybe13await_suspendENSt12experimental16coroutine_handleIvEE(%struct.suspend_maybe* {{[^,]*}} %[[AWAITABLE]], i8* %[[CH]])
  // -------------------------------------------
  // See if await_suspend decided not to suspend
  // -------------------------------------------
  // CHECK: br i1 %[[YES]], label %[[SUSPEND_PLEASE:.+]], label %[[READY_BB]]

  // CHECK: [[SUSPEND_PLEASE]]:
  // CHECK:    call i8 @llvm.coro.suspend(token %[[SUSPEND_ID]], i1 false)

  // CHECK: [[READY_BB]]:
  // CHECK:     call void @_ZN13suspend_maybe12await_resumeEv(%struct.suspend_maybe* {{[^,]*}} %[[AWAITABLE]])
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
  // CHECK: call <2 x float> @_ZN14ComplexAwaiter12await_resumeEv(%struct.ComplexAwaiter*
  // CHECK: call void @UseComplex(<2 x float> %{{.+}})

  co_await ComplexAwaiter{};
  // CHECK: call <2 x float> @_ZN14ComplexAwaiter12await_resumeEv(%struct.ComplexAwaiter*

  _Complex float Val = co_await ComplexAwaiter{};
  // CHECK: call <2 x float> @_ZN14ComplexAwaiter12await_resumeEv(%struct.ComplexAwaiter*
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
  // CHECK: call void @_ZN11AggrAwaiter12await_resumeEv(%struct.Aggr* sret(%struct.Aggr) align 4 %[[AwaitResume:.+]],
  // CHECK: call void @UseAggr(%struct.Aggr* nonnull align 4 dereferenceable(12) %[[AwaitResume]])
  // CHECK: call void @_ZN4AggrD1Ev(%struct.Aggr* {{[^,]*}} %[[AwaitResume]])
  // CHECK: call void @Whatever()

  co_await AggrAwaiter{};
  Whatever();
  // CHECK: call void @_ZN11AggrAwaiter12await_resumeEv(%struct.Aggr* sret(%struct.Aggr) align 4 %[[AwaitResume2:.+]],
  // CHECK: call void @_ZN4AggrD1Ev(%struct.Aggr* {{[^,]*}} %[[AwaitResume2]])
  // CHECK: call void @Whatever()

  Aggr Val = co_await AggrAwaiter{};
  Whatever();
  // CHECK: call void @_ZN11AggrAwaiter12await_resumeEv(%struct.Aggr* sret(%struct.Aggr) align 4 %[[AwaitResume3:.+]],
  // CHECK: call void @Whatever()
  // CHECK: call void @_ZN4AggrD1Ev(%struct.Aggr* {{[^,]*}} %[[AwaitResume3]])
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
  // CHECK: %[[Result:.+]] = call i32 @_ZN13ScalarAwaiter12await_resumeEv(%struct.ScalarAwaiter*
  // CHECK: call void @UseScalar(i32 %[[Result]])

  int Val = co_await ScalarAwaiter{};
  // CHECK: %[[Result2:.+]] = call i32 @_ZN13ScalarAwaiter12await_resumeEv(%struct.ScalarAwaiter*
  // CHECK: store i32 %[[Result2]], i32* %[[TMP_EXPRCLEANUP:.+]],
  // CHECK: %[[TMP:.+]] = load i32, i32* %[[TMP_EXPRCLEANUP]],
  // CHECK: store i32 %[[TMP]], i32* %Val,

  co_await ScalarAwaiter{};
  // CHECK: call i32 @_ZN13ScalarAwaiter12await_resumeEv(%struct.ScalarAwaiter*
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
  // CHECK: call i32 @_ZN13ScalarAwaiter12await_resumeEv(%struct.ScalarAwaiter* {{[^,]*}} %

  co_await MyAgg{};
  // CHECK: call void @_ZN5MyAggawEv(%struct.MyAgg* {{[^,]*}} %
  // CHECK: call void @_ZN11AggrAwaiter12await_resumeEv(%struct.Aggr* sret(%struct.Aggr) align 4 %
}

// CHECK-LABEL: EndlessLoop(
extern "C" void EndlessLoop() {
  // CHECK: %[[FRAME:.+]] = call i8* @llvm.coro.begin(

  // See if initial_suspend was issued:
  // ----------------------------------
  // CHECK: call void @_ZNSt12experimental16coroutine_traitsIJvEE12promise_type15initial_suspendEv(
  // CHECK-NEXT: call zeroext i1 @_ZN9init_susp11await_readyEv(%struct.init_susp*

  for (;;)
    co_await suspend_always{};

  // Verify that final_suspend was NOT issued:
  // ----------------------------------
  // CHECK-NOT: call void @_ZNSt12experimental16coroutine_traitsIJvEE12promise_type13final_suspendEv(
  // CHECK-NOT: call zeroext i1 @_ZN10final_susp11await_readyEv(%struct.final_susp*
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
  void await_suspend(std::experimental::coroutine_handle<>);
  RefTag& await_resume();
};


template<>
struct std::experimental::coroutine_traits<void,double> {
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
  // CHECK: %[[XVAR:.+]] = alloca %struct.RefTag*,

  // CHECK: %[[YVAR:.+]] = alloca %struct.RefTag*,
  // CHECK-NEXT: %[[TMP1:.+]] = alloca %struct.AwaitResumeReturnsLValue,

  // CHECK: %[[TMP_EXPRCLEANUP1:.+]] = alloca %struct.RefTag*,
  // CHECK: %[[ZVAR:.+]] = alloca %struct.RefTag*,
  // CHECK-NEXT: %[[TMP2:.+]] = alloca %struct.AwaitResumeReturnsLValue,
  // CHECK: %[[TMP_EXPRCLEANUP2:.+]] = alloca %struct.RefTag*,

  // CHECK: %[[RES1:.+]] = call nonnull align 1 dereferenceable({{.*}}) %struct.RefTag* @_ZN24AwaitResumeReturnsLValue12await_resumeEv(%struct.AwaitResumeReturnsLValue* {{[^,]*}} %[[AVAR]])
  // CHECK-NEXT: store %struct.RefTag* %[[RES1]], %struct.RefTag** %[[XVAR]],
  RefTag& x = co_await a;

  // CHECK: %[[RES2:.+]] = call nonnull align 1 dereferenceable({{.*}}) %struct.RefTag* @_ZN24AwaitResumeReturnsLValue12await_resumeEv(%struct.AwaitResumeReturnsLValue* {{[^,]*}} %[[TMP1]])
  // CHECK-NEXT: store %struct.RefTag* %[[RES2]], %struct.RefTag** %[[TMP_EXPRCLEANUP1]],
  // CHECK: %[[LOAD_TMP1:.+]] = load %struct.RefTag*, %struct.RefTag** %[[TMP_EXPRCLEANUP1]],
  // CHECK: store %struct.RefTag* %[[LOAD_TMP1]], %struct.RefTag** %[[YVAR]],

  RefTag& y = co_await AwaitResumeReturnsLValue{};
  // CHECK: %[[RES3:.+]] = call nonnull align 1 dereferenceable({{.*}}) %struct.RefTag* @_ZN24AwaitResumeReturnsLValue12await_resumeEv(%struct.AwaitResumeReturnsLValue* {{[^,]*}} %[[TMP2]])
  // CHECK-NEXT: store %struct.RefTag* %[[RES3]], %struct.RefTag** %[[TMP_EXPRCLEANUP2]],
  // CHECK: %[[LOAD_TMP2:.+]] = load %struct.RefTag*, %struct.RefTag** %[[TMP_EXPRCLEANUP2]],
  // CHECK: store %struct.RefTag* %[[LOAD_TMP2]], %struct.RefTag** %[[ZVAR]],
  RefTag& z = co_yield 42;
}

struct TailCallAwait {
  bool await_ready();
  std::experimental::coroutine_handle<> await_suspend(std::experimental::coroutine_handle<>);
  void await_resume();
};

// CHECK-LABEL: @TestTailcall(
extern "C" void TestTailcall() {
  co_await TailCallAwait{};

  // CHECK: %[[RESULT:.+]] = call i8* @_ZN13TailCallAwait13await_suspendENSt12experimental16coroutine_handleIvEE(%struct.TailCallAwait*
  // CHECK: %[[COERCE:.+]] = getelementptr inbounds %"struct.std::experimental::coroutine_handle", %"struct.std::experimental::coroutine_handle"* %[[TMP:.+]], i32 0, i32 0
  // CHECK: store i8* %[[RESULT]], i8** %[[COERCE]]
  // CHECK: %[[ADDR:.+]] = call i8* @_ZNSt12experimental16coroutine_handleIvE7addressEv(%"struct.std::experimental::coroutine_handle"* {{[^,]*}} %[[TMP]])
  // CHECK: call void @llvm.coro.resume(i8* %[[ADDR]])
}
