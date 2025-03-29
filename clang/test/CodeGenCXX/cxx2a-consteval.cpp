// RUN: %clang_cc1 -emit-llvm %s -std=c++2a -triple x86_64-unknown-linux-gnu -o %t.ll
// RUN: FileCheck -check-prefix=EVAL -input-file=%t.ll %s
// RUN: FileCheck -check-prefix=EVAL-STATIC -input-file=%t.ll %s
// RUN: FileCheck -check-prefix=EVAL-FN -input-file=%t.ll %s
//
// RUN: %clang_cc1 -emit-llvm %s -Dconsteval="" -std=c++2a -triple x86_64-unknown-linux-gnu -o %t.ll
// RUN: FileCheck -check-prefix=EXPR -input-file=%t.ll %s

// RUN: %clang_cc1 -emit-llvm %s -std=c++2a -triple x86_64-unknown-linux-gnu -o %t.ll -fexperimental-new-constant-interpreter
// RUN: FileCheck -check-prefix=EVAL -input-file=%t.ll %s
// RUN: FileCheck -check-prefix=EVAL-STATIC -input-file=%t.ll %s
// RUN: FileCheck -check-prefix=EVAL-FN -input-file=%t.ll %s
//
// RUN: %clang_cc1 -emit-llvm %s -Dconsteval="" -std=c++2a -triple x86_64-unknown-linux-gnu -o %t.ll -fexperimental-new-constant-interpreter
// RUN: FileCheck -check-prefix=EXPR -input-file=%t.ll %s

// there is two version of symbol checks to ensure
// that the symbol we are looking for are correct
// EVAL-NOT: @__cxx_global_var_init()
// EXPR: @__cxx_global_var_init()

// EVAL-NOT: @_Z4ret7v()
// EXPR: @_Z4ret7v()
consteval int ret7() {
  return 7;
}

// EVAL-FN-LABEL: @_Z9test_ret7v(
// EVAL-FN-NEXT:  entry:
// EVAL-FN-NEXT:    [[I:%.*]] = alloca i32, align 4
// EVAL-FN-NEXT:    store i32 7, ptr [[I]], align 4
// EVAL-FN-NEXT:    [[TMP0:%.*]] = load i32, ptr [[I]], align 4
// EVAL-FN-NEXT:    ret i32 [[TMP0]]
//
int test_ret7() {
  int i = ret7();
  return i;
}

int global_i = ret7();

constexpr int i_const = 5;

// EVAL-NOT: @_Z4retIv()
// EXPR: @_Z4retIv()
consteval const int &retI() {
  return i_const;
}

// EVAL-FN-LABEL: @_Z12test_retRefIv(
// EVAL-FN-NEXT:  entry:
// EVAL-FN-NEXT:    ret ptr @_ZL7i_const
//
const int &test_retRefI() {
  return retI();
}

// EVAL-FN-LABEL: @_Z9test_retIv(
// EVAL-FN-NEXT:  entry:
// EVAL-FN-NEXT:    [[TMP0:%.*]] = load i32, ptr @_ZL7i_const, align 4
// EVAL-FN-NEXT:    ret i32 [[TMP0]]
//
int test_retI() {
  return retI();
}

// EVAL-NOT: @_Z4retIv()
// EXPR: @_Z4retIv()
consteval const int *retIPtr() {
  return &i_const;
}

// EVAL-FN-LABEL: @_Z12test_retIPtrv(
// EVAL-FN-NEXT:  entry:
// EVAL-FN-NEXT:    [[TMP0:%.*]] = load i32, ptr @_ZL7i_const, align 4
// EVAL-FN-NEXT:    ret i32 [[TMP0]]
//
int test_retIPtr() {
  return *retIPtr();
}

// EVAL-FN-LABEL: @_Z13test_retPIPtrv(
// EVAL-FN-NEXT:  entry:
// EVAL-FN-NEXT:    ret ptr @_ZL7i_const
//
const int *test_retPIPtr() {
  return retIPtr();
}

// EVAL-NOT: @_Z4retIv()
// EXPR: @_Z4retIv()
consteval const int &&retIRRef() {
  return static_cast<const int &&>(i_const);
}

// EVAL-FN-LABEL: @_Z13test_retIRRefv(
// EVAL-FN-NEXT:  entry:
// EVAL-FN-NEXT:    ret ptr @_ZL7i_const
//
const int &&test_retIRRef() {
  return static_cast<const int &&>(retIRRef());
}

// EVAL-FN-LABEL: @_Z14test_retIRRefIv(
// EVAL-FN-NEXT:  entry:
// EVAL-FN-NEXT:    [[TMP0:%.*]] = load i32, ptr @_ZL7i_const, align 4
// EVAL-FN-NEXT:    ret i32 [[TMP0]]
//
int test_retIRRefI() {
  return retIRRef();
}

struct Agg {
  int a;
  long b;
};

// EVAL-NOT: @_Z6retAggv()
// EXPR: @_Z6retAggv()
consteval Agg retAgg() {
  return {13, 17};
}

// EVAL-FN-LABEL: @_Z11test_retAggv(
// EVAL-FN-NEXT:  entry:
// EVAL-FN-NEXT:    [[B:%.*]] = alloca i64, align 8
// EVAL-FN-NEXT:    [[REF_TMP:%.*]] = alloca [[STRUCT_AGG:%.*]], align 8
// EVAL-FN-NEXT:    [[TMP0:%.*]] = getelementptr inbounds nuw [[STRUCT_AGG]], ptr [[REF_TMP]], i32 0, i32 0
// EVAL-FN-NEXT:    store i32 13, ptr [[TMP0]], align 8
// EVAL-FN-NEXT:    [[TMP1:%.*]] = getelementptr inbounds nuw [[STRUCT_AGG]], ptr [[REF_TMP]], i32 0, i32 1
// EVAL-FN-NEXT:    store i64 17, ptr [[TMP1]], align 8
// EVAL-FN-NEXT:    store i64 17, ptr [[B]], align 8
// EVAL-FN-NEXT:    [[TMP2:%.*]] = load i64, ptr [[B]], align 8
// EVAL-FN-NEXT:    ret i64 [[TMP2]]
//
long test_retAgg() {
  long b = retAgg().b;
  return b;
}

// EVAL-STATIC: @A ={{.*}} global %struct.Agg { i32 13, i64 17 }, align 8
Agg A = retAgg();

// EVAL-NOT: @_Z9retRefAggv()
// EXPR: @_Z9retRefAggv()
consteval const Agg &retRefAgg() {
  const Agg &tmp = A;
  return A;
}

// EVAL-FN-LABEL: @_Z14test_retRefAggv(
// EVAL-FN-NEXT:  entry:
// EVAL-FN-NEXT:    [[B:%.*]] = alloca i64, align 8
// EVAL-FN-NEXT:    [[REF_TMP:%.*]] = alloca [[STRUCT_AGG:%.*]], align 8
// EVAL-FN-NEXT:    [[TMP0:%.*]] = getelementptr inbounds nuw [[STRUCT_AGG]], ptr [[REF_TMP]], i32 0, i32 0
// EVAL-FN-NEXT:    store i32 13, ptr [[TMP0]], align 8
// EVAL-FN-NEXT:    [[TMP1:%.*]] = getelementptr inbounds nuw [[STRUCT_AGG]], ptr [[REF_TMP]], i32 0, i32 1
// EVAL-FN-NEXT:    store i64 17, ptr [[TMP1]], align 8
// EVAL-FN-NEXT:    store i64 17, ptr [[B]], align 8
// EVAL-FN-NEXT:    [[TMP2:%.*]] = load i64, ptr [[B]], align 8
// EVAL-FN-NEXT:    ret i64 [[TMP2]]
//
long test_retRefAgg() {
  long b = retAgg().b;
  return b;
}

// EVAL-NOT: @_Z8is_constv()
// EXPR: @_Z8is_constv()
consteval Agg is_const() {
  return {5, 19 * __builtin_is_constant_evaluated()};
}

// EVAL-FN-LABEL: @_Z13test_is_constv(
// EVAL-FN-NEXT:  entry:
// EVAL-FN-NEXT:    [[B:%.*]] = alloca i64, align 8
// EVAL-FN-NEXT:    [[REF_TMP:%.*]] = alloca [[STRUCT_AGG:%.*]], align 8
// EVAL-FN-NEXT:    [[TMP0:%.*]] = getelementptr inbounds nuw [[STRUCT_AGG]], ptr [[REF_TMP]], i32 0, i32 0
// EVAL-FN-NEXT:    store i32 5, ptr [[TMP0]], align 8
// EVAL-FN-NEXT:    [[TMP1:%.*]] = getelementptr inbounds nuw [[STRUCT_AGG]], ptr [[REF_TMP]], i32 0, i32 1
// EVAL-FN-NEXT:    store i64 19, ptr [[TMP1]], align 8
// EVAL-FN-NEXT:    store i64 19, ptr [[B]], align 8
// EVAL-FN-NEXT:    [[TMP2:%.*]] = load i64, ptr [[B]], align 8
// EVAL-FN-NEXT:    ret i64 [[TMP2]]
//
long test_is_const() {
  long b = is_const().b;
  return b;
}

// EVAL-NOT: @_ZN7AggCtorC
// EXPR: @_ZN7AggCtorC
struct AggCtor {
  consteval AggCtor(int a = 3, long b = 5) : a(a * a), b(a * b) {}
  int a;
  long b;
};

// EVAL-FN-LABEL: @_Z12test_AggCtorv(
// EVAL-FN-NEXT:  entry:
// EVAL-FN-NEXT:    [[I:%.*]] = alloca i32, align 4
// EVAL-FN-NEXT:    [[C:%.*]] = alloca [[STRUCT_AGGCTOR:%.*]], align 8
// EVAL-FN-NEXT:    store i32 2, ptr [[I]], align 4
// EVAL-FN-NEXT:    [[TMP0:%.*]] = getelementptr inbounds nuw [[STRUCT_AGGCTOR]], ptr [[C]], i32 0, i32 0
// EVAL-FN-NEXT:    store i32 4, ptr [[TMP0]], align 8
// EVAL-FN-NEXT:    [[TMP1:%.*]] = getelementptr inbounds nuw [[STRUCT_AGGCTOR]], ptr [[C]], i32 0, i32 1
// EVAL-FN-NEXT:    store i64 10, ptr [[TMP1]], align 8
// EVAL-FN-NEXT:    [[A:%.*]] = getelementptr inbounds nuw [[STRUCT_AGGCTOR]], ptr [[C]], i32 0, i32 0
// EVAL-FN-NEXT:    [[TMP2:%.*]] = load i32, ptr [[A]], align 8
// EVAL-FN-NEXT:    [[CONV:%.*]] = sext i32 [[TMP2]] to i64
// EVAL-FN-NEXT:    [[B:%.*]] = getelementptr inbounds nuw [[STRUCT_AGGCTOR]], ptr [[C]], i32 0, i32 1
// EVAL-FN-NEXT:    [[TMP3:%.*]] = load i64, ptr [[B]], align 8
// EVAL-FN-NEXT:    [[ADD:%.*]] = add nsw i64 [[CONV]], [[TMP3]]
// EVAL-FN-NEXT:    ret i64 [[ADD]]
//
long test_AggCtor() {
  const int i = 2;
  AggCtor C(i);
  return C.a + C.b;
}

struct UserConv {
  consteval operator int() const noexcept { return 42; }
};

// EVAL-FN-LABEL: @_Z13test_UserConvv(
// EVAL-FN-NEXT:  entry:
// EVAL-FN-NEXT:    ret i32 42
//
int test_UserConv() {
  return UserConv();
}

// EVAL-FN-LABEL: @_Z28test_UserConvOverload_helperi(
// EVAL-FN-NEXT:  entry:
// EVAL-FN-NEXT:    [[A_ADDR:%.*]] = alloca i32, align 4
// EVAL-FN-NEXT:    store i32 [[A:%.*]], ptr [[A_ADDR]], align 4
// EVAL-FN-NEXT:    [[TMP0:%.*]] = load i32, ptr [[A_ADDR]], align 4
// EVAL-FN-NEXT:    ret i32 [[TMP0]]
//
int test_UserConvOverload_helper(int a) { return a; }

// EVAL-FN-LABEL: @_Z21test_UserConvOverloadv(
// EVAL-FN-NEXT:  entry:
// EVAL-FN-NEXT:    [[CALL:%.*]] = call noundef i32 @_Z28test_UserConvOverload_helperi(i32 noundef 42)
// EVAL-FN-NEXT:    ret i32 [[CALL]]
//
int test_UserConvOverload() {
  return test_UserConvOverload_helper(UserConv());
}

consteval int test_UserConvOverload_helper_ceval(int a) { return a; }

// EVAL-FN-LABEL: @_Z27test_UserConvOverload_cevalv(
// EVAL-FN-NEXT:  entry:
// EVAL-FN-NEXT:    ret i32 42
//
int test_UserConvOverload_ceval() {
  return test_UserConvOverload_helper_ceval(UserConv());
}

consteval void void_test() {}
void void_call() { // EVAL-FN-LABEL: define {{.*}} @_Z9void_call
  // EVAL-FN-NOT: call
  void_test();
  // EVAL-FN: {{^}}}
}


namespace GH82154 {
struct S1 { consteval S1(int) {} };
struct S3 { constexpr S3(int) {} };

void f() {
    struct S2 {
        S1 s = 0;
        S3 s2 = 0;
    };
    S2 s;
    // EVAL-FN-LABEL: define {{.*}} void @_ZZN7GH821541fEvEN2S2C2Ev
    // EVAL-FN-NOT: call void @_ZN7GH821542S1C2Ei
    // EVAL-FN:     call void @_ZN7GH821542S3C2Ei
}
}

namespace GH93040 {
struct C { char c = 1; };
struct Empty { consteval Empty() {} };
struct Empty2 { consteval Empty2() {} };
struct Test : C, Empty {
  [[no_unique_address]] Empty2 e;
};
static_assert(sizeof(Test) == 1);
void f() {
  Test test;

// Make sure we don't overwrite the initialization of c.

// EVAL-FN-LABEL: define {{.*}} void @_ZN7GH930404TestC2Ev
// EVAL-FN: entry:
// EVAL-FN-NEXT:  [[THIS_ADDR:%.*]] = alloca ptr, align 8
// EVAL-FN-NEXT:  store ptr {{.*}}, ptr [[THIS_ADDR]], align 8
// EVAL-FN-NEXT:  [[THIS:%.*]] = load ptr, ptr [[THIS_ADDR]], align 8
// EVAL-FN-NEXT:  call void @_ZN7GH930401CC2Ev(ptr noundef nonnull align 1 dereferenceable(1) [[THIS]])
// EVAL-FN-NEXT:  ret void
}
}
