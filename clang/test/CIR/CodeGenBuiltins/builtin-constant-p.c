// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

int a = 42;

/* --- Compound literals */

struct foo { int x, y; };

int y;
struct foo f = (struct foo){ __builtin_constant_p(y), 42 };

// CIR: cir.global external @f = #cir.const_record<{#cir.int<0> : !s32i, #cir.int<42> : !s32i}> : !rec_foo
// LLVM: @f = global %struct.foo { i32 0, i32 42 }
// OGCG: @f = global %struct.foo { i32 0, i32 42 }

struct foo test0(int expr) {
  struct foo f = (struct foo){ __builtin_constant_p(expr), 42 };
  return f;
}

// CIR: cir.func {{.*}} @test0(%[[ARG0:.*]]: !s32i {{.*}}) -> !rec_foo
// CIR:   %[[EXPR_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["expr", init]
// CIR:   cir.store %[[ARG0]], %[[EXPR_ADDR]]
// CIR:   %[[EXPR:.*]] = cir.load{{.*}} %[[EXPR_ADDR]]
// CIR:   %[[IS_CONSTANT:.*]] = cir.is_constant %[[EXPR]] : !s32i -> !cir.bool

// LLVM: define{{.*}} %struct.foo @test0(i32 %[[ARG0:.*]])
// LLVM:   %[[EXPR_ADDR:.*]] = alloca i32
// LLVM:   store i32 %[[ARG0]], ptr %[[EXPR_ADDR]]
// LLVM:   %[[EXPR:.*]] = load i32, ptr %[[EXPR_ADDR]]
// LLVM:   %[[IS_CONSTANT:.*]] = call i1 @llvm.is.constant.i32(i32 %[[EXPR]])

// OGCG: define{{.*}} i64 @test0(i32 {{.*}} %[[ARG0:.*]])
// OGCG:   %[[EXPR_ADDR:.*]] = alloca i32
// OGCG:   store i32 %[[ARG0]], ptr %[[EXPR_ADDR]]
// OGCG:   %[[EXPR:.*]] = load i32, ptr %[[EXPR_ADDR]]
// OGCG:   %[[IS_CONSTANT:.*]] = call i1 @llvm.is.constant.i32(i32 %[[EXPR]])

/* --- Pointer types */

int test1(void) {
  return __builtin_constant_p(&a - 13);
}

// CIR: cir.func {{.*}} @test1() -> !s32i
// CIR:   %[[TMP1:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"]
// CIR:   %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR:   cir.store %[[ZERO]], %[[TMP1]] : !s32i, !cir.ptr<!s32i>
// CIR:   %[[TMP2:.*]] = cir.load %[[TMP1]] : !cir.ptr<!s32i>, !s32i
// CIR:   cir.return %[[TMP2]] : !s32i

// LLVM: define{{.*}} i32 @test1()
// LLVM:   %[[TMP1:.*]] = alloca i32
// LLVM:   store i32 0, ptr %[[TMP1]]
// LLVM:   %[[TMP2:.*]] = load i32, ptr %[[TMP1]]
// LLVM:   ret i32 %[[TMP2]]

// OGCG: define{{.*}} i32 @test1()
// OGCG:   ret i32 0

/* --- Aggregate types */

int b[] = {1, 2, 3};

int test2(void) {
  return __builtin_constant_p(b);
}

// CIR: cir.func {{.*}} @test2() -> !s32i
// CIR:   %[[TMP1:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"]
// CIR:   %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR:   cir.store %[[ZERO]], %[[TMP1]] : !s32i, !cir.ptr<!s32i>
// CIR:   %[[TMP2:.*]] = cir.load %[[TMP1]] : !cir.ptr<!s32i>, !s32i
// CIR:   cir.return %[[TMP2]] : !s32i

// LLVM: define{{.*}} i32 @test2()
// LLVM:   %[[TMP1:.*]] = alloca i32
// LLVM:   store i32 0, ptr %[[TMP1]]
// LLVM:   %[[TMP2:.*]] = load i32, ptr %[[TMP1]]
// LLVM:   ret i32 %[[TMP2]]

// OGCG: define{{.*}} i32 @test2()
// OGCG:   ret i32 0

const char test3_c[] = {1, 2, 3, 0};

int test3(void) {
  return __builtin_constant_p(test3_c);
}

// CIR: cir.func {{.*}} @test3() -> !s32i
// CIR:   %[[TMP1:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"]
// CIR:   %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR:   cir.store %[[ZERO]], %[[TMP1]] : !s32i, !cir.ptr<!s32i>
// CIR:   %[[TMP2:.*]] = cir.load %[[TMP1]] : !cir.ptr<!s32i>, !s32i
// CIR:   cir.return %[[TMP2]] : !s32i

// LLVM: define{{.*}} i32 @test3()
// LLVM:   %[[TMP1:.*]] = alloca i32
// LLVM:   store i32 0, ptr %[[TMP1]]
// LLVM:   %[[TMP2:.*]] = load i32, ptr %[[TMP1]]
// LLVM:   ret i32 %[[TMP2]]

// OGCG: define{{.*}} i32 @test3()
// OGCG:   ret i32 0

inline char test4_i(const char *x) {
  return x[1];
}

int test4(void) {
  return __builtin_constant_p(test4_i(test3_c));
}

// CIR: cir.func {{.*}} @test4() -> !s32i
// CIR:   %[[TMP1:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"]
// CIR:   %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR:   cir.store %[[ZERO]], %[[TMP1]] : !s32i, !cir.ptr<!s32i>
// CIR:   %[[TMP2:.*]] = cir.load %[[TMP1]] : !cir.ptr<!s32i>, !s32i
// CIR:   cir.return %[[TMP2]] : !s32i

// LLVM: define{{.*}} i32 @test4()
// LLVM:   %[[TMP1:.*]] = alloca i32
// LLVM:   store i32 0, ptr %[[TMP1]]
// LLVM:   %[[TMP2:.*]] = load i32, ptr %[[TMP1]]
// LLVM:   ret i32 %[[TMP2]]

// OGCG: define{{.*}} i32 @test4()
// OGCG:   ret i32 0

/* --- Constant global variables */

const int c = 42;

int test5(void) {
  return __builtin_constant_p(c);
}

// CIR: cir.func {{.*}} @test5() -> !s32i
// CIR:   %[[TMP1:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"]
// CIR:   %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CIR:   cir.store %[[ONE]], %[[TMP1]] : !s32i, !cir.ptr<!s32i>
// CIR:   %[[TMP2:.*]] = cir.load %[[TMP1]] : !cir.ptr<!s32i>, !s32i
// CIR:   cir.return %[[TMP2]] : !s32i

// LLVM: define{{.*}} i32 @test5()
// LLVM:   %[[TMP1:.*]] = alloca i32
// LLVM:   store i32 1, ptr %[[TMP1]]
// LLVM:   %[[TMP2:.*]] = load i32, ptr %[[TMP1]]
// LLVM:   ret i32 %[[TMP2]]

// OGCG: define{{.*}} i32 @test5()
// OGCG:   ret i32 1

/* --- Array types */

int arr[] = { 1, 2, 3 };

int test6(void) {
  return __builtin_constant_p(arr[2]);
}

// CIR: cir.func {{.*}} @test6() -> !s32i
// CIR:   %[[TWO:.*]] = cir.const #cir.int<2> : !s32i
// CIR:   %[[ARR:.*]] = cir.get_global @arr : !cir.ptr<!cir.array<!s32i x 3>>
// CIR:   %[[ELE_PTR:.*]] = cir.get_element %[[ARR]][%[[TWO]]] : (!cir.ptr<!cir.array<!s32i x 3>>, !s32i) -> !cir.ptr<!s32i>
// CIR:   %[[ELE:.*]] = cir.load{{.*}} %[[ELE_PTR]] : !cir.ptr<!s32i>, !s32i
// CIR:   %[[IS_CONSTANT:.*]] = cir.is_constant %[[ELE]] : !s32i -> !cir.bool

// LLVM: define {{.*}} i32 @test6()
// LLVM:   %[[TMP1:.*]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @arr, i64 8)
// LLVM:   %[[TMP2:.*]] = call i1 @llvm.is.constant.i32(i32 %[[TMP1]])

// OGCG: define {{.*}} i32 @test6()
// OGCG:   %[[TMP1:.*]] = load i32, ptr getelementptr inbounds ([3 x i32], ptr @arr, i64 0, i64 2)
// OGCG:   %[[TMP2:.*]] = call i1 @llvm.is.constant.i32(i32 %[[TMP1]])

const int c_arr[] = { 1, 2, 3 };

int test7(void) {
  return __builtin_constant_p(c_arr[2]);
}

// CIR: cir.func {{.*}} @test7() -> !s32i
// CIR:   %[[TMP1:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"]
// CIR:   %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CIR:   cir.store %[[ONE]], %[[TMP1]] : !s32i, !cir.ptr<!s32i>
// CIR:   %[[TMP2:.*]] = cir.load %[[TMP1]] : !cir.ptr<!s32i>, !s32i
// CIR:   cir.return %[[TMP2]] : !s32i

// LLVM: define{{.*}} i32 @test7()
// LLVM:   %[[TMP1:.*]] = alloca i32
// LLVM:   store i32 1, ptr %[[TMP1]]
// LLVM:   %[[TMP2:.*]] = load i32, ptr %[[TMP1]]
// LLVM:   ret i32 %[[TMP2]]

// OGCG: define{{.*}} i32 @test7()
// OGCG:   ret i32 1

int test8(void) {
  return __builtin_constant_p(c_arr);
}

// CIR: cir.func {{.*}} @test8() -> !s32i
// CIR:   %[[TMP1:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"]
// CIR:   %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR:   cir.store %[[ZERO]], %[[TMP1]] : !s32i, !cir.ptr<!s32i>
// CIR:   %[[TMP2:.*]] = cir.load %[[TMP1]] : !cir.ptr<!s32i>, !s32i
// CIR:   cir.return %[[TMP2]] : !s32i

// LLVM: define{{.*}} i32 @test8()
// LLVM:   %[[TMP1:.*]] = alloca i32
// LLVM:   store i32 0, ptr %[[TMP1]]
// LLVM:   %[[TMP2:.*]] = load i32, ptr %[[TMP1]]
// LLVM:   ret i32 %[[TMP2]]

// OGCG: define{{.*}} i32 @test8()
// OGCG:   ret i32 0

/* --- Function pointers */

int test9(void) {
  return __builtin_constant_p(&test9);
}

// CIR: cir.func {{.*}} @test9() -> !s32i
// CIR:   %[[TMP1:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"]
// CIR:   %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR:   cir.store %[[ZERO]], %[[TMP1]] : !s32i, !cir.ptr<!s32i>
// CIR:   %[[TMP2:.*]] = cir.load %[[TMP1]] : !cir.ptr<!s32i>, !s32i
// CIR:   cir.return %[[TMP2]] : !s32i

// LLVM: define{{.*}} i32 @test9()
// LLVM:   %[[TMP1:.*]] = alloca i32
// LLVM:   store i32 0, ptr %[[TMP1]]
// LLVM:   %[[TMP2:.*]] = load i32, ptr %[[TMP1]]
// LLVM:   ret i32 %[[TMP2]]

// OGCG: define{{.*}} i32 @test9()
// OGCG:   ret i32 0

int test10(void) {
  return __builtin_constant_p(&test10 != 0);
}

// CIR: cir.func {{.*}} @test10() -> !s32i
// CIR:   %[[TMP1:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"]
// CIR:   %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CIR:   cir.store %[[ONE]], %[[TMP1]] : !s32i, !cir.ptr<!s32i>
// CIR:   %[[TMP2:.*]] = cir.load %[[TMP1]] : !cir.ptr<!s32i>, !s32i
// CIR:   cir.return %[[TMP2]] : !s32i

// LLVM: define{{.*}} i32 @test10()
// LLVM:   %[[TMP1:.*]] = alloca i32
// LLVM:   store i32 1, ptr %[[TMP1]]
// LLVM:   %[[TMP2:.*]] = load i32, ptr %[[TMP1]]
// LLVM:   ret i32 %[[TMP2]]

// OGCG: define{{.*}} i32 @test10()
// OGCG:   ret i32 1

int test11_f(void);
void test11(void) {
  int a, b;
  (void)__builtin_constant_p((a = b, test11_f()));
}

// CIR: cir.func {{.*}} @test11()
// CIR-NOT: call {{.*}}test11_f

// LLVM: define{{.*}} void @test11()
// LLVM-NOT: call {{.*}}test11_f

// OGCG: define{{.*}} void @test11()
// OGCG-NOT: call {{.*}}test11_f
