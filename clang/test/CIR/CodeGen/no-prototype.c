// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c11 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c11 -fclangir -emit-llvm %s -o - | FileCheck --check-prefix=LLVM %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c11 -fclangir -fno-clangir-call-conv-lowering \
// RUN:   -DTEST_WIDE -emit-llvm %s -o - | FileCheck --check-prefix=WIDE %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c11 -emit-llvm %s -o - | FileCheck --check-prefix=CLASSIC %s
// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -std=c11 -fclangir -emit-llvm -DTEST_ZERO %s -o - | FileCheck --check-prefix=FIXED %s
// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -std=c11 -fclangir -clangir-disable-passes -emit-cir -DTEST_ZERO %s -o - | FileCheck --check-prefix=WIN %s

#ifndef TEST_ZERO

//===----------------------------------------------------------------------===//
// DEFINED BEHAVIOUR
//===----------------------------------------------------------------------===//

// No-proto definition followed by a correct call.
int noProto0(x) int x; { return x; }
// CHECK: cir.func {{.*}} no_proto {{.*}} @noProto0(%arg0: !s32i {{.+}}) -> !s32i
int test0(int x) {
  // CHECK-LABEL: cir.func {{.*}} @test0(
  // CHECK: cir.call @noProto0(%{{.+}}) : (!s32i {llvm.noundef})
  return noProto0(x);
}

// Declaration without prototype followed by its definition, then a correct call.
//
// Prototyped definition overrides no-proto declaration before any call is made,
// only allowing calls with proper arguments. This is the only case where the
// definition is not marked as no-proto.
int noProto1();
int noProto1(int x) { return x; }
// CHECK: cir.func {{.*}} @noProto1(%arg0: !s32i {{.+}}) -> !s32i
int test1(int x) {
  // CHECK: cir.func {{.*}} @test1
  return noProto1(x);
  // CHECK: %{{.+}} = cir.call @noProto1(%{{[0-9]+}}) : (!s32i {{.*}}) -> !s32i
}

// Declaration without prototype followed by a correct call, then its definition.
//
// A matching definition allows CIR to use a direct call.
int noProto2();
int test2(int x) {
  // CHECK-LABEL: cir.func {{.*}} @test2(
  // CHECK: cir.call @noProto2(%{{.+}}) : (!s32i {llvm.noundef})
  return noProto2(x);
}
int noProto2(int x) { return x; }
// CHECK: cir.func {{.*}} no_proto {{.*}} @noProto2(%arg0: !s32i {{.+}}) -> !s32i

// No-proto declaration without definition (any call here is "correct").
//
// The call signature includes the promoted arguments and the ellipsis.
int noProto3();
int test3(short x, float y) {
  // CHECK-LABEL: cir.func {{.*}} @test3(
  // CHECK: [[GGO:%.*]] = cir.get_global @noProto3 : !cir.ptr<!cir.func<(...) -> !s32i>>
  // CHECK: [[CAST:%.*]] = cir.cast bitcast [[GGO]] : !cir.ptr<!cir.func<(...) -> !s32i>> -> !cir.ptr<!cir.func<(!s32i, !cir.double, ...) -> !s32i>>
  // CHECK: cir.call [[CAST]](%{{[0-9]+}}, %{{[0-9]+}}) : (!cir.ptr<!cir.func<(!s32i, !cir.double, ...) -> !s32i>>
  // LLVM-LABEL: define{{.*}} @test3(
  // LLVM: call i32 (i32, double, ...) @noProto3(i32{{.*}}, double{{.*}})
  // CLASSIC-LABEL: define{{.*}} @test3(
  // CLASSIC: call i32 (i32, double, ...) @noProto3(i32{{.*}}, double{{.*}})
  return noProto3(x, y);
}


//===----------------------------------------------------------------------===//
// UNDEFINED BEHAVIOUR
//
// No-proto definitions followed by incorrect calls.
//===----------------------------------------------------------------------===//

// No-proto definition followed by an incorrect call due to extra args.
int noProto4() { return 0; }
// cir.func {{.*}} no_proto {{.*}} @noProto4() -> !s32i
int test4(int x) {
  return noProto4(x); // Even if we know the definition, this should compile.
  // CHECK:  [[GGO:%.*]] = cir.get_global @noProto4 : !cir.ptr<!cir.func<() -> !s32i>>
  // CHECK:  [[CAST:%.*]] = cir.cast bitcast [[GGO]] : !cir.ptr<!cir.func<() -> !s32i>> -> !cir.ptr<!cir.func<(!s32i, ...) -> !s32i>>
  // CHECK:  {{%.*}} = cir.call [[CAST]]({{%.*}}) : (!cir.ptr<!cir.func<(!s32i, ...) -> !s32i>>, !s32i {{.*}}) -> !s32i
}

// No-proto declaration and call with too few arguments for the later definition.
int noProto5();
int test5(int x) {
  // LLVM-LABEL: define{{.*}} @test5(
  // LLVM: call i32 (...) @noProto5()
  return noProto5();
  // CHECK:  [[GGO:%.*]] = cir.get_global @noProto5 : !cir.ptr<!cir.func<(!s32i) -> !s32i>>
  // CHECK:  [[CAST:%.*]] = cir.cast bitcast [[GGO]] : !cir.ptr<!cir.func<(!s32i) -> !s32i>> -> !cir.ptr<!cir.func<(...) -> !s32i>>
  // CHECK:  {{%.*}} = cir.call [[CAST]]() : (!cir.ptr<!cir.func<(...) -> !s32i>>) -> !s32i
}
int noProto5(int x) { return x; }
// CHECK: cir.func {{.*}} no_proto {{.*}} @noProto5(%arg0: !s32i {{.+}}) -> !s32i

// A compatible definition permits a direct call and inlining at -O0.
static __attribute__((always_inline)) int inline_before(x) int x; {
  return x + 1;
}

int test_inline_before(int x) {
  // LLVM-LABEL: define{{.*}} @test_inline_before(
  // LLVM-NOT: call
  // LLVM: add nsw i32 {{.*}}, 1
  // LLVM: ret i32
  return inline_before(x);
}

static __attribute__((always_inline)) int inline_after();

int test_inline_after(int x) {
  // LLVM-LABEL: define{{.*}} @test_inline_after(
  // LLVM-NOT: call
  // LLVM: add nsw i32 {{.*}}, 2
  // LLVM: ret i32
  return inline_after(x);
}

static int inline_after(x) int x; { return x + 2; }

// The positive signed argument can be passed to the corresponding unsigned
// parameter of the old-style definition.
static __attribute__((always_inline)) unsigned inline_unsigned();

unsigned test_inline_unsigned(void) {
  // LLVM-LABEL: define{{.*}} @test_inline_unsigned(
  // LLVM-NOT: call
  // LLVM: add i32 {{.*}}, 3
  // LLVM: ret i32
  return inline_unsigned(1);
}

static unsigned inline_unsigned(x) unsigned x; { return x + 3; }

// Wide register arguments use fixed call types with caller-local AVX.
typedef float v8f __attribute__((vector_size(32)));
typedef float v16f __attribute__((vector_size(64)));

__attribute__((target("avx"))) int test_local_avx_scalar(int (*fn)(), double x) {
  // LLVM-LABEL: define{{.*}} @test_local_avx_scalar(
  // LLVM: call i32 (double, ...) %{{[0-9]+}}(double
  return fn(x);
}

__attribute__((target("avx"))) int test_local_avx(int (*fn)(), v8f x) {
  // LLVM-LABEL: define{{.*}} @test_local_avx(
  // LLVM: call i32 %{{[0-9]+}}(<8 x float>
  return fn(x);
}

__attribute__((target("avx512f"))) int test_local_avx512(int (*fn)(), v16f x) {
  // LLVM-LABEL: define{{.*}} @test_local_avx512(
  // LLVM: call i32 %{{[0-9]+}}(<16 x float>
  return fn(x);
}

#ifdef TEST_WIDE
// Without CIR call-convention lowering, the backend can split a wide vector
// across several registers in a fixed call.
typedef float v32f __attribute__((vector_size(128)));

__attribute__((target("avx"))) int test_local_avx_wide(int (*fn)(), v32f x) {
  // WIDE-LABEL: define{{.*}} @test_local_avx_wide(
  // WIDE: call i32 %{{[0-9]+}}(<32 x float>
  return fn(x);
}

// The x86 backend widens packed Boolean lanes to bytes in the call ABI.
typedef _Bool bool32 __attribute__((ext_vector_type(32)));
int wide_bool_target();

__attribute__((target("avx512f"))) int test_local_avx_bool(bool32 x) {
  // WIDE-LABEL: define{{.*}} @test_local_avx_bool(
  // WIDE: call i32 @wide_bool_target(<32 x i1>
  return wide_bool_target(x);
}
#endif

#endif // TEST_ZERO

// No prototype or definition is available at a call with no arguments.
int noProtoZero();
int testZero(void) {
  // LLVM-LABEL: define{{.*}} @testZero(
  // LLVM: call i32 (...) @noProtoZero()
  // CLASSIC-LABEL: define{{.*}} @testZero(
  // CLASSIC: call i32 (...) @noProtoZero()
  // FIXED-LABEL: define{{.*}} @testZero(
  // FIXED: call i32 @noProtoZero()
  // WIN-LABEL: cir.func {{.*}} @testZero(
  // WIN: cir.call %{{[0-9]+}}() : (!cir.ptr<!cir.func<() -> !s32i>>) -> !s32i
  return noProtoZero();
}
