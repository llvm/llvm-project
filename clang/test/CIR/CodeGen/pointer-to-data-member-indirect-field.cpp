// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir -Wno-unused-value -emit-cir -mmlir -mlir-print-ir-before=cir-cxxabi-lowering %s -o %t.cir 2> %t-before.cir
// RUN: FileCheck --check-prefix=CIR-BEFORE --input-file=%t-before.cir %s
// RUN: FileCheck --check-prefix=CIR-AFTER --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir -Wno-unused-value -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

struct A {
  int b;
  union { int c; };
};

int A::*pt_anon_union_member = &A::c;
// CIR-BEFORE: cir.global external @pt_anon_union_member = #cir.data_member<[1, 0]> : !cir.data_member<!s32i in !rec_A>
// CIR-AFTER: cir.global external @pt_anon_union_member = #cir.int<4> : !s64i
// LLVM: @pt_anon_union_member = global i64 4

struct B {
  int b;
  union {
    struct {
      int c;
      int e;
    };
    float f;
  };
};

int B::*pt_nested_anon_first = &B::c;
// CIR-BEFORE: cir.global external @pt_nested_anon_first = #cir.data_member<[1, 0, 0]> : !cir.data_member<!s32i in !rec_B>
// CIR-AFTER: cir.global external @pt_nested_anon_first = #cir.int<4> : !s64i
// LLVM: @pt_nested_anon_first = global i64 4

int B::*pt_nested_anon_second = &B::e;
// CIR-BEFORE: cir.global external @pt_nested_anon_second = #cir.data_member<[1, 0, 1]> : !cir.data_member<!s32i in !rec_B>
// CIR-AFTER: cir.global external @pt_nested_anon_second = #cir.int<8> : !s64i
// LLVM: @pt_nested_anon_second = global i64 8

static union { int gx; float gy; };
// CIR-BEFORE: cir.global "private" internal dso_local @_Z2gx = #cir.zero : !rec_anon2E3 {alignment = 4 : i64}
// CIR-AFTER: cir.global "private" internal dso_local @_Z2gx = #cir.zero : !rec_anon2E3 {alignment = 4 : i64}
// LLVM: @_Z2gx = internal global %union.anon{{.*}} zeroinitializer, align 4

int test_use(A &a, int A::*member) {
  return a.*member;
}

// CIR-BEFORE-LABEL: cir.func {{.*}}@_Z8test_useR1AMS_i(
// CIR-BEFORE: %[[LOAD_ARG:.*]] = cir.load %{{.*}} : !cir.ptr<!cir.ptr<!rec_A>>, !cir.ptr<!rec_A>
// CIR-BEFORE: %[[LOAD_PTR:.*]] = cir.load align(8) %{{.*}} : !cir.ptr<!cir.data_member<!s32i in !rec_A>>, !cir.data_member<!s32i in !rec_A>
// CIR-BEFORE: cir.get_runtime_member %[[LOAD_ARG]][%[[LOAD_PTR]] : !cir.data_member<!s32i in !rec_A>] : !cir.ptr<!rec_A> -> !cir.ptr<!s32i>

// CIR-AFTER-LABEL: cir.func {{.*}}@_Z8test_useR1AMS_i(
// CIR-AFTER: %[[LOAD_ARG:.*]] = cir.load %{{.*}} : !cir.ptr<!cir.ptr<!rec_A>>, !cir.ptr<!rec_A>
// CIR-AFTER: %[[LOAD_PTR:.*]] = cir.load align(8) %{{.*}} : !cir.ptr<!s64i>, !s64i
// CIR-AFTER: %[[ARG_TO_CHARPTR:.*]] = cir.cast bitcast %[[LOAD_ARG]] : !cir.ptr<!rec_A> -> !cir.ptr<!s8i>
// CIR-AFTER: %[[OFFSET:.*]] = cir.ptr_stride %[[ARG_TO_CHARPTR]], %[[LOAD_PTR]] : (!cir.ptr<!s8i>, !s64i) -> !cir.ptr<!s8i>
// CIR-AFTER: cir.cast bitcast %[[OFFSET]] : !cir.ptr<!s8i> -> !cir.ptr<!s32i>

// LLVM-LABEL: define {{.*}}i32 @_Z8test_useR1AMS_i(
// LLVM: %[[LOAD_ARG:.*]] = load ptr, ptr %{{.*}}, align 8
// LLVM: %[[LOAD_PTR:.*]] = load i64, ptr %{{.*}}, align 8
// LLVM: getelementptr {{.*}}i8, ptr %[[LOAD_ARG]], i64 %[[LOAD_PTR]]

int test_call_use(A &a) {
  return test_use(a, &A::c);
}

// CIR-BEFORE-LABEL: cir.func {{.*}}@_Z13test_call_useR1A
// CIR-BEFORE: %[[MEMBER:.*]] = cir.const #cir.data_member<[1, 0]> : !cir.data_member<!s32i in !rec_A>
// CIR-BEFORE: cir.call @_Z8test_useR1AMS_i(%{{.*}}, %[[MEMBER]])

// CIR-AFTER-LABEL: cir.func {{.*}}@_Z13test_call_useR1A
// CIR-AFTER:   %[[MEMBER:.*]] = cir.const #cir.int<4> : !s64i
// CIR-AFTER:   cir.call @_Z8test_useR1AMS_i({{.*}}, %[[MEMBER]])

// LLVM-LABEL: define {{.*}} i32 @_Z13test_call_useR1A(
// LLVM:   call {{.*}} i32 @_Z8test_useR1AMS_i({{.*}}, i64 4)

int use_global_anon_union() { return gy; }

// CIR-BEFORE-LABEL: cir.func {{.*}}@_Z21use_global_anon_unionv()
// CIR-BEFORE: %[[GET_GLOB:.*]] = cir.get_global @_Z2gx : !cir.ptr<!rec_anon2E3>
// CIR-BEFORE: cir.get_member %[[GET_GLOB]][1] {name = "gy"} : !cir.ptr<!rec_anon2E3> -> !cir.ptr<!cir.float>

// CIR-AFTER-LABEL: cir.func {{.*}}@_Z21use_global_anon_unionv()
// CIR-AFTER: %[[GET_GLOB:.*]] = cir.get_global @_Z2gx : !cir.ptr<!rec_anon2E3>
// CIR-AFTER: cir.get_member %[[GET_GLOB]][1] {name = "gy"} : !cir.ptr<!rec_anon2E3> -> !cir.ptr<!cir.float>

// LLVM-LABEL: define {{.*}}i32 @_Z21use_global_anon_unionv()
// LLVM: load float, ptr @_Z2gx, align 4
