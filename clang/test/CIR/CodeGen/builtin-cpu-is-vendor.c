// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fclangir -emit-cir %s -o - | FileCheck %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM

// Test that __builtin_cpu_is emits the correct ABI value and field offset for
// every vendor (field offset 0) in llvm/include/llvm/TargetParser/
// X86TargetParser.def.
extern void a(const char *);

// CIR: ![[MODEL_TY:.*]] = !cir.struct<{!u32i, !u32i, !u32i, !cir.array<!u32i x 1>}>
// CIR: cir.global "private" external dso_local @__cpu_model : ![[MODEL_TY]]
// LLVM: @__cpu_model = external dso_local global { i32, i32, i32, [1 x i32] }

#define TEST_CPU_IS(NAME, STR)                                                 \
  void test_##NAME(void) {                                                     \
    if (__builtin_cpu_is(STR))                                                 \
      a(STR);                                                                  \
  }

// CIR-LABEL: cir.func {{.*}}@test_intel()
// CIR: %[[GET_MODEL:.*]] = cir.get_global @__cpu_model : !cir.ptr<![[MODEL_TY]]>
// CIR: %[[GET_VENDOR:.*]] = cir.get_member %[[GET_MODEL]][0] {name = "__cpu_vendor"} : !cir.ptr<![[MODEL_TY]]> -> !cir.ptr<!u32i>
// CIR: %[[LOAD_VENDOR:.*]] = cir.load {{.*}}%[[GET_VENDOR]] : !cir.ptr<!u32i>, !u32i
// CIR: %[[MASK:.*]] = cir.const #cir.int<1> : !u32i
// CIR: cir.cmp eq %[[LOAD_VENDOR]], %[[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_intel(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_model
// LLVM: = icmp eq i32 [[LOAD]], 1
TEST_CPU_IS(intel, "intel")

// CIR-LABEL: cir.func {{.*}}@test_amd()
// CIR: %[[GET_MODEL:.*]] = cir.get_global @__cpu_model : !cir.ptr<![[MODEL_TY]]>
// CIR: %[[GET_VENDOR:.*]] = cir.get_member %[[GET_MODEL]][0] {name = "__cpu_vendor"} : !cir.ptr<![[MODEL_TY]]> -> !cir.ptr<!u32i>
// CIR: %[[LOAD_VENDOR:.*]] = cir.load {{.*}}%[[GET_VENDOR]] : !cir.ptr<!u32i>, !u32i
// CIR: %[[MASK:.*]] = cir.const #cir.int<2> : !u32i
// CIR: cir.cmp eq %[[LOAD_VENDOR]], %[[MASK]] : !u32i
//
// LLVM-LABEL: define{{.*}} void @test_amd(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_model
// LLVM: = icmp eq i32 [[LOAD]], 2
TEST_CPU_IS(amd, "amd")

// CIR-LABEL: cir.func {{.*}}@test_other()
// CIR: %[[GET_MODEL:.*]] = cir.get_global @__cpu_model : !cir.ptr<![[MODEL_TY]]>
// CIR: %[[GET_VENDOR:.*]] = cir.get_member %[[GET_MODEL]][0] {name = "__cpu_vendor"} : !cir.ptr<![[MODEL_TY]]> -> !cir.ptr<!u32i>
// CIR: %[[LOAD_VENDOR:.*]] = cir.load {{.*}}%[[GET_VENDOR]] : !cir.ptr<!u32i>, !u32i
// CIR: %[[MASK:.*]] = cir.const #cir.int<5> : !u32i
// CIR: cir.cmp eq %[[LOAD_VENDOR]], %[[MASK]] : !u32i
//
// LLVM-LABEL: define{{.*}} void @test_other(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_model
// LLVM: = icmp eq i32 [[LOAD]], 5
TEST_CPU_IS(other, "other")
