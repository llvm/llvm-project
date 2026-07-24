// RUN: %clang_cc1 -x c -ffreestanding -triple x86_64-unknown-linux -Wno-implicit-function-declaration -fclangir -emit-cir -o %t.cir %s
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -x c -ffreestanding -triple x86_64-unknown-linux -Wno-implicit-function-declaration -fclangir -emit-llvm -o %t.ll %s
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -ffreestanding -triple=x86_64-pc-linux-gnu -emit-llvm -Wall -Werror %s -o - | FileCheck %s -check-prefix=OGCG

// Test that __builtin_cpu_is emits the correct ABI value and field offset for
// every vendor (field offset 0) in llvm/include/llvm/TargetParser/
// X86TargetParser.def.
extern void a(const char *);

// CIR: !rec_anon_struct = !cir.struct<{!u32i, !u32i, !u32i, !cir.array<!u32i x 1>}>
// CIR: cir.global "private" external dso_local @__cpu_model : !rec_anon_struct
// LLVM: @__cpu_model = external dso_local global { i32, i32, i32, [1 x i32] }
// OGCG: @__cpu_model = external dso_local global { i32, i32, i32, [1 x i32] }

#define TEST_CPU_IS(NAME, STR)                                                 \
  void test_##NAME(void) {                                                     \
    if (__builtin_cpu_is(STR))                                                 \
      a(STR);                                                                  \
  }
// CIR-LABEL: cir.func no_inline dso_local @test_intel()
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][0] {name = "__cpu_vendor"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALUE:%.]] = cir.load align(4) [[CPUTYPE]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<1> : !u32i
// CIR-NEXT: {{.*}} = cir.cmp eq [[VALUE]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_intel(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_model
// LLVM: = icmp eq i32 [[LOAD]], 1

// OGCG-LABEL: define{{.*}} void @test_intel(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_model
// OGCG: = icmp eq i32 [[LOAD]], 1
TEST_CPU_IS(intel, "intel")

// CIR-LABEL: cir.func no_inline dso_local @test_amd()
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][0] {name = "__cpu_vendor"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALUE:%.]] = cir.load align(4) [[CPUTYPE]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<2> : !u32i
// CIR-NEXT: {{.*}} = cir.cmp eq [[VALUE]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_amd(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_model
// LLVM: = icmp eq i32 [[LOAD]], 2

// OGCG-LABEL: define{{.*}} void @test_amd(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_model
// OGCG: = icmp eq i32 [[LOAD]], 2
TEST_CPU_IS(amd, "amd")

// CIR-LABEL: cir.func no_inline dso_local @test_other()
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][0] {name = "__cpu_vendor"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALUE:%.]] = cir.load align(4) [[CPUTYPE]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<5> : !u32i
// CIR-NEXT: {{.*}} = cir.cmp eq [[VALUE]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_other(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_model
// LLVM: = icmp eq i32 [[LOAD]], 5

// OGCG-LABEL: define{{.*}} void @test_other(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_model
// OGCG: = icmp eq i32 [[LOAD]], 5
TEST_CPU_IS(other, "other")

