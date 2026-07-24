// RUN: %clang_cc1 -x c -ffreestanding -triple x86_64-unknown-linux -Wno-implicit-function-declaration -fclangir -emit-cir -o %t.cir %s
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -x c -ffreestanding -triple x86_64-unknown-linux -Wno-implicit-function-declaration -fclangir -emit-llvm -o %t.ll %s
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -ffreestanding -triple=x86_64-pc-linux-gnu -emit-llvm -Wall -Werror %s -o - | FileCheck %s -check-prefix=OGCG

// Test that __builtin_cpu_is emits the correct ABI value for every CPU
// subtype, llvm/include/llvm/TargetParser/X86TargetParser.def.
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

// CIR-LABEL: cir.func no_inline dso_local @test_nehalem()
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][2] {name = "__cpu_subtype"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALUE:%.]] = cir.load align(4) [[CPUTYPE]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<1> : !u32i
// CIR-NEXT: {{.*}} = cir.cmp eq [[VALUE]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_nehalem(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// LLVM: = icmp eq i32 [[LOAD]], 1

// OGCG-LABEL: define{{.*}} void @test_nehalem(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// OGCG: = icmp eq i32 [[LOAD]], 1
TEST_CPU_IS(nehalem, "nehalem")

// CIR-LABEL: cir.func no_inline dso_local @test_westmere()
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][2] {name = "__cpu_subtype"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALUE:%.]] = cir.load align(4) [[CPUTYPE]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<2> : !u32i
// CIR-NEXT: {{.*}} = cir.cmp eq [[VALUE]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_westmere(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// LLVM: = icmp eq i32 [[LOAD]], 2

// OGCG-LABEL: define{{.*}} void @test_westmere(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// OGCG: = icmp eq i32 [[LOAD]], 2
TEST_CPU_IS(westmere, "westmere")

// CIR-LABEL: cir.func no_inline dso_local @test_sandybridge()
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][2] {name = "__cpu_subtype"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALUE:%.]] = cir.load align(4) [[CPUTYPE]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<3> : !u32i
// CIR-NEXT: {{.*}} = cir.cmp eq [[VALUE]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_sandybridge(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// LLVM: = icmp eq i32 [[LOAD]], 3

// OGCG-LABEL: define{{.*}} void @test_sandybridge(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// OGCG: = icmp eq i32 [[LOAD]], 3
TEST_CPU_IS(sandybridge, "sandybridge")

// CIR-LABEL: cir.func no_inline dso_local @test_barcelona()
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][2] {name = "__cpu_subtype"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALUE:%.]] = cir.load align(4) [[CPUTYPE]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<4> : !u32i
// CIR-NEXT: {{.*}} = cir.cmp eq [[VALUE]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_barcelona(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// LLVM: = icmp eq i32 [[LOAD]], 4

// OGCG-LABEL: define{{.*}} void @test_barcelona(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// OGCG: = icmp eq i32 [[LOAD]], 4
TEST_CPU_IS(barcelona, "barcelona")

// CIR-LABEL: cir.func no_inline dso_local @test_shanghai()
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][2] {name = "__cpu_subtype"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALUE:%.]] = cir.load align(4) [[CPUTYPE]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<5> : !u32i
// CIR-NEXT: {{.*}} = cir.cmp eq [[VALUE]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_shanghai(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// LLVM: = icmp eq i32 [[LOAD]], 5

// OGCG-LABEL: define{{.*}} void @test_shanghai(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// OGCG: = icmp eq i32 [[LOAD]], 5
TEST_CPU_IS(shanghai, "shanghai")

// CIR-LABEL: cir.func no_inline dso_local @test_istanbul()
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][2] {name = "__cpu_subtype"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALUE:%.]] = cir.load align(4) [[CPUTYPE]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<6> : !u32i
// CIR-NEXT: {{.*}} = cir.cmp eq [[VALUE]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_istanbul(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// LLVM: = icmp eq i32 [[LOAD]], 6

// OGCG-LABEL: define{{.*}} void @test_istanbul(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// OGCG: = icmp eq i32 [[LOAD]], 6
TEST_CPU_IS(istanbul, "istanbul")

// CIR-LABEL: cir.func no_inline dso_local @test_bdver1()
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][2] {name = "__cpu_subtype"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALUE:%.]] = cir.load align(4) [[CPUTYPE]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<7> : !u32i
// CIR-NEXT: {{.*}} = cir.cmp eq [[VALUE]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_bdver1(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// LLVM: = icmp eq i32 [[LOAD]], 7

// OGCG-LABEL: define{{.*}} void @test_bdver1(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// OGCG: = icmp eq i32 [[LOAD]], 7
TEST_CPU_IS(bdver1, "bdver1")

// CIR-LABEL: cir.func no_inline dso_local @test_bdver2()
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][2] {name = "__cpu_subtype"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALUE:%.]] = cir.load align(4) [[CPUTYPE]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<8> : !u32i
// CIR-NEXT: {{.*}} = cir.cmp eq [[VALUE]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_bdver2(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// LLVM: = icmp eq i32 [[LOAD]], 8

// OGCG-LABEL: define{{.*}} void @test_bdver2(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// OGCG: = icmp eq i32 [[LOAD]], 8
TEST_CPU_IS(bdver2, "bdver2")

// CIR-LABEL: cir.func no_inline dso_local @test_bdver3()
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][2] {name = "__cpu_subtype"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALUE:%.]] = cir.load align(4) [[CPUTYPE]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<9> : !u32i
// CIR-NEXT: {{.*}} = cir.cmp eq [[VALUE]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_bdver3(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// LLVM: = icmp eq i32 [[LOAD]], 9

// OGCG-LABEL: define{{.*}} void @test_bdver3(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// OGCG: = icmp eq i32 [[LOAD]], 9
TEST_CPU_IS(bdver3, "bdver3")

// CIR-LABEL: cir.func no_inline dso_local @test_bdver4()
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][2] {name = "__cpu_subtype"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALUE:%.]] = cir.load align(4) [[CPUTYPE]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<10> : !u32i
// CIR-NEXT: {{.*}} = cir.cmp eq [[VALUE]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_bdver4(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// LLVM: = icmp eq i32 [[LOAD]], 10

// OGCG-LABEL: define{{.*}} void @test_bdver4(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// OGCG: = icmp eq i32 [[LOAD]], 10
TEST_CPU_IS(bdver4, "bdver4")

// CIR-LABEL: cir.func no_inline dso_local @test_znver1()
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][2] {name = "__cpu_subtype"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALUE:%.]] = cir.load align(4) [[CPUTYPE]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<11> : !u32i
// CIR-NEXT: {{.*}} = cir.cmp eq [[VALUE]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_znver1(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// LLVM: = icmp eq i32 [[LOAD]], 11

// OGCG-LABEL: define{{.*}} void @test_znver1(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// OGCG: = icmp eq i32 [[LOAD]], 11
TEST_CPU_IS(znver1, "znver1")

// CIR-LABEL: cir.func no_inline dso_local @test_ivybridge()
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][2] {name = "__cpu_subtype"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALUE:%.]] = cir.load align(4) [[CPUTYPE]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<12> : !u32i
// CIR-NEXT: {{.*}} = cir.cmp eq [[VALUE]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_ivybridge(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// LLVM: = icmp eq i32 [[LOAD]], 12

// OGCG-LABEL: define{{.*}} void @test_ivybridge(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// OGCG: = icmp eq i32 [[LOAD]], 12
TEST_CPU_IS(ivybridge, "ivybridge")

// CIR-LABEL: cir.func no_inline dso_local @test_haswell()
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][2] {name = "__cpu_subtype"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALUE:%.]] = cir.load align(4) [[CPUTYPE]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<13> : !u32i
// CIR-NEXT: {{.*}} = cir.cmp eq [[VALUE]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_haswell(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// LLVM: = icmp eq i32 [[LOAD]], 13

// OGCG-LABEL: define{{.*}} void @test_haswell(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// OGCG: = icmp eq i32 [[LOAD]], 13
TEST_CPU_IS(haswell, "haswell")

// CIR-LABEL: cir.func no_inline dso_local @test_broadwell()
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][2] {name = "__cpu_subtype"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALUE:%.]] = cir.load align(4) [[CPUTYPE]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<14> : !u32i
// CIR-NEXT: {{.*}} = cir.cmp eq [[VALUE]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_broadwell(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// LLVM: = icmp eq i32 [[LOAD]], 14

// OGCG-LABEL: define{{.*}} void @test_broadwell(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// OGCG: = icmp eq i32 [[LOAD]], 14
TEST_CPU_IS(broadwell, "broadwell")

// CIR-LABEL: cir.func no_inline dso_local @test_skylake()
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][2] {name = "__cpu_subtype"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALUE:%.]] = cir.load align(4) [[CPUTYPE]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<15> : !u32i
// CIR-NEXT: {{.*}} = cir.cmp eq [[VALUE]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_skylake(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// LLVM: = icmp eq i32 [[LOAD]], 15

// OGCG-LABEL: define{{.*}} void @test_skylake(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// OGCG: = icmp eq i32 [[LOAD]], 15
TEST_CPU_IS(skylake, "skylake")

// CIR-LABEL: cir.func no_inline dso_local @test_skylake_avx512()
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][2] {name = "__cpu_subtype"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALUE:%.]] = cir.load align(4) [[CPUTYPE]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<16> : !u32i
// CIR-NEXT: {{.*}} = cir.cmp eq [[VALUE]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_skylake_avx512(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// LLVM: = icmp eq i32 [[LOAD]], 16

// OGCG-LABEL: define{{.*}} void @test_skylake_avx512(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// OGCG: = icmp eq i32 [[LOAD]], 16
TEST_CPU_IS(skylake_avx512, "skylake-avx512")

// CIR-LABEL: cir.func no_inline dso_local @test_cannonlake()
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][2] {name = "__cpu_subtype"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALUE:%.]] = cir.load align(4) [[CPUTYPE]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<17> : !u32i
// CIR-NEXT: {{.*}} = cir.cmp eq [[VALUE]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_cannonlake(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// LLVM: = icmp eq i32 [[LOAD]], 17

// OGCG-LABEL: define{{.*}} void @test_cannonlake(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// OGCG: = icmp eq i32 [[LOAD]], 17
TEST_CPU_IS(cannonlake, "cannonlake")

// CIR-LABEL: cir.func no_inline dso_local @test_icelake_client()
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][2] {name = "__cpu_subtype"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALUE:%.]] = cir.load align(4) [[CPUTYPE]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<18> : !u32i
// CIR-NEXT: {{.*}} = cir.cmp eq [[VALUE]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_icelake_client(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// LLVM: = icmp eq i32 [[LOAD]], 18

// OGCG-LABEL: define{{.*}} void @test_icelake_client(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// OGCG: = icmp eq i32 [[LOAD]], 18
TEST_CPU_IS(icelake_client, "icelake-client")

// CIR-LABEL: cir.func no_inline dso_local @test_icelake_server()
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][2] {name = "__cpu_subtype"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALUE:%.]] = cir.load align(4) [[CPUTYPE]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<19> : !u32i
// CIR-NEXT: {{.*}} = cir.cmp eq [[VALUE]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_icelake_server(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// LLVM: = icmp eq i32 [[LOAD]], 19

// OGCG-LABEL: define{{.*}} void @test_icelake_server(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// OGCG: = icmp eq i32 [[LOAD]], 19
TEST_CPU_IS(icelake_server, "icelake-server")

// CIR-LABEL: cir.func no_inline dso_local @test_znver2()
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][2] {name = "__cpu_subtype"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALUE:%.]] = cir.load align(4) [[CPUTYPE]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<20> : !u32i
// CIR-NEXT: {{.*}} = cir.cmp eq [[VALUE]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_znver2(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// LLVM: = icmp eq i32 [[LOAD]], 20

// OGCG-LABEL: define{{.*}} void @test_znver2(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// OGCG: = icmp eq i32 [[LOAD]], 20
TEST_CPU_IS(znver2, "znver2")

// CIR-LABEL: cir.func no_inline dso_local @test_cascadelake()
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][2] {name = "__cpu_subtype"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALUE:%.]] = cir.load align(4) [[CPUTYPE]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<21> : !u32i
// CIR-NEXT: {{.*}} = cir.cmp eq [[VALUE]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_cascadelake(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// LLVM: = icmp eq i32 [[LOAD]], 21

// OGCG-LABEL: define{{.*}} void @test_cascadelake(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// OGCG: = icmp eq i32 [[LOAD]], 21
TEST_CPU_IS(cascadelake, "cascadelake")

// CIR-LABEL: cir.func no_inline dso_local @test_tigerlake()
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][2] {name = "__cpu_subtype"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALUE:%.]] = cir.load align(4) [[CPUTYPE]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<22> : !u32i
// CIR-NEXT: {{.*}} = cir.cmp eq [[VALUE]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_tigerlake(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// LLVM: = icmp eq i32 [[LOAD]], 22

// OGCG-LABEL: define{{.*}} void @test_tigerlake(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// OGCG: = icmp eq i32 [[LOAD]], 22
TEST_CPU_IS(tigerlake, "tigerlake")

// CIR-LABEL: cir.func no_inline dso_local @test_cooperlake()
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][2] {name = "__cpu_subtype"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALUE:%.]] = cir.load align(4) [[CPUTYPE]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<23> : !u32i
// CIR-NEXT: {{.*}} = cir.cmp eq [[VALUE]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_cooperlake(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// LLVM: = icmp eq i32 [[LOAD]], 23

// OGCG-LABEL: define{{.*}} void @test_cooperlake(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// OGCG: = icmp eq i32 [[LOAD]], 23
TEST_CPU_IS(cooperlake, "cooperlake")

// CIR-LABEL: cir.func no_inline dso_local @test_sapphirerapids()
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][2] {name = "__cpu_subtype"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALUE:%.]] = cir.load align(4) [[CPUTYPE]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<24> : !u32i
// CIR-NEXT: {{.*}} = cir.cmp eq [[VALUE]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_sapphirerapids(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// LLVM: = icmp eq i32 [[LOAD]], 24

// OGCG-LABEL: define{{.*}} void @test_sapphirerapids(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// OGCG: = icmp eq i32 [[LOAD]], 24
TEST_CPU_IS(sapphirerapids, "sapphirerapids")

// CIR-LABEL: cir.func no_inline dso_local @test_alderlake()
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][2] {name = "__cpu_subtype"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALUE:%.]] = cir.load align(4) [[CPUTYPE]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<25> : !u32i
// CIR-NEXT: {{.*}} = cir.cmp eq [[VALUE]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_alderlake(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// LLVM: = icmp eq i32 [[LOAD]], 25

// OGCG-LABEL: define{{.*}} void @test_alderlake(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// OGCG: = icmp eq i32 [[LOAD]], 25
TEST_CPU_IS(alderlake, "alderlake")

// CIR-LABEL: cir.func no_inline dso_local @test_znver3()
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][2] {name = "__cpu_subtype"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALUE:%.]] = cir.load align(4) [[CPUTYPE]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<26> : !u32i
// CIR-NEXT: {{.*}} = cir.cmp eq [[VALUE]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_znver3(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// LLVM: = icmp eq i32 [[LOAD]], 26

// OGCG-LABEL: define{{.*}} void @test_znver3(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// OGCG: = icmp eq i32 [[LOAD]], 26
TEST_CPU_IS(znver3, "znver3")

// CIR-LABEL: cir.func no_inline dso_local @test_rocketlake()
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][2] {name = "__cpu_subtype"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALUE:%.]] = cir.load align(4) [[CPUTYPE]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<27> : !u32i
// CIR-NEXT: {{.*}} = cir.cmp eq [[VALUE]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_rocketlake(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// LLVM: = icmp eq i32 [[LOAD]], 27

// OGCG-LABEL: define{{.*}} void @test_rocketlake(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// OGCG: = icmp eq i32 [[LOAD]], 27
TEST_CPU_IS(rocketlake, "rocketlake")

// CIR-LABEL: cir.func no_inline dso_local @test_zhaoxin_fam7h_lujiazui()
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][2] {name = "__cpu_subtype"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALUE:%.]] = cir.load align(4) [[CPUTYPE]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<28> : !u32i
// CIR-NEXT: {{.*}} = cir.cmp eq [[VALUE]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_zhaoxin_fam7h_lujiazui(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// LLVM: = icmp eq i32 [[LOAD]], 28

// OGCG-LABEL: define{{.*}} void @test_zhaoxin_fam7h_lujiazui(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// OGCG: = icmp eq i32 [[LOAD]], 28
TEST_CPU_IS(zhaoxin_fam7h_lujiazui, "zhaoxin_fam7h_lujiazui")

// CIR-LABEL: cir.func no_inline dso_local @test_znver4()
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][2] {name = "__cpu_subtype"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALUE:%.]] = cir.load align(4) [[CPUTYPE]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<29> : !u32i
// CIR-NEXT: {{.*}} = cir.cmp eq [[VALUE]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_znver4(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// LLVM: = icmp eq i32 [[LOAD]], 29

// OGCG-LABEL: define{{.*}} void @test_znver4(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// OGCG: = icmp eq i32 [[LOAD]], 29
TEST_CPU_IS(znver4, "znver4")

// CIR-LABEL: cir.func no_inline dso_local @test_graniterapids()
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][2] {name = "__cpu_subtype"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALUE:%.]] = cir.load align(4) [[CPUTYPE]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<30> : !u32i
// CIR-NEXT: {{.*}} = cir.cmp eq [[VALUE]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_graniterapids(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// LLVM: = icmp eq i32 [[LOAD]], 30

// OGCG-LABEL: define{{.*}} void @test_graniterapids(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// OGCG: = icmp eq i32 [[LOAD]], 30
TEST_CPU_IS(graniterapids, "graniterapids")

// CIR-LABEL: cir.func no_inline dso_local @test_graniterapids_d()
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][2] {name = "__cpu_subtype"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALUE:%.]] = cir.load align(4) [[CPUTYPE]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<31> : !u32i
// CIR-NEXT: {{.*}} = cir.cmp eq [[VALUE]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_graniterapids_d(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// LLVM: = icmp eq i32 [[LOAD]], 31

// OGCG-LABEL: define{{.*}} void @test_graniterapids_d(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// OGCG: = icmp eq i32 [[LOAD]], 31
TEST_CPU_IS(graniterapids_d, "graniterapids-d")

// CIR-LABEL: cir.func no_inline dso_local @test_arrowlake()
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][2] {name = "__cpu_subtype"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALUE:%.]] = cir.load align(4) [[CPUTYPE]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<32> : !u32i
// CIR-NEXT: {{.*}} = cir.cmp eq [[VALUE]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_arrowlake(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// LLVM: = icmp eq i32 [[LOAD]], 32

// OGCG-LABEL: define{{.*}} void @test_arrowlake(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// OGCG: = icmp eq i32 [[LOAD]], 32
TEST_CPU_IS(arrowlake, "arrowlake")

// CIR-LABEL: cir.func no_inline dso_local @test_arrowlake_s()
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][2] {name = "__cpu_subtype"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALUE:%.]] = cir.load align(4) [[CPUTYPE]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<33> : !u32i
// CIR-NEXT: {{.*}} = cir.cmp eq [[VALUE]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_arrowlake_s(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// LLVM: = icmp eq i32 [[LOAD]], 33

// OGCG-LABEL: define{{.*}} void @test_arrowlake_s(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// OGCG: = icmp eq i32 [[LOAD]], 33
TEST_CPU_IS(arrowlake_s, "arrowlake-s")

// CIR-LABEL: cir.func no_inline dso_local @test_pantherlake()
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][2] {name = "__cpu_subtype"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALUE:%.]] = cir.load align(4) [[CPUTYPE]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<34> : !u32i
// CIR-NEXT: {{.*}} = cir.cmp eq [[VALUE]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_pantherlake(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// LLVM: = icmp eq i32 [[LOAD]], 34

// OGCG-LABEL: define{{.*}} void @test_pantherlake(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// OGCG: = icmp eq i32 [[LOAD]], 34
TEST_CPU_IS(pantherlake, "pantherlake")

// CIR-LABEL: cir.func no_inline dso_local @test_znver5()
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][2] {name = "__cpu_subtype"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALUE:%.]] = cir.load align(4) [[CPUTYPE]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<36> : !u32i
// CIR-NEXT: {{.*}} = cir.cmp eq [[VALUE]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_znver5(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// LLVM: = icmp eq i32 [[LOAD]], 36

// OGCG-LABEL: define{{.*}} void @test_znver5(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// OGCG: = icmp eq i32 [[LOAD]], 36
TEST_CPU_IS(znver5, "znver5")

// CIR-LABEL: cir.func no_inline dso_local @test_diamondrapids()
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][2] {name = "__cpu_subtype"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALUE:%.]] = cir.load align(4) [[CPUTYPE]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<38> : !u32i
// CIR-NEXT: {{.*}} = cir.cmp eq [[VALUE]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_diamondrapids(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// LLVM: = icmp eq i32 [[LOAD]], 38

// OGCG-LABEL: define{{.*}} void @test_diamondrapids(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// OGCG: = icmp eq i32 [[LOAD]], 38
TEST_CPU_IS(diamondrapids, "diamondrapids")

// CIR-LABEL: cir.func no_inline dso_local @test_novalake()
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][2] {name = "__cpu_subtype"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALUE:%.]] = cir.load align(4) [[CPUTYPE]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<39> : !u32i
// CIR-NEXT: {{.*}} = cir.cmp eq [[VALUE]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_novalake(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// LLVM: = icmp eq i32 [[LOAD]], 39

// OGCG-LABEL: define{{.*}} void @test_novalake(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// OGCG: = icmp eq i32 [[LOAD]], 39
TEST_CPU_IS(novalake, "novalake")

// CIR-LABEL: cir.func no_inline dso_local @test_znver6()
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][2] {name = "__cpu_subtype"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALUE:%.]] = cir.load align(4) [[CPUTYPE]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<40> : !u32i
// CIR-NEXT: {{.*}} = cir.cmp eq [[VALUE]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_znver6(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// LLVM: = icmp eq i32 [[LOAD]], 40

// OGCG-LABEL: define{{.*}} void @test_znver6(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// OGCG: = icmp eq i32 [[LOAD]], 40
TEST_CPU_IS(znver6, "znver6")

// CIR-LABEL: cir.func no_inline dso_local @test_c86_4g_m4()
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][2] {name = "__cpu_subtype"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALUE:%.]] = cir.load align(4) [[CPUTYPE]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<41> : !u32i
// CIR-NEXT: {{.*}} = cir.cmp eq [[VALUE]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_c86_4g_m4(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// LLVM: = icmp eq i32 [[LOAD]], 41

// OGCG-LABEL: define{{.*}} void @test_c86_4g_m4(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// OGCG: = icmp eq i32 [[LOAD]], 41
TEST_CPU_IS(c86_4g_m4, "c86-4g-m4")

// CIR-LABEL: cir.func no_inline dso_local @test_c86_4g_m6()
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][2] {name = "__cpu_subtype"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALUE:%.]] = cir.load align(4) [[CPUTYPE]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<42> : !u32i
// CIR-NEXT: {{.*}} = cir.cmp eq [[VALUE]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_c86_4g_m6(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// LLVM: = icmp eq i32 [[LOAD]], 42

// OGCG-LABEL: define{{.*}} void @test_c86_4g_m6(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// OGCG: = icmp eq i32 [[LOAD]], 42
TEST_CPU_IS(c86_4g_m6, "c86-4g-m6")

// CIR-LABEL: cir.func no_inline dso_local @test_c86_4g_m7()
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][2] {name = "__cpu_subtype"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALUE:%.]] = cir.load align(4) [[CPUTYPE]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<43> : !u32i
// CIR-NEXT: {{.*}} = cir.cmp eq [[VALUE]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_c86_4g_m7(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// LLVM: = icmp eq i32 [[LOAD]], 43

// OGCG-LABEL: define{{.*}} void @test_c86_4g_m7(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// OGCG: = icmp eq i32 [[LOAD]], 43
TEST_CPU_IS(c86_4g_m7, "c86-4g-m7")

// CIR-LABEL: cir.func no_inline dso_local @test_c86_4g_m8()
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][2] {name = "__cpu_subtype"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALUE:%.]] = cir.load align(4) [[CPUTYPE]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<44> : !u32i
// CIR-NEXT: {{.*}} = cir.cmp eq [[VALUE]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_c86_4g_m8(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// LLVM: = icmp eq i32 [[LOAD]], 44

// OGCG-LABEL: define{{.*}} void @test_c86_4g_m8(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// OGCG: = icmp eq i32 [[LOAD]], 44
TEST_CPU_IS(c86_4g_m8, "c86-4g-m8")

// Aliases
// CIR-LABEL: cir.func no_inline dso_local @test_emeraldrapids()
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][2] {name = "__cpu_subtype"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALUE:%.]] = cir.load align(4) [[CPUTYPE]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<24> : !u32i
// CIR-NEXT: {{.*}} = cir.cmp eq [[VALUE]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_emeraldrapids(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// LLVM: = icmp eq i32 [[LOAD]], 24

// OGCG-LABEL: define{{.*}} void @test_emeraldrapids(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// OGCG: = icmp eq i32 [[LOAD]], 24
TEST_CPU_IS(emeraldrapids, "emeraldrapids")

// CIR-LABEL: cir.func no_inline dso_local @test_raptorlake()
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][2] {name = "__cpu_subtype"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALUE:%.]] = cir.load align(4) [[CPUTYPE]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<25> : !u32i
// CIR-NEXT: {{.*}} = cir.cmp eq [[VALUE]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_raptorlake(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// LLVM: = icmp eq i32 [[LOAD]], 25

// OGCG-LABEL: define{{.*}} void @test_raptorlake(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// OGCG: = icmp eq i32 [[LOAD]], 25
TEST_CPU_IS(raptorlake, "raptorlake")

// CIR-LABEL: cir.func no_inline dso_local @test_meteorlake()
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][2] {name = "__cpu_subtype"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALUE:%.]] = cir.load align(4) [[CPUTYPE]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<25> : !u32i
// CIR-NEXT: {{.*}} = cir.cmp eq [[VALUE]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_meteorlake(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// LLVM: = icmp eq i32 [[LOAD]], 25

// OGCG-LABEL: define{{.*}} void @test_meteorlake(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// OGCG: = icmp eq i32 [[LOAD]], 25
TEST_CPU_IS(meteorlake, "meteorlake")

// CIR-LABEL: cir.func no_inline dso_local @test_gracemont()
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][2] {name = "__cpu_subtype"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALUE:%.]] = cir.load align(4) [[CPUTYPE]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<25> : !u32i
// CIR-NEXT: {{.*}} = cir.cmp eq [[VALUE]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_gracemont(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// LLVM: = icmp eq i32 [[LOAD]], 25

// OGCG-LABEL: define{{.*}} void @test_gracemont(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// OGCG: = icmp eq i32 [[LOAD]], 25
TEST_CPU_IS(gracemont, "gracemont")

// CIR-LABEL: cir.func no_inline dso_local @test_lunarlake()
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][2] {name = "__cpu_subtype"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALUE:%.]] = cir.load align(4) [[CPUTYPE]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<33> : !u32i
// CIR-NEXT: {{.*}} = cir.cmp eq [[VALUE]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_lunarlake(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// LLVM: = icmp eq i32 [[LOAD]], 33

// OGCG-LABEL: define{{.*}} void @test_lunarlake(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// OGCG: = icmp eq i32 [[LOAD]], 33
TEST_CPU_IS(lunarlake, "lunarlake")

// CIR-LABEL: cir.func no_inline dso_local @test_wildcatlake()
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][2] {name = "__cpu_subtype"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALUE:%.]] = cir.load align(4) [[CPUTYPE]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<34> : !u32i
// CIR-NEXT: {{.*}} = cir.cmp eq [[VALUE]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_wildcatlake(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// LLVM: = icmp eq i32 [[LOAD]], 34

// OGCG-LABEL: define{{.*}} void @test_wildcatlake(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
// OGCG: = icmp eq i32 [[LOAD]], 34
TEST_CPU_IS(wildcatlake, "wildcatlake")

