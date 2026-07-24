// RUN: %clang_cc1 -x c -ffreestanding -triple x86_64-unknown-linux -Wno-implicit-function-declaration -fclangir -emit-cir -o %t.cir %s
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -x c -ffreestanding -triple x86_64-unknown-linux -Wno-implicit-function-declaration -fclangir -emit-llvm -o %t.ll %s
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -ffreestanding -triple=x86_64-pc-linux-gnu -emit-llvm -Wall -Werror %s -o - | FileCheck %s -check-prefix=OGCG

// Test that __builtin_cpu_is emits the correct ABI value for every CPU type,
// in llvm/include/llvm/TargetParser/X86TargetParser.def.
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

// CIR-LABEL: cir.func no_inline dso_local @test_bonnell(
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][1] {name = "__cpu_type"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALUE:%.]] = cir.load align(4) [[CPUTYPE]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<1> : !u32i
// CIR-NEXT: {{.*}} = cir.cmp eq [[VALUE]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_bonnell(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// LLVM: = icmp eq i32 [[LOAD]], 1

// OGCG-LABEL: define{{.*}} void @test_bonnell(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// OGCG: = icmp eq i32 [[LOAD]], 1
TEST_CPU_IS(bonnell, "bonnell")

// CIR-LABEL: cir.func no_inline dso_local @test_core2(
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][1] {name = "__cpu_type"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALUE:%.]] = cir.load align(4) [[CPUTYPE]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<2> : !u32i
// CIR-NEXT: {{.*}} = cir.cmp eq [[VALUE]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_core2(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// LLVM: = icmp eq i32 [[LOAD]], 2

// OGCG-LABEL: define{{.*}} void @test_core2(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// OGCG: = icmp eq i32 [[LOAD]], 2
TEST_CPU_IS(core2, "core2")

// CIR-LABEL: cir.func no_inline dso_local @test_corei7(
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][1] {name = "__cpu_type"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALUE:%.]] = cir.load align(4) [[CPUTYPE]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<3> : !u32i
// CIR-NEXT: {{.*}} = cir.cmp eq [[VALUE]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_corei7(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// LLVM: = icmp eq i32 [[LOAD]], 3

// OGCG-LABEL: define{{.*}} void @test_corei7(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// OGCG: = icmp eq i32 [[LOAD]], 3
TEST_CPU_IS(corei7, "corei7")

// CIR-LABEL: cir.func no_inline dso_local @test_amdfam10h(
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][1] {name = "__cpu_type"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALUE:%.]] = cir.load align(4) [[CPUTYPE]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<4> : !u32i
// CIR-NEXT: {{.*}} = cir.cmp eq [[VALUE]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_amdfam10h(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// LLVM: = icmp eq i32 [[LOAD]], 4

// OGCG-LABEL: define{{.*}} void @test_amdfam10h(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// OGCG: = icmp eq i32 [[LOAD]], 4
TEST_CPU_IS(amdfam10h, "amdfam10h")

// CIR-LABEL: cir.func no_inline dso_local @test_amdfam15h(
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][1] {name = "__cpu_type"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALUE:%.]] = cir.load align(4) [[CPUTYPE]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<5> : !u32i
// CIR-NEXT: {{.*}} = cir.cmp eq [[VALUE]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_amdfam15h(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// LLVM: = icmp eq i32 [[LOAD]], 5

// OGCG-LABEL: define{{.*}} void @test_amdfam15h(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// OGCG: = icmp eq i32 [[LOAD]], 5
TEST_CPU_IS(amdfam15h, "amdfam15h")

// CIR-LABEL: cir.func no_inline dso_local @test_silvermont(
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][1] {name = "__cpu_type"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALUE:%.]] = cir.load align(4) [[CPUTYPE]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<6> : !u32i
// CIR-NEXT: {{.*}} = cir.cmp eq [[VALUE]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_silvermont(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// LLVM: = icmp eq i32 [[LOAD]], 6

// OGCG-LABEL: define{{.*}} void @test_silvermont(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// OGCG: = icmp eq i32 [[LOAD]], 6
TEST_CPU_IS(silvermont, "silvermont")

// CIR-LABEL: cir.func no_inline dso_local @test_knl(
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][1] {name = "__cpu_type"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALUE:%.]] = cir.load align(4) [[CPUTYPE]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<7> : !u32i
// CIR-NEXT: {{.*}} = cir.cmp eq [[VALUE]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_knl(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// LLVM: = icmp eq i32 [[LOAD]], 7

// OGCG-LABEL: define{{.*}} void @test_knl(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// OGCG: = icmp eq i32 [[LOAD]], 7
TEST_CPU_IS(knl, "knl")

// CIR-LABEL: cir.func no_inline dso_local @test_btver1(
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][1] {name = "__cpu_type"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALUE:%.]] = cir.load align(4) [[CPUTYPE]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<8> : !u32i
// CIR-NEXT: {{.*}} = cir.cmp eq [[VALUE]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_btver1(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// LLVM: = icmp eq i32 [[LOAD]], 8

// OGCG-LABEL: define{{.*}} void @test_btver1(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// OGCG: = icmp eq i32 [[LOAD]], 8
TEST_CPU_IS(btver1, "btver1")

// CIR-LABEL: cir.func no_inline dso_local @test_btver2(
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][1] {name = "__cpu_type"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALUE:%.]] = cir.load align(4) [[CPUTYPE]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<9> : !u32i
// CIR-NEXT: {{.*}} = cir.cmp eq [[VALUE]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_btver2(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// LLVM: = icmp eq i32 [[LOAD]], 9

// OGCG-LABEL: define{{.*}} void @test_btver2(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// OGCG: = icmp eq i32 [[LOAD]], 9
TEST_CPU_IS(btver2, "btver2")

// CIR-LABEL: cir.func no_inline dso_local @test_amdfam17h(
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][1] {name = "__cpu_type"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALUE:%.]] = cir.load align(4) [[CPUTYPE]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<10> : !u32i
// CIR-NEXT: {{.*}} = cir.cmp eq [[VALUE]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_amdfam17h(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// LLVM: = icmp eq i32 [[LOAD]], 10

// OGCG-LABEL: define{{.*}} void @test_amdfam17h(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// OGCG: = icmp eq i32 [[LOAD]], 10
TEST_CPU_IS(amdfam17h, "amdfam17h")

// CIR-LABEL: cir.func no_inline dso_local @test_knm(
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][1] {name = "__cpu_type"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALUE:%.]] = cir.load align(4) [[CPUTYPE]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<11> : !u32i
// CIR-NEXT: {{.*}} = cir.cmp eq [[VALUE]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_knm(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// LLVM: = icmp eq i32 [[LOAD]], 11

// OGCG-LABEL: define{{.*}} void @test_knm(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// OGCG: = icmp eq i32 [[LOAD]], 11
TEST_CPU_IS(knm, "knm")

// CIR-LABEL: cir.func no_inline dso_local @test_goldmont(
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][1] {name = "__cpu_type"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALUE:%.]] = cir.load align(4) [[CPUTYPE]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<12> : !u32i
// CIR-NEXT: {{.*}} = cir.cmp eq [[VALUE]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_goldmont(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// LLVM: = icmp eq i32 [[LOAD]], 12

// OGCG-LABEL: define{{.*}} void @test_goldmont(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// OGCG: = icmp eq i32 [[LOAD]], 12
TEST_CPU_IS(goldmont, "goldmont")

// CIR-LABEL: cir.func no_inline dso_local @test_goldmont_plus(
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][1] {name = "__cpu_type"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALUE:%.]] = cir.load align(4) [[CPUTYPE]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<13> : !u32i
// CIR-NEXT: {{.*}} = cir.cmp eq [[VALUE]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_goldmont_plus(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// LLVM: = icmp eq i32 [[LOAD]], 13

// OGCG-LABEL: define{{.*}} void @test_goldmont_plus(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// OGCG: = icmp eq i32 [[LOAD]], 13
TEST_CPU_IS(goldmont_plus, "goldmont-plus")

// CIR-LABEL: cir.func no_inline dso_local @test_tremont(
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][1] {name = "__cpu_type"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALUE:%.]] = cir.load align(4) [[CPUTYPE]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<14> : !u32i
// CIR-NEXT: {{.*}} = cir.cmp eq [[VALUE]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_tremont(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// LLVM: = icmp eq i32 [[LOAD]], 14

// OGCG-LABEL: define{{.*}} void @test_tremont(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// OGCG: = icmp eq i32 [[LOAD]], 14
TEST_CPU_IS(tremont, "tremont")

// CIR-LABEL: cir.func no_inline dso_local @test_amdfam19h(
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][1] {name = "__cpu_type"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALUE:%.]] = cir.load align(4) [[CPUTYPE]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<15> : !u32i
// CIR-NEXT: {{.*}} = cir.cmp eq [[VALUE]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_amdfam19h(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// LLVM: = icmp eq i32 [[LOAD]], 15

// OGCG-LABEL: define{{.*}} void @test_amdfam19h(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// OGCG: = icmp eq i32 [[LOAD]], 15
TEST_CPU_IS(amdfam19h, "amdfam19h")

// CIR-LABEL: cir.func no_inline dso_local @test_zhaoxin_fam7h(
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][1] {name = "__cpu_type"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALUE:%.]] = cir.load align(4) [[CPUTYPE]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<16> : !u32i
// CIR-NEXT: {{.*}} = cir.cmp eq [[VALUE]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_zhaoxin_fam7h(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// LLVM: = icmp eq i32 [[LOAD]], 16

// OGCG-LABEL: define{{.*}} void @test_zhaoxin_fam7h(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// OGCG: = icmp eq i32 [[LOAD]], 16
TEST_CPU_IS(zhaoxin_fam7h, "zhaoxin_fam7h")

// CIR-LABEL: cir.func no_inline dso_local @test_sierraforest(
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][1] {name = "__cpu_type"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALUE:%.]] = cir.load align(4) [[CPUTYPE]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<17> : !u32i
// CIR-NEXT: {{.*}} = cir.cmp eq [[VALUE]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_sierraforest(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// LLVM: = icmp eq i32 [[LOAD]], 17

// OGCG-LABEL: define{{.*}} void @test_sierraforest(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// OGCG: = icmp eq i32 [[LOAD]], 17
TEST_CPU_IS(sierraforest, "sierraforest")

// CIR-LABEL: cir.func no_inline dso_local @test_grandridge(
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][1] {name = "__cpu_type"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALUE:%.]] = cir.load align(4) [[CPUTYPE]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<18> : !u32i
// CIR-NEXT: {{.*}} = cir.cmp eq [[VALUE]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_grandridge(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// LLVM: = icmp eq i32 [[LOAD]], 18

// OGCG-LABEL: define{{.*}} void @test_grandridge(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// OGCG: = icmp eq i32 [[LOAD]], 18
TEST_CPU_IS(grandridge, "grandridge")

// CIR-LABEL: cir.func no_inline dso_local @test_clearwaterforest(
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][1] {name = "__cpu_type"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALUE:%.]] = cir.load align(4) [[CPUTYPE]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<19> : !u32i
// CIR-NEXT: {{.*}} = cir.cmp eq [[VALUE]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_clearwaterforest(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// LLVM: = icmp eq i32 [[LOAD]], 19

// OGCG-LABEL: define{{.*}} void @test_clearwaterforest(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// OGCG: = icmp eq i32 [[LOAD]], 19
TEST_CPU_IS(clearwaterforest, "clearwaterforest")

// CIR-LABEL: cir.func no_inline dso_local @test_amdfam1ah(
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][1] {name = "__cpu_type"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALUE:%.]] = cir.load align(4) [[CPUTYPE]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<20> : !u32i
// CIR-NEXT: {{.*}} = cir.cmp eq [[VALUE]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_amdfam1ah(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// LLVM: = icmp eq i32 [[LOAD]], 20

// OGCG-LABEL: define{{.*}} void @test_amdfam1ah(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// OGCG: = icmp eq i32 [[LOAD]], 20
TEST_CPU_IS(amdfam1ah, "amdfam1ah")

// CIR-LABEL: cir.func no_inline dso_local @test_hygonfam18h(
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][1] {name = "__cpu_type"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALUE:%.]] = cir.load align(4) [[CPUTYPE]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<21> : !u32i
// CIR-NEXT: {{.*}} = cir.cmp eq [[VALUE]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_hygonfam18h(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// LLVM: = icmp eq i32 [[LOAD]], 21

// OGCG-LABEL: define{{.*}} void @test_hygonfam18h(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// OGCG: = icmp eq i32 [[LOAD]], 21
TEST_CPU_IS(hygonfam18h, "hygonfam18h")

// Aliases
// CIR-LABEL: cir.func no_inline dso_local @test_atom(
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][1] {name = "__cpu_type"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALUE:%.]] = cir.load align(4) [[CPUTYPE]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<1> : !u32i
// CIR-NEXT: {{.*}} = cir.cmp eq [[VALUE]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_atom(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// LLVM: = icmp eq i32 [[LOAD]], 1

// OGCG-LABEL: define{{.*}} void @test_atom(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// OGCG: = icmp eq i32 [[LOAD]], 1
TEST_CPU_IS(atom, "atom")

// CIR-LABEL: cir.func no_inline dso_local @test_amdfam10(
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][1] {name = "__cpu_type"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALUE:%.]] = cir.load align(4) [[CPUTYPE]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<4> : !u32i
// CIR-NEXT: {{.*}} = cir.cmp eq [[VALUE]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_amdfam10(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// LLVM: = icmp eq i32 [[LOAD]], 4

// OGCG-LABEL: define{{.*}} void @test_amdfam10(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// OGCG: = icmp eq i32 [[LOAD]], 4
TEST_CPU_IS(amdfam10, "amdfam10")

// CIR-LABEL: cir.func no_inline dso_local @test_amdfam15(
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][1] {name = "__cpu_type"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALUE:%.]] = cir.load align(4) [[CPUTYPE]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<5> : !u32i
// CIR-NEXT: {{.*}} = cir.cmp eq [[VALUE]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_amdfam15(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// LLVM: = icmp eq i32 [[LOAD]], 5

// OGCG-LABEL: define{{.*}} void @test_amdfam15(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// OGCG: = icmp eq i32 [[LOAD]], 5
TEST_CPU_IS(amdfam15, "amdfam15")

// CIR-LABEL: cir.func no_inline dso_local @test_slm(
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][1] {name = "__cpu_type"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALUE:%.]] = cir.load align(4) [[CPUTYPE]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<6> : !u32i
// CIR-NEXT: {{.*}} = cir.cmp eq [[VALUE]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_slm(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// LLVM: = icmp eq i32 [[LOAD]], 6

// OGCG-LABEL: define{{.*}} void @test_slm(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// OGCG: = icmp eq i32 [[LOAD]], 6
TEST_CPU_IS(slm, "slm")

// CIR-LABEL: cir.func no_inline dso_local @test_amdfam1a(
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][1] {name = "__cpu_type"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALUE:%.]] = cir.load align(4) [[CPUTYPE]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<20> : !u32i
// CIR-NEXT: {{.*}} = cir.cmp eq [[VALUE]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_amdfam1a(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// LLVM: = icmp eq i32 [[LOAD]], 20

// OGCG-LABEL: define{{.*}} void @test_amdfam1a(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
// OGCG: = icmp eq i32 [[LOAD]], 20
TEST_CPU_IS(amdfam1a, "amdfam1a")

