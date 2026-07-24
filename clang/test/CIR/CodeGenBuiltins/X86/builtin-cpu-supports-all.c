// RUN: %clang_cc1 -x c -ffreestanding -triple x86_64-unknown-linux -Wno-implicit-function-declaration -fclangir -emit-cir -o %t.cir %s
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -x c -ffreestanding -triple x86_64-unknown-linux -Wno-implicit-function-declaration -fclangir -emit-llvm -o %t.ll %s
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -ffreestanding -triple=x86_64-pc-linux-gnu -emit-llvm -Wall -Werror %s -o - | FileCheck %s -check-prefix=OGCG

// Test that __builtin_cpu_supports emits the correct field and bit for every
// feature listed in llvm/include/llvm/TargetParser/X86TargetParser.def. 
extern void a(const char *);

// CIR: !rec_anon_struct = !cir.struct<{!u32i, !u32i, !u32i, !cir.array<!u32i x 1>}>
// CIR: cir.global "private" external dso_local @__cpu_model : !rec_anon_struct
// LLVM: @__cpu_model = external dso_local global { i32, i32, i32, [1 x i32] }
// OGCG: @__cpu_model = external dso_local global { i32, i32, i32, [1 x i32] }

#define TEST_CPU_SUPPORTS(NAME, STR)                                           \
  void test_##NAME(void) {                                                     \
    if (__builtin_cpu_supports(STR))                                           \
      a(STR);                                                                  \
  }

// CIR-LABEL: cir.func no_inline dso_local @test_cmov()
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][3] {name = "__cpu_feature"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!cir.array<!u32i x 1>
// CIR-NEXT: [[STRIDE:%.]] = cir.const #cir.int<0> : !u32i
// CIR-NEXT: {{.*}} = cir.ptr_stride [[CPUTYPE]], [[STRIDE]] : (!cir.ptr<!cir.array<!u32i x 1>>, !u32i) -> !cir.ptr<!cir.array<!u32i x 1>>
// CIR: [[VALUE0:%.]] = cir.load align(4) {{.*}} : !cir.ptr<!u32i>, !u32i
// CIR: [[MASK:%.]] = cir.const #cir.int<1> : !u32i
// CIR: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR: {{.*}} = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_cmov(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// LLVM: = and i32 [[LOAD]], 1

// OGCG-LABEL: define{{.*}} void @test_cmov(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// OGCG: = and i32 [[LOAD]], 1
TEST_CPU_SUPPORTS(cmov, "cmov")

// CIR-LABEL: cir.func no_inline dso_local @test_mmx()
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][3] {name = "__cpu_feature"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!cir.array<!u32i x 1>
// CIR-NEXT: [[STRIDE:%.]] = cir.const #cir.int<0> : !u32i
// CIR-NEXT: {{.*}} = cir.ptr_stride [[CPUTYPE]], [[STRIDE]] : (!cir.ptr<!cir.array<!u32i x 1>>, !u32i) -> !cir.ptr<!cir.array<!u32i x 1>>
// CIR: [[VALUE0:%.]] = cir.load align(4) {{.*}} : !cir.ptr<!u32i>, !u32i
// CIR: [[MASK:%.]] = cir.const #cir.int<2> : !u32i
// CIR: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR: {{.*}} = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_mmx(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// LLVM: = and i32 [[LOAD]], 2

// OGCG-LABEL: define{{.*}} void @test_mmx(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// OGCG: = and i32 [[LOAD]], 2
TEST_CPU_SUPPORTS(mmx, "mmx")

// CIR-LABEL: cir.func no_inline dso_local @test_popcnt()
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][3] {name = "__cpu_feature"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!cir.array<!u32i x 1>
// CIR-NEXT: [[STRIDE:%.]] = cir.const #cir.int<0> : !u32i
// CIR-NEXT: {{.*}} = cir.ptr_stride [[CPUTYPE]], [[STRIDE]] : (!cir.ptr<!cir.array<!u32i x 1>>, !u32i) -> !cir.ptr<!cir.array<!u32i x 1>>
// CIR: [[VALUE0:%.]] = cir.load align(4) {{.*}} : !cir.ptr<!u32i>, !u32i
// CIR: [[MASK:%.]] = cir.const #cir.int<4> : !u32i
// CIR: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR: {{.*}} = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_popcnt(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// LLVM: = and i32 [[LOAD]], 4

// OGCG-LABEL: define{{.*}} void @test_popcnt(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// OGCG: = and i32 [[LOAD]], 4
TEST_CPU_SUPPORTS(popcnt, "popcnt")

// CIR-LABEL: cir.func no_inline dso_local @test_sse()
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][3] {name = "__cpu_feature"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!cir.array<!u32i x 1>
// CIR-NEXT: [[STRIDE:%.]] = cir.const #cir.int<0> : !u32i
// CIR-NEXT: {{.*}} = cir.ptr_stride [[CPUTYPE]], [[STRIDE]] : (!cir.ptr<!cir.array<!u32i x 1>>, !u32i) -> !cir.ptr<!cir.array<!u32i x 1>>
// CIR: [[VALUE0:%.]] = cir.load align(4) {{.*}} : !cir.ptr<!u32i>, !u32i
// CIR: [[MASK:%.]] = cir.const #cir.int<8> : !u32i
// CIR: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR: {{.*}} = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_sse(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// LLVM: = and i32 [[LOAD]], 8

// OGCG-LABEL: define{{.*}} void @test_sse(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// OGCG: = and i32 [[LOAD]], 8
TEST_CPU_SUPPORTS(sse, "sse")

// CIR-LABEL: cir.func no_inline dso_local @test_sse2()
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][3] {name = "__cpu_feature"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!cir.array<!u32i x 1>
// CIR-NEXT: [[STRIDE:%.]] = cir.const #cir.int<0> : !u32i
// CIR-NEXT: {{.*}} = cir.ptr_stride [[CPUTYPE]], [[STRIDE]] : (!cir.ptr<!cir.array<!u32i x 1>>, !u32i) -> !cir.ptr<!cir.array<!u32i x 1>>
// CIR: [[VALUE0:%.]] = cir.load align(4) {{.*}} : !cir.ptr<!u32i>, !u32i
// CIR: [[MASK:%.]] = cir.const #cir.int<16> : !u32i
// CIR: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR: {{.*}} = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_sse2(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// LLVM: = and i32 [[LOAD]], 16

// OGCG-LABEL: define{{.*}} void @test_sse2(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// OGCG: = and i32 [[LOAD]], 16
TEST_CPU_SUPPORTS(sse2, "sse2")

// CIR-LABEL: cir.func no_inline dso_local @test_sse3()
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][3] {name = "__cpu_feature"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!cir.array<!u32i x 1>
// CIR-NEXT: [[STRIDE:%.]] = cir.const #cir.int<0> : !u32i
// CIR-NEXT: {{.*}} = cir.ptr_stride [[CPUTYPE]], [[STRIDE]] : (!cir.ptr<!cir.array<!u32i x 1>>, !u32i) -> !cir.ptr<!cir.array<!u32i x 1>>
// CIR: [[VALUE0:%.]] = cir.load align(4) {{.*}} : !cir.ptr<!u32i>, !u32i
// CIR: [[MASK:%.]] = cir.const #cir.int<32> : !u32i
// CIR: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR: {{.*}} = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_sse3(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// LLVM: = and i32 [[LOAD]], 32

// OGCG-LABEL: define{{.*}} void @test_sse3(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// OGCG: = and i32 [[LOAD]], 32
TEST_CPU_SUPPORTS(sse3, "sse3")

// CIR-LABEL: cir.func no_inline dso_local @test_ssse3()
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][3] {name = "__cpu_feature"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!cir.array<!u32i x 1>
// CIR-NEXT: [[STRIDE:%.]] = cir.const #cir.int<0> : !u32i
// CIR-NEXT: {{.*}} = cir.ptr_stride [[CPUTYPE]], [[STRIDE]] : (!cir.ptr<!cir.array<!u32i x 1>>, !u32i) -> !cir.ptr<!cir.array<!u32i x 1>>
// CIR: [[VALUE0:%.]] = cir.load align(4) {{.*}} : !cir.ptr<!u32i>, !u32i
// CIR: [[MASK:%.]] = cir.const #cir.int<64> : !u32i
// CIR: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR: {{.*}} = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_ssse3(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// LLVM: = and i32 [[LOAD]], 64

// OGCG-LABEL: define{{.*}} void @test_ssse3(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// OGCG: = and i32 [[LOAD]], 64
TEST_CPU_SUPPORTS(ssse3, "ssse3")

// CIR-LABEL: cir.func no_inline dso_local @test_sse4_1()
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][3] {name = "__cpu_feature"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!cir.array<!u32i x 1>
// CIR-NEXT: [[STRIDE:%.]] = cir.const #cir.int<0> : !u32i
// CIR-NEXT: {{.*}} = cir.ptr_stride [[CPUTYPE]], [[STRIDE]] : (!cir.ptr<!cir.array<!u32i x 1>>, !u32i) -> !cir.ptr<!cir.array<!u32i x 1>>
// CIR: [[VALUE0:%.]] = cir.load align(4) {{.*}} : !cir.ptr<!u32i>, !u32i
// CIR: [[MASK:%.]] = cir.const #cir.int<128> : !u32i
// CIR: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR: {{.*}} = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_sse4_1(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// LLVM: = and i32 [[LOAD]], 128

// OGCG-LABEL: define{{.*}} void @test_sse4_1(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// OGCG: = and i32 [[LOAD]], 128
TEST_CPU_SUPPORTS(sse4_1, "sse4.1")

// CIR-LABEL: cir.func no_inline dso_local @test_sse4_2()
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][3] {name = "__cpu_feature"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!cir.array<!u32i x 1>
// CIR-NEXT: [[STRIDE:%.]] = cir.const #cir.int<0> : !u32i
// CIR-NEXT: {{.*}} = cir.ptr_stride [[CPUTYPE]], [[STRIDE]] : (!cir.ptr<!cir.array<!u32i x 1>>, !u32i) -> !cir.ptr<!cir.array<!u32i x 1>>
// CIR: [[VALUE0:%.]] = cir.load align(4) {{.*}} : !cir.ptr<!u32i>, !u32i
// CIR: [[MASK:%.]] = cir.const #cir.int<256> : !u32i
// CIR: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR: {{.*}} = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_sse4_2(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// LLVM: = and i32 [[LOAD]], 256

// OGCG-LABEL: define{{.*}} void @test_sse4_2(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// OGCG: = and i32 [[LOAD]], 256
TEST_CPU_SUPPORTS(sse4_2, "sse4.2")

// CIR-LABEL: cir.func no_inline dso_local @test_avx()
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][3] {name = "__cpu_feature"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!cir.array<!u32i x 1>
// CIR-NEXT: [[STRIDE:%.]] = cir.const #cir.int<0> : !u32i
// CIR-NEXT: {{.*}} = cir.ptr_stride [[CPUTYPE]], [[STRIDE]] : (!cir.ptr<!cir.array<!u32i x 1>>, !u32i) -> !cir.ptr<!cir.array<!u32i x 1>>
// CIR: [[VALUE0:%.]] = cir.load align(4) {{.*}} : !cir.ptr<!u32i>, !u32i
// CIR: [[MASK:%.]] = cir.const #cir.int<512> : !u32i
// CIR: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR: {{.*}} = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_avx(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// LLVM: = and i32 [[LOAD]], 512

// OGCG-LABEL: define{{.*}} void @test_avx(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// OGCG: = and i32 [[LOAD]], 512
TEST_CPU_SUPPORTS(avx, "avx")

// CIR-LABEL: cir.func no_inline dso_local @test_avx2()
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][3] {name = "__cpu_feature"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!cir.array<!u32i x 1>
// CIR-NEXT: [[STRIDE:%.]] = cir.const #cir.int<0> : !u32i
// CIR-NEXT: {{.*}} = cir.ptr_stride [[CPUTYPE]], [[STRIDE]] : (!cir.ptr<!cir.array<!u32i x 1>>, !u32i) -> !cir.ptr<!cir.array<!u32i x 1>>
// CIR: [[VALUE0:%.]] = cir.load align(4) {{.*}} : !cir.ptr<!u32i>, !u32i
// CIR: [[MASK:%.]] = cir.const #cir.int<1024> : !u32i
// CIR: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR: {{.*}} = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_avx2(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// LLVM: = and i32 [[LOAD]], 1024

// OGCG-LABEL: define{{.*}} void @test_avx2(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// OGCG: = and i32 [[LOAD]], 1024
TEST_CPU_SUPPORTS(avx2, "avx2")

// CIR-LABEL: cir.func no_inline dso_local @test_sse4a()
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][3] {name = "__cpu_feature"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!cir.array<!u32i x 1>
// CIR-NEXT: [[STRIDE:%.]] = cir.const #cir.int<0> : !u32i
// CIR-NEXT: {{.*}} = cir.ptr_stride [[CPUTYPE]], [[STRIDE]] : (!cir.ptr<!cir.array<!u32i x 1>>, !u32i) -> !cir.ptr<!cir.array<!u32i x 1>>
// CIR: [[VALUE0:%.]] = cir.load align(4) {{.*}} : !cir.ptr<!u32i>, !u32i
// CIR: [[MASK:%.]] = cir.const #cir.int<2048> : !u32i
// CIR: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR: {{.*}} = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_sse4a(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// LLVM: = and i32 [[LOAD]], 2048

// OGCG-LABEL: define{{.*}} void @test_sse4a(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// OGCG: = and i32 [[LOAD]], 2048
TEST_CPU_SUPPORTS(sse4a, "sse4a")

// CIR-LABEL: cir.func no_inline dso_local @test_fma4()
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][3] {name = "__cpu_feature"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!cir.array<!u32i x 1>
// CIR-NEXT: [[STRIDE:%.]] = cir.const #cir.int<0> : !u32i
// CIR-NEXT: {{.*}} = cir.ptr_stride [[CPUTYPE]], [[STRIDE]] : (!cir.ptr<!cir.array<!u32i x 1>>, !u32i) -> !cir.ptr<!cir.array<!u32i x 1>>
// CIR: [[VALUE0:%.]] = cir.load align(4) {{.*}} : !cir.ptr<!u32i>, !u32i
// CIR: [[MASK:%.]] = cir.const #cir.int<4096> : !u32i
// CIR: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR: {{.*}} = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_fma4(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// LLVM: = and i32 [[LOAD]], 4096

// OGCG-LABEL: define{{.*}} void @test_fma4(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// OGCG: = and i32 [[LOAD]], 4096
TEST_CPU_SUPPORTS(fma4, "fma4")

// CIR-LABEL: cir.func no_inline dso_local @test_xop()
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][3] {name = "__cpu_feature"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!cir.array<!u32i x 1>
// CIR-NEXT: [[STRIDE:%.]] = cir.const #cir.int<0> : !u32i
// CIR-NEXT: {{.*}} = cir.ptr_stride [[CPUTYPE]], [[STRIDE]] : (!cir.ptr<!cir.array<!u32i x 1>>, !u32i) -> !cir.ptr<!cir.array<!u32i x 1>>
// CIR: [[VALUE0:%.]] = cir.load align(4) {{.*}} : !cir.ptr<!u32i>, !u32i
// CIR: [[MASK:%.]] = cir.const #cir.int<8192> : !u32i
// CIR: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR: {{.*}} = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_xop(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// LLVM: = and i32 [[LOAD]], 8192

// OGCG-LABEL: define{{.*}} void @test_xop(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// OGCG: = and i32 [[LOAD]], 8192
TEST_CPU_SUPPORTS(xop, "xop")

// CIR-LABEL: cir.func no_inline dso_local @test_fma()
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][3] {name = "__cpu_feature"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!cir.array<!u32i x 1>
// CIR-NEXT: [[STRIDE:%.]] = cir.const #cir.int<0> : !u32i
// CIR-NEXT: {{.*}} = cir.ptr_stride [[CPUTYPE]], [[STRIDE]] : (!cir.ptr<!cir.array<!u32i x 1>>, !u32i) -> !cir.ptr<!cir.array<!u32i x 1>>
// CIR: [[VALUE0:%.]] = cir.load align(4) {{.*}} : !cir.ptr<!u32i>, !u32i
// CIR: [[MASK:%.]] = cir.const #cir.int<16384> : !u32i
// CIR: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR: {{.*}} = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_fma(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// LLVM: = and i32 [[LOAD]], 16384

// OGCG-LABEL: define{{.*}} void @test_fma(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// OGCG: = and i32 [[LOAD]], 16384
TEST_CPU_SUPPORTS(fma, "fma")

// CIR-LABEL: cir.func no_inline dso_local @test_avx512f()
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][3] {name = "__cpu_feature"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!cir.array<!u32i x 1>
// CIR-NEXT: [[STRIDE:%.]] = cir.const #cir.int<0> : !u32i
// CIR-NEXT: {{.*}} = cir.ptr_stride [[CPUTYPE]], [[STRIDE]] : (!cir.ptr<!cir.array<!u32i x 1>>, !u32i) -> !cir.ptr<!cir.array<!u32i x 1>>
// CIR: [[VALUE0:%.]] = cir.load align(4) {{.*}} : !cir.ptr<!u32i>, !u32i
// CIR: [[MASK:%.]] = cir.const #cir.int<32768> : !u32i
// CIR: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR: {{.*}} = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_avx512f(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// LLVM: = and i32 [[LOAD]], 32768

// OGCG-LABEL: define{{.*}} void @test_avx512f(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// OGCG: = and i32 [[LOAD]], 32768
TEST_CPU_SUPPORTS(avx512f, "avx512f")

// CIR-LABEL: cir.func no_inline dso_local @test_bmi()
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][3] {name = "__cpu_feature"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!cir.array<!u32i x 1>
// CIR-NEXT: [[STRIDE:%.]] = cir.const #cir.int<0> : !u32i
// CIR-NEXT: {{.*}} = cir.ptr_stride [[CPUTYPE]], [[STRIDE]] : (!cir.ptr<!cir.array<!u32i x 1>>, !u32i) -> !cir.ptr<!cir.array<!u32i x 1>>
// CIR: [[VALUE0:%.]] = cir.load align(4) {{.*}} : !cir.ptr<!u32i>, !u32i
// CIR: [[MASK:%.]] = cir.const #cir.int<65536> : !u32i
// CIR: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR: {{.*}} = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_bmi(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// LLVM: = and i32 [[LOAD]], 65536

// OGCG-LABEL: define{{.*}} void @test_bmi(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// OGCG: = and i32 [[LOAD]], 65536
TEST_CPU_SUPPORTS(bmi, "bmi")

// CIR-LABEL: cir.func no_inline dso_local @test_bmi2()
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][3] {name = "__cpu_feature"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!cir.array<!u32i x 1>
// CIR-NEXT: [[STRIDE:%.]] = cir.const #cir.int<0> : !u32i
// CIR-NEXT: {{.*}} = cir.ptr_stride [[CPUTYPE]], [[STRIDE]] : (!cir.ptr<!cir.array<!u32i x 1>>, !u32i) -> !cir.ptr<!cir.array<!u32i x 1>>
// CIR: [[VALUE0:%.]] = cir.load align(4) {{.*}} : !cir.ptr<!u32i>, !u32i
// CIR: [[MASK:%.]] = cir.const #cir.int<131072> : !u32i
// CIR: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR: {{.*}} = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_bmi2(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// LLVM: = and i32 [[LOAD]], 131072

// OGCG-LABEL: define{{.*}} void @test_bmi2(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// OGCG: = and i32 [[LOAD]], 131072
TEST_CPU_SUPPORTS(bmi2, "bmi2")

// CIR-LABEL: cir.func no_inline dso_local @test_aes()
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][3] {name = "__cpu_feature"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!cir.array<!u32i x 1>
// CIR-NEXT: [[STRIDE:%.]] = cir.const #cir.int<0> : !u32i
// CIR-NEXT: {{.*}} = cir.ptr_stride [[CPUTYPE]], [[STRIDE]] : (!cir.ptr<!cir.array<!u32i x 1>>, !u32i) -> !cir.ptr<!cir.array<!u32i x 1>>
// CIR: [[VALUE0:%.]] = cir.load align(4) {{.*}} : !cir.ptr<!u32i>, !u32i
// CIR: [[MASK:%.]] = cir.const #cir.int<262144> : !u32i
// CIR: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR: {{.*}} = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_aes(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// LLVM: = and i32 [[LOAD]], 262144

// OGCG-LABEL: define{{.*}} void @test_aes(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// OGCG: = and i32 [[LOAD]], 262144
TEST_CPU_SUPPORTS(aes, "aes")

// CIR-LABEL: cir.func no_inline dso_local @test_pclmul()
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][3] {name = "__cpu_feature"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!cir.array<!u32i x 1>
// CIR-NEXT: [[STRIDE:%.]] = cir.const #cir.int<0> : !u32i
// CIR-NEXT: {{.*}} = cir.ptr_stride [[CPUTYPE]], [[STRIDE]] : (!cir.ptr<!cir.array<!u32i x 1>>, !u32i) -> !cir.ptr<!cir.array<!u32i x 1>>
// CIR: [[VALUE0:%.]] = cir.load align(4) {{.*}} : !cir.ptr<!u32i>, !u32i
// CIR: [[MASK:%.]] = cir.const #cir.int<524288> : !u32i
// CIR: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR: {{.*}} = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_pclmul(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// LLVM: = and i32 [[LOAD]], 524288

// OGCG-LABEL: define{{.*}} void @test_pclmul(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// OGCG: = and i32 [[LOAD]], 524288
TEST_CPU_SUPPORTS(pclmul, "pclmul")

// CIR-LABEL: cir.func no_inline dso_local @test_avx512vl()
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][3] {name = "__cpu_feature"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!cir.array<!u32i x 1>
// CIR-NEXT: [[STRIDE:%.]] = cir.const #cir.int<0> : !u32i
// CIR-NEXT: {{.*}} = cir.ptr_stride [[CPUTYPE]], [[STRIDE]] : (!cir.ptr<!cir.array<!u32i x 1>>, !u32i) -> !cir.ptr<!cir.array<!u32i x 1>>
// CIR: [[VALUE0:%.]] = cir.load align(4) {{.*}} : !cir.ptr<!u32i>, !u32i
// CIR: [[MASK:%.]] = cir.const #cir.int<1048576> : !u32i
// CIR: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR: {{.*}} = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_avx512vl(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// LLVM: = and i32 [[LOAD]], 1048576

// OGCG-LABEL: define{{.*}} void @test_avx512vl(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// OGCG: = and i32 [[LOAD]], 1048576
TEST_CPU_SUPPORTS(avx512vl, "avx512vl")

// CIR-LABEL: cir.func no_inline dso_local @test_avx512bw()
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][3] {name = "__cpu_feature"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!cir.array<!u32i x 1>
// CIR-NEXT: [[STRIDE:%.]] = cir.const #cir.int<0> : !u32i
// CIR-NEXT: {{.*}} = cir.ptr_stride [[CPUTYPE]], [[STRIDE]] : (!cir.ptr<!cir.array<!u32i x 1>>, !u32i) -> !cir.ptr<!cir.array<!u32i x 1>>
// CIR: [[VALUE0:%.]] = cir.load align(4) {{.*}} : !cir.ptr<!u32i>, !u32i
// CIR: [[MASK:%.]] = cir.const #cir.int<2097152> : !u32i
// CIR: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR: {{.*}} = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_avx512bw(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// LLVM: = and i32 [[LOAD]], 2097152

// OGCG-LABEL: define{{.*}} void @test_avx512bw(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// OGCG: = and i32 [[LOAD]], 2097152
TEST_CPU_SUPPORTS(avx512bw, "avx512bw")

// CIR-LABEL: cir.func no_inline dso_local @test_avx512dq()
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][3] {name = "__cpu_feature"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!cir.array<!u32i x 1>
// CIR-NEXT: [[STRIDE:%.]] = cir.const #cir.int<0> : !u32i
// CIR-NEXT: {{.*}} = cir.ptr_stride [[CPUTYPE]], [[STRIDE]] : (!cir.ptr<!cir.array<!u32i x 1>>, !u32i) -> !cir.ptr<!cir.array<!u32i x 1>>
// CIR: [[VALUE0:%.]] = cir.load align(4) {{.*}} : !cir.ptr<!u32i>, !u32i
// CIR: [[MASK:%.]] = cir.const #cir.int<4194304> : !u32i
// CIR: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR: {{.*}} = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_avx512dq(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// LLVM: = and i32 [[LOAD]], 4194304

// OGCG-LABEL: define{{.*}} void @test_avx512dq(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// OGCG: = and i32 [[LOAD]], 4194304
TEST_CPU_SUPPORTS(avx512dq, "avx512dq")

// CIR-LABEL: cir.func no_inline dso_local @test_avx512cd()
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][3] {name = "__cpu_feature"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!cir.array<!u32i x 1>
// CIR-NEXT: [[STRIDE:%.]] = cir.const #cir.int<0> : !u32i
// CIR-NEXT: {{.*}} = cir.ptr_stride [[CPUTYPE]], [[STRIDE]] : (!cir.ptr<!cir.array<!u32i x 1>>, !u32i) -> !cir.ptr<!cir.array<!u32i x 1>>
// CIR: [[VALUE0:%.]] = cir.load align(4) {{.*}} : !cir.ptr<!u32i>, !u32i
// CIR: [[MASK:%.]] = cir.const #cir.int<8388608> : !u32i
// CIR: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR: {{.*}} = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_avx512cd(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// LLVM: = and i32 [[LOAD]], 8388608

// OGCG-LABEL: define{{.*}} void @test_avx512cd(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// OGCG: = and i32 [[LOAD]], 8388608
TEST_CPU_SUPPORTS(avx512cd, "avx512cd")

// CIR-LABEL: cir.func no_inline dso_local @test_avx512vbmi()
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][3] {name = "__cpu_feature"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!cir.array<!u32i x 1>
// CIR-NEXT: [[STRIDE:%.]] = cir.const #cir.int<0> : !u32i
// CIR-NEXT: {{.*}} = cir.ptr_stride [[CPUTYPE]], [[STRIDE]] : (!cir.ptr<!cir.array<!u32i x 1>>, !u32i) -> !cir.ptr<!cir.array<!u32i x 1>>
// CIR: [[VALUE0:%.]] = cir.load align(4) {{.*}} : !cir.ptr<!u32i>, !u32i
// CIR: [[MASK:%.]] = cir.const #cir.int<67108864> : !u32i
// CIR: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR: {{.*}} = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_avx512vbmi(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// LLVM: = and i32 [[LOAD]], 67108864

// OGCG-LABEL: define{{.*}} void @test_avx512vbmi(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// OGCG: = and i32 [[LOAD]], 67108864
TEST_CPU_SUPPORTS(avx512vbmi, "avx512vbmi")

// CIR-LABEL: cir.func no_inline dso_local @test_avx512ifma()
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][3] {name = "__cpu_feature"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!cir.array<!u32i x 1>
// CIR-NEXT: [[STRIDE:%.]] = cir.const #cir.int<0> : !u32i
// CIR-NEXT: {{.*}} = cir.ptr_stride [[CPUTYPE]], [[STRIDE]] : (!cir.ptr<!cir.array<!u32i x 1>>, !u32i) -> !cir.ptr<!cir.array<!u32i x 1>>
// CIR: [[VALUE0:%.]] = cir.load align(4) {{.*}} : !cir.ptr<!u32i>, !u32i
// CIR: [[MASK:%.]] = cir.const #cir.int<134217728> : !u32i
// CIR: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR: {{.*}} = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_avx512ifma(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// LLVM: = and i32 [[LOAD]], 134217728

// OGCG-LABEL: define{{.*}} void @test_avx512ifma(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// OGCG: = and i32 [[LOAD]], 134217728
TEST_CPU_SUPPORTS(avx512ifma, "avx512ifma")

// CIR-LABEL: cir.func no_inline dso_local @test_avx512vpopcntdq()
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][3] {name = "__cpu_feature"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!cir.array<!u32i x 1>
// CIR-NEXT: [[STRIDE:%.]] = cir.const #cir.int<0> : !u32i
// CIR-NEXT: {{.*}} = cir.ptr_stride [[CPUTYPE]], [[STRIDE]] : (!cir.ptr<!cir.array<!u32i x 1>>, !u32i) -> !cir.ptr<!cir.array<!u32i x 1>>
// CIR: [[VALUE0:%.]] = cir.load align(4) {{.*}} : !cir.ptr<!u32i>, !u32i
// CIR: [[MASK:%.]] = cir.const #cir.int<1073741824> : !u32i
// CIR: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR: {{.*}} = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_avx512vpopcntdq(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// LLVM: = and i32 [[LOAD]], 1073741824

// OGCG-LABEL: define{{.*}} void @test_avx512vpopcntdq(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// OGCG: = and i32 [[LOAD]], 1073741824
TEST_CPU_SUPPORTS(avx512vpopcntdq, "avx512vpopcntdq")

// CIR-LABEL: cir.func no_inline dso_local @test_avx512vbmi2()
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][3] {name = "__cpu_feature"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!cir.array<!u32i x 1>
// CIR-NEXT: [[STRIDE:%.]] = cir.const #cir.int<0> : !u32i
// CIR-NEXT: {{.*}} = cir.ptr_stride [[CPUTYPE]], [[STRIDE]] : (!cir.ptr<!cir.array<!u32i x 1>>, !u32i) -> !cir.ptr<!cir.array<!u32i x 1>>
// CIR: [[VALUE0:%.]] = cir.load align(4) {{.*}} : !cir.ptr<!u32i>, !u32i
// CIR: [[MASK:%.]] = cir.const #cir.int<2147483648> : !u32i
// CIR: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR: {{.*}} = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i

// LLVM-LABEL: define{{.*}} void @test_avx512vbmi2(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// LLVM: = and i32 [[LOAD]], -2147483648

// OGCG-LABEL: define{{.*}} void @test_avx512vbmi2(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12)
// OGCG: = and i32 [[LOAD]], -2147483648
TEST_CPU_SUPPORTS(avx512vbmi2, "avx512vbmi2")

// CIR-LABEL: cir.func no_inline dso_local @test_gfni()
// CIR: [[TRUE:%.]] = cir.const #true
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: [[IDX:%.]] = cir.const #cir.int<0> : !u32i
// CIR-NEXT: [[ARRPTR:%.]] = cir.cast array_to_ptrdecay [[GLOBAL]] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALPTR:%.]] = cir.ptr_stride [[ARRPTR]], [[IDX]] : (!cir.ptr<!u32i>, !u32i) -> !cir.ptr<!u32i> 
// CIR-NEXT: [[VALUE0:%.]] = cir.load align(4) [[VALPTR]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<1> : !u32i
// CIR-NEXT: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR-NEXT: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE]] : !cir.bool

// LLVM-LABEL: define{{.*}} void @test_gfni(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// LLVM: = and i32 [[LOAD]], 1

// OGCG-LABEL: define{{.*}} void @test_gfni(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// OGCG: = and i32 [[LOAD]], 1
TEST_CPU_SUPPORTS(gfni, "gfni")

// CIR-LABEL: cir.func no_inline dso_local @test_vpclmulqdq()
// CIR: [[TRUE:%.]] = cir.const #true
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: [[IDX:%.]] = cir.const #cir.int<0> : !u32i
// CIR-NEXT: [[ARRPTR:%.]] = cir.cast array_to_ptrdecay [[GLOBAL]] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALPTR:%.]] = cir.ptr_stride [[ARRPTR]], [[IDX]] : (!cir.ptr<!u32i>, !u32i) -> !cir.ptr<!u32i> 
// CIR-NEXT: [[VALUE0:%.]] = cir.load align(4) [[VALPTR]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<2> : !u32i
// CIR-NEXT: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR-NEXT: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE]] : !cir.bool

// LLVM-LABEL: define{{.*}} void @test_vpclmulqdq(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// LLVM: = and i32 [[LOAD]], 2

// OGCG-LABEL: define{{.*}} void @test_vpclmulqdq(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// OGCG: = and i32 [[LOAD]], 2
TEST_CPU_SUPPORTS(vpclmulqdq, "vpclmulqdq")

// CIR-LABEL: cir.func no_inline dso_local @test_avx512vnni()
// CIR: [[TRUE:%.]] = cir.const #true
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: [[IDX:%.]] = cir.const #cir.int<0> : !u32i
// CIR-NEXT: [[ARRPTR:%.]] = cir.cast array_to_ptrdecay [[GLOBAL]] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALPTR:%.]] = cir.ptr_stride [[ARRPTR]], [[IDX]] : (!cir.ptr<!u32i>, !u32i) -> !cir.ptr<!u32i> 
// CIR-NEXT: [[VALUE0:%.]] = cir.load align(4) [[VALPTR]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<4> : !u32i
// CIR-NEXT: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR-NEXT: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE]] : !cir.bool

// LLVM-LABEL: define{{.*}} void @test_avx512vnni(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// LLVM: = and i32 [[LOAD]], 4

// OGCG-LABEL: define{{.*}} void @test_avx512vnni(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// OGCG: = and i32 [[LOAD]], 4
TEST_CPU_SUPPORTS(avx512vnni, "avx512vnni")

// CIR-LABEL: cir.func no_inline dso_local @test_avx512bitalg()
// CIR: [[TRUE:%.]] = cir.const #true
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: [[IDX:%.]] = cir.const #cir.int<0> : !u32i
// CIR-NEXT: [[ARRPTR:%.]] = cir.cast array_to_ptrdecay [[GLOBAL]] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALPTR:%.]] = cir.ptr_stride [[ARRPTR]], [[IDX]] : (!cir.ptr<!u32i>, !u32i) -> !cir.ptr<!u32i> 
// CIR-NEXT: [[VALUE0:%.]] = cir.load align(4) [[VALPTR]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<8> : !u32i
// CIR-NEXT: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR-NEXT: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE]] : !cir.bool

// LLVM-LABEL: define{{.*}} void @test_avx512bitalg(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// LLVM: = and i32 [[LOAD]], 8

// OGCG-LABEL: define{{.*}} void @test_avx512bitalg(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// OGCG: = and i32 [[LOAD]], 8
TEST_CPU_SUPPORTS(avx512bitalg, "avx512bitalg")

// CIR-LABEL: cir.func no_inline dso_local @test_avx512bf16()
// CIR: [[TRUE:%.]] = cir.const #true
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: [[IDX:%.]] = cir.const #cir.int<0> : !u32i
// CIR-NEXT: [[ARRPTR:%.]] = cir.cast array_to_ptrdecay [[GLOBAL]] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALPTR:%.]] = cir.ptr_stride [[ARRPTR]], [[IDX]] : (!cir.ptr<!u32i>, !u32i) -> !cir.ptr<!u32i> 
// CIR-NEXT: [[VALUE0:%.]] = cir.load align(4) [[VALPTR]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<16> : !u32i
// CIR-NEXT: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR-NEXT: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE]] : !cir.bool

// LLVM-LABEL: define{{.*}} void @test_avx512bf16(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// LLVM: = and i32 [[LOAD]], 16

// OGCG-LABEL: define{{.*}} void @test_avx512bf16(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// OGCG: = and i32 [[LOAD]], 16
TEST_CPU_SUPPORTS(avx512bf16, "avx512bf16")

// CIR-LABEL: cir.func no_inline dso_local @test_avx512vp2intersect()
// CIR: [[TRUE:%.]] = cir.const #true
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: [[IDX:%.]] = cir.const #cir.int<0> : !u32i
// CIR-NEXT: [[ARRPTR:%.]] = cir.cast array_to_ptrdecay [[GLOBAL]] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALPTR:%.]] = cir.ptr_stride [[ARRPTR]], [[IDX]] : (!cir.ptr<!u32i>, !u32i) -> !cir.ptr<!u32i> 
// CIR-NEXT: [[VALUE0:%.]] = cir.load align(4) [[VALPTR]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<32> : !u32i
// CIR-NEXT: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR-NEXT: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE]] : !cir.bool

// LLVM-LABEL: define{{.*}} void @test_avx512vp2intersect(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// LLVM: = and i32 [[LOAD]], 32

// OGCG-LABEL: define{{.*}} void @test_avx512vp2intersect(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// OGCG: = and i32 [[LOAD]], 32
TEST_CPU_SUPPORTS(avx512vp2intersect, "avx512vp2intersect")

// CIR-LABEL: cir.func no_inline dso_local @test_adx()
// CIR: [[TRUE:%.]] = cir.const #true
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: [[IDX:%.]] = cir.const #cir.int<0> : !u32i
// CIR-NEXT: [[ARRPTR:%.]] = cir.cast array_to_ptrdecay [[GLOBAL]] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALPTR:%.]] = cir.ptr_stride [[ARRPTR]], [[IDX]] : (!cir.ptr<!u32i>, !u32i) -> !cir.ptr<!u32i> 
// CIR-NEXT: [[VALUE0:%.]] = cir.load align(4) [[VALPTR]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<256> : !u32i
// CIR-NEXT: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR-NEXT: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE]] : !cir.bool

// LLVM-LABEL: define{{.*}} void @test_adx(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// LLVM: = and i32 [[LOAD]], 256

// OGCG-LABEL: define{{.*}} void @test_adx(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// OGCG: = and i32 [[LOAD]], 256
TEST_CPU_SUPPORTS(adx, "adx")

// CIR-LABEL: cir.func no_inline dso_local @test_cldemote()
// CIR: [[TRUE:%.]] = cir.const #true
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: [[IDX:%.]] = cir.const #cir.int<0> : !u32i
// CIR-NEXT: [[ARRPTR:%.]] = cir.cast array_to_ptrdecay [[GLOBAL]] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALPTR:%.]] = cir.ptr_stride [[ARRPTR]], [[IDX]] : (!cir.ptr<!u32i>, !u32i) -> !cir.ptr<!u32i> 
// CIR-NEXT: [[VALUE0:%.]] = cir.load align(4) [[VALPTR]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<1024> : !u32i
// CIR-NEXT: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR-NEXT: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE]] : !cir.bool

// LLVM-LABEL: define{{.*}} void @test_cldemote(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// LLVM: = and i32 [[LOAD]], 1024

// OGCG-LABEL: define{{.*}} void @test_cldemote(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// OGCG: = and i32 [[LOAD]], 1024
TEST_CPU_SUPPORTS(cldemote, "cldemote")

// CIR-LABEL: cir.func no_inline dso_local @test_clflushopt()
// CIR: [[TRUE:%.]] = cir.const #true
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: [[IDX:%.]] = cir.const #cir.int<0> : !u32i
// CIR-NEXT: [[ARRPTR:%.]] = cir.cast array_to_ptrdecay [[GLOBAL]] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALPTR:%.]] = cir.ptr_stride [[ARRPTR]], [[IDX]] : (!cir.ptr<!u32i>, !u32i) -> !cir.ptr<!u32i> 
// CIR-NEXT: [[VALUE0:%.]] = cir.load align(4) [[VALPTR]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<2048> : !u32i
// CIR-NEXT: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR-NEXT: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE]] : !cir.bool

// LLVM-LABEL: define{{.*}} void @test_clflushopt(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// LLVM: = and i32 [[LOAD]], 2048

// OGCG-LABEL: define{{.*}} void @test_clflushopt(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// OGCG: = and i32 [[LOAD]], 2048
TEST_CPU_SUPPORTS(clflushopt, "clflushopt")

// CIR-LABEL: cir.func no_inline dso_local @test_clwb()
// CIR: [[TRUE:%.]] = cir.const #true
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: [[IDX:%.]] = cir.const #cir.int<0> : !u32i
// CIR-NEXT: [[ARRPTR:%.]] = cir.cast array_to_ptrdecay [[GLOBAL]] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALPTR:%.]] = cir.ptr_stride [[ARRPTR]], [[IDX]] : (!cir.ptr<!u32i>, !u32i) -> !cir.ptr<!u32i> 
// CIR-NEXT: [[VALUE0:%.]] = cir.load align(4) [[VALPTR]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<4096> : !u32i
// CIR-NEXT: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR-NEXT: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE]] : !cir.bool

// LLVM-LABEL: define{{.*}} void @test_clwb(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// LLVM: = and i32 [[LOAD]], 4096

// OGCG-LABEL: define{{.*}} void @test_clwb(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// OGCG: = and i32 [[LOAD]], 4096
TEST_CPU_SUPPORTS(clwb, "clwb")

// CIR-LABEL: cir.func no_inline dso_local @test_clzero()
// CIR: [[TRUE:%.]] = cir.const #true
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: [[IDX:%.]] = cir.const #cir.int<0> : !u32i
// CIR-NEXT: [[ARRPTR:%.]] = cir.cast array_to_ptrdecay [[GLOBAL]] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALPTR:%.]] = cir.ptr_stride [[ARRPTR]], [[IDX]] : (!cir.ptr<!u32i>, !u32i) -> !cir.ptr<!u32i> 
// CIR-NEXT: [[VALUE0:%.]] = cir.load align(4) [[VALPTR]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<8192> : !u32i
// CIR-NEXT: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR-NEXT: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE]] : !cir.bool

// LLVM-LABEL: define{{.*}} void @test_clzero(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// LLVM: = and i32 [[LOAD]], 8192

// OGCG-LABEL: define{{.*}} void @test_clzero(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// OGCG: = and i32 [[LOAD]], 8192
TEST_CPU_SUPPORTS(clzero, "clzero")

// CIR-LABEL: cir.func no_inline dso_local @test_cx16()
// CIR: [[TRUE:%.]] = cir.const #true
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: [[IDX:%.]] = cir.const #cir.int<0> : !u32i
// CIR-NEXT: [[ARRPTR:%.]] = cir.cast array_to_ptrdecay [[GLOBAL]] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALPTR:%.]] = cir.ptr_stride [[ARRPTR]], [[IDX]] : (!cir.ptr<!u32i>, !u32i) -> !cir.ptr<!u32i> 
// CIR-NEXT: [[VALUE0:%.]] = cir.load align(4) [[VALPTR]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<16384> : !u32i
// CIR-NEXT: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR-NEXT: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE]] : !cir.bool

// LLVM-LABEL: define{{.*}} void @test_cx16(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// LLVM: = and i32 [[LOAD]], 16384

// OGCG-LABEL: define{{.*}} void @test_cx16(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// OGCG: = and i32 [[LOAD]], 16384
TEST_CPU_SUPPORTS(cx16, "cx16")

// CIR-LABEL: cir.func no_inline dso_local @test_enqcmd()
// CIR: [[TRUE:%.]] = cir.const #true
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: [[IDX:%.]] = cir.const #cir.int<0> : !u32i
// CIR-NEXT: [[ARRPTR:%.]] = cir.cast array_to_ptrdecay [[GLOBAL]] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALPTR:%.]] = cir.ptr_stride [[ARRPTR]], [[IDX]] : (!cir.ptr<!u32i>, !u32i) -> !cir.ptr<!u32i> 
// CIR-NEXT: [[VALUE0:%.]] = cir.load align(4) [[VALPTR]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<65536> : !u32i
// CIR-NEXT: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR-NEXT: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE]] : !cir.bool

// LLVM-LABEL: define{{.*}} void @test_enqcmd(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// LLVM: = and i32 [[LOAD]], 65536

// OGCG-LABEL: define{{.*}} void @test_enqcmd(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// OGCG: = and i32 [[LOAD]], 65536
TEST_CPU_SUPPORTS(enqcmd, "enqcmd")

// CIR-LABEL: cir.func no_inline dso_local @test_f16c()
// CIR: [[TRUE:%.]] = cir.const #true
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: [[IDX:%.]] = cir.const #cir.int<0> : !u32i
// CIR-NEXT: [[ARRPTR:%.]] = cir.cast array_to_ptrdecay [[GLOBAL]] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALPTR:%.]] = cir.ptr_stride [[ARRPTR]], [[IDX]] : (!cir.ptr<!u32i>, !u32i) -> !cir.ptr<!u32i> 
// CIR-NEXT: [[VALUE0:%.]] = cir.load align(4) [[VALPTR]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<131072> : !u32i
// CIR-NEXT: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR-NEXT: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE]] : !cir.bool

// LLVM-LABEL: define{{.*}} void @test_f16c(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// LLVM: = and i32 [[LOAD]], 131072

// OGCG-LABEL: define{{.*}} void @test_f16c(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// OGCG: = and i32 [[LOAD]], 131072
TEST_CPU_SUPPORTS(f16c, "f16c")

// CIR-LABEL: cir.func no_inline dso_local @test_fsgsbase()
// CIR: [[TRUE:%.]] = cir.const #true
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: [[IDX:%.]] = cir.const #cir.int<0> : !u32i
// CIR-NEXT: [[ARRPTR:%.]] = cir.cast array_to_ptrdecay [[GLOBAL]] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALPTR:%.]] = cir.ptr_stride [[ARRPTR]], [[IDX]] : (!cir.ptr<!u32i>, !u32i) -> !cir.ptr<!u32i> 
// CIR-NEXT: [[VALUE0:%.]] = cir.load align(4) [[VALPTR]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<262144> : !u32i
// CIR-NEXT: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR-NEXT: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE]] : !cir.bool

// LLVM-LABEL: define{{.*}} void @test_fsgsbase(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// LLVM: = and i32 [[LOAD]], 262144

// OGCG-LABEL: define{{.*}} void @test_fsgsbase(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// OGCG: = and i32 [[LOAD]], 262144
TEST_CPU_SUPPORTS(fsgsbase, "fsgsbase")

// CIR-LABEL: cir.func no_inline dso_local @test_sahf()
// CIR: [[TRUE:%.]] = cir.const #true
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: [[IDX:%.]] = cir.const #cir.int<0> : !u32i
// CIR-NEXT: [[ARRPTR:%.]] = cir.cast array_to_ptrdecay [[GLOBAL]] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALPTR:%.]] = cir.ptr_stride [[ARRPTR]], [[IDX]] : (!cir.ptr<!u32i>, !u32i) -> !cir.ptr<!u32i> 
// CIR-NEXT: [[VALUE0:%.]] = cir.load align(4) [[VALPTR]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<4194304> : !u32i
// CIR-NEXT: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR-NEXT: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE]] : !cir.bool

// LLVM-LABEL: define{{.*}} void @test_sahf(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// LLVM: = and i32 [[LOAD]], 4194304

// OGCG-LABEL: define{{.*}} void @test_sahf(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// OGCG: = and i32 [[LOAD]], 4194304
TEST_CPU_SUPPORTS(sahf, "sahf")

// CIR-LABEL: cir.func no_inline dso_local @test_64bit()
// CIR: [[TRUE:%.]] = cir.const #true
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: [[IDX:%.]] = cir.const #cir.int<0> : !u32i
// CIR-NEXT: [[ARRPTR:%.]] = cir.cast array_to_ptrdecay [[GLOBAL]] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALPTR:%.]] = cir.ptr_stride [[ARRPTR]], [[IDX]] : (!cir.ptr<!u32i>, !u32i) -> !cir.ptr<!u32i> 
// CIR-NEXT: [[VALUE0:%.]] = cir.load align(4) [[VALPTR]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<8388608> : !u32i
// CIR-NEXT: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR-NEXT: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE]] : !cir.bool

// LLVM-LABEL: define{{.*}} void @test_64bit(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// LLVM: = and i32 [[LOAD]], 8388608

// OGCG-LABEL: define{{.*}} void @test_64bit(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// OGCG: = and i32 [[LOAD]], 8388608
TEST_CPU_SUPPORTS(64bit, "64bit")

// CIR-LABEL: cir.func no_inline dso_local @test_lwp()
// CIR: [[TRUE:%.]] = cir.const #true
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: [[IDX:%.]] = cir.const #cir.int<0> : !u32i
// CIR-NEXT: [[ARRPTR:%.]] = cir.cast array_to_ptrdecay [[GLOBAL]] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALPTR:%.]] = cir.ptr_stride [[ARRPTR]], [[IDX]] : (!cir.ptr<!u32i>, !u32i) -> !cir.ptr<!u32i> 
// CIR-NEXT: [[VALUE0:%.]] = cir.load align(4) [[VALPTR]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<16777216> : !u32i
// CIR-NEXT: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR-NEXT: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE]] : !cir.bool

// LLVM-LABEL: define{{.*}} void @test_lwp(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// LLVM: = and i32 [[LOAD]], 16777216

// OGCG-LABEL: define{{.*}} void @test_lwp(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// OGCG: = and i32 [[LOAD]], 16777216
TEST_CPU_SUPPORTS(lwp, "lwp")

// CIR-LABEL: cir.func no_inline dso_local @test_lzcnt()
// CIR: [[TRUE:%.]] = cir.const #true
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: [[IDX:%.]] = cir.const #cir.int<0> : !u32i
// CIR-NEXT: [[ARRPTR:%.]] = cir.cast array_to_ptrdecay [[GLOBAL]] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALPTR:%.]] = cir.ptr_stride [[ARRPTR]], [[IDX]] : (!cir.ptr<!u32i>, !u32i) -> !cir.ptr<!u32i> 
// CIR-NEXT: [[VALUE0:%.]] = cir.load align(4) [[VALPTR]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<33554432> : !u32i
// CIR-NEXT: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR-NEXT: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE]] : !cir.bool

// LLVM-LABEL: define{{.*}} void @test_lzcnt(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// LLVM: = and i32 [[LOAD]], 33554432

// OGCG-LABEL: define{{.*}} void @test_lzcnt(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// OGCG: = and i32 [[LOAD]], 33554432
TEST_CPU_SUPPORTS(lzcnt, "lzcnt")

// CIR-LABEL: cir.func no_inline dso_local @test_movbe()
// CIR: [[TRUE:%.]] = cir.const #true
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: [[IDX:%.]] = cir.const #cir.int<0> : !u32i
// CIR-NEXT: [[ARRPTR:%.]] = cir.cast array_to_ptrdecay [[GLOBAL]] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALPTR:%.]] = cir.ptr_stride [[ARRPTR]], [[IDX]] : (!cir.ptr<!u32i>, !u32i) -> !cir.ptr<!u32i> 
// CIR-NEXT: [[VALUE0:%.]] = cir.load align(4) [[VALPTR]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<67108864> : !u32i
// CIR-NEXT: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR-NEXT: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE]] : !cir.bool

// LLVM-LABEL: define{{.*}} void @test_movbe(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// LLVM: = and i32 [[LOAD]], 67108864

// OGCG-LABEL: define{{.*}} void @test_movbe(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// OGCG: = and i32 [[LOAD]], 67108864
TEST_CPU_SUPPORTS(movbe, "movbe")

// CIR-LABEL: cir.func no_inline dso_local @test_movdir64b()
// CIR: [[TRUE:%.]] = cir.const #true
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: [[IDX:%.]] = cir.const #cir.int<0> : !u32i
// CIR-NEXT: [[ARRPTR:%.]] = cir.cast array_to_ptrdecay [[GLOBAL]] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALPTR:%.]] = cir.ptr_stride [[ARRPTR]], [[IDX]] : (!cir.ptr<!u32i>, !u32i) -> !cir.ptr<!u32i> 
// CIR-NEXT: [[VALUE0:%.]] = cir.load align(4) [[VALPTR]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<134217728> : !u32i
// CIR-NEXT: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR-NEXT: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE]] : !cir.bool

// LLVM-LABEL: define{{.*}} void @test_movdir64b(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// LLVM: = and i32 [[LOAD]], 134217728

// OGCG-LABEL: define{{.*}} void @test_movdir64b(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// OGCG: = and i32 [[LOAD]], 134217728
TEST_CPU_SUPPORTS(movdir64b, "movdir64b")

// CIR-LABEL: cir.func no_inline dso_local @test_movdiri()
// CIR: [[TRUE:%.]] = cir.const #true
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: [[IDX:%.]] = cir.const #cir.int<0> : !u32i
// CIR-NEXT: [[ARRPTR:%.]] = cir.cast array_to_ptrdecay [[GLOBAL]] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALPTR:%.]] = cir.ptr_stride [[ARRPTR]], [[IDX]] : (!cir.ptr<!u32i>, !u32i) -> !cir.ptr<!u32i> 
// CIR-NEXT: [[VALUE0:%.]] = cir.load align(4) [[VALPTR]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<268435456> : !u32i
// CIR-NEXT: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR-NEXT: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE]] : !cir.bool

// LLVM-LABEL: define{{.*}} void @test_movdiri(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// LLVM: = and i32 [[LOAD]], 268435456

// OGCG-LABEL: define{{.*}} void @test_movdiri(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// OGCG: = and i32 [[LOAD]], 268435456
TEST_CPU_SUPPORTS(movdiri, "movdiri")

// CIR-LABEL: cir.func no_inline dso_local @test_mwaitx()
// CIR: [[TRUE:%.]] = cir.const #true
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: [[IDX:%.]] = cir.const #cir.int<0> : !u32i
// CIR-NEXT: [[ARRPTR:%.]] = cir.cast array_to_ptrdecay [[GLOBAL]] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALPTR:%.]] = cir.ptr_stride [[ARRPTR]], [[IDX]] : (!cir.ptr<!u32i>, !u32i) -> !cir.ptr<!u32i> 
// CIR-NEXT: [[VALUE0:%.]] = cir.load align(4) [[VALPTR]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<536870912> : !u32i
// CIR-NEXT: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR-NEXT: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE]] : !cir.bool

// LLVM-LABEL: define{{.*}} void @test_mwaitx(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// LLVM: = and i32 [[LOAD]], 536870912

// OGCG-LABEL: define{{.*}} void @test_mwaitx(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// OGCG: = and i32 [[LOAD]], 536870912
TEST_CPU_SUPPORTS(mwaitx, "mwaitx")

// CIR-LABEL: cir.func no_inline dso_local @test_pconfig()
// CIR: [[TRUE:%.]] = cir.const #true
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: [[IDX:%.]] = cir.const #cir.int<0> : !u32i
// CIR-NEXT: [[ARRPTR:%.]] = cir.cast array_to_ptrdecay [[GLOBAL]] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALPTR:%.]] = cir.ptr_stride [[ARRPTR]], [[IDX]] : (!cir.ptr<!u32i>, !u32i) -> !cir.ptr<!u32i> 
// CIR-NEXT: [[VALUE0:%.]] = cir.load align(4) [[VALPTR]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<2147483648> : !u32i
// CIR-NEXT: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR-NEXT: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE]] : !cir.bool

// LLVM-LABEL: define{{.*}} void @test_pconfig(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// LLVM: = and i32 [[LOAD]], -2147483648

// OGCG-LABEL: define{{.*}} void @test_pconfig(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
// OGCG: = and i32 [[LOAD]], -2147483648
TEST_CPU_SUPPORTS(pconfig, "pconfig")

// CIR-LABEL: cir.func no_inline dso_local @test_pku()
// CIR: [[TRUE:%.]] = cir.const #true
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: [[IDX:%.]] = cir.const #cir.int<1> : !u32i
// CIR-NEXT: [[ARRPTR:%.]] = cir.cast array_to_ptrdecay [[GLOBAL]] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALPTR:%.]] = cir.ptr_stride [[ARRPTR]], [[IDX]] : (!cir.ptr<!u32i>, !u32i) -> !cir.ptr<!u32i> 
// CIR-NEXT: [[VALUE0:%.]] = cir.load align(4) [[VALPTR]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<1> : !u32i
// CIR-NEXT: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR-NEXT: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE]] : !cir.bool

// LLVM-LABEL: define{{.*}} void @test_pku(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// LLVM: = and i32 [[LOAD]], 1

// OGCG-LABEL: define{{.*}} void @test_pku(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// OGCG: = and i32 [[LOAD]], 1
TEST_CPU_SUPPORTS(pku, "pku")

// CIR-LABEL: cir.func no_inline dso_local @test_prfchw()
// CIR: [[TRUE:%.]] = cir.const #true
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: [[IDX:%.]] = cir.const #cir.int<1> : !u32i
// CIR-NEXT: [[ARRPTR:%.]] = cir.cast array_to_ptrdecay [[GLOBAL]] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALPTR:%.]] = cir.ptr_stride [[ARRPTR]], [[IDX]] : (!cir.ptr<!u32i>, !u32i) -> !cir.ptr<!u32i> 
// CIR-NEXT: [[VALUE0:%.]] = cir.load align(4) [[VALPTR]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<4> : !u32i
// CIR-NEXT: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR-NEXT: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE]] : !cir.bool

// LLVM-LABEL: define{{.*}} void @test_prfchw(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// LLVM: = and i32 [[LOAD]], 4

// OGCG-LABEL: define{{.*}} void @test_prfchw(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// OGCG: = and i32 [[LOAD]], 4
TEST_CPU_SUPPORTS(prfchw, "prfchw")

// CIR-LABEL: cir.func no_inline dso_local @test_ptwrite()
// CIR: [[TRUE:%.]] = cir.const #true
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: [[IDX:%.]] = cir.const #cir.int<1> : !u32i
// CIR-NEXT: [[ARRPTR:%.]] = cir.cast array_to_ptrdecay [[GLOBAL]] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALPTR:%.]] = cir.ptr_stride [[ARRPTR]], [[IDX]] : (!cir.ptr<!u32i>, !u32i) -> !cir.ptr<!u32i> 
// CIR-NEXT: [[VALUE0:%.]] = cir.load align(4) [[VALPTR]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<8> : !u32i
// CIR-NEXT: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR-NEXT: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE]] : !cir.bool

// LLVM-LABEL: define{{.*}} void @test_ptwrite(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// LLVM: = and i32 [[LOAD]], 8

// OGCG-LABEL: define{{.*}} void @test_ptwrite(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// OGCG: = and i32 [[LOAD]], 8
TEST_CPU_SUPPORTS(ptwrite, "ptwrite")

// CIR-LABEL: cir.func no_inline dso_local @test_rdpid()
// CIR: [[TRUE:%.]] = cir.const #true
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: [[IDX:%.]] = cir.const #cir.int<1> : !u32i
// CIR-NEXT: [[ARRPTR:%.]] = cir.cast array_to_ptrdecay [[GLOBAL]] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALPTR:%.]] = cir.ptr_stride [[ARRPTR]], [[IDX]] : (!cir.ptr<!u32i>, !u32i) -> !cir.ptr<!u32i> 
// CIR-NEXT: [[VALUE0:%.]] = cir.load align(4) [[VALPTR]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<16> : !u32i
// CIR-NEXT: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR-NEXT: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE]] : !cir.bool

// LLVM-LABEL: define{{.*}} void @test_rdpid(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// LLVM: = and i32 [[LOAD]], 16

// OGCG-LABEL: define{{.*}} void @test_rdpid(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// OGCG: = and i32 [[LOAD]], 16
TEST_CPU_SUPPORTS(rdpid, "rdpid")

// CIR-LABEL: cir.func no_inline dso_local @test_rdrnd()
// CIR: [[TRUE:%.]] = cir.const #true
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: [[IDX:%.]] = cir.const #cir.int<1> : !u32i
// CIR-NEXT: [[ARRPTR:%.]] = cir.cast array_to_ptrdecay [[GLOBAL]] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALPTR:%.]] = cir.ptr_stride [[ARRPTR]], [[IDX]] : (!cir.ptr<!u32i>, !u32i) -> !cir.ptr<!u32i> 
// CIR-NEXT: [[VALUE0:%.]] = cir.load align(4) [[VALPTR]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<32> : !u32i
// CIR-NEXT: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR-NEXT: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE]] : !cir.bool

// LLVM-LABEL: define{{.*}} void @test_rdrnd(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// LLVM: = and i32 [[LOAD]], 32

// OGCG-LABEL: define{{.*}} void @test_rdrnd(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// OGCG: = and i32 [[LOAD]], 32
TEST_CPU_SUPPORTS(rdrnd, "rdrnd")

// CIR-LABEL: cir.func no_inline dso_local @test_rdseed()
// CIR: [[TRUE:%.]] = cir.const #true
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: [[IDX:%.]] = cir.const #cir.int<1> : !u32i
// CIR-NEXT: [[ARRPTR:%.]] = cir.cast array_to_ptrdecay [[GLOBAL]] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALPTR:%.]] = cir.ptr_stride [[ARRPTR]], [[IDX]] : (!cir.ptr<!u32i>, !u32i) -> !cir.ptr<!u32i> 
// CIR-NEXT: [[VALUE0:%.]] = cir.load align(4) [[VALPTR]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<64> : !u32i
// CIR-NEXT: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR-NEXT: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE]] : !cir.bool

// LLVM-LABEL: define{{.*}} void @test_rdseed(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// LLVM: = and i32 [[LOAD]], 64

// OGCG-LABEL: define{{.*}} void @test_rdseed(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// OGCG: = and i32 [[LOAD]], 64
TEST_CPU_SUPPORTS(rdseed, "rdseed")

// CIR-LABEL: cir.func no_inline dso_local @test_rtm()
// CIR: [[TRUE:%.]] = cir.const #true
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: [[IDX:%.]] = cir.const #cir.int<1> : !u32i
// CIR-NEXT: [[ARRPTR:%.]] = cir.cast array_to_ptrdecay [[GLOBAL]] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALPTR:%.]] = cir.ptr_stride [[ARRPTR]], [[IDX]] : (!cir.ptr<!u32i>, !u32i) -> !cir.ptr<!u32i> 
// CIR-NEXT: [[VALUE0:%.]] = cir.load align(4) [[VALPTR]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<128> : !u32i
// CIR-NEXT: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR-NEXT: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE]] : !cir.bool

// LLVM-LABEL: define{{.*}} void @test_rtm(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// LLVM: = and i32 [[LOAD]], 128

// OGCG-LABEL: define{{.*}} void @test_rtm(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// OGCG: = and i32 [[LOAD]], 128
TEST_CPU_SUPPORTS(rtm, "rtm")

// CIR-LABEL: cir.func no_inline dso_local @test_serialize()
// CIR: [[TRUE:%.]] = cir.const #true
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: [[IDX:%.]] = cir.const #cir.int<1> : !u32i
// CIR-NEXT: [[ARRPTR:%.]] = cir.cast array_to_ptrdecay [[GLOBAL]] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALPTR:%.]] = cir.ptr_stride [[ARRPTR]], [[IDX]] : (!cir.ptr<!u32i>, !u32i) -> !cir.ptr<!u32i> 
// CIR-NEXT: [[VALUE0:%.]] = cir.load align(4) [[VALPTR]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<256> : !u32i
// CIR-NEXT: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR-NEXT: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE]] : !cir.bool

// LLVM-LABEL: define{{.*}} void @test_serialize(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// LLVM: = and i32 [[LOAD]], 256

// OGCG-LABEL: define{{.*}} void @test_serialize(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// OGCG: = and i32 [[LOAD]], 256
TEST_CPU_SUPPORTS(serialize, "serialize")

// CIR-LABEL: cir.func no_inline dso_local @test_sgx()
// CIR: [[TRUE:%.]] = cir.const #true
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: [[IDX:%.]] = cir.const #cir.int<1> : !u32i
// CIR-NEXT: [[ARRPTR:%.]] = cir.cast array_to_ptrdecay [[GLOBAL]] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALPTR:%.]] = cir.ptr_stride [[ARRPTR]], [[IDX]] : (!cir.ptr<!u32i>, !u32i) -> !cir.ptr<!u32i> 
// CIR-NEXT: [[VALUE0:%.]] = cir.load align(4) [[VALPTR]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<512> : !u32i
// CIR-NEXT: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR-NEXT: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE]] : !cir.bool

// LLVM-LABEL: define{{.*}} void @test_sgx(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// LLVM: = and i32 [[LOAD]], 512

// OGCG-LABEL: define{{.*}} void @test_sgx(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// OGCG: = and i32 [[LOAD]], 512
TEST_CPU_SUPPORTS(sgx, "sgx")

// CIR-LABEL: cir.func no_inline dso_local @test_sha()
// CIR: [[TRUE:%.]] = cir.const #true
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: [[IDX:%.]] = cir.const #cir.int<1> : !u32i
// CIR-NEXT: [[ARRPTR:%.]] = cir.cast array_to_ptrdecay [[GLOBAL]] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALPTR:%.]] = cir.ptr_stride [[ARRPTR]], [[IDX]] : (!cir.ptr<!u32i>, !u32i) -> !cir.ptr<!u32i> 
// CIR-NEXT: [[VALUE0:%.]] = cir.load align(4) [[VALPTR]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<1024> : !u32i
// CIR-NEXT: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR-NEXT: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE]] : !cir.bool

// LLVM-LABEL: define{{.*}} void @test_sha(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// LLVM: = and i32 [[LOAD]], 1024

// OGCG-LABEL: define{{.*}} void @test_sha(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// OGCG: = and i32 [[LOAD]], 1024
TEST_CPU_SUPPORTS(sha, "sha")

// CIR-LABEL: cir.func no_inline dso_local @test_shstk()
// CIR: [[TRUE:%.]] = cir.const #true
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: [[IDX:%.]] = cir.const #cir.int<1> : !u32i
// CIR-NEXT: [[ARRPTR:%.]] = cir.cast array_to_ptrdecay [[GLOBAL]] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALPTR:%.]] = cir.ptr_stride [[ARRPTR]], [[IDX]] : (!cir.ptr<!u32i>, !u32i) -> !cir.ptr<!u32i> 
// CIR-NEXT: [[VALUE0:%.]] = cir.load align(4) [[VALPTR]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<2048> : !u32i
// CIR-NEXT: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR-NEXT: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE]] : !cir.bool

// LLVM-LABEL: define{{.*}} void @test_shstk(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// LLVM: = and i32 [[LOAD]], 2048

// OGCG-LABEL: define{{.*}} void @test_shstk(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// OGCG: = and i32 [[LOAD]], 2048
TEST_CPU_SUPPORTS(shstk, "shstk")

// CIR-LABEL: cir.func no_inline dso_local @test_tbm()
// CIR: [[TRUE:%.]] = cir.const #true
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: [[IDX:%.]] = cir.const #cir.int<1> : !u32i
// CIR-NEXT: [[ARRPTR:%.]] = cir.cast array_to_ptrdecay [[GLOBAL]] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALPTR:%.]] = cir.ptr_stride [[ARRPTR]], [[IDX]] : (!cir.ptr<!u32i>, !u32i) -> !cir.ptr<!u32i> 
// CIR-NEXT: [[VALUE0:%.]] = cir.load align(4) [[VALPTR]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<4096> : !u32i
// CIR-NEXT: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR-NEXT: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE]] : !cir.bool

// LLVM-LABEL: define{{.*}} void @test_tbm(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// LLVM: = and i32 [[LOAD]], 4096

// OGCG-LABEL: define{{.*}} void @test_tbm(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// OGCG: = and i32 [[LOAD]], 4096
TEST_CPU_SUPPORTS(tbm, "tbm")

// CIR-LABEL: cir.func no_inline dso_local @test_tsxldtrk()
// CIR: [[TRUE:%.]] = cir.const #true
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: [[IDX:%.]] = cir.const #cir.int<1> : !u32i
// CIR-NEXT: [[ARRPTR:%.]] = cir.cast array_to_ptrdecay [[GLOBAL]] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALPTR:%.]] = cir.ptr_stride [[ARRPTR]], [[IDX]] : (!cir.ptr<!u32i>, !u32i) -> !cir.ptr<!u32i> 
// CIR-NEXT: [[VALUE0:%.]] = cir.load align(4) [[VALPTR]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<8192> : !u32i
// CIR-NEXT: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR-NEXT: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE]] : !cir.bool

// LLVM-LABEL: define{{.*}} void @test_tsxldtrk(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// LLVM: = and i32 [[LOAD]], 8192

// OGCG-LABEL: define{{.*}} void @test_tsxldtrk(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// OGCG: = and i32 [[LOAD]], 8192
TEST_CPU_SUPPORTS(tsxldtrk, "tsxldtrk")

// CIR-LABEL: cir.func no_inline dso_local @test_vaes()
// CIR: [[TRUE:%.]] = cir.const #true
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: [[IDX:%.]] = cir.const #cir.int<1> : !u32i
// CIR-NEXT: [[ARRPTR:%.]] = cir.cast array_to_ptrdecay [[GLOBAL]] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALPTR:%.]] = cir.ptr_stride [[ARRPTR]], [[IDX]] : (!cir.ptr<!u32i>, !u32i) -> !cir.ptr<!u32i> 
// CIR-NEXT: [[VALUE0:%.]] = cir.load align(4) [[VALPTR]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<16384> : !u32i
// CIR-NEXT: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR-NEXT: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE]] : !cir.bool

// LLVM-LABEL: define{{.*}} void @test_vaes(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// LLVM: = and i32 [[LOAD]], 16384

// OGCG-LABEL: define{{.*}} void @test_vaes(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// OGCG: = and i32 [[LOAD]], 16384
TEST_CPU_SUPPORTS(vaes, "vaes")

// CIR-LABEL: cir.func no_inline dso_local @test_waitpkg()
// CIR: [[TRUE:%.]] = cir.const #true
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: [[IDX:%.]] = cir.const #cir.int<1> : !u32i
// CIR-NEXT: [[ARRPTR:%.]] = cir.cast array_to_ptrdecay [[GLOBAL]] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALPTR:%.]] = cir.ptr_stride [[ARRPTR]], [[IDX]] : (!cir.ptr<!u32i>, !u32i) -> !cir.ptr<!u32i> 
// CIR-NEXT: [[VALUE0:%.]] = cir.load align(4) [[VALPTR]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<32768> : !u32i
// CIR-NEXT: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR-NEXT: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE]] : !cir.bool

// LLVM-LABEL: define{{.*}} void @test_waitpkg(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// LLVM: = and i32 [[LOAD]], 32768

// OGCG-LABEL: define{{.*}} void @test_waitpkg(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// OGCG: = and i32 [[LOAD]], 32768
TEST_CPU_SUPPORTS(waitpkg, "waitpkg")

// CIR-LABEL: cir.func no_inline dso_local @test_wbnoinvd()
// CIR: [[TRUE:%.]] = cir.const #true
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: [[IDX:%.]] = cir.const #cir.int<1> : !u32i
// CIR-NEXT: [[ARRPTR:%.]] = cir.cast array_to_ptrdecay [[GLOBAL]] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALPTR:%.]] = cir.ptr_stride [[ARRPTR]], [[IDX]] : (!cir.ptr<!u32i>, !u32i) -> !cir.ptr<!u32i> 
// CIR-NEXT: [[VALUE0:%.]] = cir.load align(4) [[VALPTR]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<65536> : !u32i
// CIR-NEXT: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR-NEXT: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE]] : !cir.bool

// LLVM-LABEL: define{{.*}} void @test_wbnoinvd(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// LLVM: = and i32 [[LOAD]], 65536

// OGCG-LABEL: define{{.*}} void @test_wbnoinvd(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// OGCG: = and i32 [[LOAD]], 65536
TEST_CPU_SUPPORTS(wbnoinvd, "wbnoinvd")

// CIR-LABEL: cir.func no_inline dso_local @test_xsave()
// CIR: [[TRUE:%.]] = cir.const #true
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: [[IDX:%.]] = cir.const #cir.int<1> : !u32i
// CIR-NEXT: [[ARRPTR:%.]] = cir.cast array_to_ptrdecay [[GLOBAL]] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALPTR:%.]] = cir.ptr_stride [[ARRPTR]], [[IDX]] : (!cir.ptr<!u32i>, !u32i) -> !cir.ptr<!u32i> 
// CIR-NEXT: [[VALUE0:%.]] = cir.load align(4) [[VALPTR]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<131072> : !u32i
// CIR-NEXT: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR-NEXT: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE]] : !cir.bool

// LLVM-LABEL: define{{.*}} void @test_xsave(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// LLVM: = and i32 [[LOAD]], 131072

// OGCG-LABEL: define{{.*}} void @test_xsave(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// OGCG: = and i32 [[LOAD]], 131072
TEST_CPU_SUPPORTS(xsave, "xsave")

// CIR-LABEL: cir.func no_inline dso_local @test_xsavec()
// CIR: [[TRUE:%.]] = cir.const #true
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: [[IDX:%.]] = cir.const #cir.int<1> : !u32i
// CIR-NEXT: [[ARRPTR:%.]] = cir.cast array_to_ptrdecay [[GLOBAL]] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALPTR:%.]] = cir.ptr_stride [[ARRPTR]], [[IDX]] : (!cir.ptr<!u32i>, !u32i) -> !cir.ptr<!u32i> 
// CIR-NEXT: [[VALUE0:%.]] = cir.load align(4) [[VALPTR]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<262144> : !u32i
// CIR-NEXT: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR-NEXT: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE]] : !cir.bool

// LLVM-LABEL: define{{.*}} void @test_xsavec(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// LLVM: = and i32 [[LOAD]], 262144

// OGCG-LABEL: define{{.*}} void @test_xsavec(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// OGCG: = and i32 [[LOAD]], 262144
TEST_CPU_SUPPORTS(xsavec, "xsavec")

// CIR-LABEL: cir.func no_inline dso_local @test_xsaveopt()
// CIR: [[TRUE:%.]] = cir.const #true
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: [[IDX:%.]] = cir.const #cir.int<1> : !u32i
// CIR-NEXT: [[ARRPTR:%.]] = cir.cast array_to_ptrdecay [[GLOBAL]] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALPTR:%.]] = cir.ptr_stride [[ARRPTR]], [[IDX]] : (!cir.ptr<!u32i>, !u32i) -> !cir.ptr<!u32i> 
// CIR-NEXT: [[VALUE0:%.]] = cir.load align(4) [[VALPTR]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<524288> : !u32i
// CIR-NEXT: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR-NEXT: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE]] : !cir.bool

// LLVM-LABEL: define{{.*}} void @test_xsaveopt(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// LLVM: = and i32 [[LOAD]], 524288

// OGCG-LABEL: define{{.*}} void @test_xsaveopt(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// OGCG: = and i32 [[LOAD]], 524288
TEST_CPU_SUPPORTS(xsaveopt, "xsaveopt")

// CIR-LABEL: cir.func no_inline dso_local @test_xsaves()
// CIR: [[TRUE:%.]] = cir.const #true
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: [[IDX:%.]] = cir.const #cir.int<1> : !u32i
// CIR-NEXT: [[ARRPTR:%.]] = cir.cast array_to_ptrdecay [[GLOBAL]] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALPTR:%.]] = cir.ptr_stride [[ARRPTR]], [[IDX]] : (!cir.ptr<!u32i>, !u32i) -> !cir.ptr<!u32i> 
// CIR-NEXT: [[VALUE0:%.]] = cir.load align(4) [[VALPTR]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<1048576> : !u32i
// CIR-NEXT: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR-NEXT: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE]] : !cir.bool

// LLVM-LABEL: define{{.*}} void @test_xsaves(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// LLVM: = and i32 [[LOAD]], 1048576

// OGCG-LABEL: define{{.*}} void @test_xsaves(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// OGCG: = and i32 [[LOAD]], 1048576
TEST_CPU_SUPPORTS(xsaves, "xsaves")

// CIR-LABEL: cir.func no_inline dso_local @test_amx_tile()
// CIR: [[TRUE:%.]] = cir.const #true
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: [[IDX:%.]] = cir.const #cir.int<1> : !u32i
// CIR-NEXT: [[ARRPTR:%.]] = cir.cast array_to_ptrdecay [[GLOBAL]] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALPTR:%.]] = cir.ptr_stride [[ARRPTR]], [[IDX]] : (!cir.ptr<!u32i>, !u32i) -> !cir.ptr<!u32i> 
// CIR-NEXT: [[VALUE0:%.]] = cir.load align(4) [[VALPTR]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<2097152> : !u32i
// CIR-NEXT: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR-NEXT: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE]] : !cir.bool

// LLVM-LABEL: define{{.*}} void @test_amx_tile(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// LLVM: = and i32 [[LOAD]], 2097152

// OGCG-LABEL: define{{.*}} void @test_amx_tile(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// OGCG: = and i32 [[LOAD]], 2097152
TEST_CPU_SUPPORTS(amx_tile, "amx-tile")

// CIR-LABEL: cir.func no_inline dso_local @test_amx_int8()
// CIR: [[TRUE:%.]] = cir.const #true
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: [[IDX:%.]] = cir.const #cir.int<1> : !u32i
// CIR-NEXT: [[ARRPTR:%.]] = cir.cast array_to_ptrdecay [[GLOBAL]] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALPTR:%.]] = cir.ptr_stride [[ARRPTR]], [[IDX]] : (!cir.ptr<!u32i>, !u32i) -> !cir.ptr<!u32i> 
// CIR-NEXT: [[VALUE0:%.]] = cir.load align(4) [[VALPTR]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<4194304> : !u32i
// CIR-NEXT: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR-NEXT: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE]] : !cir.bool

// LLVM-LABEL: define{{.*}} void @test_amx_int8(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// LLVM: = and i32 [[LOAD]], 4194304

// OGCG-LABEL: define{{.*}} void @test_amx_int8(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// OGCG: = and i32 [[LOAD]], 4194304
TEST_CPU_SUPPORTS(amx_int8, "amx-int8")

// CIR-LABEL: cir.func no_inline dso_local @test_amx_bf16()
// CIR: [[TRUE:%.]] = cir.const #true
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: [[IDX:%.]] = cir.const #cir.int<1> : !u32i
// CIR-NEXT: [[ARRPTR:%.]] = cir.cast array_to_ptrdecay [[GLOBAL]] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALPTR:%.]] = cir.ptr_stride [[ARRPTR]], [[IDX]] : (!cir.ptr<!u32i>, !u32i) -> !cir.ptr<!u32i> 
// CIR-NEXT: [[VALUE0:%.]] = cir.load align(4) [[VALPTR]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<8388608> : !u32i
// CIR-NEXT: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR-NEXT: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE]] : !cir.bool

// LLVM-LABEL: define{{.*}} void @test_amx_bf16(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// LLVM: = and i32 [[LOAD]], 8388608

// OGCG-LABEL: define{{.*}} void @test_amx_bf16(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// OGCG: = and i32 [[LOAD]], 8388608
TEST_CPU_SUPPORTS(amx_bf16, "amx-bf16")

// CIR-LABEL: cir.func no_inline dso_local @test_uintr()
// CIR: [[TRUE:%.]] = cir.const #true
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: [[IDX:%.]] = cir.const #cir.int<1> : !u32i
// CIR-NEXT: [[ARRPTR:%.]] = cir.cast array_to_ptrdecay [[GLOBAL]] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALPTR:%.]] = cir.ptr_stride [[ARRPTR]], [[IDX]] : (!cir.ptr<!u32i>, !u32i) -> !cir.ptr<!u32i> 
// CIR-NEXT: [[VALUE0:%.]] = cir.load align(4) [[VALPTR]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<16777216> : !u32i
// CIR-NEXT: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR-NEXT: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE]] : !cir.bool

// LLVM-LABEL: define{{.*}} void @test_uintr(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// LLVM: = and i32 [[LOAD]], 16777216

// OGCG-LABEL: define{{.*}} void @test_uintr(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// OGCG: = and i32 [[LOAD]], 16777216
TEST_CPU_SUPPORTS(uintr, "uintr")

// CIR-LABEL: cir.func no_inline dso_local @test_hreset()
// CIR: [[TRUE:%.]] = cir.const #true
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: [[IDX:%.]] = cir.const #cir.int<1> : !u32i
// CIR-NEXT: [[ARRPTR:%.]] = cir.cast array_to_ptrdecay [[GLOBAL]] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALPTR:%.]] = cir.ptr_stride [[ARRPTR]], [[IDX]] : (!cir.ptr<!u32i>, !u32i) -> !cir.ptr<!u32i> 
// CIR-NEXT: [[VALUE0:%.]] = cir.load align(4) [[VALPTR]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<33554432> : !u32i
// CIR-NEXT: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR-NEXT: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE]] : !cir.bool

// LLVM-LABEL: define{{.*}} void @test_hreset(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// LLVM: = and i32 [[LOAD]], 33554432

// OGCG-LABEL: define{{.*}} void @test_hreset(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// OGCG: = and i32 [[LOAD]], 33554432
TEST_CPU_SUPPORTS(hreset, "hreset")

// CIR-LABEL: cir.func no_inline dso_local @test_kl()
// CIR: [[TRUE:%.]] = cir.const #true
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: [[IDX:%.]] = cir.const #cir.int<1> : !u32i
// CIR-NEXT: [[ARRPTR:%.]] = cir.cast array_to_ptrdecay [[GLOBAL]] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALPTR:%.]] = cir.ptr_stride [[ARRPTR]], [[IDX]] : (!cir.ptr<!u32i>, !u32i) -> !cir.ptr<!u32i> 
// CIR-NEXT: [[VALUE0:%.]] = cir.load align(4) [[VALPTR]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<67108864> : !u32i
// CIR-NEXT: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR-NEXT: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE]] : !cir.bool

// LLVM-LABEL: define{{.*}} void @test_kl(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// LLVM: = and i32 [[LOAD]], 67108864

// OGCG-LABEL: define{{.*}} void @test_kl(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// OGCG: = and i32 [[LOAD]], 67108864
TEST_CPU_SUPPORTS(kl, "kl")

// CIR-LABEL: cir.func no_inline dso_local @test_widekl()
// CIR: [[TRUE:%.]] = cir.const #true
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: [[IDX:%.]] = cir.const #cir.int<1> : !u32i
// CIR-NEXT: [[ARRPTR:%.]] = cir.cast array_to_ptrdecay [[GLOBAL]] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALPTR:%.]] = cir.ptr_stride [[ARRPTR]], [[IDX]] : (!cir.ptr<!u32i>, !u32i) -> !cir.ptr<!u32i> 
// CIR-NEXT: [[VALUE0:%.]] = cir.load align(4) [[VALPTR]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<268435456> : !u32i
// CIR-NEXT: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR-NEXT: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE]] : !cir.bool

// LLVM-LABEL: define{{.*}} void @test_widekl(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// LLVM: = and i32 [[LOAD]], 268435456

// OGCG-LABEL: define{{.*}} void @test_widekl(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// OGCG: = and i32 [[LOAD]], 268435456
TEST_CPU_SUPPORTS(widekl, "widekl")

// CIR-LABEL: cir.func no_inline dso_local @test_avxvnni()
// CIR: [[TRUE:%.]] = cir.const #true
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: [[IDX:%.]] = cir.const #cir.int<1> : !u32i
// CIR-NEXT: [[ARRPTR:%.]] = cir.cast array_to_ptrdecay [[GLOBAL]] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALPTR:%.]] = cir.ptr_stride [[ARRPTR]], [[IDX]] : (!cir.ptr<!u32i>, !u32i) -> !cir.ptr<!u32i> 
// CIR-NEXT: [[VALUE0:%.]] = cir.load align(4) [[VALPTR]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<536870912> : !u32i
// CIR-NEXT: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR-NEXT: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE]] : !cir.bool

// LLVM-LABEL: define{{.*}} void @test_avxvnni(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// LLVM: = and i32 [[LOAD]], 536870912

// OGCG-LABEL: define{{.*}} void @test_avxvnni(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// OGCG: = and i32 [[LOAD]], 536870912
TEST_CPU_SUPPORTS(avxvnni, "avxvnni")

// CIR-LABEL: cir.func no_inline dso_local @test_avx512fp16()
// CIR: [[TRUE:%.]] = cir.const #true
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: [[IDX:%.]] = cir.const #cir.int<1> : !u32i
// CIR-NEXT: [[ARRPTR:%.]] = cir.cast array_to_ptrdecay [[GLOBAL]] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALPTR:%.]] = cir.ptr_stride [[ARRPTR]], [[IDX]] : (!cir.ptr<!u32i>, !u32i) -> !cir.ptr<!u32i> 
// CIR-NEXT: [[VALUE0:%.]] = cir.load align(4) [[VALPTR]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<1073741824> : !u32i
// CIR-NEXT: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR-NEXT: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE]] : !cir.bool

// LLVM-LABEL: define{{.*}} void @test_avx512fp16(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// LLVM: = and i32 [[LOAD]], 1073741824

// OGCG-LABEL: define{{.*}} void @test_avx512fp16(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// OGCG: = and i32 [[LOAD]], 1073741824
TEST_CPU_SUPPORTS(avx512fp16, "avx512fp16")

// CIR-LABEL: cir.func no_inline dso_local @test_x86_64()
// CIR: [[TRUE:%.]] = cir.const #true
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: [[IDX:%.]] = cir.const #cir.int<1> : !u32i
// CIR-NEXT: [[ARRPTR:%.]] = cir.cast array_to_ptrdecay [[GLOBAL]] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALPTR:%.]] = cir.ptr_stride [[ARRPTR]], [[IDX]] : (!cir.ptr<!u32i>, !u32i) -> !cir.ptr<!u32i> 
// CIR-NEXT: [[VALUE0:%.]] = cir.load align(4) [[VALPTR]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<2147483648> : !u32i
// CIR-NEXT: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR-NEXT: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE]] : !cir.bool

// LLVM-LABEL: define{{.*}} void @test_x86_64(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// LLVM: = and i32 [[LOAD]], -2147483648

// OGCG-LABEL: define{{.*}} void @test_x86_64(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4)
// OGCG: = and i32 [[LOAD]], -2147483648
TEST_CPU_SUPPORTS(x86_64, "x86-64")

// CIR-LABEL: cir.func no_inline dso_local @test_x86_64_v2()
// CIR: [[TRUE:%.]] = cir.const #true
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: [[IDX:%.]] = cir.const #cir.int<2> : !u32i
// CIR-NEXT: [[ARRPTR:%.]] = cir.cast array_to_ptrdecay [[GLOBAL]] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALPTR:%.]] = cir.ptr_stride [[ARRPTR]], [[IDX]] : (!cir.ptr<!u32i>, !u32i) -> !cir.ptr<!u32i> 
// CIR-NEXT: [[VALUE0:%.]] = cir.load align(4) [[VALPTR]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<1> : !u32i
// CIR-NEXT: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR-NEXT: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE]] : !cir.bool

// LLVM-LABEL: define{{.*}} void @test_x86_64_v2(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// LLVM: = and i32 [[LOAD]], 1

// OGCG-LABEL: define{{.*}} void @test_x86_64_v2(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// OGCG: = and i32 [[LOAD]], 1
TEST_CPU_SUPPORTS(x86_64_v2, "x86-64-v2")

// CIR-LABEL: cir.func no_inline dso_local @test_x86_64_v3()
// CIR: [[TRUE:%.]] = cir.const #true
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: [[IDX:%.]] = cir.const #cir.int<2> : !u32i
// CIR-NEXT: [[ARRPTR:%.]] = cir.cast array_to_ptrdecay [[GLOBAL]] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALPTR:%.]] = cir.ptr_stride [[ARRPTR]], [[IDX]] : (!cir.ptr<!u32i>, !u32i) -> !cir.ptr<!u32i> 
// CIR-NEXT: [[VALUE0:%.]] = cir.load align(4) [[VALPTR]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<2> : !u32i
// CIR-NEXT: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR-NEXT: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE]] : !cir.bool

// LLVM-LABEL: define{{.*}} void @test_x86_64_v3(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// LLVM: = and i32 [[LOAD]], 2

// OGCG-LABEL: define{{.*}} void @test_x86_64_v3(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// OGCG: = and i32 [[LOAD]], 2
TEST_CPU_SUPPORTS(x86_64_v3, "x86-64-v3")

// CIR-LABEL: cir.func no_inline dso_local @test_x86_64_v4()
// CIR: [[TRUE:%.]] = cir.const #true
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: [[IDX:%.]] = cir.const #cir.int<2> : !u32i
// CIR-NEXT: [[ARRPTR:%.]] = cir.cast array_to_ptrdecay [[GLOBAL]] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALPTR:%.]] = cir.ptr_stride [[ARRPTR]], [[IDX]] : (!cir.ptr<!u32i>, !u32i) -> !cir.ptr<!u32i> 
// CIR-NEXT: [[VALUE0:%.]] = cir.load align(4) [[VALPTR]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<4> : !u32i
// CIR-NEXT: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR-NEXT: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE]] : !cir.bool

// LLVM-LABEL: define{{.*}} void @test_x86_64_v4(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// LLVM: = and i32 [[LOAD]], 4

// OGCG-LABEL: define{{.*}} void @test_x86_64_v4(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// OGCG: = and i32 [[LOAD]], 4
TEST_CPU_SUPPORTS(x86_64_v4, "x86-64-v4")

// CIR-LABEL: cir.func no_inline dso_local @test_avxifma()
// CIR: [[TRUE:%.]] = cir.const #true
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: [[IDX:%.]] = cir.const #cir.int<2> : !u32i
// CIR-NEXT: [[ARRPTR:%.]] = cir.cast array_to_ptrdecay [[GLOBAL]] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALPTR:%.]] = cir.ptr_stride [[ARRPTR]], [[IDX]] : (!cir.ptr<!u32i>, !u32i) -> !cir.ptr<!u32i> 
// CIR-NEXT: [[VALUE0:%.]] = cir.load align(4) [[VALPTR]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<8> : !u32i
// CIR-NEXT: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR-NEXT: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE]] : !cir.bool

// LLVM-LABEL: define{{.*}} void @test_avxifma(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// LLVM: = and i32 [[LOAD]], 8

// OGCG-LABEL: define{{.*}} void @test_avxifma(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// OGCG: = and i32 [[LOAD]], 8
TEST_CPU_SUPPORTS(avxifma, "avxifma")

// CIR-LABEL: cir.func no_inline dso_local @test_avxvnniint8()
// CIR: [[TRUE:%.]] = cir.const #true
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: [[IDX:%.]] = cir.const #cir.int<2> : !u32i
// CIR-NEXT: [[ARRPTR:%.]] = cir.cast array_to_ptrdecay [[GLOBAL]] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALPTR:%.]] = cir.ptr_stride [[ARRPTR]], [[IDX]] : (!cir.ptr<!u32i>, !u32i) -> !cir.ptr<!u32i> 
// CIR-NEXT: [[VALUE0:%.]] = cir.load align(4) [[VALPTR]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<16> : !u32i
// CIR-NEXT: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR-NEXT: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE]] : !cir.bool

// LLVM-LABEL: define{{.*}} void @test_avxvnniint8(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// LLVM: = and i32 [[LOAD]], 16

// OGCG-LABEL: define{{.*}} void @test_avxvnniint8(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// OGCG: = and i32 [[LOAD]], 16
TEST_CPU_SUPPORTS(avxvnniint8, "avxvnniint8")

// CIR-LABEL: cir.func no_inline dso_local @test_avxneconvert()
// CIR: [[TRUE:%.]] = cir.const #true
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: [[IDX:%.]] = cir.const #cir.int<2> : !u32i
// CIR-NEXT: [[ARRPTR:%.]] = cir.cast array_to_ptrdecay [[GLOBAL]] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALPTR:%.]] = cir.ptr_stride [[ARRPTR]], [[IDX]] : (!cir.ptr<!u32i>, !u32i) -> !cir.ptr<!u32i> 
// CIR-NEXT: [[VALUE0:%.]] = cir.load align(4) [[VALPTR]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<32> : !u32i
// CIR-NEXT: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR-NEXT: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE]] : !cir.bool

// LLVM-LABEL: define{{.*}} void @test_avxneconvert(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// LLVM: = and i32 [[LOAD]], 32

// OGCG-LABEL: define{{.*}} void @test_avxneconvert(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// OGCG: = and i32 [[LOAD]], 32
TEST_CPU_SUPPORTS(avxneconvert, "avxneconvert")

// CIR-LABEL: cir.func no_inline dso_local @test_cmpccxadd()
// CIR: [[TRUE:%.]] = cir.const #true
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: [[IDX:%.]] = cir.const #cir.int<2> : !u32i
// CIR-NEXT: [[ARRPTR:%.]] = cir.cast array_to_ptrdecay [[GLOBAL]] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALPTR:%.]] = cir.ptr_stride [[ARRPTR]], [[IDX]] : (!cir.ptr<!u32i>, !u32i) -> !cir.ptr<!u32i> 
// CIR-NEXT: [[VALUE0:%.]] = cir.load align(4) [[VALPTR]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<64> : !u32i
// CIR-NEXT: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR-NEXT: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE]] : !cir.bool

// LLVM-LABEL: define{{.*}} void @test_cmpccxadd(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// LLVM: = and i32 [[LOAD]], 64

// OGCG-LABEL: define{{.*}} void @test_cmpccxadd(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// OGCG: = and i32 [[LOAD]], 64
TEST_CPU_SUPPORTS(cmpccxadd, "cmpccxadd")

// CIR-LABEL: cir.func no_inline dso_local @test_amx_fp16()
// CIR: [[TRUE:%.]] = cir.const #true
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: [[IDX:%.]] = cir.const #cir.int<2> : !u32i
// CIR-NEXT: [[ARRPTR:%.]] = cir.cast array_to_ptrdecay [[GLOBAL]] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALPTR:%.]] = cir.ptr_stride [[ARRPTR]], [[IDX]] : (!cir.ptr<!u32i>, !u32i) -> !cir.ptr<!u32i> 
// CIR-NEXT: [[VALUE0:%.]] = cir.load align(4) [[VALPTR]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<128> : !u32i
// CIR-NEXT: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR-NEXT: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE]] : !cir.bool

// LLVM-LABEL: define{{.*}} void @test_amx_fp16(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// LLVM: = and i32 [[LOAD]], 128

// OGCG-LABEL: define{{.*}} void @test_amx_fp16(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// OGCG: = and i32 [[LOAD]], 128
TEST_CPU_SUPPORTS(amx_fp16, "amx-fp16")

// CIR-LABEL: cir.func no_inline dso_local @test_prefetchi()
// CIR: [[TRUE:%.]] = cir.const #true
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: [[IDX:%.]] = cir.const #cir.int<2> : !u32i
// CIR-NEXT: [[ARRPTR:%.]] = cir.cast array_to_ptrdecay [[GLOBAL]] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALPTR:%.]] = cir.ptr_stride [[ARRPTR]], [[IDX]] : (!cir.ptr<!u32i>, !u32i) -> !cir.ptr<!u32i> 
// CIR-NEXT: [[VALUE0:%.]] = cir.load align(4) [[VALPTR]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<256> : !u32i
// CIR-NEXT: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR-NEXT: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE]] : !cir.bool

// LLVM-LABEL: define{{.*}} void @test_prefetchi(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// LLVM: = and i32 [[LOAD]], 256

// OGCG-LABEL: define{{.*}} void @test_prefetchi(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// OGCG: = and i32 [[LOAD]], 256
TEST_CPU_SUPPORTS(prefetchi, "prefetchi")

// CIR-LABEL: cir.func no_inline dso_local @test_raoint()
// CIR: [[TRUE:%.]] = cir.const #true
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: [[IDX:%.]] = cir.const #cir.int<2> : !u32i
// CIR-NEXT: [[ARRPTR:%.]] = cir.cast array_to_ptrdecay [[GLOBAL]] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALPTR:%.]] = cir.ptr_stride [[ARRPTR]], [[IDX]] : (!cir.ptr<!u32i>, !u32i) -> !cir.ptr<!u32i> 
// CIR-NEXT: [[VALUE0:%.]] = cir.load align(4) [[VALPTR]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<512> : !u32i
// CIR-NEXT: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR-NEXT: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE]] : !cir.bool

// LLVM-LABEL: define{{.*}} void @test_raoint(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// LLVM: = and i32 [[LOAD]], 512

// OGCG-LABEL: define{{.*}} void @test_raoint(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// OGCG: = and i32 [[LOAD]], 512
TEST_CPU_SUPPORTS(raoint, "raoint")

// CIR-LABEL: cir.func no_inline dso_local @test_amx_complex()
// CIR: [[TRUE:%.]] = cir.const #true
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: [[IDX:%.]] = cir.const #cir.int<2> : !u32i
// CIR-NEXT: [[ARRPTR:%.]] = cir.cast array_to_ptrdecay [[GLOBAL]] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALPTR:%.]] = cir.ptr_stride [[ARRPTR]], [[IDX]] : (!cir.ptr<!u32i>, !u32i) -> !cir.ptr<!u32i> 
// CIR-NEXT: [[VALUE0:%.]] = cir.load align(4) [[VALPTR]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<1024> : !u32i
// CIR-NEXT: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR-NEXT: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE]] : !cir.bool

// LLVM-LABEL: define{{.*}} void @test_amx_complex(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// LLVM: = and i32 [[LOAD]], 1024

// OGCG-LABEL: define{{.*}} void @test_amx_complex(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// OGCG: = and i32 [[LOAD]], 1024
TEST_CPU_SUPPORTS(amx_complex, "amx-complex")

// CIR-LABEL: cir.func no_inline dso_local @test_avxvnniint16()
// CIR: [[TRUE:%.]] = cir.const #true
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: [[IDX:%.]] = cir.const #cir.int<2> : !u32i
// CIR-NEXT: [[ARRPTR:%.]] = cir.cast array_to_ptrdecay [[GLOBAL]] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALPTR:%.]] = cir.ptr_stride [[ARRPTR]], [[IDX]] : (!cir.ptr<!u32i>, !u32i) -> !cir.ptr<!u32i> 
// CIR-NEXT: [[VALUE0:%.]] = cir.load align(4) [[VALPTR]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<2048> : !u32i
// CIR-NEXT: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR-NEXT: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE]] : !cir.bool

// LLVM-LABEL: define{{.*}} void @test_avxvnniint16(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// LLVM: = and i32 [[LOAD]], 2048

// OGCG-LABEL: define{{.*}} void @test_avxvnniint16(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// OGCG: = and i32 [[LOAD]], 2048
TEST_CPU_SUPPORTS(avxvnniint16, "avxvnniint16")

// CIR-LABEL: cir.func no_inline dso_local @test_sm3()
// CIR: [[TRUE:%.]] = cir.const #true
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: [[IDX:%.]] = cir.const #cir.int<2> : !u32i
// CIR-NEXT: [[ARRPTR:%.]] = cir.cast array_to_ptrdecay [[GLOBAL]] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALPTR:%.]] = cir.ptr_stride [[ARRPTR]], [[IDX]] : (!cir.ptr<!u32i>, !u32i) -> !cir.ptr<!u32i> 
// CIR-NEXT: [[VALUE0:%.]] = cir.load align(4) [[VALPTR]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<4096> : !u32i
// CIR-NEXT: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR-NEXT: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE]] : !cir.bool

// LLVM-LABEL: define{{.*}} void @test_sm3(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// LLVM: = and i32 [[LOAD]], 4096

// OGCG-LABEL: define{{.*}} void @test_sm3(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// OGCG: = and i32 [[LOAD]], 4096
TEST_CPU_SUPPORTS(sm3, "sm3")

// CIR-LABEL: cir.func no_inline dso_local @test_sha512()
// CIR: [[TRUE:%.]] = cir.const #true
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: [[IDX:%.]] = cir.const #cir.int<2> : !u32i
// CIR-NEXT: [[ARRPTR:%.]] = cir.cast array_to_ptrdecay [[GLOBAL]] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALPTR:%.]] = cir.ptr_stride [[ARRPTR]], [[IDX]] : (!cir.ptr<!u32i>, !u32i) -> !cir.ptr<!u32i> 
// CIR-NEXT: [[VALUE0:%.]] = cir.load align(4) [[VALPTR]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<8192> : !u32i
// CIR-NEXT: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR-NEXT: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE]] : !cir.bool

// LLVM-LABEL: define{{.*}} void @test_sha512(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// LLVM: = and i32 [[LOAD]], 8192

// OGCG-LABEL: define{{.*}} void @test_sha512(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// OGCG: = and i32 [[LOAD]], 8192
TEST_CPU_SUPPORTS(sha512, "sha512")

// CIR-LABEL: cir.func no_inline dso_local @test_sm4()
// CIR: [[TRUE:%.]] = cir.const #true
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: [[IDX:%.]] = cir.const #cir.int<2> : !u32i
// CIR-NEXT: [[ARRPTR:%.]] = cir.cast array_to_ptrdecay [[GLOBAL]] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALPTR:%.]] = cir.ptr_stride [[ARRPTR]], [[IDX]] : (!cir.ptr<!u32i>, !u32i) -> !cir.ptr<!u32i> 
// CIR-NEXT: [[VALUE0:%.]] = cir.load align(4) [[VALPTR]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<16384> : !u32i
// CIR-NEXT: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR-NEXT: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE]] : !cir.bool

// LLVM-LABEL: define{{.*}} void @test_sm4(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// LLVM: = and i32 [[LOAD]], 16384

// OGCG-LABEL: define{{.*}} void @test_sm4(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// OGCG: = and i32 [[LOAD]], 16384
TEST_CPU_SUPPORTS(sm4, "sm4")

// CIR-LABEL: cir.func no_inline dso_local @test_apxf()
// CIR: [[TRUE:%.]] = cir.const #true
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: [[IDX:%.]] = cir.const #cir.int<2> : !u32i
// CIR-NEXT: [[ARRPTR:%.]] = cir.cast array_to_ptrdecay [[GLOBAL]] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALPTR:%.]] = cir.ptr_stride [[ARRPTR]], [[IDX]] : (!cir.ptr<!u32i>, !u32i) -> !cir.ptr<!u32i> 
// CIR-NEXT: [[VALUE0:%.]] = cir.load align(4) [[VALPTR]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<32768> : !u32i
// CIR-NEXT: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR-NEXT: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE]] : !cir.bool

// LLVM-LABEL: define{{.*}} void @test_apxf(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// LLVM: = and i32 [[LOAD]], 32768

// OGCG-LABEL: define{{.*}} void @test_apxf(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// OGCG: = and i32 [[LOAD]], 32768
TEST_CPU_SUPPORTS(apxf, "apxf")

// CIR-LABEL: cir.func no_inline dso_local @test_usermsr()
// CIR: [[TRUE:%.]] = cir.const #true
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: [[IDX:%.]] = cir.const #cir.int<2> : !u32i
// CIR-NEXT: [[ARRPTR:%.]] = cir.cast array_to_ptrdecay [[GLOBAL]] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALPTR:%.]] = cir.ptr_stride [[ARRPTR]], [[IDX]] : (!cir.ptr<!u32i>, !u32i) -> !cir.ptr<!u32i> 
// CIR-NEXT: [[VALUE0:%.]] = cir.load align(4) [[VALPTR]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<65536> : !u32i
// CIR-NEXT: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR-NEXT: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE]] : !cir.bool

// LLVM-LABEL: define{{.*}} void @test_usermsr(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// LLVM: = and i32 [[LOAD]], 65536

// OGCG-LABEL: define{{.*}} void @test_usermsr(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// OGCG: = and i32 [[LOAD]], 65536
TEST_CPU_SUPPORTS(usermsr, "usermsr")

// CIR-LABEL: cir.func no_inline dso_local @test_avx10_1()
// CIR: [[TRUE:%.]] = cir.const #true
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: [[IDX:%.]] = cir.const #cir.int<2> : !u32i
// CIR-NEXT: [[ARRPTR:%.]] = cir.cast array_to_ptrdecay [[GLOBAL]] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALPTR:%.]] = cir.ptr_stride [[ARRPTR]], [[IDX]] : (!cir.ptr<!u32i>, !u32i) -> !cir.ptr<!u32i> 
// CIR-NEXT: [[VALUE0:%.]] = cir.load align(4) [[VALPTR]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<262144> : !u32i
// CIR-NEXT: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR-NEXT: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE]] : !cir.bool

// LLVM-LABEL: define{{.*}} void @test_avx10_1(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// LLVM: = and i32 [[LOAD]], 262144

// OGCG-LABEL: define{{.*}} void @test_avx10_1(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// OGCG: = and i32 [[LOAD]], 262144
TEST_CPU_SUPPORTS(avx10_1, "avx10.1")

// CIR-LABEL: cir.func no_inline dso_local @test_avx10_2()
// CIR: [[TRUE:%.]] = cir.const #true
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: [[IDX:%.]] = cir.const #cir.int<2> : !u32i
// CIR-NEXT: [[ARRPTR:%.]] = cir.cast array_to_ptrdecay [[GLOBAL]] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALPTR:%.]] = cir.ptr_stride [[ARRPTR]], [[IDX]] : (!cir.ptr<!u32i>, !u32i) -> !cir.ptr<!u32i> 
// CIR-NEXT: [[VALUE0:%.]] = cir.load align(4) [[VALPTR]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<1048576> : !u32i
// CIR-NEXT: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR-NEXT: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE]] : !cir.bool

// LLVM-LABEL: define{{.*}} void @test_avx10_2(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// LLVM: = and i32 [[LOAD]], 1048576

// OGCG-LABEL: define{{.*}} void @test_avx10_2(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// OGCG: = and i32 [[LOAD]], 1048576
TEST_CPU_SUPPORTS(avx10_2, "avx10.2")

// CIR-LABEL: cir.func no_inline dso_local @test_amx_avx512()
// CIR: [[TRUE:%.]] = cir.const #true
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: [[IDX:%.]] = cir.const #cir.int<2> : !u32i
// CIR-NEXT: [[ARRPTR:%.]] = cir.cast array_to_ptrdecay [[GLOBAL]] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALPTR:%.]] = cir.ptr_stride [[ARRPTR]], [[IDX]] : (!cir.ptr<!u32i>, !u32i) -> !cir.ptr<!u32i> 
// CIR-NEXT: [[VALUE0:%.]] = cir.load align(4) [[VALPTR]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<2097152> : !u32i
// CIR-NEXT: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR-NEXT: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE]] : !cir.bool

// LLVM-LABEL: define{{.*}} void @test_amx_avx512(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// LLVM: = and i32 [[LOAD]], 2097152

// OGCG-LABEL: define{{.*}} void @test_amx_avx512(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// OGCG: = and i32 [[LOAD]], 2097152
TEST_CPU_SUPPORTS(amx_avx512, "amx-avx512")

// CIR-LABEL: cir.func no_inline dso_local @test_amx_fp8()
// CIR: [[TRUE:%.]] = cir.const #true
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: [[IDX:%.]] = cir.const #cir.int<2> : !u32i
// CIR-NEXT: [[ARRPTR:%.]] = cir.cast array_to_ptrdecay [[GLOBAL]] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALPTR:%.]] = cir.ptr_stride [[ARRPTR]], [[IDX]] : (!cir.ptr<!u32i>, !u32i) -> !cir.ptr<!u32i> 
// CIR-NEXT: [[VALUE0:%.]] = cir.load align(4) [[VALPTR]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<16777216> : !u32i
// CIR-NEXT: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR-NEXT: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE]] : !cir.bool

// LLVM-LABEL: define{{.*}} void @test_amx_fp8(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// LLVM: = and i32 [[LOAD]], 16777216

// OGCG-LABEL: define{{.*}} void @test_amx_fp8(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// OGCG: = and i32 [[LOAD]], 16777216
TEST_CPU_SUPPORTS(amx_fp8, "amx-fp8")

// CIR-LABEL: cir.func no_inline dso_local @test_movrs()
// CIR: [[TRUE:%.]] = cir.const #true
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: [[IDX:%.]] = cir.const #cir.int<2> : !u32i
// CIR-NEXT: [[ARRPTR:%.]] = cir.cast array_to_ptrdecay [[GLOBAL]] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALPTR:%.]] = cir.ptr_stride [[ARRPTR]], [[IDX]] : (!cir.ptr<!u32i>, !u32i) -> !cir.ptr<!u32i> 
// CIR-NEXT: [[VALUE0:%.]] = cir.load align(4) [[VALPTR]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<33554432> : !u32i
// CIR-NEXT: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR-NEXT: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE]] : !cir.bool

// LLVM-LABEL: define{{.*}} void @test_movrs(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// LLVM: = and i32 [[LOAD]], 33554432

// OGCG-LABEL: define{{.*}} void @test_movrs(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// OGCG: = and i32 [[LOAD]], 33554432
TEST_CPU_SUPPORTS(movrs, "movrs")

// CIR-LABEL: cir.func no_inline dso_local @test_amx_movrs()
// CIR: [[TRUE:%.]] = cir.const #true
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: [[IDX:%.]] = cir.const #cir.int<2> : !u32i
// CIR-NEXT: [[ARRPTR:%.]] = cir.cast array_to_ptrdecay [[GLOBAL]] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALPTR:%.]] = cir.ptr_stride [[ARRPTR]], [[IDX]] : (!cir.ptr<!u32i>, !u32i) -> !cir.ptr<!u32i> 
// CIR-NEXT: [[VALUE0:%.]] = cir.load align(4) [[VALPTR]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<67108864> : !u32i
// CIR-NEXT: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR-NEXT: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE]] : !cir.bool

// LLVM-LABEL: define{{.*}} void @test_amx_movrs(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// LLVM: = and i32 [[LOAD]], 67108864

// OGCG-LABEL: define{{.*}} void @test_amx_movrs(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// OGCG: = and i32 [[LOAD]], 67108864
TEST_CPU_SUPPORTS(amx_movrs, "amx-movrs")

// CIR-LABEL: cir.func no_inline dso_local @test_avx512bmm()
// CIR: [[TRUE:%.]] = cir.const #true
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: [[IDX:%.]] = cir.const #cir.int<2> : !u32i
// CIR-NEXT: [[ARRPTR:%.]] = cir.cast array_to_ptrdecay [[GLOBAL]] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALPTR:%.]] = cir.ptr_stride [[ARRPTR]], [[IDX]] : (!cir.ptr<!u32i>, !u32i) -> !cir.ptr<!u32i> 
// CIR-NEXT: [[VALUE0:%.]] = cir.load align(4) [[VALPTR]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<134217728> : !u32i
// CIR-NEXT: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR-NEXT: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE]] : !cir.bool

// LLVM-LABEL: define{{.*}} void @test_avx512bmm(
// LLVM: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// LLVM: = and i32 [[LOAD]], 134217728

// OGCG-LABEL: define{{.*}} void @test_avx512bmm(
// OGCG: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8)
// OGCG: = and i32 [[LOAD]], 134217728
TEST_CPU_SUPPORTS(avx512bmm, "avx512bmm")

