// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM,LLVMCIR --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM,OGCG --input-file=%t.ll %s

union PtrToIntUnion { int id; char *str; };
struct HasPtoIU { int info; union PtrToIntUnion u; };
struct HasPtoIU ptoIU = { 101, { 1 } };
// CIR-DAG: cir.global external @ptoIU = #cir.const_record<{#cir.int<101> : !s32i, #cir.const_record<{#cir.int<1> : !s32i}> : !rec_PtrToIntUnion}> : !rec_HasPtoIU
// LLVM-DAG: @ptoIU = global { i32, [4 x i8], { i32, [4 x i8] } } { i32 101, [4 x i8] zeroinitializer, { i32, [4 x i8] } { i32 1, [4 x i8] zeroinitializer } }

struct WithTailPadding { int info; union PtrToIntUnion u; int tail; };
struct WithTailPadding  tailPadding = { 101, { 1 }, 42 };
// CIR-DAG: cir.global external @tailPadding = #cir.const_record<{#cir.int<101> : !s32i, #cir.const_record<{#cir.int<1> : !s32i}> : !rec_PtrToIntUnion, #cir.int<42> : !s32i}> : !rec_WithTailPadding
// LLVM-DAG: @tailPadding = global { i32, [4 x i8], { i32, [4 x i8] }, i32, [4 x i8] } { i32 101, [4 x i8] zeroinitializer, { i32, [4 x i8] } { i32 1, [4 x i8] zeroinitializer }, i32 42, [4 x i8] zeroinitializer }

struct AtStart { union PtrToIntUnion u; int x; };
struct AtStart start = { 7, 9 };
// CIR-DAG: cir.global external @start = #cir.const_record<{#cir.const_record<{#cir.int<7> : !s32i}> : !rec_PtrToIntUnion, #cir.int<9> : !s32i}> : !rec_AtStart
// LLVM-DAG: @start = global { { i32, [4 x i8] }, i32, [4 x i8] } { { i32, [4 x i8] } { i32 7, [4 x i8] zeroinitializer }, i32 9, [4 x i8] zeroinitializer }

struct NotToplevel { char c; struct WithTailPadding inner; };
struct NotToplevel notTop = { 'x', { 101, { 1 }, 42 } };
// CIR-DAG: cir.global external @notTop = #cir.const_record<{#cir.int<120> : !s8i, #cir.const_record<{#cir.int<101> : !s32i, #cir.const_record<{#cir.int<1> : !s32i}> : !rec_PtrToIntUnion, #cir.int<42> : !s32i}> : !rec_WithTailPadding}> : !rec_NotToplevel
// LLVM-DAG: @notTop = global { i8, [7 x i8], { i32, [4 x i8], { i32, [4 x i8] }, i32, [4 x i8] } } { i8 120, [7 x i8] zeroinitializer, { i32, [4 x i8], { i32, [4 x i8] }, i32, [4 x i8] } { i32 101, [4 x i8] zeroinitializer, { i32, [4 x i8] } { i32 1, [4 x i8] zeroinitializer }, i32 42, [4 x i8] zeroinitializer } }

struct TwoUnions {
  int a;
  union PtrToIntUnion u1;
  int b;
  union PtrToIntUnion u2;
};
struct TwoUnions two_unions = { 1, { 2 }, 3, { 4 } };
// CIR-DAG: cir.global external @two_unions = #cir.const_record<{#cir.int<1> : !s32i, #cir.const_record<{#cir.int<2> : !s32i}> : !rec_PtrToIntUnion, #cir.int<3> : !s32i, #cir.const_record<{#cir.int<4> : !s32i}> : !rec_PtrToIntUnion}> : !rec_TwoUnions
// LLVM-DAG: @two_unions = global { i32, [4 x i8], { i32, [4 x i8] }, i32, [4 x i8], { i32, [4 x i8] } } { i32 1, [4 x i8] zeroinitializer, { i32, [4 x i8] } { i32 2, [4 x i8] zeroinitializer }, i32 3, [4 x i8] zeroinitializer, { i32, [4 x i8] } { i32 4, [4 x i8] zeroinitializer } }

struct Anon { int info; union { int id; char *str; } u; };
struct Anon anon = { 101, 1 };
// CIR-DAG: cir.global external @anon = #cir.const_record<{#cir.int<101> : !s32i, #cir.const_record<{#cir.int<1> : !s32i}> : !rec_anon2E0}> : !rec_Anon
// LLVM-DAG: @anon = global { i32, [4 x i8], { i32, [4 x i8] } } { i32 101, [4 x i8] zeroinitializer, { i32, [4 x i8] } { i32 1, [4 x i8] zeroinitializer } }

struct Bitfields { int a : 3; int b : 4; union PtrToIntUnion u; };
struct Bitfields bitfields = { 1, 2, { 9 } };
// CIR-DAG: cir.global external @bitfields = #cir.const_record<{#cir.int<17> : !u8i, #cir.const_record<{#cir.int<9> : !s32i}> : !rec_PtrToIntUnion}> : !rec_Bitfields {alignment = 8 : i64} loc(#loc42)
// LLVM-DAG: @bitfields = global { i8, [7 x i8], { i32, [4 x i8] } } { i8 17, [7 x i8] zeroinitializer, { i32, [4 x i8] } { i32 9, [4 x i8] zeroinitializer } }

struct FamUnion { int n; union PtrToIntUnion u; char fam[]; };
struct FamUnion fam_union = { 3, { 7 }, { 'a','b','c' } };
// CIR-DAG: cir.global external @fam_union = #cir.const_record<{#cir.int<3> : !s32i, #cir.const_record<{#cir.int<7> : !s32i}> : !rec_PtrToIntUnion, #cir.const_array<[#cir.int<97> : !s8i, #cir.int<98> : !s8i, #cir.int<99> : !s8i]> : !cir.array<!s8i x 3>}> : !rec_FamUnion
// LLVM-DAG: @fam_union = global <{ i32, [4 x i8], { i32, [4 x i8] }, [3 x i8] }> <{ i32 3, [4 x i8] zeroinitializer, { i32, [4 x i8] } { i32 7, [4 x i8] zeroinitializer }, [3 x i8] c"abc" }>

struct FamMoves { char c; union PtrToIntUnion u; char fam[]; };
struct FamMoves fam_realign = { 'q', { 7 }, { 'a','b','c' } };
// CIR-DAG: cir.global external @fam_realign = #cir.const_record<{#cir.int<113> : !s8i, #cir.const_record<{#cir.int<7> : !s32i}> : !rec_PtrToIntUnion, #cir.const_array<[#cir.int<97> : !s8i, #cir.int<98> : !s8i, #cir.int<99> : !s8i]> : !cir.array<!s8i x 3>}> : !rec_FamMoves
// LLVM-DAG: @fam_realign = global <{ i8, [7 x i8], { i32, [4 x i8] }, [3 x i8] }> <{ i8 113, [7 x i8] zeroinitializer, { i32, [4 x i8] } { i32 7, [4 x i8] zeroinitializer }, [3 x i8] c"abc" }>


typedef union vec3 {
  struct { double x, y, z; };
  double component[3];
} vec3;

// LLVMCIR-DAG: @__const.ret_outer.__retval = {{.*}}%struct.outer { %union.needs_padding zeroinitializer, i32 1 }, align 8
// OGCG-DAG: @__const.ret_outer.o = {{.*}}{ { i32, [4 x i8] }, i32, [4 x i8] } { { i32, [4 x i8] } zeroinitializer, i32 1, [4 x i8] zeroinitializer }, align 8

// CIR-DAG: cir.global "private" constant cir_private @__const.ret_outer.__retval = #cir.const_record<{#cir.zero : !rec_needs_padding, #cir.int<1> : !s32i}> : !rec_outer {alignment = 8 : i64}

// In C mode, this does do zero padding.
vec3 ret_vec3() {
  // CIR-LABEL: cir.func {{.*}} @ret_vec3
  // CIR-SAME: (%[[RET_ALLOCA:.*]]: !cir.ptr<!rec_vec3> {{.*}}llvm.sret = !rec_vec3{{.*}})
  // CIR: %[[GET_ANON:.*]] = cir.get_member %[[RET_ALLOCA]][0] {name = ""}
  // CIR: %[[GET_X:.*]] = cir.get_member %[[GET_ANON]][0] {name = "x"}
  // CIR: %[[FIVE:.*]] = cir.const #cir.fp<5.{{.*}}> : !cir.double
  // CIR: cir.store{{.*}} %[[FIVE]], %[[GET_X]]
  // CIR: %[[GET_Y:.*]] = cir.get_member %[[GET_ANON]][1] {name = "y"}
  // CIR: %[[ZERO:.*]] = cir.const #cir.fp<0.{{.*}}> : !cir.double
  // CIR: cir.store{{.*}} %[[ZERO]], %[[GET_Y]]
  // CIR: %[[GET_Z:.*]] = cir.get_member %[[GET_ANON]][2] {name = "z"}
  // CIR: %[[ZERO:.*]] = cir.const #cir.fp<0.{{.*}}> : !cir.double
  // CIR: cir.store{{.*}} %[[ZERO]], %[[GET_Z]]

  // LLVM-LABEL: define dso_local void @ret_vec3
  // LLVM-SAME: (ptr dead_on_unwind noalias writable sret(%union.vec3) align 8 %[[RET_ALLOCA:.*]])
  // LLVM: %[[GET_X:.*]] = getelementptr {{.*}}, ptr %[[RET_ALLOCA]], i32 0, i32 0
  // LLVM: store double 5{{.*}}, ptr %[[GET_X]]
  // LLVM: %[[GET_Y:.*]] = getelementptr {{.*}}, ptr %[[RET_ALLOCA]], i32 0, i32 1
  // LLVM: store double 0{{.*}}, ptr %[[GET_Y]]
  // LLVM: %[[GET_Z:.*]] = getelementptr {{.*}}, ptr %[[RET_ALLOCA]], i32 0, i32 2
  // LLVM: store double 0{{.*}}, ptr %[[GET_Z]]
  return (vec3) {{ .x = 5.0 }};
}

union needs_padding {
  int a;
  long long b;
};
struct outer {
  union needs_padding np;
  int x;
};

struct outer ret_outer() {
  struct outer o = {{}, 1};
  return o;

  // CIR-LABEL: ret_outer
  // CIR: %[[RET_ALLOCA:.*]] = cir.alloca "__retval" {{.*}} init : !cir.ptr<!rec_outer>
  // CIR: %[[GET_GLOB:.*]] = cir.get_global @__const.ret_outer.__retval : !cir.ptr<!rec_outer>
  // CIR: cir.copy %[[GET_GLOB]] to %[[RET_ALLOCA]] : !cir.ptr<!rec_outer>

  // LLVM-LABEL: define dso_local { i64, i32 } @ret_outer()
  // LLVM: call void @llvm.memcpy.p0.p0.i64(ptr {{.*}}%[[RET_ALLOCA:.*]], ptr {{.*}}@__const.ret_outer.{{.*}}, i64 16, i1 false)
  // LLVMCIR: %[[OUTER:.*]] = load %struct.outer, ptr %[[RET_ALLOCA]]
  // LLVMCIR: store %struct.outer %[[OUTER]], ptr %[[COERCE:.*]], align 8
  // LLVMCIR: %[[RET:.*]] = load { i64, i32 }, ptr %[[COERCE]]
  // OGCG: %[[RET:.*]] = load { i64, i32 }, ptr %[[RET_ALLOCA]]
  // LLVM: ret { i64, i32 } %[[RET]]
}

