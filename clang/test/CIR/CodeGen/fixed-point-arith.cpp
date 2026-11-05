// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -ffixed-point -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -ffixed-point -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefixes=LLVM,LLVMCIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -ffixed-point -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=LLVM,OGCG
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -ffixed-point -fpadding-on-unsigned-fixed-point -fclangir -emit-cir %s -o %t-pad.cir
// RUN: FileCheck --input-file=%t-pad.cir %s --check-prefix=PAD-CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -ffixed-point -fpadding-on-unsigned-fixed-point -fclangir -emit-llvm %s -o %t-pad-cir.ll
// RUN: FileCheck --input-file=%t-pad-cir.ll %s --check-prefix=PAD-LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -ffixed-point -fpadding-on-unsigned-fixed-point -emit-llvm %s -o %t-pad.ll
// RUN: FileCheck --input-file=%t-pad.ll %s --check-prefix=PAD-LLVM

extern "C" {
// CIR-LABEL: cir.func {{.*}}@add(
// CIR-NEXT: %[[RET:.*]] = cir.alloca "__retval" align(4) : !cir.ptr<!s32i>
// CIR-NEXT: %[[A:.*]] = cir.alloca "a" align(4) init : !cir.ptr<!s32i>
// CIR-NEXT: %[[F:.*]] = cir.alloca "f" align(2) init : !cir.ptr<!s16i>
// CIR:      %[[LOAD_A:.*]] = cir.load align(4) %[[A]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: %[[LOAD_F:.*]] = cir.load align(2) %[[F]] : !cir.ptr<!s16i>, !s16i
// CIR-NEXT: %[[WIDEN_F:.*]] = cir.cast integral %[[LOAD_F]] : !s16i -> !s32i
// CIR-NEXT: %[[RESULT:.*]] = cir.add %[[LOAD_A]], %[[WIDEN_F]] : !s32i
// CIR-NEXT: cir.store %[[RESULT]], %[[RET]] : !s32i, !cir.ptr<!s32i>

// LLVM-LABEL: define {{.*}}i32 @add(
// OGCG-NEXT: entry:
// Classic codegen doesn't alloca for a return slot.
// LLVMCIR-NEXT: %{{.+}} = alloca i32, align 4
// LLVM-NEXT: %[[A:.*]] = alloca i32, align 4
// LLVM-NEXT: %[[F:.*]] = alloca i16, align 2
// LLVM:      %[[LOAD_A:.*]] = load i32, ptr %[[A]], align 4
// LLVM-NEXT: %[[LOAD_F:.*]] = load i16, ptr %[[F]], align 2
// LLVM-NEXT: %[[EXT_F:.*]] = sext i16 %[[LOAD_F]] to i32
// LLVM-NEXT: %[[ADD:.*]] = add i32 %[[LOAD_A]], %[[EXT_F]]
_Accum add(_Accum a, _Fract f) {
  return a + f;
}

// CIR-LABEL: cir.func {{.*}}@add2(
// CIR-NEXT: %[[RET:.*]] = cir.alloca "__retval" align(4) : !cir.ptr<!s32i>
// CIR-NEXT: %[[A:.*]] = cir.alloca "a" align(4) init : !cir.ptr<!s32i>
// CIR-NEXT: %[[F:.*]] = cir.alloca "f" align(2) init : !cir.ptr<!s16i>
// CIR:      %[[LOAD_A:.*]] = cir.load align(4) %[[A]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: %[[LOAD_F:.*]] = cir.load align(2) %[[F]] : !cir.ptr<!s16i>, !s16i
// CIR-NEXT: %[[WIDEN_A:.*]] = cir.cast integral %[[LOAD_A]] : !s32i -> !cir.int<s, 47>
// CIR-NEXT: %[[SHIFT_VAL:.*]] = cir.const #cir.int<15> : !cir.int<s, 47>
// CIR-NEXT: %[[SHIFT_A:.*]] = cir.shift(left, %[[WIDEN_A]] : !cir.int<s, 47>, %[[SHIFT_VAL]] : !cir.int<s, 47>) -> !cir.int<s, 47>
// CIR-NEXT: %[[WIDEN_F:.*]] = cir.cast integral %[[LOAD_F]] : !s16i -> !cir.int<s, 47>
// CIR-NEXT: %[[ADD_A_F:.*]] = cir.add %[[SHIFT_A]], %[[WIDEN_F]] : !cir.int<s, 47>
// CIR-NEXT: %[[TRUNC_RES:.*]] = cir.cast integral %[[ADD_A_F]] : !cir.int<s, 47> -> !s16i
// CIR-NEXT: %[[WIDEN_RES:.*]] = cir.cast integral %[[TRUNC_RES]] : !s16i -> !s32i
// CIR-NEXT: cir.store %[[WIDEN_RES]], %[[RET]] : !s32i, !cir.ptr<!s32i>

// LLVM-LABEL: define {{.*}}i32 @add2(
// OGCG-NEXT: entry:
// LLVMCIR-NEXT: %{{.+}} = alloca i32, align 4
// LLVM-NEXT: %[[A:.*]] = alloca i32, align 4
// LLVM-NEXT: %[[F:.*]] = alloca i16, align 2
// LLVM:      %[[LOAD_A:.*]] = load i32, ptr %[[A]], align 4
// LLVM-NEXT: %[[LOAD_F:.*]] = load i16, ptr %[[F]], align 2
// LLVM-NEXT: %[[EXT_A:.*]] = sext i32 %[[LOAD_A]] to i47
// LLVM-NEXT: %[[SHIFT_A:.*]] = shl i47 %[[EXT_A]], 15
// LLVM-NEXT: %[[EXT_F:.*]] = sext i16 %[[LOAD_F]] to i47
// LLVM-NEXT: %[[ADD:.*]] = add i47 %[[SHIFT_A]], %[[EXT_F]]
// LLVM-NEXT: %[[TRUNC:.*]] = trunc i47 %[[ADD]] to i16
// LLVM-NEXT: %[[EXT:.*]] = sext i16 %[[TRUNC]] to i32
_Accum add2(int a, _Fract f) {
  return a + f;
}

// CIR-LABEL: cir.func {{.*}}@sub(
// CIR-NEXT: %[[RET:.*]] = cir.alloca "__retval" align(4) : !cir.ptr<!s32i>
// CIR-NEXT: %[[A:.*]] = cir.alloca "a" align(4) init : !cir.ptr<!s32i>
// CIR-NEXT: %[[F:.*]] = cir.alloca "f" align(2) init : !cir.ptr<!s16i>
// CIR:      %[[LOAD_A:.*]] = cir.load align(4) %[[A]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: %[[LOAD_F:.*]] = cir.load align(2) %[[F]] : !cir.ptr<!s16i>, !s16i
// CIR-NEXT: %[[WIDEN_F:.*]] = cir.cast integral %[[LOAD_F]] : !s16i -> !s32i
// CIR-NEXT: %[[DIFF:.*]] = cir.sub %[[LOAD_A]], %[[WIDEN_F]] : !s32i
// CIR-NEXT: cir.store %[[DIFF]], %[[RET]] : !s32i, !cir.ptr<!s32i>

// LLVM-LABEL: define {{.*}}i32 @sub(
// OGCG-NEXT: entry:
// LLVMCIR-NEXT: %{{.+}} = alloca i32, align 4
// LLVM-NEXT: %[[A:.*]] = alloca i32, align 4
// LLVM-NEXT: %[[F:.*]] = alloca i16, align 2
// LLVM:      %[[LOAD_A:.*]] = load i32, ptr %[[A]], align 4
// LLVM-NEXT: %[[LOAD_F:.*]] = load i16, ptr %[[F]], align 2
// LLVM-NEXT: %[[EXT_F:.*]] = sext i16 %[[LOAD_F]] to i32
// LLVM-NEXT: %[[SUB:.*]] = sub i32 %[[LOAD_A]], %[[EXT_F]]
_Accum sub(_Accum a, _Fract f) {
  return a - f;
}

// CIR-LABEL: cir.func {{.*}}@sub2(
// CIR-NEXT: %[[RET:.*]] = cir.alloca "__retval" align(4) : !cir.ptr<!s32i>
// CIR-NEXT: %[[A:.*]] = cir.alloca "a" align(4) init : !cir.ptr<!s32i>
// CIR-NEXT: %[[F:.*]] = cir.alloca "f" align(2) init : !cir.ptr<!s16i>
// CIR:      %[[LOAD_A:.*]] = cir.load align(4) %[[A]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: %[[LOAD_F:.*]] = cir.load align(2) %[[F]] : !cir.ptr<!s16i>, !s16i
// CIR-NEXT: %[[WIDEN_F:.*]] = cir.cast integral %[[LOAD_F]] : !s16i -> !s32i
// CIR-NEXT: %[[WIDEN_A47:.*]] = cir.cast integral %[[LOAD_A]] : !s32i -> !cir.int<s, 47>
// CIR-NEXT: %[[WIDEN_F47:.*]] = cir.cast integral %[[WIDEN_F]] : !s32i -> !cir.int<s, 47>
// CIR-NEXT: %[[SHIFT_NUM:.*]] = cir.const #cir.int<15> : !cir.int<s, 47>
// CIR-NEXT: %[[SHIFT_F:.*]] = cir.shift(left, %[[WIDEN_F47]] : !cir.int<s, 47>, %[[SHIFT_NUM]] : !cir.int<s, 47>) -> !cir.int<s, 47>
// CIR-NEXT: %[[DIFF:.*]] = cir.sub %[[WIDEN_A47]], %[[SHIFT_F]] : !cir.int<s, 47>
// CIR-NEXT: %[[TRUNC_DIFF:.*]] = cir.cast integral %[[DIFF]] : !cir.int<s, 47> -> !s32i
// CIR-NEXT: cir.store %[[TRUNC_DIFF]], %[[RET]] : !s32i, !cir.ptr<!s32i>

// LLVM-LABEL: define {{.*}}i32 @sub2(
// OGCG-NEXT: entry:
// LLVMCIR-NEXT: %{{.+}} = alloca i32, align 4
// LLVM-NEXT: %[[A:.*]] = alloca i32, align 4
// LLVM-NEXT: %[[F:.*]] = alloca i16, align 2
// LLVM:      %[[LOAD_A:.*]] = load i32, ptr %[[A]], align 4
// LLVM-NEXT: %[[LOAD_F:.*]] = load i16, ptr %[[F]], align 2
// LLVM-NEXT: %[[EXT_F32:.*]] = sext i16 %[[LOAD_F]] to i32
// LLVM-NEXT: %[[EXT_A:.*]] = sext i32 %[[LOAD_A]] to i47
// LLVM-NEXT: %[[EXT_F:.*]] = sext i32 %[[EXT_F32]] to i47
// LLVM-NEXT: %[[SHIFT:.*]] = shl i47 %[[EXT_F]], 15
// LLVM-NEXT: %[[SUB:.*]] = sub i47 %[[EXT_A]], %[[SHIFT]]
// LLVM-NEXT: %[[TRUNC:.*]] = trunc i47 %[[SUB]] to i32
_Accum sub2(_Accum a, short f) {
  return a - f;
}

// CIR-LABEL: cir.func {{.*}}@mul(
// CIR-NEXT: %[[RET:.*]] = cir.alloca "__retval" align(4) : !cir.ptr<!s32i>
// CIR-NEXT: %[[A:.*]] = cir.alloca "a" align(4) init : !cir.ptr<!s32i>
// CIR-NEXT: %[[F:.*]] = cir.alloca "f" align(2) init : !cir.ptr<!s16i>
// CIR:      %[[LOAD_A:.*]] = cir.load align(4) %[[A]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: %[[LOAD_F:.*]] = cir.load align(2) %[[F]] : !cir.ptr<!s16i>, !s16i
// CIR-NEXT: %[[WIDEN_F:.*]] = cir.cast integral %[[LOAD_F]] : !s16i -> !s32i
// CIR-NEXT: %[[SCALE:.*]] = cir.const #cir.int<15> : !s32i
// CIR-NEXT: %[[RES:.*]] = cir.call_llvm_intrinsic "smul.fix" %[[LOAD_A]], %[[WIDEN_F]], %[[SCALE]] : (!s32i, !s32i, !s32i) -> !s32i
// CIR-NEXT: cir.store %[[RES]], %[[RET]] : !s32i, !cir.ptr<!s32i>

// LLVM-LABEL: define {{.*}}i32 @mul(
// OGCG-NEXT: entry:
// LLVMCIR-NEXT: %{{.+}} = alloca i32, align 4
// LLVM-NEXT: %[[A:.*]] = alloca i32, align 4
// LLVM-NEXT: %[[F:.*]] = alloca i16, align 2
// LLVM:      %[[LOAD_A:.*]] = load i32, ptr %[[A]], align 4
// LLVM-NEXT: %[[LOAD_F:.*]] = load i16, ptr %[[F]], align 2
// LLVM-NEXT: %[[EXT_F:.*]] = sext i16 %[[LOAD_F]] to i32
// LLVM-NEXT: %[[MUL:.*]] = call i32 @llvm.smul.fix.i32(i32 %[[LOAD_A]], i32 %[[EXT_F]], i32 15)
_Accum mul(_Accum a, _Fract f) {
  return a * f;
}

// CIR-LABEL: cir.func {{.*}}@mul2(
// CIR-NEXT: %[[RET:.*]] = cir.alloca "__retval" align(4) : !cir.ptr<!s32i>
// CIR-NEXT: %[[S:.*]] = cir.alloca "s" align(2) init : !cir.ptr<!u16i>
// CIR-NEXT: %[[F:.*]] = cir.alloca "f" align(2) init : !cir.ptr<!s16i>
// CIR:      %[[LOAD_S:.*]] = cir.load align(2) %[[S]] : !cir.ptr<!u16i>, !u16i
// CIR-NEXT: %[[WIDEN_S:.*]] = cir.cast integral %[[LOAD_S]] : !u16i -> !s32i
// CIR-NEXT: %[[LOAD_F:.*]] = cir.load align(2) %[[F]] : !cir.ptr<!s16i>, !s16i
// CIR-NEXT: %[[WIDEN_S47:.*]] = cir.cast integral %[[WIDEN_S]] : !s32i -> !cir.int<s, 47>
// CIR-NEXT: %[[SHIFT_NUM:.*]] = cir.const #cir.int<15> : !cir.int<s, 47>
// CIR-NEXT: %[[SHIFT:.*]] = cir.shift(left, %[[WIDEN_S47]] : !cir.int<s, 47>, %[[SHIFT_NUM]] : !cir.int<s, 47>) -> !cir.int<s, 47>
// CIR-NEXT: %[[WIDEN_F47:.*]] = cir.cast integral %[[LOAD_F]] : !s16i -> !cir.int<s, 47>
// CIR-NEXT: %[[SCALE:.*]] = cir.const #cir.int<15> : !s32i
// CIR-NEXT: %[[RES:.*]] = cir.call_llvm_intrinsic "smul.fix" %[[SHIFT]], %[[WIDEN_F47]], %[[SCALE]] : (!cir.int<s, 47>, !cir.int<s, 47>, !s32i) -> !cir.int<s, 47>
// CIR-NEXT: %[[TRUNC_RES:.*]] = cir.cast integral %[[RES]] : !cir.int<s, 47> -> !s16i
// CIR-NEXT: %[[WIDEN_RES:.*]] = cir.cast integral %[[TRUNC_RES]] : !s16i -> !s32i
// CIR-NEXT: cir.store %[[WIDEN_RES]], %[[RET]] : !s32i, !cir.ptr<!s32i>

// LLVM-LABEL: define {{.*}}i32 @mul2(i16 
// OGCG-NEXT: entry:
// LLVMCIR-NEXT: %{{.+}} = alloca i32, align 4
// LLVM-NEXT: %[[S:.*]] = alloca i16, align 2
// LLVM-NEXT: %[[F:.*]] = alloca i16, align 2
// LLVM:      %[[LOAD_S:.*]] = load i16, ptr %[[S]], align 2
// LLVM-NEXT: %[[EXT_S:.*]] = zext i16 %[[LOAD_S]] to i32
// LLVM-NEXT: %[[LOAD_F:.*]] = load i16, ptr %[[F]], align 2
// LLVM-NEXT: %[[EXT_S2:.*]] = sext i32 %[[EXT_S]] to i47
// LLVM-NEXT: %[[SHIFT_S:.*]] = shl i47 %[[EXT_S2]], 15
// LLVM-NEXT: %[[EXT_F:.*]] = sext i16 %[[LOAD_F]] to i47
// LLVM-NEXT: %[[MUL:.*]] = call i47 @llvm.smul.fix.i47(i47 %[[SHIFT_S]], i47 %[[EXT_F]], i32 15)
// LLVM-NEXT: %[[TRUNC_RES:.*]] = trunc i47 %[[MUL]] to i16
// LLVM-NEXT: %[[EXT_RES:.*]] = sext i16 %[[TRUNC_RES]] to i32
_Accum mul2(unsigned short s, _Fract f) {
  return s * f;
}

// CIR-LABEL: cir.func {{.*}}@div(
// CIR-NEXT: %[[RET:.*]] = cir.alloca "__retval" align(4) : !cir.ptr<!s32i>
// CIR-NEXT: %[[A:.*]] = cir.alloca "a" align(4) init : !cir.ptr<!s32i>
// CIR-NEXT: %[[F:.*]] = cir.alloca "f" align(2) init : !cir.ptr<!s16i>
// CIR:      %[[LOAD_A:.*]] = cir.load align(4) %[[A]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: %[[LOAD_F:.*]] = cir.load align(2) %[[F]] : !cir.ptr<!s16i>, !s16i
// CIR-NEXT: %[[WIDEN_F:.*]] = cir.cast integral %[[LOAD_F]] : !s16i -> !s32i
// CIR-NEXT: %[[SCALE:.*]] = cir.const #cir.int<15> : !s32i
// CIR-NEXT: %[[DIV:.*]] = cir.call_llvm_intrinsic "sdiv.fix" %[[LOAD_A]], %[[WIDEN_F]], %[[SCALE]] : (!s32i, !s32i, !s32i) -> !s32i
// CIR-NEXT: cir.store %[[DIV]], %[[RET]] : !s32i, !cir.ptr<!s32i>

// LLVM-LABEL: define {{.*}}i32 @div(
// OGCG-NEXT: entry:
// LLVMCIR-NEXT: %{{.+}} = alloca i32, align 4
// LLVM-NEXT: %[[A:.*]] = alloca i32, align 4
// LLVM-NEXT: %[[F:.*]] = alloca i16, align 2
// LLVM:      %[[LOAD_A:.*]] = load i32, ptr %[[A]], align 4
// LLVM-NEXT: %[[LOAD_F:.*]] = load i16, ptr %[[F]], align 2
// LLVM-NEXT: %[[EXT_F:.*]] = sext i16 %[[LOAD_F]] to i32
// LLVM-NEXT: %[[DIV:.*]] = call i32 @llvm.sdiv.fix.i32(i32 %[[LOAD_A]], i32 %[[EXT_F]], i32 15)
_Accum div(_Accum a, _Fract f) {
  return a / f;
}

// CIR-LABEL: cir.func {{.*}}@div2(
// CIR-NEXT: %[[RET:.*]] = cir.alloca "__retval" align(4) : !cir.ptr<!s32i>
// CIR-NEXT: %[[A:.*]] = cir.alloca "a" align(4) init : !cir.ptr<!s32i>
// CIR-NEXT: %[[F:.*]] = cir.alloca "f" align(4) init : !cir.ptr<!cir.float>
// CIR:      %[[LOAD_A:.*]] = cir.load align(4) %[[A]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: %[[FLOAT_A:.*]] = cir.cast int_to_float %[[LOAD_A]] : !s32i -> !cir.float
// CIR-NEXT: %[[SCALE:.*]] = cir.const #cir.fp<3.05175781E-5> : !cir.float
// CIR-NEXT: %[[SCALE_A:.*]] = cir.fmul %[[FLOAT_A]], %[[SCALE]] : !cir.float
// CIR-NEXT: %[[LOAD_F:.*]] = cir.load align(4) %[[F]] : !cir.ptr<!cir.float>, !cir.float
// CIR-NEXT: %[[DIV:.*]] = cir.fdiv %[[SCALE_A]], %[[LOAD_F]] : !cir.float
// CIR-NEXT: %[[SCALE:.*]] = cir.const #cir.fp<3.276800e+04> : !cir.float
// CIR-NEXT: %[[RES:.*]] = cir.fmul %[[DIV]], %[[SCALE]] : !cir.float
// CIR-NEXT: %[[RES_TO_INT:.*]] = cir.cast float_to_int %[[RES]] : !cir.float -> !s32i
// CIR-NEXT: cir.store %[[RES_TO_INT]], %[[RET]] : !s32i, !cir.ptr<!s32i>

// LLVM-LABEL: define {{.*}}i32 @div2(
// OGCG-NEXT: entry:
// LLVMCIR-NEXT: %{{.+}} = alloca i32, align 4
// LLVM-NEXT: %[[A:.*]] = alloca i32, align 4
// LLVM-NEXT: %[[F:.*]] = alloca float, align 4
// LLVM:      %[[LOAD_A:.*]] = load i32, ptr %[[A]], align 4
// LLVM-NEXT: %[[A_TO_FLOAT:.*]] = sitofp i32 %[[LOAD_A]] to float
// LLVM-NEXT: %[[SCALE_A:.*]] = fmul float %[[A_TO_FLOAT]], f0x38000000
// LLVM-NEXT: %[[LOAD_F:.*]] = load float, ptr %[[F]], align 4
// LLVM-NEXT: %[[DIV:.*]] = fdiv float %[[SCALE_A]], %[[LOAD_F]]
// LLVM-NEXT: %[[SCALE_RES:.*]] = fmul float %[[DIV]], 3.276800e+04
// LLVM-NEXT: %[[RES_TO_INT:.*]] = fptosi float %[[SCALE_RES]] to i32
_Accum div2(_Accum a, float f) {
  return a / f;
}

// CIR-LABEL: cir.func {{.*}}@umul(
// CIR-NEXT: %[[RET:.*]] = cir.alloca "__retval" align(4) : !cir.ptr<!u32i>
// CIR-NEXT: %[[A:.*]] = cir.alloca "a" align(4) init : !cir.ptr<!u32i>
// CIR-NEXT: %[[B:.*]] = cir.alloca "b" align(4) init : !cir.ptr<!u32i>
// CIR:      %[[LOAD_A:.*]] = cir.load align(4) %[[A]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: %[[LOAD_B:.*]] = cir.load align(4) %[[B]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: %[[SCALE:.*]] = cir.const #cir.int<16> : !u32i
// CIR-NEXT: %[[MUL:.*]] = cir.call_llvm_intrinsic "umul.fix" %[[LOAD_A]], %[[LOAD_B]], %[[SCALE]] : (!u32i, !u32i, !u32i) -> !u32i
// CIR-NEXT: cir.store %[[MUL]], %[[RET]] : !u32i, !cir.ptr<!u32i>

// LLVM-LABEL: define {{.*}}i32 @umul(
// OGCG-NEXT: entry:
// LLVMCIR-NEXT: %{{.+}} = alloca i32, align 4
// LLVM-NEXT: %[[A:.*]] = alloca i32, align 4
// LLVM-NEXT: %[[B:.*]] = alloca i32, align 4
// LLVM:      %[[LOAD_A:.*]] = load i32, ptr %[[A]], align 4
// LLVM-NEXT: %[[LOAD_B:.*]] = load i32, ptr %[[B]], align 4
// LLVM-NEXT: %[[MUL:.*]] = call i32 @llvm.umul.fix.i32(i32 %[[LOAD_A]], i32 %[[LOAD_B]], i32 16)
unsigned _Accum umul(unsigned _Accum a, unsigned _Accum b) {
  return a * b;
}

// CIR-LABEL: cir.func {{.*}}@cmp(
// CIR-NEXT: %[[RET:.*]] = cir.alloca "__retval" align(1) : !cir.ptr<!cir.bool>
// CIR-NEXT: %[[A:.*]] = cir.alloca "a" align(4) init : !cir.ptr<!s32i>
// CIR-NEXT: %[[F:.*]] = cir.alloca "f" align(2) init : !cir.ptr<!s16i>
// CIR:      %[[LOAD_A:.*]] = cir.load align(4) %[[A]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: %[[LOAD_F:.*]] = cir.load align(2) %[[F]] : !cir.ptr<!s16i>, !s16i
// CIR-NEXT: %[[WIDEN_F:.*]] = cir.cast integral %[[LOAD_F]] : !s16i -> !s32i
// CIR-NEXT: %[[LT:.*]] = cir.cmp lt %[[LOAD_A]], %[[WIDEN_F]] : !s32i
// CIR-NEXT: cir.store %[[LT]], %[[RET]] : !cir.bool, !cir.ptr<!cir.bool>

// LLVM-LABEL: define {{.*}}i1 @cmp(
// OGCG-NEXT: entry:
// LLVMCIR-NEXT: %{{.+}} = alloca i8, align 1
// LLVM-NEXT: %[[A:.*]] = alloca i32, align 4
// LLVM-NEXT: %[[F:.*]] = alloca i16, align 2
// LLVM:      %[[LOAD_A:.*]] = load i32, ptr %[[A]], align 4
// LLVM-NEXT: %[[LOAD_F:.*]] = load i16, ptr %[[F]], align 2
// LLVM-NEXT: %[[EXT_F:.*]] = sext i16 %[[LOAD_F]] to i32
// LLVM-NEXT: %[[CMP:.*]] = icmp slt i32 %[[LOAD_A]], %[[EXT_F]]
bool cmp(_Accum a, _Fract f) {
  return a < f;
}

// CIR-LABEL: cir.func {{.*}}@cmp2(
// CIR-NEXT: %[[RET:.*]] = cir.alloca "__retval" align(1) : !cir.ptr<!cir.bool>
// CIR-NEXT: %[[A:.*]] = cir.alloca "a" align(4) init : !cir.ptr<!s32i>
// CIR-NEXT: %[[F:.*]] = cir.alloca "f" align(2) init : !cir.ptr<!s16i>
// CIR:      %[[LOAD_A:.*]] = cir.load align(4) %[[A]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: %[[LOAD_F:.*]] = cir.load align(2) %[[F]] : !cir.ptr<!s16i>, !s16i
// CIR-NEXT: %[[WIDEN_F:.*]] = cir.cast integral %[[LOAD_F]] : !s16i -> !s32i
// CIR-NEXT: %[[GT:.*]] = cir.cmp gt %[[LOAD_A]], %[[WIDEN_F]] : !s32i
// CIR-NEXT: cir.store %[[GT]], %[[RET]] : !cir.bool, !cir.ptr<!cir.bool>

// LLVM-LABEL: define {{.*}}i1 @cmp2(
// OGCG-NEXT: entry:
// LLVMCIR-NEXT: %{{.+}} = alloca i8, align 1
// LLVM-NEXT: %[[A:.*]] = alloca i32, align 4
// LLVM-NEXT: %[[F:.*]] = alloca i16, align 2
// LLVM:      %[[LOAD_A:.*]] = load i32, ptr %[[A]], align 4
// LLVM-NEXT: %[[LOAD_F:.*]] = load i16, ptr %[[F]], align 2
// LLVM-NEXT: %[[EXT_F:.*]] = sext i16 %[[LOAD_F]] to i32
// LLVM-NEXT: %[[CMP:.*]] = icmp sgt i32 %[[LOAD_A]], %[[EXT_F]]
bool cmp2(_Accum a, _Fract f) {
  return a > f;
}

// CIR-LABEL: cir.func {{.*}}@cmp3(
// CIR-NEXT: %[[RET:.*]] = cir.alloca "__retval" align(1) : !cir.ptr<!cir.bool>
// CIR-NEXT: %[[A:.*]] = cir.alloca "a" align(4) init : !cir.ptr<!s32i>
// CIR-NEXT: %[[F:.*]] = cir.alloca "f" align(2) init : !cir.ptr<!s16i>
// CIR:      %[[LOAD_A:.*]] = cir.load align(4) %[[A]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: %[[LOAD_F:.*]] = cir.load align(2) %[[F]] : !cir.ptr<!s16i>, !s16i
// CIR-NEXT: %[[WIDEN_F:.*]] = cir.cast integral %[[LOAD_F]] : !s16i -> !s32i
// CIR-NEXT: %[[EQ:.*]] = cir.cmp eq %[[LOAD_A]], %[[WIDEN_F]] : !s32i
// CIR-NEXT: cir.store %[[EQ]], %[[RET]] : !cir.bool, !cir.ptr<!cir.bool>

// LLVM-LABEL: define {{.*}}i1 @cmp3(
// OGCG-NEXT: entry:
// LLVMCIR-NEXT: %{{.+}} = alloca i8, align 1
// LLVM-NEXT: %[[A:.*]] = alloca i32, align 4
// LLVM-NEXT: %[[F:.*]] = alloca i16, align 2
// LLVM:      %[[LOAD_A:.*]] = load i32, ptr %[[A]], align 4
// LLVM-NEXT: %[[LOAD_F:.*]] = load i16, ptr %[[F]], align 2
// LLVM-NEXT: %[[EXT_F:.*]] = sext i16 %[[LOAD_F]] to i32
// LLVM-NEXT: %[[CMP:.*]] = icmp eq i32 %[[LOAD_A]], %[[EXT_F]]
bool cmp3(_Accum a, _Fract f) {
  return a == f;
}

// CIR-LABEL: cir.func {{.*}}@shl(
// CIR-NEXT: %[[RET:.*]] = cir.alloca "__retval" align(4) : !cir.ptr<!s32i>
// CIR-NEXT: %[[A:.*]] = cir.alloca "a" align(4) init : !cir.ptr<!s32i>
// CIR:      %[[LOAD_A:.*]] = cir.load align(4) %[[A]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: %[[SHIFT_NUM:.*]] = cir.const #cir.int<2> : !s32i
// CIR-NEXT: %[[SHIFT:.*]] = cir.shift(left, %[[LOAD_A]] : !s32i, %[[SHIFT_NUM]] : !s32i) -> !s32i
// CIR-NEXT: cir.store %[[SHIFT]], %[[RET]] : !s32i, !cir.ptr<!s32i>

// LLVM-LABEL: define {{.*}}i32 @shl(
// OGCG-NEXT: entry:
// LLVMCIR-NEXT: %{{.+}} = alloca i32, align 4
// LLVM-NEXT: %[[A:.*]] = alloca i32, align 4
// LLVM:      %[[LOAD_A:.*]] = load i32, ptr %[[A]], align 4
// LLVM-NEXT: %[[SHIFT:.*]] = shl i32 %[[LOAD_A]], 2
_Accum shl(_Accum a) {
  return a << 2;
}

// CIR-LABEL: cir.func {{.*}}@shl2(
// CIR-NEXT: %[[RET:.*]] = cir.alloca "__retval" align(4) : !cir.ptr<!s32i>
// CIR-NEXT: %[[A:.*]] = cir.alloca "a" align(4) init : !cir.ptr<!s32i>
// CIR-NEXT: %[[C:.*]] = cir.alloca "c" align(1) init : !cir.ptr<!u8i>
// CIR:      %[[LOAD_A:.*]] = cir.load align(4) %[[A]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: %[[LOAD_C:.*]] = cir.load align(1) %[[C]] : !cir.ptr<!u8i>, !u8i
// CIR-NEXT: %[[WIDEN_C:.*]] = cir.cast integral %[[LOAD_C]] : !u8i -> !s32i
// CIR-NEXT: %[[SHIFT:.*]] = cir.shift(left, %[[LOAD_A]] : !s32i, %[[WIDEN_C]] : !s32i) -> !s32i
// CIR-NEXT: cir.store %[[SHIFT]], %[[RET]] : !s32i, !cir.ptr<!s32i>

// LLVM-LABEL: define {{.*}}i32 @shl2(
// OGCG-NEXT: entry:
// LLVMCIR-NEXT: %{{.+}} = alloca i32, align 4
// LLVM-NEXT: %[[A:.*]] = alloca i32, align 4
// LLVM-NEXT: %[[C:.*]] = alloca i8, align 1
// LLVM:      %[[LOAD_A:.*]] = load i32, ptr %[[A]], align 4
// LLVM-NEXT: %[[LOAD_C:.*]] = load i8, ptr %[[C]], align 1
// LLVM-NEXT: %[[C_EXT:.*]] = zext i8 %[[LOAD_C]] to i32
// LLVM-NEXT: %[[SHIFT:.*]] = shl i32 %[[LOAD_A]], %[[C_EXT]]
_Accum shl2(_Accum a, unsigned char c) {
  return a << c;
}

// CIR-LABEL: cir.func {{.*}}@shl3(
// CIR-NEXT: %[[RET:.*]] = cir.alloca "__retval" align(2) : !cir.ptr<!s16i>
// CIR-NEXT: %[[F:.*]] = cir.alloca "f" align(2) init : !cir.ptr<!s16i>
// CIR-NEXT: %[[C:.*]] = cir.alloca "c" align(1) init : !cir.ptr<!u8i>
// CIR:      %[[LOAD_F:.*]] = cir.load align(2) %[[F]] : !cir.ptr<!s16i>, !s16i
// CIR-NEXT: %[[LOAD_C:.*]] = cir.load align(1) %[[C]] : !cir.ptr<!u8i>, !u8i
// CIR-NEXT: %[[WIDEN_C:.*]] = cir.cast integral %[[LOAD_C]] : !u8i -> !s32i
// CIR-NEXT: %[[SHIFT:.*]] = cir.shift(left, %[[LOAD_F]] : !s16i, %[[WIDEN_C]] : !s32i) -> !s16i
// CIR-NEXT: cir.store %[[SHIFT]], %[[RET]] : !s16i, !cir.ptr<!s16i>

// LLVM-LABEL: define {{.*}}i16 @shl3(
// OGCG-NEXT: entry:
// LLVMCIR-NEXT: %{{.+}} = alloca i16, align 2
// LLVM-NEXT: %[[F:.*]] = alloca i16, align 2
// LLVM-NEXT: %[[C:.*]] = alloca i8, align 1
// LLVM:      %[[LOAD_F:.*]] = load i16, ptr %[[F]], align 2
// LLVM-NEXT: %[[LOAD_C:.*]] = load i8, ptr %[[C]], align 1
// LLVM-NEXT: %[[EXT_C:.*]] = zext i8 %[[LOAD_C]] to i32
// LLVM-NEXT: %[[TRUNC_C:.*]] = trunc i32 %[[EXT_C]] to i16
// LLVM-NEXT: %[[SHIFT:.*]] = shl i16 %[[LOAD_F]], %[[TRUNC_C]]
_Fract shl3(_Fract f, unsigned char c) {
  return f << c;
}

// CIR-LABEL: cir.func {{.*}}@shr(
// CIR-NEXT: %[[RET:.*]] = cir.alloca "__retval" align(4) : !cir.ptr<!s32i>
// CIR-NEXT: %[[A:.*]] = cir.alloca "a" align(4) init : !cir.ptr<!s32i>
// CIR:      %[[LOAD_A:.*]] = cir.load align(4) %[[A]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: %[[SHIFT_NUM:.*]] = cir.const #cir.int<2> : !s32i
// CIR-NEXT: %[[SHIFT:.*]] = cir.shift(right, %[[LOAD_A]] : !s32i, %[[SHIFT_NUM]] : !s32i) -> !s32i
// CIR-NEXT: cir.store %[[SHIFT]], %[[RET]] : !s32i, !cir.ptr<!s32i>

// LLVM-LABEL: define {{.*}}i32 @shr(
// OGCG-NEXT: entry:
// LLVMCIR-NEXT: %{{.+}} = alloca i32, align 4
// LLVM-NEXT: %[[A:.*]] = alloca i32, align 4
// LLVM:      %[[LOAD_A:.*]] = load i32, ptr %[[A]], align 4
// LLVM-NEXT: %[[SHIFT:.*]] = ashr i32 %[[LOAD_A]], 2
_Accum shr(_Accum a) {
  return a >> 2;
}

// CIR-LABEL: cir.func {{.*}}@shr2(
// CIR-NEXT: %[[RET:.*]] = cir.alloca "__retval" align(4) : !cir.ptr<!s32i>
// CIR-NEXT: %[[A:.*]] = cir.alloca "a" align(4) init : !cir.ptr<!s32i>
// CIR-NEXT: %[[U:.*]] = cir.alloca "u" align(4) init : !cir.ptr<!u32i>
// CIR:      %[[LOAD_A:.*]] = cir.load align(4) %[[A]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: %[[LOAD_U:.*]] = cir.load align(4) %[[U]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: %[[SHIFT:.*]] = cir.shift(right, %[[LOAD_A]] : !s32i, %[[LOAD_U]] : !u32i) -> !s32i
// CIR-NEXT: cir.store %[[SHIFT]], %[[RET]] : !s32i, !cir.ptr<!s32i>

// LLVM-LABEL: define {{.*}}i32 @shr2(
// OGCG-NEXT: entry:
// LLVMCIR-NEXT: %{{.+}} = alloca i32, align 4
// LLVM-NEXT: %[[A:.*]] = alloca i32, align 4
// LLVM-NEXT: %[[U:.*]] = alloca i32, align 4
// LLVM:      %[[LOAD_A:.*]] = load i32, ptr %[[A]], align 4
// LLVM-NEXT: %[[LOAD_U:.*]] = load i32, ptr %[[U]], align 4
// LLVM-NEXT: %[[SHIFT:.*]] = ashr i32 %[[LOAD_A]], %[[LOAD_U]]
_Accum shr2(_Accum a, unsigned u) {
  return a >> u;
}

// CIR-LABEL: cir.func {{.*}}@shr3(
// CIR-NEXT: %[[RET:.*]] = cir.alloca "__retval" align(2) : !cir.ptr<!s16i>
// CIR-NEXT: %[[F:.*]] = cir.alloca "f" align(2) init : !cir.ptr<!s16i>
// CIR-NEXT: %[[U:.*]] = cir.alloca "u" align(4) init : !cir.ptr<!u32i>
// CIR:      %[[LOAD_F:.*]] = cir.load align(2) %[[F]] : !cir.ptr<!s16i>, !s16i
// CIR-NEXT: %[[LOAD_U:.*]] = cir.load align(4) %[[U]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: %[[SHIFT:.*]] = cir.shift(right, %[[LOAD_F]] : !s16i, %[[LOAD_U]] : !u32i) -> !s16i
// CIR-NEXT: cir.store %[[SHIFT]], %[[RET]] : !s16i, !cir.ptr<!s16i>

// LLVM-LABEL: define {{.*}}i16 @shr3(
// OGCG-NEXT: entry:
// LLVMCIR-NEXT: %{{.+}} = alloca i16, align 2
// LLVM-NEXT: %[[F:.*]] = alloca i16, align 2
// LLVM-NEXT: %[[U:.*]] = alloca i32, align 4
// LLVM:      %[[LOAD_F:.*]] = load i16, ptr %[[F]], align 2
// LLVM-NEXT: %[[LOAD_U:.*]] = load i32, ptr %[[U]], align 4
// LLVM-NEXT: %[[TRUNC_U:.*]] = trunc i32 %[[LOAD_U]] to i16
// LLVM-NEXT: %[[SHIFT:.*]] = ashr i16 %[[LOAD_F]], %[[TRUNC_U]]
_Fract shr3(_Fract f, unsigned u) {
  return f >> u;
}

// CIR-LABEL: cir.func {{.*}}@inc(
// CIR-NEXT: %[[RET:.*]] = cir.alloca "__retval" align(4) : !cir.ptr<!s32i>
// CIR-NEXT: %[[A:.*]] = cir.alloca "a" align(4) init : !cir.ptr<!s32i>
// CIR:      %[[LOAD_A:.*]] = cir.load align(4) %[[A]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: %[[NEG_1:.*]] = cir.const #cir.int<-1> : !s32i
// CIR-NEXT: %[[SHIFT_NUM:.*]] = cir.const #cir.int<15> : !s32i
// CIR-NEXT: %[[SHIFT:.*]] = cir.shift(left, %[[NEG_1]] : !s32i, %[[SHIFT_NUM]] : !s32i) -> !s32i
// CIR-NEXT: %[[SUB:.*]] = cir.sub %[[LOAD_A]], %[[SHIFT]] : !s32i
// CIR-NEXT: cir.store align(4) %[[SUB:.*]], %[[A]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT: cir.store %[[SUB]], %[[RET]] : !s32i, !cir.ptr<!s32i>

// LLVM-LABEL: define {{.*}}i32 @inc(
// OGCG-NEXT: entry:
// LLVMCIR-NEXT: %{{.+}} = alloca i32, align 4
// LLVM-NEXT: %[[A:.*]] = alloca i32, align 4
// LLVM:      %[[LOAD_A:.*]] = load i32, ptr %[[A]], align 4
// LLVM-NEXT: %[[INC:.*]] = sub i32 %[[LOAD_A]], -32768
_Accum inc(_Accum a) {
  return ++a;
}

// CIR-LABEL: cir.func {{.*}}@inc2(
// CIR-NEXT: %[[RET:.*]] = cir.alloca "__retval" align(2) : !cir.ptr<!s16i>
// CIR-NEXT: %[[F:.*]] = cir.alloca "f" align(2) init : !cir.ptr<!s16i>
// CIR: %[[LOAD_F:.*]] = cir.load align(2) %[[F]] : !cir.ptr<!s16i>, !s16i
// CIR-NEXT: %[[NEG_1:.*]] = cir.const #cir.int<-1> : !s16i
// CIR-NEXT: %[[SHIFT_NUM:.*]] = cir.const #cir.int<15> : !s16i
// CIR-NEXT: %[[SHIFT:.*]] = cir.shift(left, %[[NEG_1:.*]] : !s16i, %[[SHIFT_NUM]] : !s16i) -> !s16i
// CIR-NEXT: %[[SUB:.*]] = cir.sub %[[LOAD_F]], %[[SHIFT]] : !s16i
// CIR-NEXT: cir.store align(2) %[[SUB]], %[[F]] : !s16i, !cir.ptr<!s16i>
// CIR-NEXT: cir.store %[[LOAD_F]], %[[RET]] : !s16i, !cir.ptr<!s16i>

// LLVM-LABEL: define {{.*}}i16 @inc2(
// OGCG-NEXT: entry:
// LLVMCIR-NEXT: %{{.+}} = alloca i16, align 2
// LLVM-NEXT: %[[A:.*]] = alloca i16, align 2
// LLVM:      %[[LOAD_A:.*]] = load i16, ptr %[[A]], align 2
// LLVM-NEXT: %[[INC:.*]] = sub i16 %[[LOAD_A]], -32768
_Fract inc2(_Fract f) {
  return f++;
}

// CIR-LABEL: cir.func {{.*}}@dec(
// CIR-NEXT: %[[RET:.*]] = cir.alloca "__retval" align(4) : !cir.ptr<!s32i>
// CIR-NEXT: %[[A:.*]] = cir.alloca "a" align(4) init : !cir.ptr<!s32i>
// CIR:      %[[LOAD_A:.*]] = cir.load align(4) %[[A]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: %[[NEG_1:.*]] = cir.const #cir.int<-1> : !s32i
// CIR-NEXT: %[[SHIFT_NUM:.*]] = cir.const #cir.int<15> : !s32i
// CIR-NEXT: %[[SHIFT:.*]] = cir.shift(left, %[[NEG_1]] : !s32i, %[[SHIFT_NUM]] : !s32i) -> !s32i
// CIR-NEXT: %[[ADD:.*]] = cir.add %[[LOAD_A]], %[[SHIFT]] : !s32i
// CIR-NEXT: cir.store align(4) %[[ADD]], %[[A]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT: cir.store %[[ADD]], %[[RET]] : !s32i, !cir.ptr<!s32i>

// LLVM-LABEL: define {{.*}}i32 @dec(
// OGCG-NEXT: entry:
// LLVMCIR-NEXT: %{{.+}} = alloca i32, align 4
// LLVM-NEXT: %[[A:.*]] = alloca i32, align 4
// LLVM:      %[[LOAD_A:.*]] = load i32, ptr %[[A]], align 4
// LLVM-NEXT: %[[DEC:.*]] = add i32 %[[LOAD_A]], -32768
_Accum dec(_Accum a) {
  return --a;
}

// CIR-LABEL: cir.func {{.*}}@dec2(
// CIR-NEXT: %[[RET:.*]] = cir.alloca "__retval" align(2) : !cir.ptr<!s16i>
// CIR-NEXT: %[[A:.*]] = cir.alloca "a" align(2) init : !cir.ptr<!s16i>
// CIR:      %[[LOAD_A:.*]] = cir.load align(2) %[[A]] : !cir.ptr<!s16i>, !s16i
// CIR-NEXT: %[[NEG_1:.*]] = cir.const #cir.int<-1> : !s16i
// CIR-NEXT: %[[SHIFT_NUM:.*]] = cir.const #cir.int<15> : !s16i
// CIR-NEXT: %[[SHIFT:.*]] = cir.shift(left, %[[NEG_1]] : !s16i, %[[SHIFT_NUM]] : !s16i) -> !s16i
// CIR-NEXT: %[[SUB:.*]] = cir.add %[[LOAD_A]], %[[SHIFT]] : !s16i
// CIR-NEXT: cir.store align(2) %[[SUB]], %[[A]] : !s16i, !cir.ptr<!s16i>
// CIR-NEXT: cir.store %[[LOAD_A:.*]], %[[RET]] : !s16i, !cir.ptr<!s16i>

// LLVM-LABEL: define {{.*}}i16 @dec2(
// OGCG-NEXT: entry:
// LLVMCIR-NEXT: %{{.+}} = alloca i16, align 2
// LLVM-NEXT: %[[A:.*]] = alloca i16, align 2
// LLVM:      %[[LOAD_A:.*]] = load i16, ptr %[[A]], align 2
// LLVM-NEXT: %[[DEC:.*]] = add i16 %[[LOAD_A]], -32768
_Fract dec2(_Fract a) {
  return a--;
}

// CIR-LABEL: cir.func {{.*}}@satAdd(
// CIR-NEXT: %[[RET:.*]] = cir.alloca "__retval" align(4) : !cir.ptr<!s32i>
// CIR-NEXT: %[[A:.*]] = cir.alloca "a" align(4) init : !cir.ptr<!s32i>
// CIR-NEXT: %[[B:.*]] = cir.alloca "b" align(4) init : !cir.ptr<!s32i>
// CIR:      %[[LOAD_A:.*]] = cir.load align(4) %[[A]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: %[[LOAD_B:.*]] = cir.load align(4) %[[B]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: %[[ADD:.*]] = cir.add.sat %[[LOAD_A]], %[[LOAD_B]] : !s32i
// CIR-NEXT: cir.store %[[ADD]], %[[RET]] : !s32i, !cir.ptr<!s32i>

// LLVM-LABEL: define {{.*}}i32 @satAdd(
// OGCG-NEXT: entry:
// LLVMCIR-NEXT: %{{.+}} = alloca i32, align 4
// LLVM-NEXT: %[[A:.*]] = alloca i32, align 4
// LLVM-NEXT: %[[B:.*]] = alloca i32, align 4
// LLVM:      %[[LOAD_A:.*]] = load i32, ptr %[[A]], align 4
// LLVM-NEXT: %[[LOAD_B:.*]] = load i32, ptr %[[B]], align 4
// LLVM-NEXT: %[[SUB:.*]] = call i32 @llvm.sadd.sat.i32(i32 %[[LOAD_A]], i32 %[[LOAD_B]])
_Sat _Accum satAdd(_Sat _Accum a, _Accum b) {
  return a + b;
}

// CIR-LABEL: cir.func {{.*}}@satSub(
// CIR-NEXT: %[[RET:.*]] = cir.alloca "__retval" align(4) : !cir.ptr<!s32i>
// CIR-NEXT: %[[A:.*]] = cir.alloca "a" align(4) init : !cir.ptr<!s32i>
// CIR-NEXT: %[[B:.*]] = cir.alloca "b" align(4) init : !cir.ptr<!s32i>
// CIR:      %[[LOAD_A:.*]] = cir.load align(4) %[[A]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: %[[LOAD_B:.*]] = cir.load align(4) %[[B]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: %[[SUB:.*]] = cir.sub.sat %[[LOAD_A]], %[[LOAD_B]] : !s32i
// CIR-NEXT: cir.store %[[SUB]], %[[RET]] : !s32i, !cir.ptr<!s32i>

// LLVM-LABEL: define {{.*}}i32 @satSub(
// OGCG-NEXT: entry:
// LLVMCIR-NEXT: %{{.+}} = alloca i32, align 4
// LLVM-NEXT: %[[A:.*]] = alloca i32, align 4
// LLVM-NEXT: %[[B:.*]] = alloca i32, align 4
// LLVM:      %[[LOAD_A:.*]] = load i32, ptr %[[A]], align 4
// LLVM-NEXT: %[[LOAD_B:.*]] = load i32, ptr %[[B]], align 4
// LLVM-NEXT: %[[SUB:.*]] = call i32 @llvm.ssub.sat.i32(i32 %[[LOAD_A]], i32 %[[LOAD_B]])
_Sat _Accum satSub(_Sat _Accum a, _Sat _Accum b) {
  return a - b;
}

// CIR-LABEL: cir.func {{.*}}@satMul(
// CIR-NEXT: %[[RET:.*]] = cir.alloca "__retval" align(4) : !cir.ptr<!s32i>
// CIR-NEXT: %[[A:.*]] = cir.alloca "a" align(4) init : !cir.ptr<!s32i>
// CIR-NEXT: %[[B:.*]] = cir.alloca "b" align(4) init : !cir.ptr<!s32i>
// CIR:      %[[LOAD_A:.*]] = cir.load align(4) %[[A]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: %[[LOAD_B:.*]] = cir.load align(4) %[[B]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: %[[SCALE:.*]] = cir.const #cir.int<15> : !s32i
// CIR-NEXT: %[[MUL:.*]] = cir.call_llvm_intrinsic "smul.fix.sat" %[[LOAD_A]], %[[LOAD_B]], %[[SCALE]] : (!s32i, !s32i, !s32i) -> !s32i
// CIR-NEXT: cir.store %[[MUL]], %[[RET]] : !s32i, !cir.ptr<!s32i>

// LLVM-LABEL: define {{.*}}i32 @satMul(
// OGCG-NEXT: entry:
// LLVMCIR-NEXT: %{{.+}} = alloca i32, align 4
// LLVM-NEXT: %[[A:.*]] = alloca i32, align 4
// LLVM-NEXT: %[[B:.*]] = alloca i32, align 4
// LLVM:      %[[LOAD_A:.*]] = load i32, ptr %[[A]], align 4
// LLVM-NEXT: %[[LOAD_B:.*]] = load i32, ptr %[[B]], align 4
// LLVM-NEXT: %[[SHIFT:.*]] = call i32 @llvm.smul.fix.sat.i32(i32 %[[LOAD_A]], i32 %[[LOAD_B]], i32 15)
_Sat _Accum satMul(_Sat _Accum a, _Sat _Accum b) {
  return a * b;
}

// CIR-LABEL: cir.func {{.*}}@satDiv(
// CIR-NEXT: %[[RET:.*]] = cir.alloca "__retval" align(4) : !cir.ptr<!s32i>
// CIR-NEXT: %[[A:.*]] = cir.alloca "a" align(4) init : !cir.ptr<!s32i>
// CIR-NEXT: %[[B:.*]] = cir.alloca "b" align(4) init : !cir.ptr<!s32i>
// CIR:      %[[LOAD_A:.*]] = cir.load align(4) %[[A]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: %[[LOAD_B:.*]] = cir.load align(4) %[[B]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: %[[SCALE:.*]] = cir.const #cir.int<15> : !s32i
// CIR-NEXT: %[[DIV:.*]] = cir.call_llvm_intrinsic "sdiv.fix.sat" %[[LOAD_A]], %[[LOAD_B]], %[[SCALE]] : (!s32i, !s32i, !s32i) -> !s32i
// CIR-NEXT: cir.store %[[DIV]], %[[RET]] : !s32i, !cir.ptr<!s32i>

// LLVM-LABEL: define {{.*}}i32 @satDiv(
// OGCG-NEXT: entry:
// LLVMCIR-NEXT: %{{.+}} = alloca i32, align 4
// LLVM-NEXT: %[[A:.*]] = alloca i32, align 4
// LLVM-NEXT: %[[B:.*]] = alloca i32, align 4
// LLVM:      %[[LOAD_A:.*]] = load i32, ptr %[[A]], align 4
// LLVM-NEXT: %[[LOAD_B:.*]] = load i32, ptr %[[B]], align 4
// LLVM-NEXT: %[[SHIFT:.*]] = call i32 @llvm.sdiv.fix.sat.i32(i32 %[[LOAD_A]], i32 %[[LOAD_B]], i32 15)
_Sat _Accum satDiv(_Sat _Accum a, _Sat _Accum b) {
  return a / b;
}

// CIR-LABEL: cir.func {{.*}}@satShl(
// CIR-NEXT: %[[RET:.*]] = cir.alloca "__retval" align(4) : !cir.ptr<!s32i>
// CIR-NEXT: %[[A:.*]] = cir.alloca "a" align(4) init : !cir.ptr<!s32i>
// CIR:      %[[LOAD_A:.*]] = cir.load align(4) %[[A]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: %[[SHIFT_NUM:.*]] = cir.const #cir.int<2> : !s32i
// CIR-NEXT: %[[SHIFT:.*]] = cir.call_llvm_intrinsic "sshl.sat" %[[LOAD_A]], %[[SHIFT_NUM]] : (!s32i, !s32i) -> !s32i
// CIR-NEXT: cir.store %[[SHIFT]], %[[RET]] : !s32i, !cir.ptr<!s32i>

// LLVM-LABEL: define {{.*}}i32 @satShl(
// OGCG-NEXT: entry:
// LLVMCIR-NEXT: %{{.+}} = alloca i32, align 4
// LLVM-NEXT: %[[A:.*]] = alloca i32, align 4
// LLVM:      %[[LOAD_A:.*]] = load i32, ptr %[[A]], align 4
// LLVM-NEXT: %[[SHIFT:.*]] = call i32 @llvm.sshl.sat.i32(i32 %[[LOAD_A]], i32 2)
_Sat _Accum satShl(_Sat _Accum a) {
  return a << 2;
}

// CIR-LABEL: cir.func {{.*}}@satShl2(
// CIR-NEXT: %[[RET:.*]] = cir.alloca "__retval" align(2) : !cir.ptr<!s16i>
// CIR-NEXT: %[[A:.*]] = cir.alloca "a" align(2) init : !cir.ptr<!s16i>
// CIR-NEXT: %[[C:.*]] = cir.alloca "c" align(1) init : !cir.ptr<!s8i>
// CIR:      %[[LOAD_A:.*]] = cir.load align(2) %[[A]] : !cir.ptr<!s16i>, !s16i
// CIR-NEXT: %[[LOAD_C:.*]] = cir.load align(1) %[[C]] : !cir.ptr<!s8i>, !s8i
// CIR-NEXT: %[[WIDEN_C:.*]] = cir.cast integral %[[LOAD_C]] : !s8i -> !s32i
// CIR-NEXT: %[[UNSIGNED_C:.*]] = cir.cast integral %[[WIDEN_C]] : !s32i -> !u32i
// CIR-NEXT: %[[TRUNC_C:.*]] = cir.cast integral %[[UNSIGNED_C]] : !u32i -> !s16i
// CIR-NEXT: %[[SHIFT:.*]] = cir.call_llvm_intrinsic "sshl.sat" %[[LOAD_A]], %[[TRUNC_C]] : (!s16i, !s16i) -> !s16i
// CIR-NEXT: cir.store %[[SHIFT]], %[[RET]] : !s16i, !cir.ptr<!s16i>

// LLVM-LABEL: define {{.*}}i16 @satShl2(
// OGCG-NEXT: entry:
// LLVMCIR-NEXT: %{{.+}} = alloca i16, align 2
// LLVM-NEXT: %[[F:.*]] = alloca i16, align 2
// LLVM-NEXT: %[[C:.*]] = alloca i8, align 1
// LLVM:      %[[LOAD_F:.*]] = load i16, ptr %[[F]], align 2
// LLVM-NEXT: %[[LOAD_C:.*]] = load i8, ptr %[[C]], align 1
// LLVM-NEXT: %[[EXT_C:.*]] = sext i8 %[[LOAD_C]] to i32
// LLVM-NEXT: %[[TRUNC_C:.*]] = trunc i32 %[[EXT_C]] to i16
// LLVM-NEXT: %[[SHIFT:.*]] = call i16 @llvm.sshl.sat.i16(i16 %[[LOAD_F]], i16 %[[TRUNC_C]])
_Sat _Fract satShl2(_Sat _Fract a, char c) {
  return a << c;
}

// CIR-LABEL: cir.func {{.*}}@satShr(
// CIR-NEXT: %[[RET:.*]] = cir.alloca "__retval" align(4) : !cir.ptr<!s32i>
// CIR-NEXT: %[[A:.*]] = cir.alloca "a" align(4) init : !cir.ptr<!s32i>
// CIR:      %[[LOAD_A:.*]] = cir.load align(4) %[[A]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: %[[SHIFT_NUM:.*]] = cir.const #cir.int<2> : !s32i
// CIR-NEXT: %[[SHIFT:.*]] = cir.shift(right, %[[LOAD_A]] : !s32i, %[[SHIFT_NUM]] : !s32i) -> !s32i
// CIR-NEXT: cir.store %[[SHIFT]], %[[RET]] : !s32i, !cir.ptr<!s32i>

// LLVM-LABEL: define {{.*}}i32 @satShr(
// OGCG-NEXT: entry:
// LLVMCIR-NEXT: %{{.+}} = alloca i32, align 4
// LLVM-NEXT: %[[A:.*]] = alloca i32, align 4
// LLVM:      %[[LOAD_A:.*]] = load i32, ptr %[[A]], align 4
// LLVM-NEXT: %[[SHIFT:.*]] = ashr i32 %[[LOAD_A]], 2
_Sat _Accum satShr(_Sat _Accum a) {
  return a >> 2;
}

// CIR-LABEL: cir.func {{.*}}@satShr2(
// CIR-NEXT: %[[RET:.*]] = cir.alloca "__retval" align(2) : !cir.ptr<!s16i>
// CIR-NEXT: %[[F:.*]] = cir.alloca "f" align(2) init : !cir.ptr<!s16i>
// CIR-NEXT: %[[I:.*]] = cir.alloca "i" align(4) init : !cir.ptr<!s32i>
// CIR:      %[[LOAD_F:.*]] = cir.load align(2) %[[F]] : !cir.ptr<!s16i>, !s16i
// CIR-NEXT: %[[LOAD_I:.*]] = cir.load align(4) %[[I]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: %[[SHIFT:.*]] = cir.shift(right, %[[LOAD_F]] : !s16i, %[[LOAD_I]] : !s32i) -> !s16i
// CIR-NEXT: cir.store %[[SHIFT]], %[[RET]] : !s16i, !cir.ptr<!s16i>

// LLVM-LABEL: define {{.*}}i16 @satShr2(
// OGCG-NEXT: entry:
// LLVMCIR-NEXT: %{{.+}} = alloca i16, align 2
// LLVM-NEXT: %[[F:.*]] = alloca i16, align 2
// LLVM-NEXT: %[[I:.*]] = alloca i32, align 4
// LLVM:      %[[LOAD_F:.*]] = load i16, ptr %[[F]], align 2
// LLVM-NEXT: %[[LOAD_I:.*]] = load i32, ptr %[[I]], align 4
// LLVM-NEXT: %[[TRUNC:.*]] = trunc i32 %[[LOAD_I]] to i16
// LLVM-NEXT: %[[SHIFT:.*]] = ashr i16 %[[LOAD_F]], %[[TRUNC]]

_Sat _Fract satShr2(_Sat _Fract f, int i) {
  return f >> i;
}

// CIR-LABEL: cir.func {{.*}}@satInc(
// CIR-NEXT: %[[RET:.*]] = cir.alloca "__retval" align(4) : !cir.ptr<!s32i>
// CIR-NEXT: %[[A:.*]] = cir.alloca "a" align(4) init : !cir.ptr<!s32i>
// CIR:      %[[LOAD_A:.*]] = cir.load align(4) %[[A]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: %[[NEG_1:.*]] = cir.const #cir.int<-1> : !cir.int<s, 47>
// CIR-NEXT: %[[SHIFT_NUM:.*]] = cir.const #cir.int<15> : !cir.int<s, 47>
// CIR-NEXT: %[[SHIFT:.*]] = cir.shift(left, %[[NEG_1]] : !cir.int<s, 47>, %[[SHIFT_NUM]] : !cir.int<s, 47>) -> !cir.int<s, 47>
// CIR-NEXT: %[[MAX:.*]] = cir.const #cir.int<2147483647> : !cir.int<s, 47>
// CIR-NEXT: %[[CMP:.*]] = cir.cmp gt %[[SHIFT]], %[[MAX]] : !cir.int<s, 47>
// CIR-NEXT: %[[SELECT:.*]] = cir.select if %[[CMP]] then %[[MAX]] else %[[SHIFT]] : (!cir.bool, !cir.int<s, 47>, !cir.int<s, 47>) -> !cir.int<s, 47>
// CIR-NEXT: %[[MIN:.*]] = cir.const #cir.int<-2147483648> : !cir.int<s, 47>
// CIR-NEXT: %[[CMP:.*]] = cir.cmp lt %[[SELECT]], %[[MIN]] : !cir.int<s, 47>
// CIR-NEXT: %[[SELECT2:.*]] = cir.select if %[[CMP]] then %[[MIN]] else %[[SELECT]] : (!cir.bool, !cir.int<s, 47>, !cir.int<s, 47>) -> !cir.int<s, 47>
// CIR-NEXT: %[[TRUNC:.*]] = cir.cast integral %[[SELECT2]] : !cir.int<s, 47> -> !s32i
// CIR-NEXT: %[[SUB:.*]] = cir.sub.sat %[[LOAD_A]], %[[TRUNC]] : !s32i
// CIR-NEXT: cir.store align(4) %[[SUB]], %[[A]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT: cir.store %[[SUB]], %[[RET]] : !s32i, !cir.ptr<!s32i>

// LLVM-LABEL: define {{.*}}i32 @satInc(
// OGCG-NEXT: entry:
// LLVMCIR-NEXT: %{{.+}} = alloca i32, align 4
// LLVM-NEXT: %[[A:.*]] = alloca i32, align 4
// LLVM:      %[[LOAD_A:.*]] = load i32, ptr %[[A]], align 4
// LLVM-NEXT: %[[ADD:.*]] = call i32 @llvm.ssub.sat.i32(i32 %[[LOAD_A]], i32 -32768)
// LLVM-NEXT: store i32 %[[ADD]], ptr %[[A]], align 4
_Sat _Accum satInc(_Sat _Accum a) {
  return ++a;
}

// CIR-LABEL: cir.func {{.*}}@satDec(
// CIR-NEXT: %[[RET:.*]] = cir.alloca "__retval" align(2) : !cir.ptr<!s16i>
// CIR-NEXT: %[[F:.*]] = cir.alloca "f" align(2) init : !cir.ptr<!s16i>
// CIR:      %[[LOAD_F:.*]] = cir.load align(2) %[[F]] : !cir.ptr<!s16i>, !s16i
// CIR-NEXT: %[[NEG_1:.*]] = cir.const #cir.int<-1> : !cir.int<s, 31>
// CIR-NEXT: %[[SHIFT_NUM:.*]] = cir.const #cir.int<15> : !cir.int<s, 31>
// CIR-NEXT: %[[SHIFT:.*]] = cir.shift(left, %[[NEG_1]] : !cir.int<s, 31>, %[[SHIFT_NUM]] : !cir.int<s, 31>) -> !cir.int<s, 31>
// CIR-NEXT: %[[MAX:.*]] = cir.const #cir.int<32767> : !cir.int<s, 31>
// CIR-NEXT: %[[CMP:.*]] = cir.cmp gt %[[SHIFT]], %[[MAX]] : !cir.int<s, 31>
// CIR-NEXT: %[[SELECT:.*]] = cir.select if %[[CMP]] then %[[MAX]] else %[[SHIFT]] : (!cir.bool, !cir.int<s, 31>, !cir.int<s, 31>) -> !cir.int<s, 31>
// CIR-NEXT: %[[MIN:.*]] = cir.const #cir.int<-32768> : !cir.int<s, 31>
// CIR-NEXT: %[[CMP:.*]] = cir.cmp lt %[[SELECT]], %[[MIN]] : !cir.int<s, 31>
// CIR-NEXT: %[[SELECT2:.*]] = cir.select if %[[CMP]] then %[[MIN]] else %[[SELECT]] : (!cir.bool, !cir.int<s, 31>, !cir.int<s, 31>) -> !cir.int<s, 31>
// CIR-NEXT: %[[TRUNC:.*]] = cir.cast integral %[[SELECT2]] : !cir.int<s, 31> -> !s16i
// CIR-NEXT: %[[ADD:.*]] = cir.add.sat %[[LOAD_F]], %[[TRUNC]] : !s16i
// CIR-NEXT: cir.store align(2) %[[ADD]], %[[F]] : !s16i, !cir.ptr<!s16i>
// CIR-NEXT: cir.store %[[LOAD_F]], %[[RET]] : !s16i, !cir.ptr<!s16i>

// LLVM-LABEL: define {{.*}}i16 @satDec(
// OGCG-NEXT: entry:
// LLVMCIR-NEXT: %{{.+}} = alloca i16, align 2
// LLVM-NEXT: %[[F:.*]] = alloca i16, align 2
// LLVM:      %[[LOAD_F:.*]] = load i16, ptr %[[F]], align 2
// LLVM-NEXT: %[[ADD:.*]] = call i16 @llvm.sadd.sat.i16(i16 %[[LOAD_F]], i16 -32768)
// LLVM-NEXT: store i16 %[[ADD]], ptr %[[F]], align 2

_Sat _Fract satDec(_Sat _Fract f) {
  return f--;
}

// CIR-LABEL: cir.func {{.*}}@satIncUFract(
// CIR-NEXT: %[[RET:.*]] = cir.alloca "__retval" align(2) : !cir.ptr<!u16i>
// CIR-NEXT: %[[A:.*]] = cir.alloca "a" align(2) init : !cir.ptr<!u16i>
// CIR:      %[[LOAD_A:.*]] = cir.load align(2) %[[A]] : !cir.ptr<!u16i>, !u16i
// CIR-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CIR-NEXT: %[[SHIFT_NUM:.*]] = cir.const #cir.int<16> : !s32i
// CIR-NEXT: %[[SHIFT:.*]] = cir.shift(left, %[[ONE]] : !s32i, %[[SHIFT_NUM]] : !s32i) -> !s32i
// CIR-NEXT: %[[MAX:.*]] = cir.const #cir.int<65535> : !s32i
// CIR-NEXT: %[[CMP:.*]] = cir.cmp gt %[[SHIFT]], %[[MAX]] : !s32i
// CIR-NEXT: %[[SELECT:.*]] = cir.select if %[[CMP]] then %[[MAX]] else %[[SHIFT]] : (!cir.bool, !s32i, !s32i) -> !s32i
// CIR-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR-NEXT: %[[CMP:.*]] = cir.cmp lt %[[SELECT]], %[[ZERO]] : !s32i
// CIR-NEXT: %[[SELECT2:.*]] = cir.select if %[[CMP]] then %[[ZERO]] else %[[SELECT]] : (!cir.bool, !s32i, !s32i) -> !s32i
// CIR-NEXT: %[[TRUNC:.*]] = cir.cast integral %[[SELECT2]] : !s32i -> !u16i
// CIR-NEXT: %[[ADD:.*]] = cir.add.sat %[[LOAD_A]], %[[TRUNC]] : !u16i
// CIR-NEXT: cir.store align(2) %[[ADD]], %[[A]] : !u16i, !cir.ptr<!u16i>
// CIR-NEXT: cir.store %[[ADD]], %[[RET]] : !u16i, !cir.ptr<!u16i>

// LLVM-LABEL: define {{.*}}i16 @satIncUFract(
// OGCG-NEXT: entry:
// LLVMCIR-NEXT: %{{.+}} = alloca i16, align 2
// LLVM-NEXT: %[[A:.*]] = alloca i16, align 2
// LLVM:      %[[LOAD_A:.*]] = load i16, ptr %[[A]], align 2
// LLVM-NEXT: %[[ADD:.*]] = call i16 @llvm.uadd.sat.i16(i16 %[[LOAD_A]], i16 -1)
// LLVM-NEXT: store i16 %[[ADD]], ptr %[[A]], align 2
_Sat unsigned _Fract satIncUFract(_Sat unsigned _Fract a) {
  return ++a;
}

// With unsigned padding, saturating add/sub saturate as signed values like
// classic codegen does, and subtraction clamps at 0 while still signed.

// PAD-CIR-LABEL: cir.func {{.*}}@padSatAddUAccum(
// PAD-CIR:      %[[LHS:.*]] = cir.cast integral %{{.*}} : !u32i -> !s32i
// PAD-CIR-NEXT: %[[RHS:.*]] = cir.cast integral %{{.*}} : !u32i -> !s32i
// PAD-CIR-NEXT: %[[SUM:.*]] = cir.add.sat %[[LHS]], %[[RHS]] : !s32i
// PAD-CIR-NEXT: %{{.*}} = cir.cast integral %[[SUM]] : !s32i -> !u32i

// PAD-LLVM-LABEL: define {{.*}}i32 @padSatAddUAccum(
// PAD-LLVM:      %[[SUM:.*]] = call i32 @llvm.sadd.sat.i32(i32 %{{.*}}, i32 %{{.*}})
// PAD-LLVM-NEXT: %{{.*}} = trunc i32 %[[SUM]] to i31
_Sat unsigned _Accum padSatAddUAccum(_Sat unsigned _Accum a,
                                     _Sat unsigned _Accum b) {
  return a + b;
}

// PAD-CIR-LABEL: cir.func {{.*}}@padSatSubUAccum(
// PAD-CIR:      %[[LHS:.*]] = cir.cast integral %{{.*}} : !u32i -> !s32i
// PAD-CIR-NEXT: %[[RHS:.*]] = cir.cast integral %{{.*}} : !u32i -> !s32i
// PAD-CIR-NEXT: %[[DIFF:.*]] = cir.sub.sat %[[LHS]], %[[RHS]] : !s32i
// PAD-CIR-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// PAD-CIR-NEXT: %[[NEG:.*]] = cir.cmp lt %[[DIFF]], %[[ZERO]] : !s32i
// PAD-CIR-NEXT: %[[CLAMP:.*]] = cir.select if %[[NEG]] then %[[ZERO]] else %[[DIFF]]
// PAD-CIR-NEXT: %{{.*}} = cir.cast integral %[[CLAMP]] : !s32i -> !u32i

// PAD-LLVM-LABEL: define {{.*}}i32 @padSatSubUAccum(
// PAD-LLVM:      %[[DIFF:.*]] = call i32 @llvm.ssub.sat.i32(i32 %{{.*}}, i32 %{{.*}})
// PAD-LLVM-NEXT: %[[NEG:.*]] = icmp slt i32 %[[DIFF]], 0
// PAD-LLVM-NEXT: %[[CLAMP:.*]] = select i1 %[[NEG]], i32 0, i32 %[[DIFF]]
// PAD-LLVM-NEXT: %{{.*}} = trunc i32 %[[CLAMP]] to i31
_Sat unsigned _Accum padSatSubUAccum(_Sat unsigned _Accum a,
                                     _Sat unsigned _Accum b) {
  return a - b;
}

// Unary minus on a saturating type is a saturating 0 - x, like classic codegen.

// CIR-LABEL: cir.func {{.*}}@satNeg(
// CIR:      %[[LOAD_A:.*]] = cir.load align(2) %{{.*}} : !cir.ptr<!s16i>, !s16i
// CIR-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !s16i
// CIR-NEXT: %[[NEG:.*]] = cir.sub.sat %[[ZERO]], %[[LOAD_A]] : !s16i

// LLVM-LABEL: define {{.*}}i16 @satNeg(
// LLVM:      %[[LOAD_A:.*]] = load i16, ptr %{{.*}}, align 2
// LLVM-NEXT: %{{.*}} = call i16 @llvm.ssub.sat.i16(i16 0, i16 %[[LOAD_A]])
_Sat short _Accum satNeg(_Sat short _Accum a) {
  return -a;
}

// CIR-LABEL: cir.func {{.*}}@satNegU(
// CIR:      %[[LOAD_A:.*]] = cir.load align(2) %{{.*}} : !cir.ptr<!u16i>, !u16i
// CIR-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !u16i
// CIR-NEXT: %[[NEG:.*]] = cir.sub.sat %[[ZERO]], %[[LOAD_A]] : !u16i

// LLVM-LABEL: define {{.*}}i16 @satNegU(
// LLVM:      %[[LOAD_A:.*]] = load i16, ptr %{{.*}}, align 2
// LLVM-NEXT: %{{.*}} = call i16 @llvm.usub.sat.i16(i16 0, i16 %[[LOAD_A]])

// PAD-CIR-LABEL: cir.func {{.*}}@satNegU(
// PAD-CIR:      %[[ZERO:.*]] = cir.const #cir.int<0> : !s16i
// PAD-CIR-NEXT: %[[A:.*]] = cir.cast integral %{{.*}} : !u16i -> !s16i
// PAD-CIR-NEXT: %[[DIFF:.*]] = cir.sub.sat %[[ZERO]], %[[A]] : !s16i
// PAD-CIR-NEXT: %[[ZERO2:.*]] = cir.const #cir.int<0> : !s16i
// PAD-CIR-NEXT: %[[LT:.*]] = cir.cmp lt %[[DIFF]], %[[ZERO2]] : !s16i
// PAD-CIR-NEXT: %{{.*}} = cir.select if %[[LT]] then %[[ZERO2]] else %[[DIFF]]

// PAD-LLVM-LABEL: define {{.*}}i16 @satNegU(
// PAD-LLVM:      %[[DIFF:.*]] = call i16 @llvm.ssub.sat.i16(i16 0, i16 %{{.*}})
// PAD-LLVM-NEXT: %[[LT:.*]] = icmp slt i16 %[[DIFF]], 0
// PAD-LLVM-NEXT: %{{.*}} = select i1 %[[LT]], i16 0, i16 %[[DIFF]]
_Sat unsigned short _Accum satNegU(_Sat unsigned short _Accum a) {
  return -a;
}

// With unsigned padding, float to fixed-point conversion is signed, and the
// saturating form clamps negative values to 0.

// PAD-CIR-LABEL: cir.func {{.*}}@padFloatToSatU(
// PAD-CIR:      %[[CONV:.*]] = cir.call_llvm_intrinsic "fptosi.sat" %{{.*}} : (!cir.float) -> !s16i
// PAD-CIR-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !s16i
// PAD-CIR-NEXT: %[[LT:.*]] = cir.cmp lt %[[CONV]], %[[ZERO]] : !s16i
// PAD-CIR-NEXT: %[[CLAMP:.*]] = cir.select if %[[LT]] then %[[ZERO]] else %[[CONV]]
// PAD-CIR-NEXT: %{{.*}} = cir.cast integral %[[CLAMP]] : !s16i -> !u16i

// PAD-LLVM-LABEL: define {{.*}}i16 @padFloatToSatU(
// PAD-LLVM:      %[[CONV:.*]] = call i16 @llvm.fptosi.sat.i16.f32(float %{{.*}})
// PAD-LLVM-NEXT: %[[LT:.*]] = icmp slt i16 %[[CONV]], 0
// PAD-LLVM-NEXT: %{{.*}} = select i1 %[[LT]], i16 0, i16 %[[CONV]]
_Sat unsigned short _Accum padFloatToSatU(float f) {
  return f;
}

// PAD-CIR-LABEL: cir.func {{.*}}@padFloatToU(
// PAD-CIR:      %[[CONV:.*]] = cir.cast float_to_int %{{.*}} : !cir.float -> !s16i
// PAD-CIR-NEXT: %{{.*}} = cir.cast integral %[[CONV]] : !s16i -> !u16i

// PAD-LLVM-LABEL: define {{.*}}i16 @padFloatToU(
// PAD-LLVM:      %{{.*}} = fptosi float %{{.*}} to i16
unsigned short _Accum padFloatToU(float f) {
  return f;
}

// A saturating conversion that only changes signedness still yields a value of
// the destination type.

// PAD-CIR-LABEL: cir.func {{.*}}@padSatSignedToU(
// PAD-CIR:      %[[LT:.*]] = cir.cmp lt %[[A:.*]], %[[ZERO:.*]] : !s16i
// PAD-CIR-NEXT: %[[CLAMP:.*]] = cir.select if %[[LT]] then %[[ZERO]] else %[[A]]
// PAD-CIR-NEXT: %[[CONV:.*]] = cir.cast integral %[[CLAMP]] : !s16i -> !u16i
// PAD-CIR-NEXT: %[[B:.*]] = cir.load align(2) %{{.*}} : !cir.ptr<!u16i>, !u16i
// PAD-CIR-NEXT: %{{.*}} = cir.cmp lt %[[CONV]], %[[B]] : !u16i

// PAD-LLVM-LABEL: define {{.*}}i1 @padSatSignedToU(
// PAD-LLVM:      %[[LT:.*]] = icmp slt i16 %[[A:.*]], 0
// PAD-LLVM-NEXT: %[[CLAMP:.*]] = select i1 %[[LT]], i16 0, i16 %[[A]]
// PAD-LLVM-NEXT: %[[B:.*]] = load i16, ptr %{{.*}}, align 2
// PAD-LLVM-NEXT: %{{.*}} = icmp ult i16 %[[CLAMP]], %[[B]]
bool padSatSignedToU(_Sat short _Accum a, _Sat unsigned short _Accum b) {
  return (_Sat unsigned short _Accum)a < b;
}
}
