// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -ffixed-point -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -ffixed-point -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -ffixed-point -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=LLVM

extern "C" {

// CIR-LABEL: cir.func{{.*}} @int_to_fract
// CIR: %[[ARG:.*]] = cir.alloca "i" align(4) init : !cir.ptr<!s32i>
// CIR: %[[LOAD_ARG:.*]] = cir.load align(4) %[[ARG]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: %[[TRUNC:.*]] = cir.cast integral %[[LOAD_ARG]] : !s32i -> !s16i
// CIR-NEXT: %[[SHIFT_AMOUNT:.*]] = cir.const #cir.int<15> : !s16i
// CIR-NEXT: %[[SHIFT:.*]] = cir.shift(left, %[[TRUNC]] : !s16i, %[[SHIFT_AMOUNT]] : !s16i) -> !s16i
// CIR-NEXT: cir.store %[[SHIFT]], %[[RET:.*]] : !s16i, !cir.ptr<!s16i>
// CIR-NEXT: %[[LOAD_RET:.*]] = cir.load %[[RET]] : !cir.ptr<!s16i>, !s16i
// CIR-NEXT: cir.return %[[LOAD_RET]] : !s16i

// LLVM-LABEL: @int_to_fract
// LLVM: %[[LOAD_ARG:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM: %[[CAST:.*]] = trunc i32 %[[LOAD_ARG]] to i16
// LLVM: %[[SHIFT:.*]] = shl i16 %[[CAST]], 15
// LLVM: ret i16 %{{.*}}
_Fract int_to_fract(int i) {
  return i;
}

// CIR-LABEL: cir.func{{.*}} @int_to_accum
// CIR: %[[ARG:.*]] = cir.alloca "i" align(4) init : !cir.ptr<!s32i>
// CIR: %[[LOAD_ARG:.*]] = cir.load align(4) %[[ARG]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: %[[SHIFT_AMOUNT:.*]] = cir.const #cir.int<15> : !s32i
// CIR-NEXT: %[[SHIFT:.*]] = cir.shift(left, %[[LOAD_ARG]] : !s32i, %[[SHIFT_AMOUNT]] : !s32i) -> !s32i
// CIR-NEXT: cir.store %[[SHIFT]], %[[RET:.*]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT: %[[LOAD_RET:.*]] = cir.load %[[RET]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: cir.return %[[LOAD_RET]] : !s32i

// LLVM-LABEL: @int_to_accum
// LLVM: %[[LOAD_ARG:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM: %[[SHIFT:.*]] = shl i32 %[[LOAD_ARG]], 15
// LLVM: ret i32 %{{.*}}
_Accum int_to_accum(int i) {
  return i;
}

// CIR-LABEL: cir.func{{.*}} @int_to_ufract
// CIR: %[[ARG:.*]] = cir.alloca "i" align(4) init : !cir.ptr<!s32i>
// CIR: %[[LOAD_ARG:.*]] = cir.load align(4) %[[ARG]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: %[[TRUNC:.*]] = cir.cast integral %[[LOAD_ARG]] : !s32i -> !u16i
// CIR-NEXT: %[[SHIFT_AMOUNT:.*]] = cir.const #cir.int<16> : !u16i
// CIR-NEXT: %[[SHIFT:.*]] = cir.shift(left, %[[TRUNC]] : !u16i, %[[SHIFT_AMOUNT]] : !u16i) -> !u16i
// CIR-NEXT: cir.store %[[SHIFT]], %[[RET:.*]] : !u16i, !cir.ptr<!u16i>
// CIR-NEXT: %[[LOAD_RET:.*]] = cir.load %[[RET]] : !cir.ptr<!u16i>, !u16i
// CIR-NEXT: cir.return %[[LOAD_RET]] : !u16i

// LLVM-LABEL: @int_to_ufract
// LLVM: %[[LOAD_ARG:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM: %[[CAST:.*]] = trunc i32 %[[LOAD_ARG]] to i16
// LLVM: %[[SHIFT:.*]] = shl i16 %[[CAST]], 16
// LLVM: ret i16 %{{.*}}
unsigned _Fract int_to_ufract(int i) {
  return i;
}

// CIR-LABEL: cir.func{{.*}} @int_to_uaccum
// CIR: %[[ARG:.*]] = cir.alloca "i" align(4) init : !cir.ptr<!s32i>
// CIR: %[[LOAD_ARG:.*]] = cir.load align(4) %[[ARG]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: %[[CAST:.*]] = cir.cast integral %[[LOAD_ARG]] : !s32i -> !u32i
// CIR-NEXT: %[[SHIFT_AMOUNT:.*]] = cir.const #cir.int<16> : !u32i
// CIR-NEXT: %[[SHIFT:.*]] = cir.shift(left, %[[CAST]] : !u32i, %[[SHIFT_AMOUNT]] : !u32i) -> !u32i
// CIR-NEXT: cir.store %[[SHIFT]], %[[RET:.*]] : !u32i, !cir.ptr<!u32i>
// CIR-NEXT: %[[LOAD_RET:.*]] = cir.load %[[RET]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: cir.return %[[LOAD_RET]] : !u32i

// LLVM-LABEL: @int_to_uaccum
// LLVM: %[[LOAD_ARG:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM: %[[SHIFT:.*]] = shl i32 %[[LOAD_ARG]], 16
// LLVM: ret i32 %{{.*}}
unsigned _Accum int_to_uaccum(int i) {
  return i;
}

// CIR-LABEL: cir.func{{.*}} @int_to_sat_fract
// CIR: %[[ARG:.*]] = cir.alloca "i" align(4) init : !cir.ptr<!s32i>
// CIR: %[[LOAD_ARG:.*]] = cir.load align(4) %[[ARG]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: %[[TRUNC:.*]] = cir.cast integral %[[LOAD_ARG]] : !s32i -> !cir.int<s, 47>
// CIR-NEXT: %[[SHIFT_AMOUNT:.*]] = cir.const #cir.int<15> : !cir.int<s, 47>
// CIR-NEXT: %[[SHIFT:.*]] = cir.shift(left, %[[TRUNC]] : !cir.int<s, 47>, %[[SHIFT_AMOUNT]] : !cir.int<s, 47>) -> !cir.int<s, 47>
// CIR-NEXT: %[[MAX:.*]] = cir.const #cir.int<32767> : !cir.int<s, 47>
// CIR-NEXT: %[[GT_CMP:.*]] = cir.cmp gt %[[SHIFT]], %[[MAX]] : !cir.int<s, 47>
// CIR-NEXT: %[[MAX_SEL:.*]] = cir.select if %[[GT_CMP]] then %[[MAX]] else %[[SHIFT]] : (!cir.bool, !cir.int<s, 47>, !cir.int<s, 47>) -> !cir.int<s, 47>
// CIR-NEXT: %[[MIN:.*]] = cir.const #cir.int<-32768> : !cir.int<s, 47>
// CIR-NEXT: %[[LT_CMP:.*]] = cir.cmp lt %[[MAX_SEL]], %[[MIN]] : !cir.int<s, 47>
// CIR-NEXT: %[[BOUNDED:.*]] = cir.select if %[[LT_CMP]] then %[[MIN]] else %[[MAX_SEL]] : (!cir.bool, !cir.int<s, 47>, !cir.int<s, 47>) -> !cir.int<s, 47>
// CIR-NEXT: %[[CAST_BACK:.*]] = cir.cast integral %[[BOUNDED]] : !cir.int<s, 47> -> !s16i
// CIR-NEXT: cir.store %[[CAST_BACK]], %[[RET:.*]] : !s16i, !cir.ptr<!s16i>
// CIR-NEXT: %[[RET_LOAD:.*]] = cir.load %[[RET]] : !cir.ptr<!s16i>, !s16i
// CIR-NEXT: cir.return %[[RET_LOAD]] : !s16i

// LLVM-LABEL: @int_to_sat_fract
// LLVM: %[[LOAD_ARG:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM: %[[CAST:.*]] = sext i32 %[[LOAD_ARG]] to i47
// LLVM: %[[SHIFT:.*]] = shl i47 %[[CAST]], 15
// LLVM: %[[GT_CMP:.*]] = icmp sgt i47 %[[SHIFT]], 32767
// LLVM: %[[MAX_SEL:.*]] = select i1 %[[GT_CMP]], i47 32767, i47 %[[SHIFT]]
// LLVM: %[[LT_CMP:.*]] = icmp slt i47 %[[MAX_SEL]], -32768
// LLVM: %[[BOUNDED:.*]] = select i1 %[[LT_CMP]], i47 -32768, i47 %[[MAX_SEL]]
// LLVM: %[[CAST:.*]] = trunc i47 %[[BOUNDED]] to i16
// LLVM: ret i16 %{{.*}}
_Sat _Fract int_to_sat_fract(int i) {
  return i;
}

// CIR-LABEL: cir.func{{.*}} @int_to_sat_accum
// CIR: %[[ARG:.*]] = cir.alloca "i" align(4) init : !cir.ptr<!s32i>
// CIR: %[[LOAD_ARG:.*]] = cir.load align(4) %[[ARG]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: %[[CAST:.*]] = cir.cast integral %[[LOAD_ARG]] : !s32i -> !cir.int<s, 47>
// CIR-NEXT: %[[SHIFT_AMOUNT:.*]] = cir.const #cir.int<15> : !cir.int<s, 47>
// CIR-NEXT: %[[SHIFT:.*]] = cir.shift(left, %[[CAST]] : !cir.int<s, 47>, %[[SHIFT_AMOUNT]] : !cir.int<s, 47>) -> !cir.int<s, 47>
// CIR-NEXT: %[[MAX:.*]] = cir.const #cir.int<2147483647> : !cir.int<s, 47>
// CIR-NEXT: %[[GT_CMP:.*]] = cir.cmp gt %[[SHIFT]], %[[MAX]] : !cir.int<s, 47>
// CIR-NEXT: %[[MAX_SEL:.*]] = cir.select if %[[GT_CMP]] then %[[MAX]] else %[[SHIFT]] : (!cir.bool, !cir.int<s, 47>, !cir.int<s, 47>) -> !cir.int<s, 47>
// CIR-NEXT: %[[MIN:.*]] = cir.const #cir.int<-2147483648> : !cir.int<s, 47>
// CIR-NEXT: %[[LT_CMP:.*]] = cir.cmp lt %[[MAX_SEL]], %[[MIN]] : !cir.int<s, 47>
// CIR-NEXT: %[[BOUNDED:.*]] = cir.select if %[[LT_CMP]] then %[[MIN]] else %[[MAX_SEL]] : (!cir.bool, !cir.int<s, 47>, !cir.int<s, 47>) -> !cir.int<s, 47>
// CIR-NEXT: %[[CAST_BACK:.*]] = cir.cast integral %[[BOUNDED]] : !cir.int<s, 47> -> !s32i
// CIR-NEXT: cir.store %[[CAST_BACK]], %[[RET:.*]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT: %[[LOAD_RET:.*]] = cir.load %[[RET]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: cir.return %[[LOAD_RET]] : !s32i
// LLVM-LABEL: @int_to_sat_accum
// LLVM: %[[LOAD_ARG:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM: %[[CAST:.*]] = sext i32 %[[LOAD_ARG]] to i47
// LLVM: %[[SHIFT:.*]] = shl i47 %[[CAST]], 15
// LLVM: %[[GT_CMP:.*]] = icmp sgt i47 %[[SHIFT]], 2147483647
// LLVM: %[[MAX_SEL:.*]] = select i1 %[[GT_CMP]], i47 2147483647, i47 %[[SHIFT]]
// LLVM: %[[LT_CMP:.*]] = icmp slt i47 %[[MAX_SEL]], -2147483648
// LLVM: %[[BOUNDED:.*]] = select i1 %[[LT_CMP]], i47 -2147483648, i47 %[[MAX_SEL]]
// LLVM: %[[CAST_BACK:.*]] = trunc i47 %[[BOUNDED]] to i32
// LLVM: ret i32 %{{.*}}
_Sat _Accum int_to_sat_accum(int i) {
  return i;
}

// CIR-LABEL: cir.func{{.*}} @uint_to_fract
// CIR: %[[ARG:.*]] = cir.alloca "i" align(4) init : !cir.ptr<!u32i>
// CIR: %[[LOAD_ARG:.*]] = cir.load align(4) %[[ARG]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: %[[TRUNC:.*]] = cir.cast integral %[[LOAD_ARG]] : !u32i -> !s16i
// CIR-NEXT: %[[SHIFT_AMOUNT:.*]] = cir.const #cir.int<15> : !s16i
// CIR-NEXT: %[[SHIFT:.*]] = cir.shift(left, %[[TRUNC]] : !s16i, %[[SHIFT_AMOUNT]] : !s16i) -> !s16i
// CIR-NEXT: cir.store %[[SHIFT]], %[[RET:.*]] : !s16i, !cir.ptr<!s16i>
// CIR-NEXT: %[[LOAD_RET:.*]] = cir.load %[[RET]] : !cir.ptr<!s16i>, !s16i
// CIR-NEXT: cir.return %[[LOAD_RET]] : !s16i
// LLVM-LABEL: @uint_to_fract
// LLVM: %[[LOAD_ARG:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM: %[[CAST:.*]] = trunc i32 %[[LOAD_ARG]] to i16
// LLVM: %[[SHIFT:.*]] = shl i16 %[[CAST]], 15
// LLVM: ret i16 %{{.*}}
_Fract uint_to_fract(unsigned int i) {
  return i;
}

// CIR-LABEL: cir.func{{.*}} @uint_to_accum
// CIR: %[[ARG:.*]] = cir.alloca "i" align(4) init : !cir.ptr<!u32i>
// CIR: %[[LOAD_ARG:.*]] = cir.load align(4) %[[ARG]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: %[[CAST:.*]] = cir.cast integral %[[LOAD_ARG]] : !u32i -> !s32i
// CIR-NEXT: %[[SHIFT_AMOUNT:.*]] = cir.const #cir.int<15> : !s32i
// CIR-NEXT: %[[SHIFT:.*]] = cir.shift(left, %[[CAST]] : !s32i, %[[SHIFT_AMOUNT]] : !s32i) -> !s32i
// CIR-NEXT: cir.store %[[SHIFT]], %[[RET:.*]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT: %[[LOAD_RET:.*]] = cir.load %[[RET]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: cir.return %[[LOAD_RET]] : !s32i
// LLVM-LABEL: @uint_to_accum
// LLVM: %[[LOAD_ARG:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM: %[[SHIFT:.*]] = shl i32 %[[LOAD_ARG]], 15
// LLVM: ret i32 %{{.*}}
_Accum uint_to_accum(unsigned int i) {
  return i;
}

// CIR-LABEL: cir.func{{.*}} @uint_to_ufract
// CIR: %[[ARG:.*]] = cir.alloca "i" align(4) init : !cir.ptr<!u32i>
// CIR: %[[LOAD_ARG:.*]] = cir.load align(4) %[[ARG]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: %[[TRUNC:.*]] = cir.cast integral %[[LOAD_ARG]] : !u32i -> !u16i
// CIR-NEXT: %[[SHIFT_AMOUNT:.*]] = cir.const #cir.int<16> : !u16i
// CIR-NEXT: %[[SHIFT:.*]] = cir.shift(left, %[[TRUNC]] : !u16i, %[[SHIFT_AMOUNT]] : !u16i) -> !u16i
// CIR-NEXT: cir.store %[[SHIFT]], %[[RET:.*]] : !u16i, !cir.ptr<!u16i>
// CIR-NEXT: %[[LOAD_RET:.*]] = cir.load %[[RET]] : !cir.ptr<!u16i>, !u16i
// CIR-NEXT: cir.return %[[LOAD_RET]] : !u16i
// LLVM-LABEL: @uint_to_ufract
// LLVM: %[[LOAD_ARG:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM: %[[CAST:.*]] = trunc i32 %[[LOAD_ARG]] to i16
// LLVM: %[[SHIFT:.*]] = shl i16 %[[CAST]], 16
// LLVM: ret i16 %{{.*}}
unsigned _Fract uint_to_ufract(unsigned int i) {
  return i;
}

// CIR-LABEL: cir.func{{.*}} @uint_to_uaccum
// CIR: %[[ARG:.*]] = cir.alloca "i" align(4) init : !cir.ptr<!u32i>
// CIR: %[[LOAD_ARG:.*]] = cir.load align(4) %[[ARG]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: %[[SHIFT_AMOUNT:.*]] = cir.const #cir.int<16> : !u32i
// CIR-NEXT: %[[SHIFT:.*]] = cir.shift(left, %[[LOAD_ARG]] : !u32i, %[[SHIFT_AMOUNT]] : !u32i) -> !u32i
// CIR-NEXT: cir.store %[[SHIFT]], %[[RET:.*]] : !u32i, !cir.ptr<!u32i>
// CIR-NEXT: %[[LOAD_RET:.*]] = cir.load %[[RET]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: cir.return %[[LOAD_RET]] : !u32i
// LLVM-LABEL: @uint_to_uaccum
// LLVM: %[[LOAD_ARG:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM: %[[SHIFT:.*]] = shl i32 %[[LOAD_ARG]], 16
// LLVM: ret i32 %{{.*}}
unsigned _Accum uint_to_uaccum(unsigned int i) {
  return i;
}

// CIR-LABEL: cir.func{{.*}} @uint_to_sat_fract
// CIR: %[[ARG:.*]] = cir.alloca "i" align(4) init : !cir.ptr<!u32i>
// CIR: %[[LOAD_ARG:.*]] = cir.load align(4) %[[ARG]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: %[[CAST:.*]] = cir.cast integral %[[LOAD_ARG]] : !u32i -> !cir.int<u, 47>
// CIR-NEXT: %[[SHIFT_AMOUNT:.*]] = cir.const #cir.int<15> : !cir.int<u, 47>
// CIR-NEXT: %[[SHIFT:.*]] = cir.shift(left, %[[CAST]] : !cir.int<u, 47>, %[[SHIFT_AMOUNT]] : !cir.int<u, 47>) -> !cir.int<u, 47>
// CIR-NEXT: %[[MAX:.*]] = cir.const #cir.int<32767> : !cir.int<u, 47>
// CIR-NEXT: %[[GT_CMP:.*]] = cir.cmp gt %[[SHIFT]], %[[MAX]] : !cir.int<u, 47>
// CIR-NEXT: %[[MAX_SEL:.*]] = cir.select if %[[GT_CMP]] then %[[MAX]] else %[[SHIFT]] : (!cir.bool, !cir.int<u, 47>, !cir.int<u, 47>) -> !cir.int<u, 47>
// CIR-NEXT: %[[CAST_BACK:.*]] = cir.cast integral %[[MAX_SEL]] : !cir.int<u, 47> -> !s16i
// CIR-NEXT: cir.store %[[CAST_BACK]], %[[RET:.*]] : !s16i, !cir.ptr<!s16i>
// CIR-NEXT: %[[LOAD_RET:.*]] = cir.load %[[RET]] : !cir.ptr<!s16i>, !s16i
// CIR-NEXT: cir.return %[[LOAD_RET]] : !s16i
// LLVM-LABEL: @uint_to_sat_fract
// LLVM: %[[LOAD_ARG:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM: %[[CAST:.*]] = zext i32 %[[LOAD_ARG]] to i47
// LLVM: %[[SHIFT:.*]] = shl i47 %[[CAST]], 15
// LLVM: %[[GT_CMP:.*]] = icmp ugt i47 %[[SHIFT]], 32767
// LLVM: %[[MAX_SEL:.*]] = select i1 %[[GT_CMP]], i47 32767, i47 %[[SHIFT]]
// LLVM: %[[CAST:.*]] = trunc i47 %[[MAX_SEL]] to i16
// LLVM: ret i16 %{{.*}}
_Sat _Fract uint_to_sat_fract(unsigned int i) {
  return i;
}

// CIR-LABEL: cir.func{{.*}} @uint_to_sat_accum
// CIR: %[[ARG:.*]] = cir.alloca "i" align(4) init : !cir.ptr<!u32i>
// CIR: %[[LOAD_ARG:.*]] = cir.load align(4) %[[ARG]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: %[[CAST:.*]] = cir.cast integral %[[LOAD_ARG]] : !u32i -> !cir.int<u, 47>
// CIR-NEXT: %[[SHIFT_AMOUNT:.*]] = cir.const #cir.int<15> : !cir.int<u, 47>
// CIR-NEXT: %[[SHIFT:.*]] = cir.shift(left, %[[CAST]] : !cir.int<u, 47>, %[[SHIFT_AMOUNT]] : !cir.int<u, 47>) -> !cir.int<u, 47>
// CIR-NEXT: %[[MAX:.*]] = cir.const #cir.int<2147483647> : !cir.int<u, 47>
// CIR-NEXT: %[[GT_CMP:.*]] = cir.cmp gt %[[SHIFT]], %[[MAX]] : !cir.int<u, 47>
// CIR-NEXT: %[[MAX_SEL:.*]] = cir.select if %[[GT_CMP]] then %[[MAX]] else %[[SHIFT]] : (!cir.bool, !cir.int<u, 47>, !cir.int<u, 47>) -> !cir.int<u, 47>
// CIR-NEXT: %[[CAST_BACK:.*]] = cir.cast integral %[[MAX_SEL]] : !cir.int<u, 47> -> !s32i
// CIR-NEXT: cir.store %[[CAST_BACK]], %[[RET:.*]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT: %[[LOAD_RET:.*]] = cir.load %[[RET]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: cir.return %[[LOAD_RET]] : !s32i
// LLVM-LABEL: @uint_to_sat_accum
// LLVM: %[[LOAD_ARG:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM: %[[CAST:.*]] = zext i32 %[[LOAD_ARG]] to i47
// LLVM: %[[SHIFT:.*]] = shl i47 %[[CAST]], 15
// LLVM: %[[GT_CMP:.*]] = icmp ugt i47 %[[SHIFT]], 2147483647
// LLVM: %[[MAX_SEL:.*]] = select i1 %[[GT_CMP]], i47 2147483647, i47 %[[SHIFT]]
// LLVM: %[[CAST_BACK:.*]] = trunc i47 %[[MAX_SEL]] to i32
// LLVM: ret i32 %{{.*}}
_Sat _Accum uint_to_sat_accum(unsigned int i) {
  return i;
}

// CIR-LABEL: cir.func{{.*}} @bool_to_fract
// CIR: %[[ARG:.*]] = cir.alloca "i" align(1) init : !cir.ptr<!cir.bool>
// CIR: %[[LOAD_ARG:.*]] = cir.load align(1) %[[ARG]] : !cir.ptr<!cir.bool>, !cir.bool
// CIR-NEXT: %[[BOOL_CAST:.*]] = cir.cast bool_to_int %[[LOAD_ARG]] : !cir.bool -> !cir.int<u, 1>
// CIR-NEXT: %[[CAST:.*]] = cir.cast integral %[[BOOL_CAST]] : !cir.int<u, 1> -> !s16i
// CIR-NEXT: %[[SHIFT_AMOUNT:.*]] = cir.const #cir.int<15> : !s16i
// CIR-NEXT: %[[SHIFT:.*]] = cir.shift(left, %[[CAST]] : !s16i, %[[SHIFT_AMOUNT]] : !s16i) -> !s16i
// CIR-NEXT: cir.store %[[SHIFT]], %[[RET:.*]] : !s16i, !cir.ptr<!s16i>
// CIR-NEXT: %[[LOAD_RET:.*]] = cir.load %[[RET]] : !cir.ptr<!s16i>, !s16i
// CIR-NEXT: cir.return %[[LOAD_RET]] : !s16i
// LLVM-LABEL: @bool_to_fract
// LLVM: %[[LOAD_ARG:.*]] = load i8, ptr %{{.*}}, align 1
// LLVM: %[[CAST:.*]] = zext i1 %{{.*}} to i16
// LLVM: %[[SHIFT:.*]] = shl i16 %[[CAST]], 15
// LLVM: ret i16 %{{.*}}
_Fract bool_to_fract(bool i) {
  return i;
}

// CIR-LABEL: cir.func{{.*}} @bool_to_accum
// CIR: %[[ARG:.*]] = cir.alloca "i" align(1) init : !cir.ptr<!cir.bool>
// CIR: %[[LOAD_ARG:.*]] = cir.load align(1) %[[ARG]] : !cir.ptr<!cir.bool>, !cir.bool
// CIR-NEXT: %[[BOOL_CAST:.*]] = cir.cast bool_to_int %[[LOAD_ARG]] : !cir.bool -> !cir.int<u, 1>
// CIR-NEXT: %[[CAST:.*]] = cir.cast integral %[[BOOL_CAST]] : !cir.int<u, 1> -> !s32i
// CIR-NEXT: %[[SHIFT_AMOUNT:.*]] = cir.const #cir.int<15> : !s32i
// CIR-NEXT: %[[SHIFT:.*]] = cir.shift(left, %[[CAST]] : !s32i, %[[SHIFT_AMOUNT]] : !s32i) -> !s32i
// CIR-NEXT: cir.store %[[SHIFT]], %[[RET:.*]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT: %[[LOAD_RET:.*]] = cir.load %[[RET]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: cir.return %[[LOAD_RET]] : !s32i
// LLVM-LABEL: @bool_to_accum
// LLVM: %[[LOAD_ARG:.*]] = load i8, ptr %{{.*}}, align 1
// LLVM: %[[CAST:.*]] = zext i1 %{{.*}} to i32
// LLVM: %[[SHIFT:.*]] = shl i32 %[[CAST]], 15
// LLVM: ret i32 %{{.*}}
_Accum bool_to_accum(bool i) {
  return i;
}

// CIR-LABEL: cir.func{{.*}} @bool_to_ufract
// CIR: %[[ARG:.*]] = cir.alloca "i" align(1) init : !cir.ptr<!cir.bool>
// CIR: %[[LOAD_ARG:.*]] = cir.load align(1) %[[ARG]] : !cir.ptr<!cir.bool>, !cir.bool
// CIR-NEXT: %[[BOOL_CAST:.*]] = cir.cast bool_to_int %[[LOAD_ARG]] : !cir.bool -> !cir.int<u, 1>
// CIR-NEXT: %[[CAST:.*]] = cir.cast integral %[[BOOL_CAST]] : !cir.int<u, 1> -> !u16i
// CIR-NEXT: %[[SHIFT_AMOUNT:.*]] = cir.const #cir.int<16> : !u16i
// CIR-NEXT: %[[SHIFT:.*]] = cir.shift(left, %[[CAST]] : !u16i, %[[SHIFT_AMOUNT]] : !u16i) -> !u16i
// CIR-NEXT: cir.store %[[SHIFT]], %[[RET:.*]] : !u16i, !cir.ptr<!u16i>
// CIR-NEXT: %[[LOAD_RET:.*]] = cir.load %[[RET]] : !cir.ptr<!u16i>, !u16i
// CIR-NEXT: cir.return %[[LOAD_RET]] : !u16i
// LLVM-LABEL: @bool_to_ufract
// LLVM: %[[LOAD_ARG:.*]] = load i8, ptr %{{.*}}, align 1
// LLVM: %[[CAST:.*]] = zext i1 %{{.*}} to i16
// LLVM: %[[SHIFT:.*]] = shl i16 %[[CAST]], 16
// LLVM: ret i16 %{{.*}}
unsigned _Fract bool_to_ufract(bool i) {
  return i;
}

// CIR-LABEL: cir.func{{.*}} @bool_to_uaccum
// CIR: %[[ARG:.*]] = cir.alloca "i" align(1) init : !cir.ptr<!cir.bool>
// CIR: %[[LOAD_ARG:.*]] = cir.load align(1) %[[ARG]] : !cir.ptr<!cir.bool>, !cir.bool
// CIR-NEXT: %[[BOOL_CAST:.*]] = cir.cast bool_to_int %[[LOAD_ARG]] : !cir.bool -> !cir.int<u, 1>
// CIR-NEXT: %[[CAST:.*]] = cir.cast integral %[[BOOL_CAST]] : !cir.int<u, 1> -> !u32i
// CIR-NEXT: %[[SHIFT_AMOUNT:.*]] = cir.const #cir.int<16> : !u32i
// CIR-NEXT: %[[SHIFT:.*]] = cir.shift(left, %[[CAST]] : !u32i, %[[SHIFT_AMOUNT]] : !u32i) -> !u32i
// CIR-NEXT: cir.store %[[SHIFT]], %[[RET:.*]] : !u32i, !cir.ptr<!u32i>
// CIR-NEXT: %[[LOAD_RET:.*]] = cir.load %[[RET]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: cir.return %[[LOAD_RET]] : !u32i
// LLVM-LABEL: @bool_to_uaccum
// LLVM: %[[LOAD_ARG:.*]] = load i8, ptr %{{.*}}, align 1
// LLVM: %[[CAST:.*]] = zext i1 %{{.*}} to i32
// LLVM: %[[SHIFT:.*]] = shl i32 %[[CAST]], 16
// LLVM: ret i32 %{{.*}}
unsigned _Accum bool_to_uaccum(bool i) {
  return i;
}

// CIR-LABEL: cir.func{{.*}} @bool_to_sat_fract
// CIR: %[[ARG:.*]] = cir.alloca "i" align(1) init : !cir.ptr<!cir.bool>
// CIR: %[[LOAD_ARG:.*]] = cir.load align(1) %[[ARG]] : !cir.ptr<!cir.bool>, !cir.bool
// CIR-NEXT: %[[BOOL_CAST:.*]] = cir.cast bool_to_int %[[LOAD_ARG]] : !cir.bool -> !cir.int<u, 1>
// CIR-NEXT: %[[CAST:.*]] = cir.cast integral %[[BOOL_CAST]] : !cir.int<u, 1> -> !u16i
// CIR-NEXT: %[[SHIFT_AMOUNT:.*]] = cir.const #cir.int<15> : !u16i
// CIR-NEXT: %[[SHIFT:.*]] = cir.shift(left, %[[CAST]] : !u16i, %[[SHIFT_AMOUNT]] : !u16i) -> !u16i
// CIR-NEXT: %[[MAX:.*]] = cir.const #cir.int<32767> : !u16i
// CIR-NEXT: %[[GT_CMP:.*]] = cir.cmp gt %[[SHIFT]], %[[MAX]] : !u16i
// CIR-NEXT: %[[MAX_SEL:.*]] = cir.select if %[[GT_CMP]] then %[[MAX]] else %[[SHIFT]] : (!cir.bool, !u16i, !u16i) -> !u16i
// CIR-NEXT: %[[RET_BITCAST:.*]] = cir.cast bitcast %[[RET:.*]] : !cir.ptr<!s16i> -> !cir.ptr<!u16i>
// CIR-NEXT: cir.store %[[MAX_SEL]], %[[RET_BITCAST]] : !u16i, !cir.ptr<!u16i>
// CIR-NEXT: %[[LOAD_RET:.*]] = cir.load %[[RET]] : !cir.ptr<!s16i>, !s16i
// CIR-NEXT: cir.return %[[LOAD_RET]] : !s16i
// LLVM-LABEL: @bool_to_sat_fract
// LLVM: %[[LOAD_ARG:.*]] = load i8, ptr %{{.*}}, align 1
// LLVM: %[[CAST:.*]] = zext i1 %{{.*}} to i16
// LLVM: %[[SHIFT:.*]] = shl i16 %[[CAST]], 15
// LLVM: %[[GT_CMP:.*]] = icmp ugt i16 %[[SHIFT]], 32767
// LLVM: %[[MAX_SEL:.*]] = select i1 %[[GT_CMP]], i16 32767, i16 %[[SHIFT]]
// LLVM: ret i16 %{{.*}}
_Sat _Fract bool_to_sat_fract(bool i) {
  return i;
}

// CIR-LABEL: cir.func{{.*}} @bool_to_sat_accum
// CIR: %[[ARG:.*]] = cir.alloca "i" align(1) init : !cir.ptr<!cir.bool>
// CIR: %[[LOAD_ARG:.*]] = cir.load align(1) %[[ARG]] : !cir.ptr<!cir.bool>, !cir.bool
// CIR-NEXT: %[[BOOL_CAST:.*]] = cir.cast bool_to_int %[[LOAD_ARG]] : !cir.bool -> !cir.int<u, 1>
// CIR-NEXT: %[[CAST:.*]] = cir.cast integral %[[BOOL_CAST]] : !cir.int<u, 1> -> !u32i
// CIR-NEXT: %[[SHIFT_AMOUNT:.*]] = cir.const #cir.int<15> : !u32i
// CIR-NEXT: %[[SHIFT:.*]] = cir.shift(left, %[[CAST]] : !u32i, %[[SHIFT_AMOUNT]] : !u32i) -> !u32i
// CIR-NEXT: %[[RET_BITCAST:.*]] = cir.cast bitcast %[[RET:.*]] : !cir.ptr<!s32i> -> !cir.ptr<!u32i>
// CIR-NEXT: cir.store %[[SHIFT]], %[[RET_BITCAST]] : !u32i, !cir.ptr<!u32i>
// CIR-NEXT: %[[LOAD_RET:.*]] = cir.load %[[RET]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: cir.return %[[LOAD_RET]] : !s32i
// LLVM-LABEL: @bool_to_sat_accum
// LLVM: %[[LOAD_ARG:.*]] = load i8, ptr %{{.*}}, align 1
// LLVM: %[[CAST:.*]] = zext i1 %{{.*}} to i32
// LLVM: %[[SHIFT:.*]] = shl i32 %[[CAST]], 15
// LLVM: ret i32 %{{.*}}
_Sat _Accum bool_to_sat_accum(bool i) {
  return i;
}

// CIR-LABEL: cir.func{{.*}} @float_to_fract
// CIR: %[[ARG:.*]] = cir.alloca "i" align(4) init : !cir.ptr<!cir.float>
// CIR: %[[LOAD_ARG:.*]] = cir.load align(4) %[[ARG]] : !cir.ptr<!cir.float>, !cir.float
// CIR-NEXT: %[[SCALE:.*]] = cir.const #cir.fp<3.276800e+04> : !cir.float
// CIR-NEXT: %[[SCALED:.*]] = cir.fmul %[[LOAD_ARG]], %[[SCALE]] : !cir.float
// CIR-NEXT: %[[CAST:.*]] = cir.cast float_to_int %[[SCALED]] : !cir.float -> !s16i
// CIR-NEXT: cir.store %[[CAST]], %[[RET:.*]] : !s16i, !cir.ptr<!s16i>
// CIR-NEXT: %[[LOAD_RET:.*]] = cir.load %[[RET]] : !cir.ptr<!s16i>, !s16i
// CIR-NEXT: cir.return %[[LOAD_RET]] : !s16i
// LLVM-LABEL: @float_to_fract
// LLVM: %[[LOAD_ARG:.*]] = load float, ptr %{{.*}}, align 4
// LLVM: %[[SCALED:.*]] = fmul float %[[LOAD_ARG]], 3.276800e+04
// LLVM: %[[CAST:.*]] = fptosi float %[[SCALED]] to i16
// LLVM: ret i16 %{{.*}}
_Fract float_to_fract(float i) {
  return i;
}

// CIR-LABEL: cir.func{{.*}} @float_to_accum
// CIR: %[[ARG:.*]] = cir.alloca "i" align(4) init : !cir.ptr<!cir.float>
// CIR: %[[LOAD_ARG:.*]] = cir.load align(4) %[[ARG]] : !cir.ptr<!cir.float>, !cir.float
// CIR-NEXT: %[[SCALE:.*]] = cir.const #cir.fp<3.276800e+04> : !cir.float
// CIR-NEXT: %[[SCALED:.*]] = cir.fmul %[[LOAD_ARG]], %[[SCALE]] : !cir.float
// CIR-NEXT: %[[CAST:.*]] = cir.cast float_to_int %[[SCALED]] : !cir.float -> !s32i
// CIR-NEXT: cir.store %[[CAST]], %[[RET:.*]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT: %[[LOAD_RET:.*]] = cir.load %[[RET]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: cir.return %[[LOAD_RET]] : !s32i
// LLVM-LABEL: @float_to_accum
// LLVM: %[[LOAD_ARG:.*]] = load float, ptr %{{.*}}, align 4
// LLVM: %[[SCALED:.*]] = fmul float %[[LOAD_ARG]], 3.276800e+04
// LLVM: %[[CAST:.*]] = fptosi float %[[SCALED]] to i32
// LLVM: ret i32 %{{.*}}
_Accum float_to_accum(float i) {
  return i;
}

// CIR-LABEL: cir.func{{.*}} @float_to_ufract
// CIR: %[[ARG:.*]] = cir.alloca "i" align(4) init : !cir.ptr<!cir.float>
// CIR: %[[LOAD_ARG:.*]] = cir.load align(4) %[[ARG]] : !cir.ptr<!cir.float>, !cir.float
// CIR-NEXT: %[[SCALE:.*]] = cir.const #cir.fp<6.553600e+04> : !cir.float
// CIR-NEXT: %[[SCALED:.*]] = cir.fmul %[[LOAD_ARG]], %[[SCALE]] : !cir.float
// CIR-NEXT: %[[CAST:.*]] = cir.cast float_to_int %[[SCALED]] : !cir.float -> !u16i
// CIR-NEXT: cir.store %[[CAST]], %[[RET:.*]] : !u16i, !cir.ptr<!u16i>
// CIR-NEXT: %[[LOAD_RET:.*]] = cir.load %[[RET]] : !cir.ptr<!u16i>, !u16i
// CIR-NEXT: cir.return %[[LOAD_RET]] : !u16i
// LLVM-LABEL: @float_to_ufract
// LLVM: %[[LOAD_ARG:.*]] = load float, ptr %{{.*}}, align 4
// LLVM: %[[SCALED:.*]] = fmul float %[[LOAD_ARG]], 6.553600e+04
// LLVM: %[[CAST:.*]] = fptoui float %[[SCALED]] to i16
// LLVM: ret i16 %{{.*}}
unsigned _Fract float_to_ufract(float i) {
  return i;
}

// CIR-LABEL: cir.func{{.*}} @float_to_uaccum
// CIR: %[[ARG:.*]] = cir.alloca "i" align(4) init : !cir.ptr<!cir.float>
// CIR: %[[LOAD_ARG:.*]] = cir.load align(4) %[[ARG]] : !cir.ptr<!cir.float>, !cir.float
// CIR-NEXT: %[[SCALE:.*]] = cir.const #cir.fp<6.553600e+04> : !cir.float
// CIR-NEXT: %[[SCALED:.*]] = cir.fmul %[[LOAD_ARG]], %[[SCALE]] : !cir.float
// CIR-NEXT: %[[CAST:.*]] = cir.cast float_to_int %[[SCALED]] : !cir.float -> !u32i
// CIR-NEXT: cir.store %[[CAST]], %[[RET:.*]] : !u32i, !cir.ptr<!u32i>
// CIR-NEXT: %[[LOAD_RET:.*]] = cir.load %[[RET]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: cir.return %[[LOAD_RET]] : !u32i
// LLVM-LABEL: @float_to_uaccum
// LLVM: %[[LOAD_ARG:.*]] = load float, ptr %{{.*}}, align 4
// LLVM: %[[SCALED:.*]] = fmul float %[[LOAD_ARG]], 6.553600e+04
// LLVM: %[[CAST:.*]] = fptoui float %[[SCALED]] to i32
// LLVM: ret i32 %{{.*}}
unsigned _Accum float_to_uaccum(float i) {
  return i;
}

// CIR-LABEL: cir.func{{.*}} @float_to_sat_fract
// CIR: %[[ARG:.*]] = cir.alloca "i" align(4) init : !cir.ptr<!cir.float>
// CIR: %[[LOAD_ARG:.*]] = cir.load align(4) %[[ARG]] : !cir.ptr<!cir.float>, !cir.float
// CIR-NEXT: %[[SCALE:.*]] = cir.const #cir.fp<3.276800e+04> : !cir.float
// CIR-NEXT: %[[SCALED:.*]] = cir.fmul %[[LOAD_ARG]], %[[SCALE]] : !cir.float
// CIR-NEXT: %[[SAT_CAST:.*]] = cir.call_llvm_intrinsic "fptosi.sat" %[[SCALED]] : (!cir.float) -> !s16i
// CIR-NEXT: cir.store %[[SAT_CAST]], %[[RET:.*]] : !s16i, !cir.ptr<!s16i>
// CIR-NEXT: %[[LOAD_RET:.*]] = cir.load %[[RET]] : !cir.ptr<!s16i>, !s16i
// CIR-NEXT: cir.return %[[LOAD_RET]] : !s16i
// LLVM-LABEL: @float_to_sat_fract
// LLVM: %[[LOAD_ARG:.*]] = load float, ptr %{{.*}}, align 4
// LLVM: %[[SCALED:.*]] = fmul float %[[LOAD_ARG]], 3.276800e+04
// LLVM: %[[SAT_CAST:.*]] = call i16 @llvm.fptosi.sat.i16.f32(float %[[SCALED]])
// LLVM: ret i16 %{{.*}}
_Sat _Fract float_to_sat_fract(float i) {
  return i;
}

// CIR-LABEL: cir.func{{.*}} @float_to_sat_accum
// CIR: %[[ARG:.*]] = cir.alloca "i" align(4) init : !cir.ptr<!cir.float>
// CIR: %[[LOAD_ARG:.*]] = cir.load align(4) %[[ARG]] : !cir.ptr<!cir.float>, !cir.float
// CIR-NEXT: %[[SCALE:.*]] = cir.const #cir.fp<3.276800e+04> : !cir.float
// CIR-NEXT: %[[SCALED:.*]] = cir.fmul %[[LOAD_ARG]], %[[SCALE]] : !cir.float
// CIR-NEXT: %[[SAT_CAST:.*]] = cir.call_llvm_intrinsic "fptosi.sat" %[[SCALED]] : (!cir.float) -> !s32i
// CIR-NEXT: cir.store %[[SAT_CAST]], %[[RET:.*]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT: %[[LOAD_RET:.*]] = cir.load %[[RET]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: cir.return %[[LOAD_RET]] : !s32i
// LLVM-LABEL: @float_to_sat_accum
// LLVM: %[[LOAD_ARG:.*]] = load float, ptr %{{.*}}, align 4
// LLVM: %[[SCALED:.*]] = fmul float %[[LOAD_ARG]], 3.276800e+04
// LLVM: %[[SAT_CAST:.*]] = call i32 @llvm.fptosi.sat.i32.f32(float %[[SCALED]])
// LLVM: ret i32 %{{.*}}
_Sat _Accum float_to_sat_accum(float i) {
  return i;
}

// CIR-LABEL: cir.func{{.*}} @double_to_fract
// CIR: %[[ARG:.*]] = cir.alloca "i" align(8) init : !cir.ptr<!cir.double>
// CIR: %[[LOAD_ARG:.*]] = cir.load align(8) %[[ARG]] : !cir.ptr<!cir.double>, !cir.double
// CIR-NEXT: %[[SCALE:.*]] = cir.const #cir.fp<3.276800e+04> : !cir.double
// CIR-NEXT: %[[SCALED:.*]] = cir.fmul %[[LOAD_ARG]], %[[SCALE]] : !cir.double
// CIR-NEXT: %[[CAST:.*]] = cir.cast float_to_int %[[SCALED]] : !cir.double -> !s16i
// CIR-NEXT: cir.store %[[CAST]], %[[RET:.*]] : !s16i, !cir.ptr<!s16i>
// CIR-NEXT: %[[LOAD_RET:.*]] = cir.load %[[RET]] : !cir.ptr<!s16i>, !s16i
// CIR-NEXT: cir.return %[[LOAD_RET]] : !s16i
// LLVM-LABEL: @double_to_fract
// LLVM: %[[LOAD_ARG:.*]] = load double, ptr %{{.*}}, align 8
// LLVM: %[[SCALED:.*]] = fmul double %[[LOAD_ARG]], 3.276800e+04
// LLVM: %[[CAST:.*]] = fptosi double %[[SCALED]] to i16
// LLVM: ret i16 %{{.*}}
_Fract double_to_fract(double i) {
  return i;
}

// CIR-LABEL: cir.func{{.*}} @double_to_accum
// CIR: %[[ARG:.*]] = cir.alloca "i" align(8) init : !cir.ptr<!cir.double>
// CIR: %[[LOAD_ARG:.*]] = cir.load align(8) %[[ARG]] : !cir.ptr<!cir.double>, !cir.double
// CIR-NEXT: %[[SCALE:.*]] = cir.const #cir.fp<3.276800e+04> : !cir.double
// CIR-NEXT: %[[SCALED:.*]] = cir.fmul %[[LOAD_ARG]], %[[SCALE]] : !cir.double
// CIR-NEXT: %[[CAST:.*]] = cir.cast float_to_int %[[SCALED]] : !cir.double -> !s32i
// CIR-NEXT: cir.store %[[CAST]], %[[RET:.*]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT: %[[LOAD_RET:.*]] = cir.load %[[RET]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: cir.return %[[LOAD_RET]] : !s32i
// LLVM-LABEL: @double_to_accum
// LLVM: %[[LOAD_ARG:.*]] = load double, ptr %{{.*}}, align 8
// LLVM: %[[SCALED:.*]] = fmul double %[[LOAD_ARG]], 3.276800e+04
// LLVM: %[[CAST:.*]] = fptosi double %[[SCALED]] to i32
// LLVM: ret i32 %{{.*}}
_Accum double_to_accum(double i) {
  return i;
}

// CIR-LABEL: cir.func{{.*}} @double_to_ufract
// CIR: %[[ARG:.*]] = cir.alloca "i" align(8) init : !cir.ptr<!cir.double>
// CIR: %[[LOAD_ARG:.*]] = cir.load align(8) %[[ARG]] : !cir.ptr<!cir.double>, !cir.double
// CIR-NEXT: %[[SCALE:.*]] = cir.const #cir.fp<6.553600e+04> : !cir.double
// CIR-NEXT: %[[SCALED:.*]] = cir.fmul %[[LOAD_ARG]], %[[SCALE]] : !cir.double
// CIR-NEXT: %[[CAST:.*]] = cir.cast float_to_int %[[SCALED]] : !cir.double -> !u16i
// CIR-NEXT: cir.store %[[CAST]], %[[RET:.*]] : !u16i, !cir.ptr<!u16i>
// CIR-NEXT: %[[LOAD_RET:.*]] = cir.load %[[RET]] : !cir.ptr<!u16i>, !u16i
// CIR-NEXT: cir.return %[[LOAD_RET]] : !u16i
// LLVM-LABEL: @double_to_ufract
// LLVM: %[[LOAD_ARG:.*]] = load double, ptr %{{.*}}, align 8
// LLVM: %[[SCALED:.*]] = fmul double %[[LOAD_ARG]], 6.553600e+04
// LLVM: %[[CAST:.*]] = fptoui double %[[SCALED]] to i16
// LLVM: ret i16 %{{.*}}
unsigned _Fract double_to_ufract(double i) {
  return i;
}

// CIR-LABEL: cir.func{{.*}} @double_to_uaccum
// CIR: %[[ARG:.*]] = cir.alloca "i" align(8) init : !cir.ptr<!cir.double>
// CIR: %[[LOAD_ARG:.*]] = cir.load align(8) %[[ARG]] : !cir.ptr<!cir.double>, !cir.double
// CIR-NEXT: %[[SCALE:.*]] = cir.const #cir.fp<6.553600e+04> : !cir.double
// CIR-NEXT: %[[SCALED:.*]] = cir.fmul %[[LOAD_ARG]], %[[SCALE]] : !cir.double
// CIR-NEXT: %[[CAST:.*]] = cir.cast float_to_int %[[SCALED]] : !cir.double -> !u32i
// CIR-NEXT: cir.store %[[CAST]], %[[RET:.*]] : !u32i, !cir.ptr<!u32i>
// CIR-NEXT: %[[LOAD_RET:.*]] = cir.load %[[RET]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: cir.return %[[LOAD_RET]] : !u32i
// LLVM-LABEL: @double_to_uaccum
// LLVM: %[[LOAD_ARG:.*]] = load double, ptr %{{.*}}, align 8
// LLVM: %[[SCALED:.*]] = fmul double %[[LOAD_ARG]], 6.553600e+04
// LLVM: %[[CAST:.*]] = fptoui double %[[SCALED]] to i32
// LLVM: ret i32 %{{.*}}
unsigned _Accum double_to_uaccum(double i) {
  return i;
}

// CIR-LABEL: cir.func{{.*}} @double_to_sat_fract
// CIR: %[[ARG:.*]] = cir.alloca "i" align(8) init : !cir.ptr<!cir.double>
// CIR: %[[LOAD_ARG:.*]] = cir.load align(8) %[[ARG]] : !cir.ptr<!cir.double>, !cir.double
// CIR-NEXT: %[[SCALE:.*]] = cir.const #cir.fp<3.276800e+04> : !cir.double
// CIR-NEXT: %[[SCALED:.*]] = cir.fmul %[[LOAD_ARG]], %[[SCALE]] : !cir.double
// CIR-NEXT: %[[SAT_CAST:.*]] = cir.call_llvm_intrinsic "fptosi.sat" %[[SCALED]] : (!cir.double) -> !s16i
// CIR-NEXT: cir.store %[[SAT_CAST]], %[[RET:.*]] : !s16i, !cir.ptr<!s16i>
// CIR-NEXT: %[[LOAD_RET:.*]] = cir.load %[[RET]] : !cir.ptr<!s16i>, !s16i
// CIR-NEXT: cir.return %[[LOAD_RET]] : !s16i
// LLVM-LABEL: @double_to_sat_fract
// LLVM: %[[LOAD_ARG:.*]] = load double, ptr %{{.*}}, align 8
// LLVM: %[[SCALED:.*]] = fmul double %[[LOAD_ARG]], 3.276800e+04
// LLVM: %[[SAT_CAST:.*]] = call i16 @llvm.fptosi.sat.i16.f64(double %[[SCALED]])
// LLVM: ret i16 %{{.*}}
_Sat _Fract double_to_sat_fract(double i) {
  return i;
}

// CIR-LABEL: cir.func{{.*}} @double_to_sat_accum
// CIR: %[[ARG:.*]] = cir.alloca "i" align(8) init : !cir.ptr<!cir.double>
// CIR: %[[LOAD_ARG:.*]] = cir.load align(8) %[[ARG]] : !cir.ptr<!cir.double>, !cir.double
// CIR-NEXT: %[[SCALE:.*]] = cir.const #cir.fp<3.276800e+04> : !cir.double
// CIR-NEXT: %[[SCALED:.*]] = cir.fmul %[[LOAD_ARG]], %[[SCALE]] : !cir.double
// CIR-NEXT: %[[SAT_CAST:.*]] = cir.call_llvm_intrinsic "fptosi.sat" %[[SCALED]] : (!cir.double) -> !s32i
// CIR-NEXT: cir.store %[[SAT_CAST]], %[[RET:.*]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT: %[[LOAD_RET:.*]] = cir.load %[[RET]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: cir.return %[[LOAD_RET]] : !s32i
// LLVM-LABEL: @double_to_sat_accum
// LLVM: %[[LOAD_ARG:.*]] = load double, ptr %{{.*}}, align 8
// LLVM: %[[SCALED:.*]] = fmul double %[[LOAD_ARG]], 3.276800e+04
// LLVM: %[[SAT_CAST:.*]] = call i32 @llvm.fptosi.sat.i32.f64(double %[[SCALED]])
// LLVM: ret i32 %{{.*}}
_Sat _Accum double_to_sat_accum(double i) {
  return i;
}

// CIR-LABEL: cir.func{{.*}} @fract_to_int
// CIR: %[[ARG:.*]] = cir.alloca "f" align(2) init : !cir.ptr<!s16i>
// CIR: %[[LOAD_ARG:.*]] = cir.load align(2) %[[ARG]] : !cir.ptr<!s16i>, !s16i
// CIR-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !s16i
// CIR-NEXT: %[[NEG_CMP:.*]] = cir.cmp lt %[[LOAD_ARG]], %[[ZERO]] : !s16i
// CIR-NEXT: %[[ROUND_BIAS:.*]] = cir.const #cir.int<32767> : !s16i
// CIR-NEXT: %[[ROUNDED:.*]] = cir.add %[[LOAD_ARG]], %[[ROUND_BIAS]] : !s16i
// CIR-NEXT: %[[SEL:.*]] = cir.select if %[[NEG_CMP]] then %[[ROUNDED]] else %[[LOAD_ARG]] : (!cir.bool, !s16i, !s16i) -> !s16i
// CIR-NEXT: %[[SHIFT_AMOUNT:.*]] = cir.const #cir.int<15> : !s16i
// CIR-NEXT: %[[SHIFT:.*]] = cir.shift(right, %[[SEL]] : !s16i, %[[SHIFT_AMOUNT]] : !s16i) -> !s16i
// CIR-NEXT: %[[CAST:.*]] = cir.cast integral %[[SHIFT]] : !s16i -> !s32i
// CIR-NEXT: cir.store %[[CAST]], %[[RET:.*]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT: %[[LOAD_RET:.*]] = cir.load %[[RET]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: cir.return %[[LOAD_RET]] : !s32i
// LLVM-LABEL: @fract_to_int
// LLVM: %[[LOAD_ARG:.*]] = load i16, ptr %{{.*}}, align 2
// LLVM: %[[NEG_CMP:.*]] = icmp slt i16 %[[LOAD_ARG]], 0
// LLVM: %[[ROUNDED:.*]] = add i16 %[[LOAD_ARG]], 32767
// LLVM: %[[SEL:.*]] = select i1 %[[NEG_CMP]], i16 %[[ROUNDED]], i16 %[[LOAD_ARG]]
// LLVM: %[[SHIFT:.*]] = ashr i16 %[[SEL]], 15
// LLVM: %[[CAST:.*]] = sext i16 %[[SHIFT]] to i32
// LLVM: ret i32 %{{.*}}
int fract_to_int(_Fract f) {
  return f;
}

// CIR-LABEL: cir.func{{.*}} @accum_to_int
// CIR: %[[ARG:.*]] = cir.alloca "a" align(4) init : !cir.ptr<!s32i>
// CIR: %[[LOAD_ARG:.*]] = cir.load align(4) %[[ARG]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR-NEXT: %[[NEG_CMP:.*]] = cir.cmp lt %[[LOAD_ARG]], %[[ZERO]] : !s32i
// CIR-NEXT: %[[ROUND_BIAS:.*]] = cir.const #cir.int<32767> : !s32i
// CIR-NEXT: %[[ROUNDED:.*]] = cir.add %[[LOAD_ARG]], %[[ROUND_BIAS]] : !s32i
// CIR-NEXT: %[[SEL:.*]] = cir.select if %[[NEG_CMP]] then %[[ROUNDED]] else %[[LOAD_ARG]] : (!cir.bool, !s32i, !s32i) -> !s32i
// CIR-NEXT: %[[SHIFT_AMOUNT:.*]] = cir.const #cir.int<15> : !s32i
// CIR-NEXT: %[[SHIFT:.*]] = cir.shift(right, %[[SEL]] : !s32i, %[[SHIFT_AMOUNT]] : !s32i) -> !s32i
// CIR-NEXT: cir.store %[[SHIFT]], %[[RET:.*]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT: %[[LOAD_RET:.*]] = cir.load %[[RET]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: cir.return %[[LOAD_RET]] : !s32i
// LLVM-LABEL: @accum_to_int
// LLVM: %[[LOAD_ARG:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM: %[[NEG_CMP:.*]] = icmp slt i32 %[[LOAD_ARG]], 0
// LLVM: %[[ROUNDED:.*]] = add i32 %[[LOAD_ARG]], 32767
// LLVM: %[[SEL:.*]] = select i1 %[[NEG_CMP]], i32 %[[ROUNDED]], i32 %[[LOAD_ARG]]
// LLVM: %[[SHIFT:.*]] = ashr i32 %[[SEL]], 15
// LLVM: ret i32 %{{.*}}
int accum_to_int(_Accum a) {
  return a;
}

// CIR-LABEL: cir.func{{.*}} @ufract_to_int
// CIR: %[[ARG:.*]] = cir.alloca "f" align(2) init : !cir.ptr<!u16i>
// CIR: %[[LOAD_ARG:.*]] = cir.load align(2) %[[ARG]] : !cir.ptr<!u16i>, !u16i
// CIR-NEXT: %[[SHIFT_AMOUNT:.*]] = cir.const #cir.int<16> : !u16i
// CIR-NEXT: %[[SHIFT:.*]] = cir.shift(right, %[[LOAD_ARG]] : !u16i, %[[SHIFT_AMOUNT]] : !u16i) -> !u16i
// CIR-NEXT: %[[CAST:.*]] = cir.cast integral %[[SHIFT]] : !u16i -> !s32i
// CIR-NEXT: cir.store %[[CAST]], %[[RET:.*]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT: %[[LOAD_RET:.*]] = cir.load %[[RET]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: cir.return %[[LOAD_RET]] : !s32i
// LLVM-LABEL: @ufract_to_int
// LLVM: %[[LOAD_ARG:.*]] = load i16, ptr %{{.*}}, align 2
// LLVM: %[[SHIFT:.*]] = lshr i16 %[[LOAD_ARG]], 16
// LLVM: %[[CAST:.*]] = zext i16 %[[SHIFT]] to i32
// LLVM: ret i32 %{{.*}}
int ufract_to_int(unsigned _Fract f) {
  return f;
}

// CIR-LABEL: cir.func{{.*}} @uaccum_to_int
// CIR: %[[ARG:.*]] = cir.alloca "a" align(4) init : !cir.ptr<!u32i>
// CIR: %[[LOAD_ARG:.*]] = cir.load align(4) %[[ARG]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: %[[SHIFT_AMOUNT:.*]] = cir.const #cir.int<16> : !u32i
// CIR-NEXT: %[[SHIFT:.*]] = cir.shift(right, %[[LOAD_ARG]] : !u32i, %[[SHIFT_AMOUNT]] : !u32i) -> !u32i
// CIR-NEXT: %[[CAST:.*]] = cir.cast integral %[[SHIFT]] : !u32i -> !s32i
// CIR-NEXT: cir.store %[[CAST]], %[[RET:.*]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT: %[[LOAD_RET:.*]] = cir.load %[[RET]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: cir.return %[[LOAD_RET]] : !s32i
// LLVM-LABEL: @uaccum_to_int
// LLVM: %[[LOAD_ARG:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM: %[[SHIFT:.*]] = lshr i32 %[[LOAD_ARG]], 16
// LLVM: ret i32 %{{.*}}
int uaccum_to_int(unsigned _Accum a) {
  return a;
}

// CIR-LABEL: cir.func{{.*}} @sat_fract_to_int
// CIR: %[[ARG:.*]] = cir.alloca "f" align(2) init : !cir.ptr<!s16i>
// CIR: %[[LOAD_ARG:.*]] = cir.load align(2) %[[ARG]] : !cir.ptr<!s16i>, !s16i
// CIR-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !s16i
// CIR-NEXT: %[[NEG_CMP:.*]] = cir.cmp lt %[[LOAD_ARG]], %[[ZERO]] : !s16i
// CIR-NEXT: %[[ROUND_BIAS:.*]] = cir.const #cir.int<32767> : !s16i
// CIR-NEXT: %[[ROUNDED:.*]] = cir.add %[[LOAD_ARG]], %[[ROUND_BIAS]] : !s16i
// CIR-NEXT: %[[SEL:.*]] = cir.select if %[[NEG_CMP]] then %[[ROUNDED]] else %[[LOAD_ARG]] : (!cir.bool, !s16i, !s16i) -> !s16i
// CIR-NEXT: %[[SHIFT_AMOUNT:.*]] = cir.const #cir.int<15> : !s16i
// CIR-NEXT: %[[SHIFT:.*]] = cir.shift(right, %[[SEL]] : !s16i, %[[SHIFT_AMOUNT]] : !s16i) -> !s16i
// CIR-NEXT: %[[CAST:.*]] = cir.cast integral %[[SHIFT]] : !s16i -> !s32i
// CIR-NEXT: cir.store %[[CAST]], %[[RET:.*]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT: %[[LOAD_RET:.*]] = cir.load %[[RET]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: cir.return %[[LOAD_RET]] : !s32i
// LLVM-LABEL: @sat_fract_to_int
// LLVM: %[[LOAD_ARG:.*]] = load i16, ptr %{{.*}}, align 2
// LLVM: %[[NEG_CMP:.*]] = icmp slt i16 %[[LOAD_ARG]], 0
// LLVM: %[[ROUNDED:.*]] = add i16 %[[LOAD_ARG]], 32767
// LLVM: %[[SEL:.*]] = select i1 %[[NEG_CMP]], i16 %[[ROUNDED]], i16 %[[LOAD_ARG]]
// LLVM: %[[SHIFT:.*]] = ashr i16 %[[SEL]], 15
// LLVM: %[[CAST:.*]] = sext i16 %[[SHIFT]] to i32
// LLVM: ret i32 %{{.*}}
int sat_fract_to_int(_Sat _Fract f) {
  return f;
}

// CIR-LABEL: cir.func{{.*}} @sat_accum_to_int
// CIR: %[[ARG:.*]] = cir.alloca "a" align(4) init : !cir.ptr<!s32i>
// CIR: %[[LOAD_ARG:.*]] = cir.load align(4) %[[ARG]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR-NEXT: %[[NEG_CMP:.*]] = cir.cmp lt %[[LOAD_ARG]], %[[ZERO]] : !s32i
// CIR-NEXT: %[[ROUND_BIAS:.*]] = cir.const #cir.int<32767> : !s32i
// CIR-NEXT: %[[ROUNDED:.*]] = cir.add %[[LOAD_ARG]], %[[ROUND_BIAS]] : !s32i
// CIR-NEXT: %[[SEL:.*]] = cir.select if %[[NEG_CMP]] then %[[ROUNDED]] else %[[LOAD_ARG]] : (!cir.bool, !s32i, !s32i) -> !s32i
// CIR-NEXT: %[[SHIFT_AMOUNT:.*]] = cir.const #cir.int<15> : !s32i
// CIR-NEXT: %[[SHIFT:.*]] = cir.shift(right, %[[SEL]] : !s32i, %[[SHIFT_AMOUNT]] : !s32i) -> !s32i
// CIR-NEXT: cir.store %[[SHIFT]], %[[RET:.*]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT: %[[LOAD_RET:.*]] = cir.load %[[RET]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: cir.return %[[LOAD_RET]] : !s32i
// LLVM-LABEL: @sat_accum_to_int
// LLVM: %[[LOAD_ARG:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM: %[[NEG_CMP:.*]] = icmp slt i32 %[[LOAD_ARG]], 0
// LLVM: %[[ROUNDED:.*]] = add i32 %[[LOAD_ARG]], 32767
// LLVM: %[[SEL:.*]] = select i1 %[[NEG_CMP]], i32 %[[ROUNDED]], i32 %[[LOAD_ARG]]
// LLVM: %[[SHIFT:.*]] = ashr i32 %[[SEL]], 15
// LLVM: ret i32 %{{.*}}
int sat_accum_to_int(_Sat _Accum a) {
  return a;
}

// CIR-LABEL: cir.func{{.*}} @fract_to_uint
// CIR: %[[ARG:.*]] = cir.alloca "f" align(2) init : !cir.ptr<!s16i>
// CIR: %[[LOAD_ARG:.*]] = cir.load align(2) %[[ARG]] : !cir.ptr<!s16i>, !s16i
// CIR-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !s16i
// CIR-NEXT: %[[NEG_CMP:.*]] = cir.cmp lt %[[LOAD_ARG]], %[[ZERO]] : !s16i
// CIR-NEXT: %[[ROUND_BIAS:.*]] = cir.const #cir.int<32767> : !s16i
// CIR-NEXT: %[[ROUNDED:.*]] = cir.add %[[LOAD_ARG]], %[[ROUND_BIAS]] : !s16i
// CIR-NEXT: %[[SEL:.*]] = cir.select if %[[NEG_CMP]] then %[[ROUNDED]] else %[[LOAD_ARG]] : (!cir.bool, !s16i, !s16i) -> !s16i
// CIR-NEXT: %[[SHIFT_AMOUNT:.*]] = cir.const #cir.int<15> : !s16i
// CIR-NEXT: %[[SHIFT:.*]] = cir.shift(right, %[[SEL]] : !s16i, %[[SHIFT_AMOUNT]] : !s16i) -> !s16i
// CIR-NEXT: %[[CAST:.*]] = cir.cast integral %[[SHIFT]] : !s16i -> !u32i
// CIR-NEXT: cir.store %[[CAST]], %[[RET:.*]] : !u32i, !cir.ptr<!u32i>
// CIR-NEXT: %[[LOAD_RET:.*]] = cir.load %[[RET]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: cir.return %[[LOAD_RET]] : !u32i
// LLVM-LABEL: @fract_to_uint
// LLVM: %[[LOAD_ARG:.*]] = load i16, ptr %{{.*}}, align 2
// LLVM: %[[NEG_CMP:.*]] = icmp slt i16 %[[LOAD_ARG]], 0
// LLVM: %[[ROUNDED:.*]] = add i16 %[[LOAD_ARG]], 32767
// LLVM: %[[SEL:.*]] = select i1 %[[NEG_CMP]], i16 %[[ROUNDED]], i16 %[[LOAD_ARG]]
// LLVM: %[[SHIFT:.*]] = ashr i16 %[[SEL]], 15
// LLVM: %[[CAST:.*]] = sext i16 %[[SHIFT]] to i32
// LLVM: ret i32 %{{.*}}
unsigned int fract_to_uint(_Fract f) {
  return f;
}

// CIR-LABEL: cir.func{{.*}} @accum_to_uint
// CIR: %[[ARG:.*]] = cir.alloca "a" align(4) init : !cir.ptr<!s32i>
// CIR: %[[LOAD_ARG:.*]] = cir.load align(4) %[[ARG]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR-NEXT: %[[NEG_CMP:.*]] = cir.cmp lt %[[LOAD_ARG]], %[[ZERO]] : !s32i
// CIR-NEXT: %[[ROUND_BIAS:.*]] = cir.const #cir.int<32767> : !s32i
// CIR-NEXT: %[[ROUNDED:.*]] = cir.add %[[LOAD_ARG]], %[[ROUND_BIAS]] : !s32i
// CIR-NEXT: %[[SEL:.*]] = cir.select if %[[NEG_CMP]] then %[[ROUNDED]] else %[[LOAD_ARG]] : (!cir.bool, !s32i, !s32i) -> !s32i
// CIR-NEXT: %[[SHIFT_AMOUNT:.*]] = cir.const #cir.int<15> : !s32i
// CIR-NEXT: %[[SHIFT:.*]] = cir.shift(right, %[[SEL]] : !s32i, %[[SHIFT_AMOUNT]] : !s32i) -> !s32i
// CIR-NEXT: %[[CAST:.*]] = cir.cast integral %[[SHIFT]] : !s32i -> !u32i
// CIR-NEXT: cir.store %[[CAST]], %[[RET:.*]] : !u32i, !cir.ptr<!u32i>
// CIR-NEXT: %[[LOAD_RET:.*]] = cir.load %[[RET]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: cir.return %[[LOAD_RET]] : !u32i
// LLVM-LABEL: @accum_to_uint
// LLVM: %[[LOAD_ARG:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM: %[[NEG_CMP:.*]] = icmp slt i32 %[[LOAD_ARG]], 0
// LLVM: %[[ROUNDED:.*]] = add i32 %[[LOAD_ARG]], 32767
// LLVM: %[[SEL:.*]] = select i1 %[[NEG_CMP]], i32 %[[ROUNDED]], i32 %[[LOAD_ARG]]
// LLVM: %[[SHIFT:.*]] = ashr i32 %[[SEL]], 15
// LLVM: ret i32 %{{.*}}
unsigned int accum_to_uint(_Accum a) {
  return a;
}

// CIR-LABEL: cir.func{{.*}} @ufract_to_uint
// CIR: %[[ARG:.*]] = cir.alloca "f" align(2) init : !cir.ptr<!u16i>
// CIR: %[[LOAD_ARG:.*]] = cir.load align(2) %[[ARG]] : !cir.ptr<!u16i>, !u16i
// CIR-NEXT: %[[SHIFT_AMOUNT:.*]] = cir.const #cir.int<16> : !u16i
// CIR-NEXT: %[[SHIFT:.*]] = cir.shift(right, %[[LOAD_ARG]] : !u16i, %[[SHIFT_AMOUNT]] : !u16i) -> !u16i
// CIR-NEXT: %[[CAST:.*]] = cir.cast integral %[[SHIFT]] : !u16i -> !u32i
// CIR-NEXT: cir.store %[[CAST]], %[[RET:.*]] : !u32i, !cir.ptr<!u32i>
// CIR-NEXT: %[[LOAD_RET:.*]] = cir.load %[[RET]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: cir.return %[[LOAD_RET]] : !u32i
// LLVM-LABEL: @ufract_to_uint
// LLVM: %[[LOAD_ARG:.*]] = load i16, ptr %{{.*}}, align 2
// LLVM: %[[SHIFT:.*]] = lshr i16 %[[LOAD_ARG]], 16
// LLVM: %[[CAST:.*]] = zext i16 %[[SHIFT]] to i32
// LLVM: ret i32 %{{.*}}
unsigned int ufract_to_uint(unsigned _Fract f) {
  return f;
}

// CIR-LABEL: cir.func{{.*}} @uaccum_to_uint
// CIR: %[[ARG:.*]] = cir.alloca "a" align(4) init : !cir.ptr<!u32i>
// CIR: %[[LOAD_ARG:.*]] = cir.load align(4) %[[ARG]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: %[[SHIFT_AMOUNT:.*]] = cir.const #cir.int<16> : !u32i
// CIR-NEXT: %[[SHIFT:.*]] = cir.shift(right, %[[LOAD_ARG]] : !u32i, %[[SHIFT_AMOUNT]] : !u32i) -> !u32i
// CIR-NEXT: cir.store %[[SHIFT]], %[[RET:.*]] : !u32i, !cir.ptr<!u32i>
// CIR-NEXT: %[[LOAD_RET:.*]] = cir.load %[[RET]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: cir.return %[[LOAD_RET]] : !u32i
// LLVM-LABEL: @uaccum_to_uint
// LLVM: %[[LOAD_ARG:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM: %[[SHIFT:.*]] = lshr i32 %[[LOAD_ARG]], 16
// LLVM: ret i32 %{{.*}}
unsigned int uaccum_to_uint(unsigned _Accum a) {
  return a;
}

// CIR-LABEL: cir.func{{.*}} @sat_fract_to_uint
// CIR: %[[ARG:.*]] = cir.alloca "f" align(2) init : !cir.ptr<!s16i>
// CIR: %[[LOAD_ARG:.*]] = cir.load align(2) %[[ARG]] : !cir.ptr<!s16i>, !s16i
// CIR-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !s16i
// CIR-NEXT: %[[NEG_CMP:.*]] = cir.cmp lt %[[LOAD_ARG]], %[[ZERO]] : !s16i
// CIR-NEXT: %[[ROUND_BIAS:.*]] = cir.const #cir.int<32767> : !s16i
// CIR-NEXT: %[[ROUNDED:.*]] = cir.add %[[LOAD_ARG]], %[[ROUND_BIAS]] : !s16i
// CIR-NEXT: %[[SEL:.*]] = cir.select if %[[NEG_CMP]] then %[[ROUNDED]] else %[[LOAD_ARG]] : (!cir.bool, !s16i, !s16i) -> !s16i
// CIR-NEXT: %[[SHIFT_AMOUNT:.*]] = cir.const #cir.int<15> : !s16i
// CIR-NEXT: %[[SHIFT:.*]] = cir.shift(right, %[[SEL]] : !s16i, %[[SHIFT_AMOUNT]] : !s16i) -> !s16i
// CIR-NEXT: %[[CAST:.*]] = cir.cast integral %[[SHIFT]] : !s16i -> !u32i
// CIR-NEXT: cir.store %[[CAST]], %[[RET:.*]] : !u32i, !cir.ptr<!u32i>
// CIR-NEXT: %[[LOAD_RET:.*]] = cir.load %[[RET]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: cir.return %[[LOAD_RET]] : !u32i
// LLVM-LABEL: @sat_fract_to_uint
// LLVM: %[[LOAD_ARG:.*]] = load i16, ptr %{{.*}}, align 2
// LLVM: %[[NEG_CMP:.*]] = icmp slt i16 %[[LOAD_ARG]], 0
// LLVM: %[[ROUNDED:.*]] = add i16 %[[LOAD_ARG]], 32767
// LLVM: %[[SEL:.*]] = select i1 %[[NEG_CMP]], i16 %[[ROUNDED]], i16 %[[LOAD_ARG]]
// LLVM: %[[SHIFT:.*]] = ashr i16 %[[SEL]], 15
// LLVM: %[[CAST:.*]] = sext i16 %[[SHIFT]] to i32
// LLVM: ret i32 %{{.*}}
unsigned int sat_fract_to_uint(_Sat _Fract f) {
  return f;
}

// CIR-LABEL: cir.func{{.*}} @sat_accum_to_uint
// CIR: %[[ARG:.*]] = cir.alloca "a" align(4) init : !cir.ptr<!s32i>
// CIR: %[[LOAD_ARG:.*]] = cir.load align(4) %[[ARG]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR-NEXT: %[[NEG_CMP:.*]] = cir.cmp lt %[[LOAD_ARG]], %[[ZERO]] : !s32i
// CIR-NEXT: %[[ROUND_BIAS:.*]] = cir.const #cir.int<32767> : !s32i
// CIR-NEXT: %[[ROUNDED:.*]] = cir.add %[[LOAD_ARG]], %[[ROUND_BIAS]] : !s32i
// CIR-NEXT: %[[SEL:.*]] = cir.select if %[[NEG_CMP]] then %[[ROUNDED]] else %[[LOAD_ARG]] : (!cir.bool, !s32i, !s32i) -> !s32i
// CIR-NEXT: %[[SHIFT_AMOUNT:.*]] = cir.const #cir.int<15> : !s32i
// CIR-NEXT: %[[SHIFT:.*]] = cir.shift(right, %[[SEL]] : !s32i, %[[SHIFT_AMOUNT]] : !s32i) -> !s32i
// CIR-NEXT: %[[CAST:.*]] = cir.cast integral %[[SHIFT]] : !s32i -> !u32i
// CIR-NEXT: cir.store %[[CAST]], %[[RET:.*]] : !u32i, !cir.ptr<!u32i>
// CIR-NEXT: %[[LOAD_RET:.*]] = cir.load %[[RET]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: cir.return %[[LOAD_RET]] : !u32i
// LLVM-LABEL: @sat_accum_to_uint
// LLVM: %[[LOAD_ARG:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM: %[[NEG_CMP:.*]] = icmp slt i32 %[[LOAD_ARG]], 0
// LLVM: %[[ROUNDED:.*]] = add i32 %[[LOAD_ARG]], 32767
// LLVM: %[[SEL:.*]] = select i1 %[[NEG_CMP]], i32 %[[ROUNDED]], i32 %[[LOAD_ARG]]
// LLVM: %[[SHIFT:.*]] = ashr i32 %[[SEL]], 15
// LLVM: ret i32 %{{.*}}
unsigned int sat_accum_to_uint(_Sat _Accum a) {
  return a;
}

// CIR-LABEL: cir.func{{.*}} @fract_to_bool
// CIR: %[[ARG:.*]] = cir.alloca "f" align(2) init : !cir.ptr<!s16i>
// CIR: %[[LOAD_ARG:.*]] = cir.load align(2) %[[ARG]] : !cir.ptr<!s16i>, !s16i
// CIR-NEXT: %[[CAST:.*]] = cir.cast int_to_bool %[[LOAD_ARG]] : !s16i -> !cir.bool
// CIR-NEXT: cir.store %[[CAST]], %[[RET:.*]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR-NEXT: %[[LOAD_RET:.*]] = cir.load %[[RET]] : !cir.ptr<!cir.bool>, !cir.bool
// CIR-NEXT: cir.return %[[LOAD_RET]] : !cir.bool
// LLVM-LABEL: @fract_to_bool
// LLVM: %[[LOAD_ARG:.*]] = load i16, ptr %{{.*}}, align 2
// LLVM: %[[CAST:.*]] = icmp ne i16 %[[LOAD_ARG]], 0
// LLVM: ret i1 %{{.*}}
bool fract_to_bool(_Fract f) {
  return f;
}

// CIR-LABEL: cir.func{{.*}} @accum_to_bool
// CIR: %[[ARG:.*]] = cir.alloca "a" align(4) init : !cir.ptr<!s32i>
// CIR: %[[LOAD_ARG:.*]] = cir.load align(4) %[[ARG]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: %[[CAST:.*]] = cir.cast int_to_bool %[[LOAD_ARG]] : !s32i -> !cir.bool
// CIR-NEXT: cir.store %[[CAST]], %[[RET:.*]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR-NEXT: %[[LOAD_RET:.*]] = cir.load %[[RET]] : !cir.ptr<!cir.bool>, !cir.bool
// CIR-NEXT: cir.return %[[LOAD_RET]] : !cir.bool
// LLVM-LABEL: @accum_to_bool
// LLVM: %[[LOAD_ARG:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM: %[[CAST:.*]] = icmp ne i32 %[[LOAD_ARG]], 0
// LLVM: ret i1 %{{.*}}
bool accum_to_bool(_Accum a) {
  return a;
}

// CIR-LABEL: cir.func{{.*}} @ufract_to_bool
// CIR: %[[ARG:.*]] = cir.alloca "f" align(2) init : !cir.ptr<!u16i>
// CIR: %[[LOAD_ARG:.*]] = cir.load align(2) %[[ARG]] : !cir.ptr<!u16i>, !u16i
// CIR-NEXT: %[[CAST:.*]] = cir.cast int_to_bool %[[LOAD_ARG]] : !u16i -> !cir.bool
// CIR-NEXT: cir.store %[[CAST]], %[[RET:.*]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR-NEXT: %[[LOAD_RET:.*]] = cir.load %[[RET]] : !cir.ptr<!cir.bool>, !cir.bool
// CIR-NEXT: cir.return %[[LOAD_RET]] : !cir.bool
// LLVM-LABEL: @ufract_to_bool
// LLVM: %[[LOAD_ARG:.*]] = load i16, ptr %{{.*}}, align 2
// LLVM: %[[CAST:.*]] = icmp ne i16 %[[LOAD_ARG]], 0
// LLVM: ret i1 %{{.*}}
bool ufract_to_bool(unsigned _Fract f) {
  return f;
}

// CIR-LABEL: cir.func{{.*}} @uaccum_to_bool
// CIR: %[[ARG:.*]] = cir.alloca "a" align(4) init : !cir.ptr<!u32i>
// CIR: %[[LOAD_ARG:.*]] = cir.load align(4) %[[ARG]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: %[[CAST:.*]] = cir.cast int_to_bool %[[LOAD_ARG]] : !u32i -> !cir.bool
// CIR-NEXT: cir.store %[[CAST]], %[[RET:.*]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR-NEXT: %[[LOAD_RET:.*]] = cir.load %[[RET]] : !cir.ptr<!cir.bool>, !cir.bool
// CIR-NEXT: cir.return %[[LOAD_RET]] : !cir.bool
// LLVM-LABEL: @uaccum_to_bool
// LLVM: %[[LOAD_ARG:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM: %[[CAST:.*]] = icmp ne i32 %[[LOAD_ARG]], 0
// LLVM: ret i1 %{{.*}}
bool uaccum_to_bool(unsigned _Accum a) {
  return a;
}

// CIR-LABEL: cir.func{{.*}} @sat_fract_to_bool
// CIR: %[[ARG:.*]] = cir.alloca "f" align(2) init : !cir.ptr<!s16i>
// CIR: %[[LOAD_ARG:.*]] = cir.load align(2) %[[ARG]] : !cir.ptr<!s16i>, !s16i
// CIR-NEXT: %[[CAST:.*]] = cir.cast int_to_bool %[[LOAD_ARG]] : !s16i -> !cir.bool
// CIR-NEXT: cir.store %[[CAST]], %[[RET:.*]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR-NEXT: %[[LOAD_RET:.*]] = cir.load %[[RET]] : !cir.ptr<!cir.bool>, !cir.bool
// CIR-NEXT: cir.return %[[LOAD_RET]] : !cir.bool
// LLVM-LABEL: @sat_fract_to_bool
// LLVM: %[[LOAD_ARG:.*]] = load i16, ptr %{{.*}}, align 2
// LLVM: %[[CAST:.*]] = icmp ne i16 %[[LOAD_ARG]], 0
// LLVM: ret i1 %{{.*}}
bool sat_fract_to_bool(_Sat _Fract f) {
  return f;
}

// CIR-LABEL: cir.func{{.*}} @sat_accum_to_bool
// CIR: %[[ARG:.*]] = cir.alloca "a" align(4) init : !cir.ptr<!s32i>
// CIR: %[[LOAD_ARG:.*]] = cir.load align(4) %[[ARG]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: %[[CAST:.*]] = cir.cast int_to_bool %[[LOAD_ARG]] : !s32i -> !cir.bool
// CIR-NEXT: cir.store %[[CAST]], %[[RET:.*]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR-NEXT: %[[LOAD_RET:.*]] = cir.load %[[RET]] : !cir.ptr<!cir.bool>, !cir.bool
// CIR-NEXT: cir.return %[[LOAD_RET]] : !cir.bool
// LLVM-LABEL: @sat_accum_to_bool
// LLVM: %[[LOAD_ARG:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM: %[[CAST:.*]] = icmp ne i32 %[[LOAD_ARG]], 0
// LLVM: ret i1 %{{.*}}
bool sat_accum_to_bool(_Sat _Accum a) {
  return a;
}

// CIR-LABEL: cir.func{{.*}} @fract_to_float
// CIR: %[[ARG:.*]] = cir.alloca "f" align(2) init : !cir.ptr<!s16i>
// CIR: %[[LOAD_ARG:.*]] = cir.load align(2) %[[ARG]] : !cir.ptr<!s16i>, !s16i
// CIR-NEXT: %[[CAST:.*]] = cir.cast int_to_float %[[LOAD_ARG]] : !s16i -> !cir.float
// CIR-NEXT: %[[SCALE:.*]] = cir.const #cir.fp<3.05175781E-5> : !cir.float
// CIR-NEXT: %[[MUL:.*]] = cir.fmul %[[CAST]], %[[SCALE]] : !cir.float
// CIR-NEXT: cir.store %[[MUL]], %[[RET:.*]] : !cir.float, !cir.ptr<!cir.float>
// CIR-NEXT: %[[LOAD_RET:.*]] = cir.load %[[RET]] : !cir.ptr<!cir.float>, !cir.float
// CIR-NEXT: cir.return %[[LOAD_RET]] : !cir.float
// LLVM-LABEL: @fract_to_float
// LLVM: %[[LOAD_ARG:.*]] = load i16, ptr %{{.*}}, align 2
// LLVM: %[[CAST:.*]] = sitofp i16 %[[LOAD_ARG]] to float
// LLVM: %[[MULT:.*]] = fmul float %[[CAST]], f0x38000000
// LLVM: ret float %{{.*}}
float fract_to_float(_Fract f) {
  return f;
}

// CIR-LABEL: cir.func{{.*}} @accum_to_float
// CIR: %[[ARG:.*]] = cir.alloca "a" align(4) init : !cir.ptr<!s32i>
// CIR: %[[LOAD_ARG:.*]] = cir.load align(4) %[[ARG]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: %[[CAST:.*]] = cir.cast int_to_float %[[LOAD_ARG]] : !s32i -> !cir.float
// CIR-NEXT: %[[SCALE:.*]] = cir.const #cir.fp<3.05175781E-5> : !cir.float
// CIR-NEXT: %[[MUL:.*]] = cir.fmul %[[CAST]], %[[SCALE]] : !cir.float
// CIR-NEXT: cir.store %[[MUL]], %[[RET:.*]] : !cir.float, !cir.ptr<!cir.float>
// CIR-NEXT: %[[LOAD_RET:.*]] = cir.load %[[RET]] : !cir.ptr<!cir.float>, !cir.float
// CIR-NEXT: cir.return %[[LOAD_RET]] : !cir.float
// LLVM-LABEL: @accum_to_float
// LLVM: %[[LOAD_ARG:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM: %[[CAST:.*]] = sitofp i32 %[[LOAD_ARG]] to float
// LLVM: %[[MULT:.*]] = fmul float %[[CAST]], f0x38000000
// LLVM: ret float %{{.*}}
float accum_to_float(_Accum a) {
  return a;
}

// CIR-LABEL: cir.func{{.*}} @ufract_to_float
// CIR: %[[ARG:.*]] = cir.alloca "f" align(2) init : !cir.ptr<!u16i>
// CIR: %[[LOAD_ARG:.*]] = cir.load align(2) %[[ARG]] : !cir.ptr<!u16i>, !u16i
// CIR-NEXT: %[[CAST:.*]] = cir.cast int_to_float %[[LOAD_ARG]] : !u16i -> !cir.float
// CIR-NEXT: %[[SCALE:.*]] = cir.const #cir.fp<1.52587891E-5> : !cir.float
// CIR-NEXT: %[[MUL:.*]] = cir.fmul %[[CAST]], %[[SCALE]] : !cir.float
// CIR-NEXT: cir.store %[[MUL]], %[[RET:.*]] : !cir.float, !cir.ptr<!cir.float>
// CIR-NEXT: %[[LOAD_RET:.*]] = cir.load %[[RET]] : !cir.ptr<!cir.float>, !cir.float
// CIR-NEXT: cir.return %[[LOAD_RET]] : !cir.float
// LLVM-LABEL: @ufract_to_float
// LLVM: %[[LOAD_ARG:.*]] = load i16, ptr %{{.*}}, align 2
// LLVM: %[[CAST:.*]] = uitofp i16 %[[LOAD_ARG]] to float
// LLVM: %[[MULT:.*]] = fmul float %[[CAST]], f0x37800000
// LLVM: ret float %{{.*}}
float ufract_to_float(unsigned _Fract f) {
  return f;
}

// CIR-LABEL: cir.func{{.*}} @uaccum_to_float
// CIR: %[[ARG:.*]] = cir.alloca "a" align(4) init : !cir.ptr<!u32i>
// CIR: %[[LOAD_ARG:.*]] = cir.load align(4) %[[ARG]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: %[[CAST:.*]] = cir.cast int_to_float %[[LOAD_ARG]] : !u32i -> !cir.float
// CIR-NEXT: %[[SCALE:.*]] = cir.const #cir.fp<1.52587891E-5> : !cir.float
// CIR-NEXT: %[[MUL:.*]] = cir.fmul %[[CAST]], %[[SCALE]] : !cir.float
// CIR-NEXT: cir.store %[[MUL]], %[[RET:.*]] : !cir.float, !cir.ptr<!cir.float>
// CIR-NEXT: %[[LOAD_RET:.*]] = cir.load %[[RET]] : !cir.ptr<!cir.float>, !cir.float
// CIR-NEXT: cir.return %[[LOAD_RET]] : !cir.float
// LLVM-LABEL: @uaccum_to_float
// LLVM: %[[LOAD_ARG:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM: %[[CAST:.*]] = uitofp i32 %[[LOAD_ARG]] to float
// LLVM: %[[MULT:.*]] = fmul float %[[CAST]], f0x37800000
// LLVM: ret float %{{.*}}
float uaccum_to_float(unsigned _Accum a) {
  return a;
}

// CIR-LABEL: cir.func{{.*}} @sat_fract_to_float
// CIR: %[[ARG:.*]] = cir.alloca "f" align(2) init : !cir.ptr<!s16i>
// CIR: %[[LOAD_ARG:.*]] = cir.load align(2) %[[ARG]] : !cir.ptr<!s16i>, !s16i
// CIR-NEXT: %[[CAST:.*]] = cir.cast int_to_float %[[LOAD_ARG]] : !s16i -> !cir.float
// CIR-NEXT: %[[SCALE:.*]] = cir.const #cir.fp<3.05175781E-5> : !cir.float
// CIR-NEXT: %[[MUL:.*]] = cir.fmul %[[CAST]], %[[SCALE]] : !cir.float
// CIR-NEXT: cir.store %[[MUL]], %[[RET:.*]] : !cir.float, !cir.ptr<!cir.float>
// CIR-NEXT: %[[LOAD_RET:.*]] = cir.load %[[RET]] : !cir.ptr<!cir.float>, !cir.float
// CIR-NEXT: cir.return %[[LOAD_RET]] : !cir.float
// LLVM-LABEL: @sat_fract_to_float
// LLVM: %[[LOAD_ARG:.*]] = load i16, ptr %{{.*}}, align 2
// LLVM: %[[CAST:.*]] = sitofp i16 %[[LOAD_ARG]] to float
// LLVM: %[[MULT:.*]] = fmul float %[[CAST]], f0x38000000
// LLVM: ret float %{{.*}}
float sat_fract_to_float(_Sat _Fract f) {
  return f;
}

// CIR-LABEL: cir.func{{.*}} @sat_accum_to_float
// CIR: %[[ARG:.*]] = cir.alloca "a" align(4) init : !cir.ptr<!s32i>
// CIR: %[[LOAD_ARG:.*]] = cir.load align(4) %[[ARG]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: %[[CAST:.*]] = cir.cast int_to_float %[[LOAD_ARG]] : !s32i -> !cir.float
// CIR-NEXT: %[[SCALE:.*]] = cir.const #cir.fp<3.05175781E-5> : !cir.float
// CIR-NEXT: %[[MUL:.*]] = cir.fmul %[[CAST]], %[[SCALE]] : !cir.float
// CIR-NEXT: cir.store %[[MUL]], %[[RET:.*]] : !cir.float, !cir.ptr<!cir.float>
// CIR-NEXT: %[[LOAD_RET:.*]] = cir.load %[[RET]] : !cir.ptr<!cir.float>, !cir.float
// CIR-NEXT: cir.return %[[LOAD_RET]] : !cir.float
// LLVM-LABEL: @sat_accum_to_float
// LLVM: %[[LOAD_ARG:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM: %[[CAST:.*]] = sitofp i32 %[[LOAD_ARG]] to float
// LLVM: %[[MULT:.*]] = fmul float %[[CAST]], f0x38000000
// LLVM: ret float %{{.*}}
float sat_accum_to_float(_Sat _Accum a) {
  return a;
}

// CIR-LABEL: cir.func{{.*}} @fract_to_double
// CIR: %[[ARG:.*]] = cir.alloca "f" align(2) init : !cir.ptr<!s16i>
// CIR: %[[LOAD_ARG:.*]] = cir.load align(2) %[[ARG]] : !cir.ptr<!s16i>, !s16i
// CIR-NEXT: %[[CAST:.*]] = cir.cast int_to_float %[[LOAD_ARG]] : !s16i -> !cir.double
// CIR-NEXT: %[[SCALE:.*]] = cir.const #cir.fp<3.0517578125E-5> : !cir.double
// CIR-NEXT: %[[MUL:.*]] = cir.fmul %[[CAST]], %[[SCALE]] : !cir.double
// CIR-NEXT: cir.store %[[MUL]], %[[RET:.*]] : !cir.double, !cir.ptr<!cir.double>
// CIR-NEXT: %[[LOAD_RET:.*]] = cir.load %[[RET]] : !cir.ptr<!cir.double>, !cir.double
// CIR-NEXT: cir.return %[[LOAD_RET]] : !cir.double
// LLVM-LABEL: @fract_to_double
// LLVM: %[[LOAD_ARG:.*]] = load i16, ptr %{{.*}}, align 2
// LLVM: %[[CAST:.*]] = sitofp i16 %[[LOAD_ARG]] to double
// LLVM: %[[MULT:.*]] = fmul double %[[CAST]], f0x3F00000000000000
// LLVM: ret double %{{.*}}
double fract_to_double(_Fract f) {
  return f;
}

// CIR-LABEL: cir.func{{.*}} @accum_to_double
// CIR: %[[ARG:.*]] = cir.alloca "a" align(4) init : !cir.ptr<!s32i>
// CIR: %[[LOAD_ARG:.*]] = cir.load align(4) %[[ARG]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: %[[CAST:.*]] = cir.cast int_to_float %[[LOAD_ARG]] : !s32i -> !cir.double
// CIR-NEXT: %[[SCALE:.*]] = cir.const #cir.fp<3.0517578125E-5> : !cir.double
// CIR-NEXT: %[[MUL:.*]] = cir.fmul %[[CAST]], %[[SCALE]] : !cir.double
// CIR-NEXT: cir.store %[[MUL]], %[[RET:.*]] : !cir.double, !cir.ptr<!cir.double>
// CIR-NEXT: %[[LOAD_RET:.*]] = cir.load %[[RET]] : !cir.ptr<!cir.double>, !cir.double
// CIR-NEXT: cir.return %[[LOAD_RET]] : !cir.double
// LLVM-LABEL: @accum_to_double
// LLVM: %[[LOAD_ARG:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM: %[[CAST:.*]] = sitofp i32 %[[LOAD_ARG]] to double
// LLVM: %[[MULT:.*]] = fmul double %[[CAST]], f0x3F00000000000000
// LLVM: ret double %{{.*}}
double accum_to_double(_Accum a) {
  return a;
}

// CIR-LABEL: cir.func{{.*}} @ufract_to_double
// CIR: %[[ARG:.*]] = cir.alloca "f" align(2) init : !cir.ptr<!u16i>
// CIR: %[[LOAD_ARG:.*]] = cir.load align(2) %[[ARG]] : !cir.ptr<!u16i>, !u16i
// CIR-NEXT: %[[CAST:.*]] = cir.cast int_to_float %[[LOAD_ARG]] : !u16i -> !cir.double
// CIR-NEXT: %[[SCALE:.*]] = cir.const #cir.fp<1.52587890625E-5> : !cir.double
// CIR-NEXT: %[[MUL:.*]] = cir.fmul %[[CAST]], %[[SCALE]] : !cir.double
// CIR-NEXT: cir.store %[[MUL]], %[[RET:.*]] : !cir.double, !cir.ptr<!cir.double>
// CIR-NEXT: %[[LOAD_RET:.*]] = cir.load %[[RET]] : !cir.ptr<!cir.double>, !cir.double
// CIR-NEXT: cir.return %[[LOAD_RET]] : !cir.double
// LLVM-LABEL: @ufract_to_double
// LLVM: %[[LOAD_ARG:.*]] = load i16, ptr %{{.*}}, align 2
// LLVM: %[[CAST:.*]] = uitofp i16 %[[LOAD_ARG]] to double
// LLVM: %[[MULT:.*]] = fmul double %[[CAST]], f0x3EF0000000000000
// LLVM: ret double %{{.*}}
double ufract_to_double(unsigned _Fract f) {
  return f;
}

// CIR-LABEL: cir.func{{.*}} @uaccum_to_double
// CIR: %[[ARG:.*]] = cir.alloca "a" align(4) init : !cir.ptr<!u32i>
// CIR: %[[LOAD_ARG:.*]] = cir.load align(4) %[[ARG]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: %[[CAST:.*]] = cir.cast int_to_float %[[LOAD_ARG]] : !u32i -> !cir.double
// CIR-NEXT: %[[SCALE:.*]] = cir.const #cir.fp<1.52587890625E-5> : !cir.double
// CIR-NEXT: %[[MUL:.*]] = cir.fmul %[[CAST]], %[[SCALE]] : !cir.double
// CIR-NEXT: cir.store %[[MUL]], %[[RET:.*]] : !cir.double, !cir.ptr<!cir.double>
// CIR-NEXT: %[[LOAD_RET:.*]] = cir.load %[[RET]] : !cir.ptr<!cir.double>, !cir.double
// CIR-NEXT: cir.return %[[LOAD_RET]] : !cir.double
// LLVM-LABEL: @uaccum_to_double
// LLVM: %[[LOAD_ARG:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM: %[[CAST:.*]] = uitofp i32 %[[LOAD_ARG]] to double
// LLVM: %[[MULT:.*]] = fmul double %[[CAST]], f0x3EF0000000000000
// LLVM: ret double %{{.*}}
double uaccum_to_double(unsigned _Accum a) {
  return a;
}

// CIR-LABEL: cir.func{{.*}} @sat_fract_to_double
// CIR: %[[ARG:.*]] = cir.alloca "f" align(2) init : !cir.ptr<!s16i>
// CIR: %[[LOAD_ARG:.*]] = cir.load align(2) %[[ARG]] : !cir.ptr<!s16i>, !s16i
// CIR-NEXT: %[[CAST:.*]] = cir.cast int_to_float %[[LOAD_ARG]] : !s16i -> !cir.double
// CIR-NEXT: %[[SCALE:.*]] = cir.const #cir.fp<3.0517578125E-5> : !cir.double
// CIR-NEXT: %[[MUL:.*]] = cir.fmul %[[CAST]], %[[SCALE]] : !cir.double
// CIR-NEXT: cir.store %[[MUL]], %[[RET:.*]] : !cir.double, !cir.ptr<!cir.double>
// CIR-NEXT: %[[LOAD_RET:.*]] = cir.load %[[RET]] : !cir.ptr<!cir.double>, !cir.double
// CIR-NEXT: cir.return %[[LOAD_RET]] : !cir.double
// LLVM-LABEL: @sat_fract_to_double
// LLVM: %[[LOAD_ARG:.*]] = load i16, ptr %{{.*}}, align 2
// LLVM: %[[CAST:.*]] = sitofp i16 %[[LOAD_ARG]] to double
// LLVM: %[[MULT:.*]] = fmul double %[[CAST]], f0x3F00000000000000
// LLVM: ret double %{{.*}}
double sat_fract_to_double(_Sat _Fract f) {
  return f;
}

// CIR-LABEL: cir.func{{.*}} @sat_accum_to_double
// CIR: %[[ARG:.*]] = cir.alloca "a" align(4) init : !cir.ptr<!s32i>
// CIR: %[[LOAD_ARG:.*]] = cir.load align(4) %[[ARG]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: %[[CAST:.*]] = cir.cast int_to_float %[[LOAD_ARG]] : !s32i -> !cir.double
// CIR-NEXT: %[[SCALE:.*]] = cir.const #cir.fp<3.0517578125E-5> : !cir.double
// CIR-NEXT: %[[MUL:.*]] = cir.fmul %[[CAST]], %[[SCALE]] : !cir.double
// CIR-NEXT: cir.store %[[MUL]], %[[RET:.*]] : !cir.double, !cir.ptr<!cir.double>
// CIR-NEXT: %[[LOAD_RET:.*]] = cir.load %[[RET]] : !cir.ptr<!cir.double>, !cir.double
// CIR-NEXT: cir.return %[[LOAD_RET]] : !cir.double
// LLVM-LABEL: @sat_accum_to_double
// LLVM: %[[LOAD_ARG:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM: %[[CAST:.*]] = sitofp i32 %[[LOAD_ARG]] to double
// LLVM: %[[MULT:.*]] = fmul double %[[CAST]], f0x3F00000000000000
// LLVM: ret double %{{.*}}
double sat_accum_to_double(_Sat _Accum a) {
  return a;
}

// CIR-LABEL: cir.func{{.*}} @fract_to_fract
// CIR: %[[ARG:.*]] = cir.alloca "f" align(2) init : !cir.ptr<!s16i>
// CIR: %[[LOAD_ARG:.*]] = cir.load align(2) %[[ARG]] : !cir.ptr<!s16i>, !s16i
// CIR-NEXT: cir.store %[[LOAD_ARG]], %[[RET:.*]] : !s16i, !cir.ptr<!s16i>
// CIR-NEXT: %[[LOAD_RET:.*]] = cir.load %[[RET]] : !cir.ptr<!s16i>, !s16i
// CIR-NEXT: cir.return %[[LOAD_RET]] : !s16i
// LLVM-LABEL: @fract_to_fract
// LLVM: %[[LOAD_ARG:.*]] = load i16, ptr %{{.*}}, align 2
// LLVM: ret i16 %{{.*}}
_Fract fract_to_fract(_Fract f) {
  return f;
}

// CIR-LABEL: cir.func{{.*}} @fract_to_sat_fract
// CIR: %[[ARG:.*]] = cir.alloca "f" align(2) init : !cir.ptr<!s16i>
// CIR: %[[LOAD_ARG:.*]] = cir.load align(2) %[[ARG]] : !cir.ptr<!s16i>, !s16i
// CIR-NEXT: cir.store %[[LOAD_ARG]], %[[RET:.*]] : !s16i, !cir.ptr<!s16i>
// CIR-NEXT: %[[LOAD_RET:.*]] = cir.load %[[RET]] : !cir.ptr<!s16i>, !s16i
// CIR-NEXT: cir.return %[[LOAD_RET]] : !s16i
// LLVM-LABEL: @fract_to_sat_fract
// LLVM: %[[LOAD_ARG:.*]] = load i16, ptr %{{.*}}, align 2
// LLVM: ret i16 %{{.*}}
_Sat _Fract fract_to_sat_fract(_Fract f) {
  return f;
}

// CIR-LABEL: cir.func{{.*}} @sat_fract_to_fract
// CIR: %[[ARG:.*]] = cir.alloca "f" align(2) init : !cir.ptr<!s16i>
// CIR: %[[LOAD_ARG:.*]] = cir.load align(2) %[[ARG]] : !cir.ptr<!s16i>, !s16i
// CIR-NEXT: cir.store %[[LOAD_ARG]], %[[RET:.*]] : !s16i, !cir.ptr<!s16i>
// CIR-NEXT: %[[LOAD_RET:.*]] = cir.load %[[RET]] : !cir.ptr<!s16i>, !s16i
// CIR-NEXT: cir.return %[[LOAD_RET]] : !s16i
// LLVM-LABEL: @sat_fract_to_fract
// LLVM: %[[LOAD_ARG:.*]] = load i16, ptr %{{.*}}, align 2
// LLVM: ret i16 %{{.*}}
_Fract sat_fract_to_fract(_Sat _Fract f) {
  return f;
}

// CIR-LABEL: cir.func{{.*}} @accum_to_accum
// CIR: %[[ARG:.*]] = cir.alloca "f" align(4) init : !cir.ptr<!s32i>
// CIR: %[[LOAD_ARG:.*]] = cir.load align(4) %[[ARG]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: cir.store %[[LOAD_ARG]], %[[RET:.*]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT: %[[LOAD_RET:.*]] = cir.load %[[RET]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: cir.return %[[LOAD_RET]] : !s32i
// LLVM-LABEL: @accum_to_accum
// LLVM: %[[LOAD_ARG:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM: ret i32 %{{.*}}
_Accum accum_to_accum(_Accum f) {
  return f;
}

// CIR-LABEL: cir.func{{.*}} @accum_to_sat_accum
// CIR: %[[ARG:.*]] = cir.alloca "f" align(4) init : !cir.ptr<!s32i>
// CIR: %[[LOAD_ARG:.*]] = cir.load align(4) %[[ARG]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: cir.store %[[LOAD_ARG]], %[[RET:.*]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT: %[[LOAD_RET:.*]] = cir.load %[[RET]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: cir.return %[[LOAD_RET]] : !s32i
// LLVM-LABEL: @accum_to_sat_accum
// LLVM: %[[LOAD_ARG:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM: ret i32 %{{.*}}
_Sat _Accum accum_to_sat_accum(_Accum f) {
  return f;
}

// CIR-LABEL: cir.func{{.*}} @sat_accum_to_accum
// CIR: %[[ARG:.*]] = cir.alloca "f" align(4) init : !cir.ptr<!s32i>
// CIR: %[[LOAD_ARG:.*]] = cir.load align(4) %[[ARG]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: cir.store %[[LOAD_ARG]], %[[RET:.*]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT: %[[LOAD_RET:.*]] = cir.load %[[RET]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: cir.return %[[LOAD_RET]] : !s32i
// LLVM-LABEL: @sat_accum_to_accum
// LLVM: %[[LOAD_ARG:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM: ret i32 %{{.*}}
_Accum sat_accum_to_accum(_Sat _Accum f) {
  return f;
}

// CIR-LABEL: cir.func{{.*}} @fract_to_acccum
// CIR: %[[ARG:.*]] = cir.alloca "f" align(2) init : !cir.ptr<!s16i>
// CIR: %[[LOAD_ARG:.*]] = cir.load align(2) %[[ARG]] : !cir.ptr<!s16i>, !s16i
// CIR-NEXT: %[[CAST:.*]] = cir.cast integral %[[LOAD_ARG]] : !s16i -> !s32i
// CIR-NEXT: cir.store %[[CAST]], %[[RET:.*]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT: %[[LOAD_RET:.*]] = cir.load %[[RET]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: cir.return %[[LOAD_RET]] : !s32i
// LLVM-LABEL: @fract_to_acccum
// LLVM: %[[LOAD_ARG:.*]] = load i16, ptr %{{.*}}, align 2
// LLVM: %[[CAST:.*]] = sext i16 %[[LOAD_ARG]] to i32
// LLVM: ret i32 %{{.*}}
_Accum fract_to_acccum(_Fract f) {
  return f;
}

// CIR-LABEL: cir.func{{.*}} @accum_to_fract
// CIR: %[[ARG:.*]] = cir.alloca "f" align(4) init : !cir.ptr<!s32i>
// CIR: %[[LOAD_ARG:.*]] = cir.load align(4) %[[ARG]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: %[[TRUNC:.*]] = cir.cast integral %[[LOAD_ARG]] : !s32i -> !s16i
// CIR-NEXT: cir.store %[[TRUNC]], %[[RET:.*]] : !s16i, !cir.ptr<!s16i>
// CIR-NEXT: %[[LOAD_RET:.*]] = cir.load %[[RET]] : !cir.ptr<!s16i>, !s16i
// CIR-NEXT: cir.return %[[LOAD_RET]] : !s16i
// LLVM-LABEL: @accum_to_fract
// LLVM: %[[LOAD_ARG:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM: %[[CAST:.*]] = trunc i32 %[[LOAD_ARG]] to i16
// LLVM: ret i16 %{{.*}}
_Fract accum_to_fract(_Accum f) {
  return f;
}

// CIR-LABEL: cir.func{{.*}} @fract_to_sat_acccum
// CIR: %[[ARG:.*]] = cir.alloca "f" align(2) init : !cir.ptr<!s16i>
// CIR: %[[LOAD_ARG:.*]] = cir.load align(2) %[[ARG]] : !cir.ptr<!s16i>, !s16i
// CIR-NEXT: %[[CAST:.*]] = cir.cast integral %[[LOAD_ARG]] : !s16i -> !s32i
// CIR-NEXT: cir.store %[[CAST]], %[[RET:.*]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT: %[[LOAD_RET:.*]] = cir.load %[[RET]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: cir.return %[[LOAD_RET]] : !s32i
// LLVM-LABEL: @fract_to_sat_acccum
// LLVM: %[[LOAD_ARG:.*]] = load i16, ptr %{{.*}}, align 2
// LLVM: %[[CAST:.*]] = sext i16 %[[LOAD_ARG]] to i32
// LLVM: ret i32 %{{.*}}
_Sat _Accum fract_to_sat_acccum(_Fract f) {
  return f;
}

// CIR-LABEL: cir.func{{.*}} @accum_to_sat_fract
// CIR: %[[ARG:.*]] = cir.alloca "f" align(4) init : !cir.ptr<!s32i>
// CIR: %[[LOAD_ARG:.*]] = cir.load align(4) %[[ARG]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: %[[MAX:.*]] = cir.const #cir.int<32767> : !s32i
// CIR-NEXT: %[[GT_CMP:.*]] = cir.cmp gt %[[LOAD_ARG]], %[[MAX]] : !s32i
// CIR-NEXT: %[[MAX_SEL:.*]] = cir.select if %[[GT_CMP]] then %[[MAX]] else %[[LOAD_ARG]] : (!cir.bool, !s32i, !s32i) -> !s32i
// CIR-NEXT: %[[MIN:.*]] = cir.const #cir.int<-32768> : !s32i
// CIR-NEXT: %[[LT_CMP:.*]] = cir.cmp lt %[[MAX_SEL]], %[[MIN]] : !s32i
// CIR-NEXT: %[[BOUNDED:.*]] = cir.select if %[[LT_CMP]] then %[[MIN]] else %[[MAX_SEL]] : (!cir.bool, !s32i, !s32i) -> !s32i
// CIR-NEXT: %[[CAST_BACK:.*]] = cir.cast integral %[[BOUNDED]] : !s32i -> !s16i
// CIR-NEXT: cir.store %[[CAST_BACK]], %[[RET:.*]] : !s16i, !cir.ptr<!s16i>
// CIR-NEXT: %[[LOAD_RET:.*]] = cir.load %[[RET]] : !cir.ptr<!s16i>, !s16i
// CIR-NEXT: cir.return %[[LOAD_RET]] : !s16i
// LLVM-LABEL: @accum_to_sat_fract
// LLVM: %[[LOAD_ARG:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM: %[[GT_CMP:.*]] = icmp sgt i32 %[[LOAD_ARG]], 32767
// LLVM: %[[MAX_SEL:.*]] = select i1 %[[GT_CMP]], i32 32767, i32 %[[LOAD_ARG]]
// LLVM: %[[LT_CMP:.*]] = icmp slt i32 %[[MAX_SEL]], -32768
// LLVM: %[[BOUNDED:.*]] = select i1 %[[LT_CMP]], i32 -32768, i32 %[[MAX_SEL]]
// LLVM: %[[CAST:.*]] = trunc i32 %[[BOUNDED]] to i16
// LLVM: ret i16 %{{.*}}
_Sat _Fract accum_to_sat_fract(_Accum f) {
  return f;
}

// CIR-LABEL: cir.func{{.*}} @sat_fract_to_acccum
// CIR: %[[ARG:.*]] = cir.alloca "f" align(2) init : !cir.ptr<!s16i>
// CIR: %[[LOAD_ARG:.*]] = cir.load align(2) %[[ARG]] : !cir.ptr<!s16i>, !s16i
// CIR-NEXT: %[[CAST:.*]] = cir.cast integral %[[LOAD_ARG]] : !s16i -> !s32i
// CIR-NEXT: cir.store %[[CAST]], %[[RET:.*]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT: %[[LOAD_RET:.*]] = cir.load %[[RET]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: cir.return %[[LOAD_RET]] : !s32i
// LLVM-LABEL: @sat_fract_to_acccum
// LLVM: %[[LOAD_ARG:.*]] = load i16, ptr %{{.*}}, align 2
// LLVM: %[[CAST:.*]] = sext i16 %[[LOAD_ARG]] to i32
// LLVM: ret i32 %{{.*}}
_Accum sat_fract_to_acccum(_Sat _Fract f) {
  return f;
}

// CIR-LABEL: cir.func{{.*}} @sat_accum_to_fract
// CIR: %[[ARG:.*]] = cir.alloca "f" align(4) init : !cir.ptr<!s32i>
// CIR: %[[LOAD_ARG:.*]] = cir.load align(4) %[[ARG]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: %[[TRUNC:.*]] = cir.cast integral %[[LOAD_ARG]] : !s32i -> !s16i
// CIR-NEXT: cir.store %[[TRUNC]], %[[RET:.*]] : !s16i, !cir.ptr<!s16i>
// CIR-NEXT: %[[LOAD_RET:.*]] = cir.load %[[RET]] : !cir.ptr<!s16i>, !s16i
// CIR-NEXT: cir.return %[[LOAD_RET]] : !s16i
// LLVM-LABEL: @sat_accum_to_fract
// LLVM: %[[LOAD_ARG:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM: %[[CAST:.*]] = trunc i32 %[[LOAD_ARG]] to i16
// LLVM: ret i16 %{{.*}}
_Fract sat_accum_to_fract(_Sat _Accum f) {
  return f;
}
}
