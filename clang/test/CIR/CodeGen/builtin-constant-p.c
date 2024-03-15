// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-cir %s -o - | FileCheck %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o - | FileCheck %s -check-prefix=LLVM

int a = 0;
int foo() {
  return __builtin_constant_p(a);
}

// CIR:  cir.func no_proto @foo() -> !s32i extra(#fn_attr)
// CIR:    [[TMP0:%.*]] = cir.alloca !s32i, cir.ptr <!s32i>, ["__retval"] {alignment = 4 : i64}
// CIR:    [[TMP1:%.*]] = cir.get_global @a : cir.ptr <!s32i>
// CIR:    [[TMP2:%.*]] = cir.load [[TMP1]] : cir.ptr <!s32i>, !s32i
// CIR:    [[TMP3:%.*]] = cir.is_constant([[TMP2]] : !s32i) : !cir.bool
// CIR:    [[TMP4:%.*]] = cir.cast(bool_to_int, [[TMP3]] : !cir.bool), !s32i
// CIR:    cir.store [[TMP4]], [[TMP0]] : !s32i, cir.ptr <!s32i>
// CIR:    [[TMP5:%.*]] = cir.load [[TMP0]] : cir.ptr <!s32i>, !s32i
// CIR:    cir.return [[TMP5]] : !s32i

// LLVM:define i32 @foo()
// LLVM:  [[TMP1:%.*]] = alloca i32, i64 1
// LLVM:  [[TMP2:%.*]] = load i32, ptr @a
// LLVM:  [[TMP3:%.*]] = call i1 @llvm.is.constant.i32(i32 [[TMP2]])
// LLVM:  [[TMP4:%.*]] = zext i1 [[TMP3]] to i8
// LLVM:  [[TMP5:%.*]] = zext i8 [[TMP4]] to i32
// LLVM:  store i32 [[TMP5]], ptr [[TMP1]]
// LLVM:  [[TMP6:%.*]] = load i32, ptr [[TMP1]]
// LLVM:  ret i32 [[TMP6]]

