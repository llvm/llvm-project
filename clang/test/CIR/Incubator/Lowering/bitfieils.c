// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

typedef struct {
  int a : 4;
} B;

// LLVM: define dso_local void @set_signed
// LLVM:   [[TMP0:%.*]] = load ptr
// LLVM:   [[TMP1:%.*]] = getelementptr %struct.B, ptr [[TMP0]], i32 0, i32 0
// LLVM:   [[TMP2:%.*]] = load i8, ptr [[TMP1]]
// LLVM:   [[TMP3:%.*]] = and i8 [[TMP2]], -16
// LLVM:   [[TMP4:%.*]] = or i8 [[TMP3]], 14
// LLVM:   store i8 [[TMP4]], ptr [[TMP1]]
void set_signed(B* b) {
  b->a = -2;
}

// LLVM: define dso_local i32 @get_signed
// LLVM:   [[TMP0:%.*]] = alloca i32
// LLVM:   [[TMP1:%.*]] = load ptr
// LLVM:   [[TMP2:%.*]] = getelementptr %struct.B, ptr [[TMP1]], i32 0, i32 0
// LLVM:   [[TMP3:%.*]] = load i8, ptr [[TMP2]]
// LLVM:   [[TMP4:%.*]] = shl i8 [[TMP3]], 4
// LLVM:   [[TMP5:%.*]] = ashr i8 [[TMP4]], 4
// LLVM:   [[TMP6:%.*]] = sext i8 [[TMP5]] to i32
// LLVM:   store i32 [[TMP6]], ptr [[TMP0]]
// LLVM:   [[TMP7:%.*]] = load i32, ptr [[TMP0]]
// LLVM:   ret i32 [[TMP7]]
int get_signed(B* b) {
  return b->a;
}
