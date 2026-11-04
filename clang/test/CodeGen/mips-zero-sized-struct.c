// RUN: %clang_cc1 -triple mips-unknown-linux-gnu -emit-llvm -o - %s | FileCheck -check-prefix=O32 %s
// RUN: %clang_cc1 -triple mipsel-unknown-linux-gnu -emit-llvm -o - %s | FileCheck -check-prefix=O32 %s
// RUN: %clang_cc1 -triple mipsisa32r6-unknown-linux-gnu -emit-llvm -o - %s | FileCheck -check-prefix=O32 %s
// RUN: %clang_cc1 -triple mipsisa32r6el-unknown-linux-gnu -emit-llvm -o - %s | FileCheck -check-prefix=O32 %s
// RUN: %clang_cc1 -triple mips64-unknown-linux-gnu -emit-llvm -o - %s  -target-abi n32 | FileCheck -check-prefix=N32 %s
// RUN: %clang_cc1 -triple mips64el-unknown-linux-gnu -emit-llvm -o - %s  -target-abi n32 | FileCheck -check-prefix=N32 %s
// RUN: %clang_cc1 -triple mipsisa64r6-unknown-linux-gnu -emit-llvm -o - %s  -target-abi n32 | FileCheck -check-prefix=N32 %s
// RUN: %clang_cc1 -triple mipsisa64r6el-unknown-linux-gnu -emit-llvm -o - %s  -target-abi n32 | FileCheck -check-prefix=N32 %s
// RUN: %clang_cc1 -triple mips64-unknown-linux-gnuabin32 -emit-llvm -o - %s  | FileCheck -check-prefix=N32 %s
// RUN: %clang_cc1 -triple mips64el-unknown-linux-gnuabin32 -emit-llvm -o - %s  | FileCheck -check-prefix=N32 %s
// RUN: %clang_cc1 -triple mipsisa64r6-unknown-linux-gnuabin32 -emit-llvm -o - %s  | FileCheck -check-prefix=N32 %s
// RUN: %clang_cc1 -triple mipsisa64r6el-unknown-linux-gnuabin32 -emit-llvm -o - %s  | FileCheck -check-prefix=N32 %s
// RUN: %clang_cc1 -triple mips64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck -check-prefix=N64 %s
// RUN: %clang_cc1 -triple mips64el-unknown-linux-gnu -emit-llvm -o - %s | FileCheck -check-prefix=N64 %s
// RUN: %clang_cc1 -triple mipsisa64r6-unknown-linux-gnu -emit-llvm -o - %s | FileCheck -check-prefix=N64 %s
// RUN: %clang_cc1 -triple mipsisa64r6el-unknown-linux-gnu -emit-llvm -o - %s | FileCheck -check-prefix=N64 %s
// RUN: %clang_cc1 -triple mips64-unknown-linux-gnuabi64 -emit-llvm -o - %s | FileCheck -check-prefix=N64 %s
// RUN: %clang_cc1 -triple mips64el-unknown-linux-gnuabi64 -emit-llvm -o - %s | FileCheck -check-prefix=N64 %s
// RUN: %clang_cc1 -triple mipsisa64r6-unknown-linux-gnuabi64 -emit-llvm -o - %s | FileCheck -check-prefix=N64 %s
// RUN: %clang_cc1 -triple mipsisa64r6el-unknown-linux-gnuabi64 -emit-llvm -o - %s | FileCheck -check-prefix=N64 %s

// O32: define{{.*}} void @fn28(ptr dead_on_unwind noalias writable sret(%struct.T2) align 1 %agg.result, i8 noundef signext %arg0)
// N32: define{{.*}} void @fn28(i8 noundef signext %arg0)
// N64: define{{.*}} void @fn28(i8 noundef signext %arg0)

typedef struct T2 {  } T2;
T2 T2_retval;
T2 fn28(char arg0) {
  return T2_retval;
}

// A zero-sized argument consumes no register, but on O32 it does end the run of
// leading floating-point arguments, so the arguments after it are passed in
// integer registers.
//
// O32: define{{.*}} void @fn29(i32 noundef %arg1.coerce, i64 noundef %arg2.coerce)
// O32: declare void @fn30(i32 noundef, i64 noundef)
//
// N32: define{{.*}} void @fn29(float noundef %arg1, double noundef %arg2)
// N32: declare void @fn30(float noundef, double noundef)
//
// N64: define{{.*}} void @fn29(float noundef %arg1, double noundef %arg2)
// N64: declare void @fn30(float noundef, double noundef)

void fn30(T2 arg0, float arg1, double arg2);

void fn29(T2 arg0, float arg1, double arg2) {
  fn30(arg0, arg1, arg2);
}

// The arguments before the zero-sized one are unaffected: arg0 is still a
// leading floating-point argument and stays in a floating-point register.
//
// O32: define{{.*}} void @fn31(float noundef %arg0, i32 noundef %arg2.coerce)
// O32: declare void @fn32(float noundef, i32 noundef)
//
// N32: define{{.*}} void @fn31(float noundef %arg0, float noundef %arg2)
// N32: declare void @fn32(float noundef, float noundef)
//
// N64: define{{.*}} void @fn31(float noundef %arg0, float noundef %arg2)
// N64: declare void @fn32(float noundef, float noundef)

void fn32(float arg0, T2 arg1, float arg2);

void fn31(float arg0, T2 arg1, float arg2) {
  fn32(arg0, arg1, arg2);
}

typedef struct T3 {  } __attribute__((aligned(8))) T3;
typedef struct T4 {  } __attribute__((aligned(16))) T4;

// An over-aligned zero-sized argument has no value, but does take up the
// argument slots that its alignment requires.
//
// T3 requires an alignment of 8. On O32 that requires 4 bytes of padding
// (the bar i32), on N32/N64 the GPRs are 8 bytes and not additional
// padding is needed.
//
// O32: define{{.*}} void @fn33(i32 noundef signext %arg0, i32 %0, i32 noundef signext %arg2)
// O32: declare void @fn34(i32 noundef signext, i32, i32 noundef signext)
//
// N32: define{{.*}} void @fn33(i32 noundef signext %arg0, i32 noundef signext %arg2)
// N32: declare void @fn34(i32 noundef signext, i32 noundef signext)
//
// N64: define{{.*}} void @fn33(i32 noundef signext %arg0, i32 noundef signext %arg2)
// N64: declare void @fn34(i32 noundef signext, i32 noundef signext)

void fn34(int arg0, T3 arg1, int arg2);

void fn33(int arg0, T3 arg1, int arg2) {
  fn34(arg0, arg1, arg2);
}

// T4 is over-aligned for every ABI, so it skips a slot on all of them. The alignment
// is capped at the stack alignment of 8 bytes, so O32 skips only one slot, not three.
//
// O32: define{{.*}} void @fn35(i32 noundef signext %arg0, i32 %0, i32 noundef signext %arg2)
// O32: declare void @fn36(i32 noundef signext, i32, i32 noundef signext)
//
// N32: define{{.*}} void @fn35(i32 noundef signext %arg0, i64 %0, i32 noundef signext %arg2)
// N32: declare void @fn36(i32 noundef signext, i64, i32 noundef signext)
//
// N64: define{{.*}} void @fn35(i32 noundef signext %arg0, i64 %0, i32 noundef signext %arg2)
// N64: declare void @fn36(i32 noundef signext, i64, i32 noundef signext)

void fn36(int arg0, T4 arg1, int arg2);

void fn35(int arg0, T4 arg1, int arg2) {
  fn36(arg0, arg1, arg2);
}

// No padding is needed when the slot is already aligned.
//
// O32: define{{.*}} void @fn37(i32 noundef signext %arg1, i32 noundef signext %arg2)
// O32: declare void @fn38(i32 noundef signext, i32 noundef signext)
//
// N32: define{{.*}} void @fn37(i32 noundef signext %arg1, i32 noundef signext %arg2)
// N32: declare void @fn38(i32 noundef signext, i32 noundef signext)
//
// N64: define{{.*}} void @fn37(i32 noundef signext %arg1, i32 noundef signext %arg2)
// N64: declare void @fn38(i32 noundef signext, i32 noundef signext)

void fn38(T3 arg0, int arg1, int arg2);

void fn37(T3 arg0, int arg1, int arg2) {
  fn38(arg0, arg1, arg2);
}

// On O32 both effects apply: the skipped slot becomes padding, and the
// zero-sized argument ends the run of leading floating-point arguments.
//
// O32: define{{.*}} void @fn39(float noundef %arg0, i32 %0, i32 noundef %arg2.coerce)
// O32: declare void @fn40(float noundef, i32, i32 noundef)
//
// N32: define{{.*}} void @fn39(float noundef %arg0, float noundef %arg2)
// N32: declare void @fn40(float noundef, float noundef)
//
// N64: define{{.*}} void @fn39(float noundef %arg0, float noundef %arg2)
// N64: declare void @fn40(float noundef, float noundef)

void fn40(float arg0, T3 arg1, float arg2);

void fn39(float arg0, T3 arg1, float arg2) {
  fn40(arg0, arg1, arg2);
}
