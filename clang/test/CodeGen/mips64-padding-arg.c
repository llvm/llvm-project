// RUN: %clang_cc1 -triple mipsel-unknown-linux -O3 -o - -emit-llvm %s | FileCheck %s -check-prefix=O32
// RUN: %clang_cc1 -triple mips64el-unknown-linux -O3 -target-abi n64 -o - -emit-llvm %s | FileCheck %s -check-prefix=N64
// RUN: %clang_cc1 -triple mipsel-unknown-linux -target-feature "+fp64" -O3 -o - -emit-llvm %s | FileCheck %s -check-prefix=O32

typedef struct {
  double d;
  long double ld;
} S0;

// Insert padding to ensure arguments of type S0 are aligned to 16-byte boundaries.

// N64-LABEL: define{{.*}} void @foo1(i32 noundef signext %a0, i64 %0, double inreg %a1.coerce0, i64 inreg %a1.coerce1, i64 inreg %a1.coerce2, i64 inreg %a1.coerce3, double inreg %a2.coerce0, i64 inreg %a2.coerce1, i64 inreg %a2.coerce2, i64 inreg %a2.coerce3, i32 noundef signext %b, i64 %1, double inreg %a3.coerce0, i64 inreg %a3.coerce1, i64 inreg %a3.coerce2, i64 inreg %a3.coerce3)
// N64: tail call void @foo2(i32 noundef signext 1, i32 noundef signext 2, i32 noundef signext %a0, i64 undef, double inreg %a1.coerce0, i64 inreg %a1.coerce1, i64 inreg %a1.coerce2, i64 inreg %a1.coerce3, double inreg %a2.coerce0, i64 inreg %a2.coerce1, i64 inreg %a2.coerce2, i64 inreg %a2.coerce3, i32 noundef signext 3, i64 undef, double inreg %a3.coerce0, i64 inreg %a3.coerce1, i64 inreg %a3.coerce2, i64 inreg %a3.coerce3)
// N64: declare void @foo2(i32 noundef signext, i32 noundef signext, i32 noundef signext, i64, double inreg, i64 inreg, i64 inreg, i64 inreg, double inreg, i64 inreg, i64 inreg, i64 inreg, i32 noundef signext, i64, double inreg, i64 inreg, i64 inreg, i64 inreg)

extern void foo2(int, int, int, S0, S0, int, S0);

void foo1(int a0, S0 a1, S0 a2, int b, S0 a3) {
  foo2(1, 2, a0, a1, a2, 3, a3);
}

// Insert padding before long double argument.
//
// N64-LABEL: define{{.*}} void @foo3(i32 noundef signext %a0, i64 %0, fp128 noundef %a1)
// N64: tail call void @foo4(i32 noundef signext 1, i32 noundef signext 2, i32 noundef signext %a0, i64 undef, fp128 noundef %a1)
// N64: declare void @foo4(i32 noundef signext, i32 noundef signext, i32 noundef signext, i64, fp128 noundef)

extern void foo4(int, int, int, long double);

void foo3(int a0, long double a1) {
  foo4(1, 2, a0, a1);
}

// Insert padding after hidden argument.
//
// N64-LABEL: define{{.*}} void @foo5(ptr dead_on_unwind noalias writable sret(%struct.S0) align 16 %agg.result, i64 %0, fp128 noundef %a0)
// N64: call void @foo6(ptr dead_on_unwind writable sret(%struct.S0) align 16 %agg.result, i32 noundef signext 1, i32 noundef signext 2, i64 undef, fp128 noundef %a0)
// N64: declare void @foo6(ptr dead_on_unwind writable sret(%struct.S0) align 16, i32 noundef signext, i32 noundef signext, i64, fp128 noundef)

extern S0 foo6(int, int, long double);

S0 foo5(long double a0) {
  return foo6(1, 2, a0);
}

// Do not insert padding if ABI is O32.
//
// O32-LABEL: define{{.*}} void @foo7(float noundef %a0, double noundef %a1)
// O32: declare void @foo8(float noundef, double noundef)

extern void foo8(float, double);

void foo7(float a0, double a1) {
  foo8(a0 + 1.0f, a1 + 2.0);
}

// O32-LABEL: define{{.*}} void @foo9()
// O32: declare void @foo10(i32 noundef signext, i32

typedef struct __attribute__((aligned(16))) {
  int a;
} S16;

S16 s16;

void foo10(int, S16);

void foo9(void) {
  foo10(1, s16);
}

