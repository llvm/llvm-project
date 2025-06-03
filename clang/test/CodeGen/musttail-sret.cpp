// RUN: %clang_cc1 -triple=arm %s -emit-llvm -O3 -o - | FileCheck %s --check-prefix=CHECK-ARM
// RUN: %clang_cc1 -triple=arm64 %s -emit-llvm -O3 -o - | FileCheck %s --check-prefix=CHECK-ARM64
// RUN: %clang_cc1 -triple=i686 %s -emit-llvm -O3 -o - | FileCheck %s --check-prefix=CHECK-X86
// RUN: %clang_cc1 -triple=x86_64 %s -emit-llvm -O3 -o - | FileCheck %s --check-prefix=CHECK-X64

// Sret tests
struct Big {
  int a, b, c, d, e, f, g, h;
};

struct Big F1(signed short P0);

struct Big F2(signed short P0) {
  signed short P1 = 20391;
  [[clang::musttail]] return F1(P1);
}

// CHECK-NOT: alloca
// CHECK-ARM: musttail call arm_aapcscc void @_Z2F1s(ptr dead_on_unwind writable sret(%struct.Big) align 4 %agg.result, i16 noundef signext 20391)
// CHECK-ARM64: musttail call void @_Z2F1s(ptr dead_on_unwind writable sret(%struct.Big) align 4 %agg.result, i16 noundef 20391)
// CHECK-X86: musttail call void @_Z2F1s(ptr dead_on_unwind writable sret(%struct.Big) align 4 %agg.result, i16 noundef signext 20391)
// CHECK-X64: musttail call void @_Z2F1s(ptr dead_on_unwind writable sret(%struct.Big) align 4 %agg.result, i16 noundef signext 20391)

struct ReallyBig {
  int a[100];
};

// Indirect sret tests
// Function pointer for testing indirect musttail call.
struct FunctionPointers {
  ReallyBig (*F3)(int, int, int, int, float, double);
  ReallyBig (*F4)(int, int, int, char, float, double);
};

struct ReallyBig F3(int P0, int P1, int P2, int P3, float P4, double P5);
struct ReallyBig F4(int P0, int P1, int P2, char P3, float P4, double P5);

static struct FunctionPointers FP = {F3, F4};

struct ReallyBig F5 (int P0, int P1, int P2, int P3, float P4, double P5) {
  [[clang::musttail]] return FP.F3(P0, P1, P2, P3, P4, P5);
}

// CHECK-NOT: alloca
// CHECK-ARM: musttail call arm_aapcscc void @_Z2F3iiiifd(ptr dead_on_unwind writable sret(%struct.ReallyBig) align 4 %agg.result, i32 noundef %P0, i32 noundef %P1, i32 noundef %P2, i32 noundef %P3, float noundef %P4, double noundef %P5)
// CHECK-ARM64: musttail call void @_Z2F3iiiifd(ptr dead_on_unwind writable sret(%struct.ReallyBig) align 4 %agg.result, i32 noundef %P0, i32 noundef %P1, i32 noundef %P2, i32 noundef %P3, float noundef %P4, double noundef %P5)
// CHECK-X86: musttail call void @_Z2F3iiiifd(ptr dead_on_unwind writable sret(%struct.ReallyBig) align 4 %agg.result, i32 noundef %P0, i32 noundef %P1, i32 noundef %P2, i32 noundef %P3, float noundef %P4, double noundef %P5)
// CHECK-X64: musttail call void @_Z2F3iiiifd(ptr dead_on_unwind writable sret(%struct.ReallyBig) align 4 %agg.result, i32 noundef %P0, i32 noundef %P1, i32 noundef %P2, i32 noundef %P3, float noundef %P4, double noundef %P5)

struct ReallyBig F6 (int P0, int P1, int P2, char P3, float P4, double P5) {
  [[clang::musttail]] return FP.F4(P0, P1, P2, P3, P4, P5);
}

// Complex and BitInt. Special cases for sret.
// CHECK-NOT: alloca
// CHECK-ARM: musttail call arm_aapcscc void @_Z2F4iiicfd(ptr dead_on_unwind writable sret(%struct.ReallyBig) align 4 %agg.result, i32 noundef %P0, i32 noundef %P1, i32 noundef %P2, i8 noundef signext %P3, float noundef %P4, double noundef %P5)
// CHECK-ARM64: musttail call void @_Z2F4iiicfd(ptr dead_on_unwind writable sret(%struct.ReallyBig) align 4 %agg.result, i32 noundef %P0, i32 noundef %P1, i32 noundef %P2, i8 noundef %P3, float noundef %P4, double noundef %P5)
// CHECK-X86: musttail call void @_Z2F4iiicfd(ptr dead_on_unwind writable sret(%struct.ReallyBig) align 4 %agg.result, i32 noundef %P0, i32 noundef %P1, i32 noundef %P2, i8 noundef signext %P3, float noundef %P4, double noundef %P5)
// CHECK-X64: musttail call void @_Z2F4iiicfd(ptr dead_on_unwind writable sret(%struct.ReallyBig) align 4 %agg.result, i32 noundef %P0, i32 noundef %P1, i32 noundef %P2, i8 noundef signext %P3, float noundef %P4, double noundef %P5)

double _Complex F7(signed short P0);

double _Complex F8(signed short P0) {
  signed short P1 = 20391;
  [[clang::musttail]] return F7(P1);
}

// CHECK-NOT: alloca
// CHECK-ARM: musttail call arm_aapcscc void @_Z2F7s(ptr dead_on_unwind writable sret({ double, double }) align 8 %agg.result, i16 noundef signext 20391)
// CHECK-ARM64: musttail call noundef { double, double } @_Z2F7s(i16 noundef 20391)
// CHECK-X86: musttail call void @_Z2F7s(ptr dead_on_unwind writable sret({ double, double }) align 4 %agg.result, i16 noundef signext 20391) 
// CHECK-X64: musttail call noundef { double, double } @_Z2F7s(i16 noundef signext 20391)

signed _BitInt(100) F9(float P0, float P1, double P2, char P3);

signed _BitInt(100) F10(float P0, float P1, double P2, char P3) {
  [[clang::musttail]] return F9(P0, P1, P2, P3);
}

// CHECK-NOT: alloca
// CHECK-ARM: musttail call arm_aapcscc void @_Z2F9ffdc(ptr dead_on_unwind writable sret(i128) align 8 %agg.result, float noundef %P0, float noundef %P1, double noundef %P2, i8 noundef signext %P3)
// CHECK-ARM64: musttail call noundef i100 @_Z2F9ffdc(float noundef %P0, float noundef %P1, double noundef %P2, i8 noundef %P3)
// CHECK-X86: musttail call void @_Z2F9ffdc(ptr dead_on_unwind writable sret(i128) align 4 %agg.result, float noundef %P0, float noundef %P1, double noundef %P2, i8 noundef signext %P3)
// CHECK-X64: musttail call noundef { i64, i64 } @_Z2F9ffdc(float noundef %P0, float noundef %P1, double noundef %P2, i8 noundef signext %P3)