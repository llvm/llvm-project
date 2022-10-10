// RUN: %clang_cc1 -emit-llvm %s -o - -ffreestanding -triple=i386-pc-win32       | FileCheck %s --check-prefixes=X86,Win32
// RUN: %clang_cc1 -emit-llvm %s -o - -ffreestanding -triple=x86_64-pc-win32     | FileCheck %s --check-prefixes=X64,Win64
// RUN: %clang_cc1 -emit-llvm %s -o - -ffreestanding -triple=i386-pc-linux-gnu   | FileCheck %s --check-prefixes=X86,Lin32
// RUN: %clang_cc1 -emit-llvm %s -o - -ffreestanding -triple=x86_64-pc-linux-gnu | FileCheck %s --check-prefixes=X64,Lin64

#include <xmmintrin.h>

void __regcall v1(int a, int b) {}
// X86: define dso_local x86_regcallcc void @__regcall3__v1(i32 inreg noundef %a, i32 inreg noundef %b)
// X64: define dso_local x86_regcallcc void @__regcall3__v1(i32 noundef %a, i32 noundef %b)

void __attribute__((regcall)) v1b(int a, int b) {}
// X86: define dso_local x86_regcallcc void @__regcall3__v1b(i32 inreg noundef %a, i32 inreg noundef %b)
// X64: define dso_local x86_regcallcc void @__regcall3__v1b(i32 noundef %a, i32 noundef %b)

void __regcall v2(char a, char b) {}
// X86: define dso_local x86_regcallcc void @__regcall3__v2(i8 inreg noundef signext %a, i8 inreg noundef signext %b)
// Win64: define dso_local x86_regcallcc void @__regcall3__v2(i8 noundef %a, i8 noundef %b)
// Lin64: define dso_local x86_regcallcc void @__regcall3__v2(i8 noundef signext %a, i8 noundef signext %b)

struct Small { int x; };
void __regcall v3(int a, struct Small b, int c) {}
// Win32: define dso_local x86_regcallcc void @__regcall3__v3(i32 inreg noundef %a, i32 %b.0, i32 inreg noundef %c)
// Lin32: define dso_local x86_regcallcc void @__regcall3__v3(i32 inreg noundef %a, i32 inreg %0, i32 %b.0, i32 inreg noundef %c)
// X64: define dso_local x86_regcallcc void @__regcall3__v3(i32 noundef %a, i32 %b.coerce, i32 noundef %c)

struct Large { int a[5]; };
void __regcall v4(int a, struct Large b, int c) {}
// Win32: define dso_local x86_regcallcc void @__regcall3__v4(i32 inreg noundef %a, ptr noundef byval(%struct.Large) align 4 %b, i32 inreg noundef %c)
// Lin32: define dso_local x86_regcallcc void @__regcall3__v4(i32 inreg noundef %a, ptr noundef byval(%struct.Large) align 4 %b, i32 noundef %c)
// Win64: define dso_local x86_regcallcc void @__regcall3__v4(i32 noundef %a, ptr noundef %b, i32 noundef %c)
// Lin64: define dso_local x86_regcallcc void @__regcall3__v4(i32 noundef %a, [5 x i32] %b.coerce, i32 noundef %c)

void __regcall v5(long long a, int b, int c) {}
// X86: define dso_local x86_regcallcc void @__regcall3__v5(i64 noundef %a, i32 inreg noundef %b, i32 inreg noundef %c)
// X64: define dso_local x86_regcallcc void @__regcall3__v5(i64 noundef %a, i32 noundef %b, i32 noundef %c)

struct HFA2 { double x, y; };
struct HFA4 { double w, x, y, z; };
struct HFA5 { double v, w, x, y, z; };

void __regcall hfa1(int a, struct HFA4 b, int c) {}
// X86: define dso_local x86_regcallcc void @__regcall3__hfa1(i32 inreg noundef %a, double %b.0, double %b.1, double %b.2, double %b.3, i32 inreg noundef %c)
// X64: define dso_local x86_regcallcc void @__regcall3__hfa1(i32 noundef %a, double %{{.*}}, double %{{.*}}, double %{{.*}}, double %{{.*}}, i32 noundef %c)

// HFAs that would require more than six total SSE registers are passed
// indirectly. Additional vector arguments can consume the rest of the SSE
// registers.
void __regcall hfa2(struct HFA4 a, struct HFA4 b, double c) {}
// X86: define dso_local x86_regcallcc void @__regcall3__hfa2(double %a.0, double %a.1, double %a.2, double %a.3, double %b.0, double %b.1, double %b.2, double %b.3, ptr inreg noundef %0)
// X64: define dso_local x86_regcallcc void @__regcall3__hfa2(double %{{.*}}, double %{{.*}}, double %{{.*}}, double %{{.*}}, double %{{.*}}, double %{{.*}}, double %{{.*}}, double %{{.*}}, double noundef %c)

// Ensure that we pass builtin types directly while counting them against the
// SSE register usage.
void __regcall hfa3(double a, double b, double c, double d, double e, struct HFA2 f) {}
// X86: define dso_local x86_regcallcc void @__regcall3__hfa3(double noundef %a, double noundef %b, double noundef %c, double noundef %d, double noundef %e, double %f.0, double %f.1)
// X64: define dso_local x86_regcallcc void @__regcall3__hfa3(double noundef %a, double noundef %b, double noundef %c, double noundef %d, double noundef %e, double %{{.*}}, double %{{.*}})

// Aggregates with more than four elements are not HFAs and are passed byval(%b.3, double noundef).
// Because they are not classified as homogeneous, they don't get special
// handling to ensure alignment.
void __regcall hfa4(struct HFA5 a) {}
// X32: define dso_local x86_regcallcc void @__regcall3__hfa4(ptr noundef byval(%struct.HFA5) align 4 %{{.*}})
// Win64: define dso_local x86_regcallcc void @__regcall3__hfa4(ptr noundef %a)
// Lin64: define dso_local x86_regcallcc void @__regcall3__hfa4(double %a.coerce0, double %a.coerce1, double %a.coerce2, double %a.coerce3, double %a.coerce4)

// Return HFAs of 4 or fewer elements in registers.
static struct HFA2 g_hfa2;
struct HFA2 __regcall hfa5(void) { return g_hfa2; }
// X86: define dso_local x86_regcallcc %struct.HFA2 @__regcall3__hfa5()
// X64: define dso_local x86_regcallcc %struct.HFA2 @__regcall3__hfa5()

typedef float __attribute__((vector_size(16))) v4f32;
struct HVA2 { v4f32 x, y; };
struct HVA4 { v4f32 w, x, y, z; };

void __regcall hva1(int a, struct HVA4 b, int c) {}
// X86: define dso_local x86_regcallcc void @__regcall3__hva1(i32 inreg noundef %a, <4 x float> %b.0, <4 x float> %b.1, <4 x float> %b.2, <4 x float> %b.3, i32 inreg noundef %c)
// X64: define dso_local x86_regcallcc void @__regcall3__hva1(i32 noundef %a, <4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 noundef %c)

void __regcall hva2(struct HVA4 a, struct HVA4 b, v4f32 c) {}
// X86: define dso_local x86_regcallcc void @__regcall3__hva2(<4 x float> %a.0, <4 x float> %a.1, <4 x float> %a.2, <4 x float> %a.3, <4 x float> %b.0, <4 x float> %b.1, <4 x float> %b.2, <4 x float> %b.3, ptr inreg noundef %0)
// X64: define dso_local x86_regcallcc void @__regcall3__hva2(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> noundef %c)

void __regcall hva3(v4f32 a, v4f32 b, v4f32 c, v4f32 d, v4f32 e, struct HVA2 f) {}
// X86: define dso_local x86_regcallcc void @__regcall3__hva3(<4 x float> noundef %a, <4 x float> noundef %b, <4 x float> noundef %c, <4 x float> noundef %d, <4 x float> noundef %e, <4 x float> %f.0, <4 x float> %f.1)
// X64: define dso_local x86_regcallcc void @__regcall3__hva3(<4 x float> noundef %a, <4 x float> noundef %b, <4 x float> noundef %c, <4 x float> noundef %d, <4 x float> noundef %e, <4 x float> %{{.*}}, <4 x float> %{{.*}})

typedef float __attribute__((ext_vector_type(3))) v3f32;
struct OddSizeHVA { v3f32 x, y; };

void __regcall odd_size_hva(struct OddSizeHVA a) {}
// X86: define dso_local x86_regcallcc void @__regcall3__odd_size_hva(<3 x float> %a.0, <3 x float> %a.1)
// X64: define dso_local x86_regcallcc void @__regcall3__odd_size_hva(<3 x float> %{{.*}}, <3 x float> %{{.*}})

struct HFA6 { __m128 f[4]; };
struct HFA6 __regcall ret_reg_reused(struct HFA6 a, struct HFA6 b, struct HFA6 c, struct HFA6 d){ struct HFA6 h; return h;}
// X86: define dso_local x86_regcallcc %struct.HFA6 @__regcall3__ret_reg_reused(<4 x float> %a.0, <4 x float> %a.1, <4 x float> %a.2, <4 x float> %a.3, <4 x float> %b.0, <4 x float> %b.1, <4 x float> %b.2, <4 x float> %b.3, ptr inreg noundef %c, ptr inreg noundef %d)
// Win64: define dso_local x86_regcallcc %struct.HFA6 @__regcall3__ret_reg_reused(<4 x float> %a.0, <4 x float> %a.1, <4 x float> %a.2, <4 x float> %a.3, <4 x float> %b.0, <4 x float> %b.1, <4 x float> %b.2, <4 x float> %b.3, <4 x float> %c.0, <4 x float> %c.1, <4 x float> %c.2, <4 x float> %c.3, <4 x float> %d.0, <4 x float> %d.1, <4 x float> %d.2, <4 x float> %d.3)
// Lin64: define dso_local x86_regcallcc %struct.HFA6 @__regcall3__ret_reg_reused([4 x <4 x float>] %a.coerce, [4 x <4 x float>] %b.coerce, [4 x <4 x float>] %c.coerce, [4 x <4 x float>] %d.coerce)
