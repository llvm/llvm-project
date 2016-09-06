// RUN: %clang_cc1 -emit-llvm %s -o - -triple=i386-pc-win32 | FileCheck %s
// RUN: %clang_cc1 -emit-llvm %s -o - -triple=x86_64-pc-win32 | FileCheck %s --check-prefix=X64

void __vectorcall v1(int a, int b) {}
// CHECK: define x86_vectorcallcc void @"\01v1@@8"(i32 inreg %a, i32 inreg %b)
// X64: define x86_vectorcallcc void @"\01v1@@16"(i32 %a, i32 %b)

void __vectorcall v2(char a, char b) {}
// CHECK: define x86_vectorcallcc void @"\01v2@@8"(i8 inreg signext %a, i8 inreg signext %b)
// X64: define x86_vectorcallcc void @"\01v2@@16"(i8 %a, i8 %b)

struct Small { int x; };
void __vectorcall v3(int a, struct Small b, int c) {}
// CHECK: define x86_vectorcallcc void @"\01v3@@12"(i32 inreg %a, i32 %b.0, i32 inreg %c)
// X64: define x86_vectorcallcc void @"\01v3@@24"(i32 %a, i32 %b.coerce, i32 %c)

struct Large { int a[5]; };
void __vectorcall v4(int a, struct Large b, int c) {}
// CHECK: define x86_vectorcallcc void @"\01v4@@28"(i32 inreg %a, %struct.Large* byval align 4 %b, i32 inreg %c)
// X64: define x86_vectorcallcc void @"\01v4@@40"(i32 %a, %struct.Large* %b, i32 %c)

struct HFA2 { double x, y; };
struct HFA4 { double w, x, y, z; };
struct HFA5 { double v, w, x, y, z; };

void __vectorcall hfa1(int a, struct HFA4 b, int c) {}
// CHECK: define x86_vectorcallcc void @"\01hfa1@@40"(i32 inreg %a, double %b.0, double %b.1, double %b.2, double %b.3, i32 inreg %c)
// X64: define x86_vectorcallcc void @"\01hfa1@@48"(i32 %a, double %b.0, double %b.1, double %b.2, double %b.3, i32 %c)

// HFAs that would require more than six total SSE registers are passed
// indirectly. Additional vector arguments can consume the rest of the SSE
// registers.
void __vectorcall hfa2(struct HFA4 a, struct HFA4 b, double c) {}
// CHECK: define x86_vectorcallcc void @"\01hfa2@@72"(double %a.0, double %a.1, double %a.2, double %a.3, %struct.HFA4* inreg %b, double %c)
// X64: define x86_vectorcallcc void @"\01hfa2@@72"(double %a.0, double %a.1, double %a.2, double %a.3, %struct.HFA4* %b, double %c)

// Ensure that we pass builtin types directly while counting them against the
// SSE register usage.
void __vectorcall hfa3(double a, double b, double c, double d, double e, struct HFA2 f) {}
// CHECK: define x86_vectorcallcc void @"\01hfa3@@56"(double %a, double %b, double %c, double %d, double %e, %struct.HFA2* inreg %f)
// X64: define x86_vectorcallcc void @"\01hfa3@@56"(double %a, double %b, double %c, double %d, double %e, %struct.HFA2* %f)

// Aggregates with more than four elements are not HFAs and are passed byval.
// Because they are not classified as homogeneous, they don't get special
// handling to ensure alignment.
void __vectorcall hfa4(struct HFA5 a) {}
// CHECK: define x86_vectorcallcc void @"\01hfa4@@40"(%struct.HFA5* byval align 4)
// X64: define x86_vectorcallcc void @"\01hfa4@@40"(%struct.HFA5* %a)

// Return HFAs of 4 or fewer elements in registers.
static struct HFA2 g_hfa2;
struct HFA2 __vectorcall hfa5(void) { return g_hfa2; }
// CHECK: define x86_vectorcallcc %struct.HFA2 @"\01hfa5@@0"()
// X64: define x86_vectorcallcc %struct.HFA2 @"\01hfa5@@0"()

typedef float __attribute__((vector_size(16))) v4f32;
struct HVA2 { v4f32 x, y; };
struct HVA4 { v4f32 w, x, y, z; };

void __vectorcall hva1(int a, struct HVA4 b, int c) {}
// CHECK: define x86_vectorcallcc void @"\01hva1@@72"(i32 inreg %a, <4 x float> %b.0, <4 x float> %b.1, <4 x float> %b.2, <4 x float> %b.3, i32 inreg %c)
// X64: define x86_vectorcallcc void @"\01hva1@@80"(i32 %a, <4 x float> %b.0, <4 x float> %b.1, <4 x float> %b.2, <4 x float> %b.3, i32 %c)

void __vectorcall hva2(struct HVA4 a, struct HVA4 b, v4f32 c) {}
// CHECK: define x86_vectorcallcc void @"\01hva2@@144"(<4 x float> %a.0, <4 x float> %a.1, <4 x float> %a.2, <4 x float> %a.3, %struct.HVA4* inreg %b, <4 x float> %c)
// X64: define x86_vectorcallcc void @"\01hva2@@144"(<4 x float> %a.0, <4 x float> %a.1, <4 x float> %a.2, <4 x float> %a.3, %struct.HVA4* %b, <4 x float> %c)

void __vectorcall hva3(v4f32 a, v4f32 b, v4f32 c, v4f32 d, v4f32 e, struct HVA2 f) {}
// CHECK: define x86_vectorcallcc void @"\01hva3@@112"(<4 x float> %a, <4 x float> %b, <4 x float> %c, <4 x float> %d, <4 x float> %e, %struct.HVA2* inreg %f)
// X64: define x86_vectorcallcc void @"\01hva3@@112"(<4 x float> %a, <4 x float> %b, <4 x float> %c, <4 x float> %d, <4 x float> %e, %struct.HVA2* %f)

typedef float __attribute__((ext_vector_type(3))) v3f32;
struct OddSizeHVA { v3f32 x, y; };

void __vectorcall odd_size_hva(struct OddSizeHVA a) {}
// CHECK: define x86_vectorcallcc void @"\01odd_size_hva@@32"(<3 x float> %a.0, <3 x float> %a.1)
// X64: define x86_vectorcallcc void @"\01odd_size_hva@@32"(<3 x float> %a.0, <3 x float> %a.1)
