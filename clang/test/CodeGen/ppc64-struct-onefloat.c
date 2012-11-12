// REQUIRES: ppc64-registered-target
// RUN: %clang_cc1 -O0 -triple powerpc64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s

typedef struct s1 { float f; } Sf;
typedef struct s2 { double d; } Sd;
typedef struct s4 { Sf fs; } SSf;
typedef struct s5 { Sd ds; } SSd;

void bar(Sf a, Sd b, SSf d, SSd e) {}

// CHECK: define void @bar
// CHECK:  %a = alloca %struct.s1, align 4
// CHECK:  %b = alloca %struct.s2, align 8
// CHECK:  %d = alloca %struct.s4, align 4
// CHECK:  %e = alloca %struct.s5, align 8
// CHECK:  %{{[a-zA-Z0-9.]+}} = getelementptr %struct.s1* %a, i32 0, i32 0
// CHECK:  store float %a.coerce, float* %{{[a-zA-Z0-9.]+}}, align 1
// CHECK:  %{{[a-zA-Z0-9.]+}} = getelementptr %struct.s2* %b, i32 0, i32 0
// CHECK:  store double %b.coerce, double* %{{[a-zA-Z0-9.]+}}, align 1
// CHECK:  %{{[a-zA-Z0-9.]+}} = getelementptr %struct.s4* %d, i32 0, i32 0
// CHECK:  %{{[a-zA-Z0-9.]+}} = getelementptr %struct.s1* %{{[a-zA-Z0-9.]+}}, i32 0, i32 0
// CHECK:  store float %d.coerce, float* %{{[a-zA-Z0-9.]+}}, align 1
// CHECK:  %{{[a-zA-Z0-9.]+}} = getelementptr %struct.s5* %e, i32 0, i32 0
// CHECK:  %{{[a-zA-Z0-9.]+}} = getelementptr %struct.s2* %{{[a-zA-Z0-9.]+}}, i32 0, i32 0
// CHECK:  store double %e.coerce, double* %{{[a-zA-Z0-9.]+}}, align 1
// CHECK:  ret void

void foo(void) 
{
  Sf p1 = { 22.63f };
  Sd p2 = { 19.47 };
  SSf p4 = { { 22.63f } };
  SSd p5 = { { 19.47 } };
  bar(p1, p2, p4, p5);
}

// CHECK: define void @foo
// CHECK:  %{{[a-zA-Z0-9.]+}} = getelementptr %struct.s1* %p1, i32 0, i32 0
// CHECK:  %{{[0-9]+}} = load float* %{{[a-zA-Z0-9.]+}}, align 1
// CHECK:  %{{[a-zA-Z0-9.]+}} = getelementptr %struct.s2* %p2, i32 0, i32 0
// CHECK:  %{{[0-9]+}} = load double* %{{[a-zA-Z0-9.]+}}, align 1
// CHECK:  %{{[a-zA-Z0-9.]+}} = getelementptr %struct.s4* %p4, i32 0, i32 0
// CHECK:  %{{[a-zA-Z0-9.]+}} = getelementptr %struct.s1* %{{[a-zA-Z0-9.]+}}, i32 0, i32 0
// CHECK:  %{{[0-9]+}} = load float* %{{[a-zA-Z0-9.]+}}, align 1
// CHECK:  %{{[a-zA-Z0-9.]+}} = getelementptr %struct.s5* %p5, i32 0, i32 0
// CHECK:  %{{[a-zA-Z0-9.]+}} = getelementptr %struct.s2* %{{[a-zA-Z0-9.]+}}, i32 0, i32 0
// CHECK:  %{{[0-9]+}} = load double* %{{[a-zA-Z0-9.]+}}, align 1
// CHECK:  call void @bar(float inreg %{{[0-9]+}}, double inreg %{{[0-9]+}}, float inreg %{{[0-9]+}}, double inreg %{{[0-9]+}})
// CHECK:  ret void
