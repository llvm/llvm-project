// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -triple powerpc64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s

typedef struct s1 { float f; } Sf;
typedef struct s2 { double d; } Sd;
typedef struct s4 { Sf fs; } SSf;
typedef struct s5 { Sd ds; } SSd;

void bar(Sf a, Sd b, SSf d, SSd e) {}

// CHECK-LABEL: define{{.*}} void @bar
// CHECK:  %a = alloca %struct.s1, align 4
// CHECK:  %b = alloca %struct.s2, align 8
// CHECK:  %d = alloca %struct.s4, align 4
// CHECK:  %e = alloca %struct.s5, align 8
// CHECK:  %{{[a-zA-Z0-9.]+}} = getelementptr inbounds nuw %struct.s1, ptr %a, i32 0, i32 0
// CHECK:  store float %a.coerce, ptr %{{[a-zA-Z0-9.]+}}, align 4
// CHECK:  %{{[a-zA-Z0-9.]+}} = getelementptr inbounds nuw %struct.s2, ptr %b, i32 0, i32 0
// CHECK:  store double %b.coerce, ptr %{{[a-zA-Z0-9.]+}}, align 8
// CHECK:  %{{[a-zA-Z0-9.]+}} = getelementptr inbounds nuw %struct.s4, ptr %d, i32 0, i32 0
// CHECK:  %{{[a-zA-Z0-9.]+}} = getelementptr inbounds nuw %struct.s1, ptr %{{[a-zA-Z0-9.]+}}, i32 0, i32 0
// CHECK:  store float %d.coerce, ptr %{{[a-zA-Z0-9.]+}}, align 4
// CHECK:  %{{[a-zA-Z0-9.]+}} = getelementptr inbounds nuw %struct.s5, ptr %e, i32 0, i32 0
// CHECK:  %{{[a-zA-Z0-9.]+}} = getelementptr inbounds nuw %struct.s2, ptr %{{[a-zA-Z0-9.]+}}, i32 0, i32 0
// CHECK:  store double %e.coerce, ptr %{{[a-zA-Z0-9.]+}}, align 8
// CHECK:  ret void

void foo(void) 
{
  Sf p1 = { 22.63f };
  Sd p2 = { 19.47 };
  SSf p4 = { { 22.63f } };
  SSd p5 = { { 19.47 } };
  bar(p1, p2, p4, p5);
}

// CHECK-LABEL: define{{.*}} void @foo
// CHECK:  %{{[a-zA-Z0-9.]+}} = getelementptr inbounds nuw %struct.s1, ptr %p1, i32 0, i32 0
// CHECK:  %{{[0-9]+}} = load float, ptr %{{[a-zA-Z0-9.]+}}, align 4
// CHECK:  %{{[a-zA-Z0-9.]+}} = getelementptr inbounds nuw %struct.s2, ptr %p2, i32 0, i32 0
// CHECK:  %{{[0-9]+}} = load double, ptr %{{[a-zA-Z0-9.]+}}, align 8
// CHECK:  %{{[a-zA-Z0-9.]+}} = getelementptr inbounds nuw %struct.s4, ptr %p4, i32 0, i32 0
// CHECK:  %{{[a-zA-Z0-9.]+}} = getelementptr inbounds nuw %struct.s1, ptr %{{[a-zA-Z0-9.]+}}, i32 0, i32 0
// CHECK:  %{{[0-9]+}} = load float, ptr %{{[a-zA-Z0-9.]+}}, align 4
// CHECK:  %{{[a-zA-Z0-9.]+}} = getelementptr inbounds nuw %struct.s5, ptr %p5, i32 0, i32 0
// CHECK:  %{{[a-zA-Z0-9.]+}} = getelementptr inbounds nuw %struct.s2, ptr %{{[a-zA-Z0-9.]+}}, i32 0, i32 0
// CHECK:  %{{[0-9]+}} = load double, ptr %{{[a-zA-Z0-9.]+}}, align 8
// CHECK:  call void @bar(float inreg %{{[0-9]+}}, double inreg %{{[0-9]+}}, float inreg %{{[0-9]+}}, double inreg %{{[0-9]+}})
// CHECK:  ret void
