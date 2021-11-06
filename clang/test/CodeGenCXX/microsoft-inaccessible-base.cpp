// RUN: %clang_cc1 -fms-compatibility -triple x86_64-windows-msvc %s -emit-llvm -o - | FileCheck %s

// Make sure we choose the *direct* base path when doing these conversions.

// CHECK: %struct.C = type { %struct.A, %struct.B }
// CHECK: %struct.D = type { %struct.B, %struct.A }

struct A { int a; };
struct B : A { int b; };

struct C : A, B { };
extern "C" A *a_from_c(C *p) { return p; }
// CHECK-LABEL: define dso_local %struct.A* @a_from_c(%struct.C* noundef %{{.*}})
// CHECK: bitcast %struct.C* %{{.*}} to %struct.A*

struct D : B, A { };
extern "C" A *a_from_d(D *p) { return p; }
// CHECK-LABEL: define dso_local %struct.A* @a_from_d(%struct.D* noundef %{{.*}})
// CHECK: %[[p_i8:[^ ]*]] = bitcast %struct.D* %{{.*}} to i8*
// CHECK: getelementptr inbounds i8, i8* %[[p_i8]], i64 8
