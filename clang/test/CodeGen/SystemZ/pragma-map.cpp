// RUN: %clang_cc1 %s -triple s390x-ibm-zos -emit-llvm -o - | FileCheck %s
// REQUIRES: systemz-registered-target

// Testing pragma map after decl.
extern "C" void gf0(void);
int v0;
extern int e0;
#pragma map(gf0, "F0")
#pragma map(v0, "V0")
#pragma map(e0, "E0")

// Testing pragma map before decl.
#pragma map(gf1, "F1")
#pragma map(v1, "V1")
#pragma map(e1, "E1")
extern "C" void gf1(void);
int v1;
extern int e1;

#pragma map(gf2, "F2I")
extern "C" void gf2(int);
void gf2(void);

void f3(double);
extern "C" void f3(int, double);
#pragma map(f3, "F3ID")

extern "C" void f4(void);
extern "C" double cos(double);
double cos(double, double);

int t0() {
  gf0();
  gf1();
  gf2();
  gf2(0);
  f3(0.1);
  f3(0, 0.1);
  f4();
  return e0 + e1;
}

// Testing builtin function.
#pragma map(__builtin_sin, "SIN")
double __builtin_sin(double);

double t1(double var) {
  return __builtin_sin(0.0) + __builtin_sin(var) + cos(var) + cos(var, var);
}

// Testing pragma map after decl and usage.
#pragma map(f4, "F4")
#pragma map(cos, "COS")

// Testing pragma map with namespace.
void f5(void);
namespace N0 {
extern "C" void f0(void);
extern "C" void f1(void);
extern "C" void f2(void);
extern "C" void f5(void);
extern "C" void f6(void);
#pragma map(f0, "N0F0")
#pragma map(f1, "N0F1")
#pragma map(f5, "N0F5")
#pragma map(f2, "N0F2")
} // namespace N0

namespace N0 {
void f6(void) {}
#pragma map(f6, "N0F6")
}

void t2(void) {
  N0::f0();
  N0::f1();
  N0::f2();
  N0::f5();
  N0::f6();
  f5();
}

extern "C" void func(int);
#pragma map(func, "FUNC")


void call_func() {
  func(1);
}

void func(int) { }

// Pragma before declarations

void func1(int);
#pragma map(func1, "FUNC1")
extern "C" void func1(double);

void call_func1() {
  func1(1);
}
void func1(int) { }
void func1(double) { }

// Pragma between  declarations

#pragma map(func2, "FUNC2")
void func2(int);
extern "C" void func2(double);

void call_func2() {
  func2(1);
  func2(1.0);
}
void func2(int) { }
void func2(double) { }

// Pragma after  declarations

void func3(int);
extern "C" void func3(double);
#pragma map(func3, "FUNC3")

void call_func3() {
  func3(1);
}
void func3(int) { }
void func3(double) { }

// CHECK:       @v0 = global i32 0, align 4
// CHECK-NEXT:  @V1 = global i32 0, align 4
// CHECK-NEXT:  @E0 = external global i32, align 4
// CHECK-NEXT:  @E1 = external global i32, align 4

// CHECK-LABEL: _Z2t0v
// CHECK:       call void @F0{{.*}}
// CHECK:       call void @F1{{.*}}
// CHECK:       call void @_Z3gf2v{{.*}}
// CHECK:       call void @F2I{{.*}}
// CHECK:       call void @_Z2f3d{{.*}}
// CHECK:       call void @F3ID{{.*}}
// CHECK:       call void @f4{{.*}}


// CHECK-LABEL: _Z2t1d
// This change turns off optimization of math functions, therefore the next 2 lines change.
// CHECK:       %call{{.*}} call double @SIN{{.*}}
// CHECK:       %call1{{.*}} call double @SIN{{.*}}
// CHECK:       %add = fadd double %call, %call1
// CHECK:       call double @llvm.cos{{.*}}
// CHECK:       call noundef double @_Z3cosdd{{.*}}

// CHECK:      declare double @SIN(double noundef){{.*}}
// CHECK:      declare double @llvm.cos.f64(double){{.*}}
// CHECK:      declare noundef double @_Z3cosdd(double noundef, double noundef){{.*}}

// CHECK-LABEL: _Z2t2v
// CHECK:       call void @N0F0()
// CHECK:       call void @N0F1()
// CHECK:       call void @N0F2()
// CHECK:       call void @N0F5()
// CHECK:       call void @f6()
// CHECK:       call void @_Z2f5v()

// CHECK-LABEL: _Z9call_funcv
// CHECK:       call void @FUNC

// CHECK-LABEL: _Z10call_func1v
// CHECK:       call void @_Z5func1i

// CHECK-LABEL: _Z10call_func2v
// CHECK:       call void @_Z5func2i
// CHECK:       call void @FUNC2

// CHECK-LABEL: _Z10call_func3v
// CHECK:       call void @_Z5func3i

// CHECK:       !zos_mapped_names = !{![[n1:[0-9]+]], ![[n2:[0-9]+]], ![[n3:[0-9]+]], ![[n4:[0-9]+]], ![[n5:[0-9]+]], ![[n6:[0-9]+]], ![[n7:[0-9]+]], ![[n8:[0-9]+]], ![[n9:[0-9]+]], ![[n10:[0-9]+]], ![[n11:[0-9]+]], ![[n12:[0-9]+]], ![[n13:[0-9]+]], ![[n14:[0-9]+]], ![[n15:[0-9]+]], ![[n16:[0-9]+]], ![[n17:[0-9]+]]}
// CHECK: ![[n1]] = !{!"v0", !"v0"}
// CHECK-NEXT: ![[n2]] = !{!"V1", !"v1"}
// CHECK-NEXT: ![[n3]] = !{!"F0", !"gf0"}
// CHECK-NEXT: ![[n4]] = !{!"F1", !"gf1"}
// CHECK-NEXT: ![[n5]] = !{!"F2I", !"gf2"}
// CHECK-NEXT: ![[n6]] = !{!"F3ID", !"f3"}
// CHECK-NEXT: ![[n7]] = !{!"E0", !"e0"}
// CHECK-NEXT: ![[n8]] = !{!"E1", !"e1"}
// CHECK-NEXT: ![[n9]] = !{!"SIN", !"__builtin_sin"}
// CHECK-NEXT: ![[n10]] = !{!"N0F0", !"f0"}
// CHECK-NEXT: ![[n11]] = !{!"N0F1", !"f1"}
// CHECK-NEXT: ![[n12]] = !{!"N0F2", !"f2"}
// CHECK-NEXT: ![[n13]] = !{!"N0F5", !"f5"}
// CHECK-NEXT: ![[n14]] = !{!"FUNC", !"func"}
// CHECK-NEXT: ![[n15]] = !{!"FUNC1", !"func1"}
// CHECK-NEXT: ![[n16]] = !{!"FUNC2", !"func2"}
// CHECK-NEXT: ![[n17]] = !{!"FUNC3", !"func3"}
