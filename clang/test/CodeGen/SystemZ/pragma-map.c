// RUN: %clang_cc1 %s -triple s390x-ibm-zos -emit-llvm -O1 -o - | FileCheck %s
// REQUIRES: systemz-registered-target

// Testing pragma map after decl.
void f0(void);
int v0;
extern int e0;
#pragma map(f0, "F0")
#pragma map(v0, "V0")
#pragma map(e0, "E0")

// Testing pragma map before decl.
#pragma map(f1, "F1")
#pragma map(v1, "V1")
#pragma map(e1, "E1")
void f1(void);
int v1;
extern int e1;

void f2(void);

// Test that mapping to empty string is ignored.
void f3(void);
#pragma map(f3, "")

int t0() {
  f0();
  f1();
  f2();
  f3();
  return e0 + e1;
}

// Testing pragma map after decl and usage.
#pragma map(f2, "F2")

// Testing builtin function.
#pragma map(__builtin_sin, "SIN")
double __builtin_sin(double);

double t1(double var) {
  return __builtin_sin(0.0) + __builtin_sin(var);
}

// Testing that the source name is kept for a mapped function definition.
#pragma map(f4, "F4")
void f4(void) {}

// CHECK:       @E0 = external local_unnamed_addr global i32, align 4
// CHECK-NEXT:  @E1 = external local_unnamed_addr global i32, align 4
// CHECK:       @V0 = local_unnamed_addr global i32 0, align 4
// CHECK-NEXT:  @V1 = local_unnamed_addr global i32 0, align 4
// CHECK:       call void @F0()
// CHECK:       call void @F1()
// CHECK:       call void @f2()
// CHECK:       declare void @F0{{.*}}
// CHECK:       declare void @F1{{.*}}
// CHECK:       declare void @f2{{.*}}
// This change turns off optimization of math functions, therefore the next 2 lines change.
// CHECK:       %call1 = tail call double @SIN(double noundef %var)
// CHECK-NEXT:  %add = fadd double %call, %call1
// CHECK:       declare double @SIN(double noundef) {{.*}}
// The mapped name is the symbol name; the source name is in "zos-ppa1-name".
// CHECK:       define{{.*}} void @F4(){{.*}} #[[F4:[0-9]+]]
// CHECK:       attributes #[[F4]] = {{{.*}}"zos-ppa1-name"="f4"{{.*}}}
