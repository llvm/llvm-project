// RUN: %clang_cc1 %s -triple s390x-ibm-zos -emit-llvm -o - | FileCheck %s
// REQUIRES: systemz-registered-target

// Overloadable functions are mangled, so #pragma map only applies to the
// overload that lacks __attribute__((overloadable)).

// Pragma after the declarations.
void a(double) __attribute__((overloadable));
void a(int);
#pragma map(a, "A")

// Pragma before the declarations, overloadable function first. The pragma must
// not be attached to the overloadable declaration (ProcessPragmaMap); it is
// applied when the unmarked overload is declared.
#pragma map(b, "B")
void b(double) __attribute__((overloadable));
void b(int);

// Unmarked overload declared first.
void c(int);
void c(double) __attribute__((overloadable));
#pragma map(c, "C")

// Every overload is marked; the pragma is not applied.
void d(int) __attribute__((overloadable));
void d(double) __attribute__((overloadable));
#pragma map(d, "D")

// Pragma before the unmarked overload, which is declared first.
#pragma map(e, "E")
void e(int);
void e(double) __attribute__((overloadable));

// Pragma between the declarations, marked overload first.
void f(double) __attribute__((overloadable));
#pragma map(f, "F")
void f(int);

// Pragma between the declarations, unmarked overload first.
void g(int);
#pragma map(g, "G")
void g(double) __attribute__((overloadable));

// Using the overloadable overload does not prevent mapping the unmarked one.
void h(double) __attribute__((overloadable));
void h(int);
void use_h(void) {
  h(1.0);
}
#pragma map(h, "H")

void t(void) {
  a(1);
  a(1.0);
  b(1);
  b(1.0);
  c(1);
  c(1.0);
  d(1);
  d(1.0);
  e(1);
  e(1.0);
  f(1);
  f(1.0);
  g(1);
  g(1.0);
  h(1);
}

// CHECK: define {{.*}}void @use_h()
// CHECK: call void @_Z1hd(double {{.*}})
// CHECK: define {{.*}}void @t()
// CHECK: call void @A(i32 {{.*}})
// CHECK: call void @_Z1ad(double {{.*}})
// CHECK: call void @B(i32 {{.*}})
// CHECK: call void @_Z1bd(double {{.*}})
// CHECK: call void @C(i32 {{.*}})
// CHECK: call void @_Z1cd(double {{.*}})
// CHECK: call void @_Z1di(i32 {{.*}})
// CHECK: call void @_Z1dd(double {{.*}})
// CHECK: call void @E(i32 {{.*}})
// CHECK: call void @_Z1ed(double {{.*}})
// CHECK: call void @F(i32 {{.*}})
// CHECK: call void @_Z1fd(double {{.*}})
// CHECK: call void @G(i32 {{.*}})
// CHECK: call void @_Z1gd(double {{.*}})
// CHECK: call void @H(i32 {{.*}})
