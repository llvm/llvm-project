// RUN: %clang_cc1 -no-opaque-pointers -fexceptions -fcxx-exceptions %s -triple=x86_64-pc-linux-gnu -funwind-tables=2 -emit-llvm -o - -O1 -disable-llvm-passes | FileCheck %s

// When generating the thunk for secondary, do not push terminate scopes for
// either the varargs or non-varargs case. Related to PR44987.

struct A {
  virtual void primary_key();
};
struct B {
  virtual void secondary();
  virtual void secondary_vararg(int, ...);
};
class C : A, B {
  virtual void primary_key();
  void secondary() noexcept;
  void secondary_vararg(int, ...) noexcept;
};
void C::primary_key() {}

// CHECK-LABEL: define available_externally void @_ZThn8_N1C9secondaryEv(%class.C* noundef %this) {{.*}} #2
// CHECK-NOT: invoke
// CHECK: tail call void @_ZN1C9secondaryEv(%class.C* {{[^,]*}} %{{.*}})
// CHECK-NOT: invoke
// CHECK: ret void

// CHECK-LABEL: define available_externally void @_ZThn8_N1C16secondary_varargEiz(%class.C* noundef %this, i32 noundef %0, ...) {{.*}} #2
// CHECK-NOT: invoke
// CHECK:  musttail call void (%class.C*, i32, ...) @_ZN1C16secondary_varargEiz(%class.C* {{[^,]*}} %{{.*}}, i32 noundef %{{.*}}, ...) #3
// CHECK-NEXT:  ret void
