// RUN: %clang_cc1 -emit-llvm %s -o - -triple=i386-pc-win32 | FileCheck %s

class C {
 public:
  void simple_method() {}

  void __cdecl cdecl_method() {}

  void vararg_method(const char *fmt, ...) {}

  static void static_method() {}

  int a;
};

void call_simple_method() {
  C instance;

  instance.simple_method();
// Make sure that the call uses the right calling convention:
// CHECK: call x86_thiscallcc void @"?simple_method@C@@QAEXXZ"
// CHECK: ret

// Make sure that the definition uses the right calling convention:
// CHECK: define linkonce_odr dso_local x86_thiscallcc void @"?simple_method@C@@QAEXXZ"
// CHECK: ret
}

void call_cdecl_method() {
  C instance;
  instance.cdecl_method();
// Make sure that the call uses the right calling convention:
// CHECK: call void @"?cdecl_method@C@@QAAXXZ"
// CHECK: ret

// Make sure that the definition uses the right calling convention:
// CHECK: define linkonce_odr dso_local void @"?cdecl_method@C@@QAAXXZ"
// CHECK: ret
}

void call_vararg_method() {
  C instance;
  instance.vararg_method("Hello");
// Make sure that the call uses the right calling convention:
// CHECK: call void (%class.C*, i8*, ...) @"?vararg_method@C@@QAAXPBDZZ"
// CHECK: ret

// Make sure that the definition uses the right calling convention:
// CHECK: define linkonce_odr dso_local void @"?vararg_method@C@@QAAXPBDZZ"
}

void call_static_method() {
  C::static_method();
// Make sure that the call uses the right calling convention:
// CHECK: call void @"?static_method@C@@SAXXZ"
// CHECK: ret

// Make sure that the definition uses the right calling convention:
// CHECK: define linkonce_odr dso_local void @"?static_method@C@@SAXXZ"
}

class Base {
 public:
  Base() {}
  ~Base() {}
};

class Child: public Base { };

void constructors() {
  Child c;
// Make sure that the Base constructor call noundef in the Child constructor uses
// the right calling convention:
// CHECK: define linkonce_odr dso_local x86_thiscallcc noundef %class.Child* @"??0Child@@QAE@XZ"
// CHECK: %{{[.0-9A-Z_a-z]+}} = call x86_thiscallcc noundef %class.Base* @"??0Base@@QAE@XZ"
// CHECK: ret

// Make sure that the Base destructor call noundef in the Child denstructor uses
// the right calling convention:
// CHECK: define linkonce_odr dso_local x86_thiscallcc void @"??1Child@@QAE@XZ"
// CHECK: call x86_thiscallcc void @"??1Base@@QAE@XZ"
// CHECK: ret

// Make sure that the Base constructor definition uses the right CC:
// CHECK: define linkonce_odr dso_local x86_thiscallcc noundef %class.Base* @"??0Base@@QAE@XZ"

// Make sure that the Base destructor definition uses the right CC:
// CHECK: define linkonce_odr dso_local x86_thiscallcc void @"??1Base@@QAE@XZ"
}
