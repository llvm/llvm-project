// RUN: %clang_cc1 -triple thumbv7-windows-msvc -fdeclspec -std=c++11 -fobjc-runtime=ios-6.0 -o - -emit-llvm %s | FileCheck %s

@class I;

id kid;
// CHECK: @"\01?kid@@3PAUobjc_object@@A" = global

Class klass;
// CHECK: @"\01?klass@@3PAUobjc_class@@A" = global

I *kI;
// CHECK: @"\01?kI@@3PAUI@@A" = global

void f(I *) {}
// CHECK-LABEL: "\01?f@@YAXPAUI@@@Z"

void f(const I *) {}
// CHECK-LABEL: "\01?f@@YAXPBUI@@@Z"

void f(I &) {}
// CHECK-LABEL: "\01?f@@YAXAAUI@@@Z"

void f(const I &) {}
// CHECK-LABEL: "\01?f@@YAXABUI@@@Z"

void f(const I &&) {}
// CHECK-LABEL: "\01?f@@YAX$$QBUI@@@Z"

void g(id) {}
// CHECK-LABEL: "\01?g@@YAXPAUobjc_object@@@Z"

void g(id &) {}
// CHECK-LABEL: "\01?g@@YAXAAPAUobjc_object@@@Z"

void g(const id &) {}
// CHECK-LABEL: "\01?g@@YAXABPAUobjc_object@@@Z"

void g(id &&) {}
// CHECK-LABEL: "\01?g@@YAX$$QAPAUobjc_object@@@Z"

void h(Class) {}
// CHECK-LABEL: "\01?h@@YAXPAUobjc_class@@@Z"

void h(Class &) {}
// CHECK-LABEL: "\01?h@@YAXAAPAUobjc_class@@@Z"

void h(const Class &) {}
// CHECK-LABEL: "\01?h@@YAXABPAUobjc_class@@@Z"

void h(Class &&) {}
// CHECK-LABEL: "\01?h@@YAX$$QAPAUobjc_class@@@Z"

I *i() { return nullptr; }
// CHECK-LABEL: "\01?i@@YAPAUI@@XZ"

const I *j() { return nullptr; }
// CHECK-LABEL: "\01?j@@YAPBUI@@XZ"

I &k() { return *kI; }
// CHECK-LABEL: "\01?k@@YAAAUI@@XZ"

const I &l() { return *kI; }
// CHECK-LABEL: "\01?l@@YAABUI@@XZ"

struct __declspec(dllexport) s {
  struct s &operator=(const struct s &) = delete;

  void m(I *) {}
  // CHECK-LABEL: "\01?m@s@@QAAXPAUI@@@Z"

  void m(const I *) {}
  // CHECK-LABEL: "\01?m@s@@QAAXPBUI@@@Z"

  void m(I &) {}
  // CHECK-LABEL: "\01?m@s@@QAAXAAUI@@@Z"

  void m(const I &) {}
  // CHECK-LABEL: "\01?m@s@@QAAXABUI@@@Z"

  void m(I &&) {}
  // CHECK-LABEL: "\01?m@s@@QAAX$$QAUI@@@Z"

  void m(const I &&) {}
  // CHECK-LABEL: "\01?m@s@@QAAX$$QBUI@@@Z"

  void m(id) {}
  // CHECK-LABEL: "\01?m@s@@QAAXPAUobjc_object@@@Z"

  void m(id &) {}
  // CHECK-LABEL: "\01?m@s@@QAAXAAPAUobjc_object@@@Z"

  void m(id &&) {}
  // CHECK-LABEL: "\01?m@s@@QAAX$$QAPAUobjc_object@@@Z"

  void m(const id &) {}
  // CHECK-LABEL: "\01?m@s@@QAAXABPAUobjc_object@@@Z"

  void m(const id &&) {}
  // CHECK-LABEL: "\01?m@s@@QAAX$$QBPAUobjc_object@@@Z"

  void m(Class *) {}
  // CHECK-LABEL: "\01?m@s@@QAAXPAPAUobjc_class@@@Z"

  void m(const Class *) {}
  // CHECK-LABEL: "\01?m@s@@QAAXPBPAUobjc_class@@@Z"

  void m(Class) {}
  // CHECK-LABEL: "\01?m@s@@QAAXPAUobjc_class@@@Z"

  void m(Class &) {}
  // CHECK-LABEL: "\01?m@s@@QAAXAAPAUobjc_class@@@Z"

  void m(const Class &) {}
  // CHECK-LABEL: "\01?m@s@@QAAXABPAUobjc_class@@@Z"

  void m(Class &&) {}
  // CHECK-LABEL: "\01?m@s@@QAAX$$QAPAUobjc_class@@@Z"

  void m(const Class &&) {}
  // CHECK-LABEL: "\01?m@s@@QAAX$$QBPAUobjc_class@@@Z"
};

