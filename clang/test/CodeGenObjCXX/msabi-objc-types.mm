// RUN: %clang_cc1 -triple thumbv7-windows-msvc -fdeclspec -std=c++11 -fobjc-runtime=ios-6.0 -o - -emit-llvm %s | FileCheck %s

@class I;

id kid;
// CHECK: @"?kid@@3PAU.objc_object@@A" =  dso_local global

Class klass;
// CHECK: @"?klass@@3PAU.objc_class@@A" = dso_local global

I *kI;
// CHECK: @"?kI@@3PAU.objc_cls_I@@A" = dso_local global

void f(I *) {}
// CHECK-LABEL: "?f@@YAXPAU.objc_cls_I@@@Z"

void f(const I *) {}
// CHECK-LABEL: "?f@@YAXPBU.objc_cls_I@@@Z"

void f(I &) {}
// CHECK-LABEL: "?f@@YAXAAU.objc_cls_I@@@Z"

void f(const I &) {}
// CHECK-LABEL: "?f@@YAXABU.objc_cls_I@@@Z"

void f(const I &&) {}
// CHECK-LABEL: "?f@@YAX$$QBU.objc_cls_I@@@Z"

void g(id) {}
// CHECK-LABEL: "?g@@YAXPAU.objc_object@@@Z"

void g(id &) {}
// CHECK-LABEL: "?g@@YAXAAPAU.objc_object@@@Z"

void g(const id &) {}
// CHECK-LABEL: "?g@@YAXABQAU.objc_object@@@Z"

void g(id &&) {}
// CHECK-LABEL: "?g@@YAX$$QAPAU.objc_object@@@Z"

void h(Class) {}
// CHECK-LABEL: "?h@@YAXPAU.objc_class@@@Z"

void h(Class &) {}
// CHECK-LABEL: "?h@@YAXAAPAU.objc_class@@@Z"

void h(const Class &) {}
// CHECK-LABEL: "?h@@YAXABQAU.objc_class@@@Z"

void h(Class &&) {}
// CHECK-LABEL: "?h@@YAX$$QAPAU.objc_class@@@Z"

I *i() { return nullptr; }
// CHECK-LABEL: "?i@@YAPAU.objc_cls_I@@XZ"

const I *j() { return nullptr; }
// CHECK-LABEL: "?j@@YAPBU.objc_cls_I@@XZ"

I &k() { return *kI; }
// CHECK-LABEL: "?k@@YAAAU.objc_cls_I@@XZ"

const I &l() { return *kI; }
// CHECK-LABEL: "?l@@YAABU.objc_cls_I@@XZ"

void m(const id) {}
// CHECK-LABEL: "?m@@YAXQAU.objc_object@@@Z"

void m(const I *) {}
// CHECK-LABEL: "?m@@YAXPBU.objc_cls_I@@@Z"

void n(SEL) {}
// CHECK-LABEL: "?n@@YAXPAU.objc_selector@@@Z"

void n(SEL *) {}
// CHECK-LABEL: "?n@@YAXPAPAU.objc_selector@@@Z"

void n(const SEL *) {}
// CHECK-LABEL: "?n@@YAXPBQAU.objc_selector@@@Z"

void n(SEL &) {}
// CHECK-LABEL: "?n@@YAXAAPAU.objc_selector@@@Z"

void n(const SEL &) {}
// CHECK-LABEL: "?n@@YAXABQAU.objc_selector@@@Z"

void n(SEL &&) {}
// CHECK-LABEL: "?n@@YAX$$QAPAU.objc_selector@@@Z"

struct __declspec(dllexport) s {
  struct s &operator=(const struct s &) = delete;

  void m(I *) {}
  // CHECK-LABEL: "?m@s@@QAAXPAU.objc_cls_I@@@Z"

  void m(const I *) {}
  // CHECK-LABEL: "?m@s@@QAAXPBU.objc_cls_I@@@Z"

  void m(I &) {}
  // CHECK-LABEL: "?m@s@@QAAXAAU.objc_cls_I@@@Z"

  void m(const I &) {}
  // CHECK-LABEL: "?m@s@@QAAXABU.objc_cls_I@@@Z"

  void m(I &&) {}
  // CHECK-LABEL: "?m@s@@QAAX$$QAU.objc_cls_I@@@Z"

  void m(const I &&) {}
  // CHECK-LABEL: "?m@s@@QAAX$$QBU.objc_cls_I@@@Z"

  void m(id) {}
  // CHECK-LABEL: "?m@s@@QAAXPAU.objc_object@@@Z"

  void m(id &) {}
  // CHECK-LABEL: "?m@s@@QAAXAAPAU.objc_object@@@Z"

  void m(id &&) {}
  // CHECK-LABEL: "?m@s@@QAAX$$QAPAU.objc_object@@@Z"

  void m(const id &) {}
  // CHECK-LABEL: "?m@s@@QAAXABQAU.objc_object@@@Z"

  void m(const id &&) {}
  // CHECK-LABEL: "?m@s@@QAAX$$QBQAU.objc_object@@@Z"

  void m(Class *) {}
  // CHECK-LABEL: "?m@s@@QAAXPAPAU.objc_class@@@Z"

  void m(const Class *) {}
  // CHECK-LABEL: "?m@s@@QAAXPBQAU.objc_class@@@Z"

  void m(Class) {}
  // CHECK-LABEL: "?m@s@@QAAXPAU.objc_class@@@Z"

  void m(Class &) {}
  // CHECK-LABEL: "?m@s@@QAAXAAPAU.objc_class@@@Z"

  void m(const Class &) {}
  // CHECK-LABEL: "?m@s@@QAAXABQAU.objc_class@@@Z"

  void m(Class &&) {}
  // CHECK-LABEL: "?m@s@@QAAX$$QAPAU.objc_class@@@Z"

  void m(const Class &&) {}
  // CHECK-LABEL: "?m@s@@QAAX$$QBQAU.objc_class@@@Z"

  void m(SEL) {}
  // CHECK-LABEL: "?m@s@@QAAXPAU.objc_selector@@@Z"

  void m(SEL *) {}
  // CHECK-LABEL: "?m@s@@QAAXPAPAU.objc_selector@@@Z"

  void m(const SEL *) {}
  // CHECK-LABEL: "?m@s@@QAAXPBQAU.objc_selector@@@Z"

  void m(SEL &) {}
  // CHECK-LABEL: "?m@s@@QAAXAAPAU.objc_selector@@@Z"

  void m(const SEL &) {}
  // CHECK-LABEL: "?m@s@@QAAXABQAU.objc_selector@@@Z"

  void m(SEL &&) {}
  // CHECK-LABEL: "?m@s@@QAAX$$QAPAU.objc_selector@@@Z"

  void m(const SEL &&) {}
  // CHECK-LABEL: "?m@s@@QAAX$$QBQAU.objc_selector@@@Z"
};

template <typename T>
struct remove_pointer { typedef T type; };

template <typename T>
struct remove_pointer<T *> {
  typedef T type;
};

template <typename T>
struct t {
  t() {}
};

template struct t<id>;
// CHECK-LABEL: "??0?$t@PAU.objc_object@@@@QAA@XZ"

template struct t<remove_pointer<id>::type>;
// CHECK-LABEL: "??0?$t@U.objc_object@@@@QAA@XZ"

template struct t<SEL>;
// CHECK-LABEL: "??0?$t@PAU.objc_selector@@@@QAA@XZ"

template struct t<remove_pointer<SEL>::type>;
// CHECK-LABEL: "??0?$t@U.objc_selector@@@@QAA@XZ"

