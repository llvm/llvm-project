// RUN: mkdir -p %T/move-template-class
// RUN: cp %S/Inputs/template_class_test*  %T/move-template-class
// RUN: cd %T/move-template-class
// RUN: clang-move -names="A,B" -new_cc=%T/move-template-class/new_template_class_test.cpp -new_header=%T/move-template-class/new_template_class_test.h -old_cc=%T/move-template-class/template_class_test.cpp -old_header=../move-template-class/template_class_test.h %T/move-template-class/template_class_test.cpp --
// RUN: FileCheck -input-file=%T/move-template-class/template_class_test.cpp -check-prefix=CHECK-OLD-TEST-EMPTY -allow-empty %s
// RUN: FileCheck -input-file=%T/move-template-class/template_class_test.h -check-prefix=CHECK-OLD-TEST-EMPTY -allow-empty %s
// RUN: FileCheck -input-file=%T/move-template-class/new_template_class_test.cpp -check-prefix=CHECK-NEW-TEST-CPP-CASE1 %s
// RUN: FileCheck -input-file=%T/move-template-class/new_template_class_test.h -check-prefix=CHECK-NEW-TEST-H-CASE1 %s
//
// RUN: cp %S/Inputs/template_class_test*  %T/move-template-class
// RUN: clang-move -names="A" -new_cc=%T/move-template-class/new_template_class_test.cpp -new_header=%T/move-template-class/new_template_class_test.h -old_cc=%T/move-template-class/template_class_test.cpp -old_header=../move-template-class/template_class_test.h %T/move-template-class/template_class_test.cpp --
// RUN: FileCheck -input-file=%T/move-template-class/template_class_test.h -check-prefix=CHECK-OLD-TEST-H-CASE2 %s
// RUN: FileCheck -input-file=%T/move-template-class/template_class_test.cpp -check-prefix=CHECK-OLD-TEST-CPP-CASE2 %s
// RUN: FileCheck -input-file=%T/move-template-class/new_template_class_test.h -check-prefix=CHECK-NEW-TEST-H-CASE2 %s
// RUN: FileCheck -input-file=%T/move-template-class/new_template_class_test.cpp -check-prefix=CHECK-NEW-TEST-CPP-CASE2 %s
//
//
// CHECK-OLD-TEST-EMPTY: {{^}}{{$}}
//
// CHECK-NEW-TEST-H-CASE1: #ifndef TEMPLATE_CLASS_TEST_H // comment 1
// CHECK-NEW-TEST-H-CASE1: #define TEMPLATE_CLASS_TEST_H
// CHECK-NEW-TEST-H-CASE1: template <typename T>
// CHECK-NEW-TEST-H-CASE1: class A {
// CHECK-NEW-TEST-H-CASE1:  public:
// CHECK-NEW-TEST-H-CASE1:   void f();
// CHECK-NEW-TEST-H-CASE1:   void g();
// CHECK-NEW-TEST-H-CASE1:   template <typename U> void h();
// CHECK-NEW-TEST-H-CASE1:   template <typename U> void k();
// CHECK-NEW-TEST-H-CASE1:   static int b;
// CHECK-NEW-TEST-H-CASE1:   static int c;
// CHECK-NEW-TEST-H-CASE1: };
// CHECK-NEW-TEST-H-CASE1: template <typename T>
// CHECK-NEW-TEST-H-CASE1: void A<T>::f() {}
// CHECK-NEW-TEST-H-CASE1: template <typename T>
// CHECK-NEW-TEST-H-CASE1: template <typename U>
// CHECK-NEW-TEST-H-CASE1: void A<T>::h() {}
// CHECK-NEW-TEST-H-CASE1: template <typename T>
// CHECK-NEW-TEST-H-CASE1: int A<T>::b = 2;
// CHECK-NEW-TEST-H-CASE1: class B {
// CHECK-NEW-TEST-H-CASE1:  public:
// CHECK-NEW-TEST-H-CASE1:   void f();
// CHECK-NEW-TEST-H-CASE1: };
// CHECK-NEW-TEST-H-CASE1: #endif // TEMPLATE_CLASS_TEST_H
//
// CHECK-NEW-TEST-CPP-CASE1: #include "{{.*}}new_template_class_test.h"
// CHECK-NEW-TEST-CPP-CASE1: template <typename T>
// CHECK-NEW-TEST-CPP-CASE1: void A<T>::g() {}
// CHECK-NEW-TEST-CPP-CASE1: template <typename T>
// CHECK-NEW-TEST-CPP-CASE1: template <typename U>
// CHECK-NEW-TEST-CPP-CASE1: void A<T>::k() {}
// CHECK-NEW-TEST-CPP-CASE1: template <typename T>
// CHECK-NEW-TEST-CPP-CASE1: int A<T>::c = 2;
// CHECK-NEW-TEST-CPP-CASE1: void B::f() {}
//
// CHECK-OLD-TEST-H-CASE2: #ifndef TEMPLATE_CLASS_TEST_H // comment 1
// CHECK-OLD-TEST-H-CASE2: #define TEMPLATE_CLASS_TEST_H
// CHECK-OLD-TEST-H-CASE2: class B {
// CHECK-OLD-TEST-H-CASE2:  public:
// CHECK-OLD-TEST-H-CASE2:   void f();
// CHECK-OLD-TEST-H-CASE2: };
// CHECK-OLD-TEST-H-CASE2: #endif // TEMPLATE_CLASS_TEST_H
//
// CHECK-OLD-TEST-CPP-CASE2: #include "template_class_test.h"
// CHECK-OLD-TEST-CPP-CASE2:  void B::f() {}
//
// CHECK-NEW-TEST-H-CASE2: #ifndef {{.*}}NEW_TEMPLATE_CLASS_TEST_H
// CHECK-NEW-TEST-H-CASE2: #define {{.*}}NEW_TEMPLATE_CLASS_TEST_H
// CHECK-NEW-TEST-H-CASE2: template <typename T>
// CHECK-NEW-TEST-H-CASE2: class A {
// CHECK-NEW-TEST-H-CASE2:  public:
// CHECK-NEW-TEST-H-CASE2:   void f();
// CHECK-NEW-TEST-H-CASE2:   void g();
// CHECK-NEW-TEST-H-CASE2:   template <typename U> void h();
// CHECK-NEW-TEST-H-CASE2:   template <typename U> void k();
// CHECK-NEW-TEST-H-CASE2:   static int b;
// CHECK-NEW-TEST-H-CASE2:   static int c;
// CHECK-NEW-TEST-H-CASE2: };
// CHECK-NEW-TEST-H-CASE2: template <typename T> void A<T>::f() {}
// CHECK-NEW-TEST-H-CASE2: template <typename T> template <typename U> void A<T>::h() {}
// CHECK-NEW-TEST-H-CASE2: template <typename T> int A<T>::b = 2;
// CHECK-NEW-TEST-H-CASE2: #endif // {{.*}}NEW_TEMPLATE_CLASS_TEST_H
//
// CHECK-NEW-TEST-CPP-CASE2: #include "{{.*}}new_template_class_test.h"
// CHECK-NEW-TEST-CPP-CASE2: template <typename T> void A<T>::g() {}
// CHECK-NEW-TEST-CPP-CASE2: template <typename T> template <typename U> void A<T>::k() {}
// CHECK-NEW-TEST-CPP-CASE2: template <typename T> int A<T>::c = 2;
