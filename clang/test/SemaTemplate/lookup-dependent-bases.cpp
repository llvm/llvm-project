// RUN: %clang_cc1 -fms-extensions -fsyntax-only -verify %s
// expected-no-diagnostics

class C {
public:
   static void foo2() {  }
};
template <class T>
class A {
public:
   typedef C D;
};

template <class T>
class B : public A<T> {
public:
   void foo() {
    D::foo2();
   }
};
