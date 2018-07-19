// RUN: %clang_cc1 -fsyntax-only -fms-extensions -std=c++11 -verify %s

namespace A {

template <class T>
class ClassTemplate; // expected-note {{explicitly specialized declaration is here}}

template <class T1, class T2>
class ClassTemplatePartial; // expected-note {{explicitly specialized declaration is here}}

template <typename T> struct X {
  struct MemberClass; // expected-note {{explicitly specialized declaration is here}}
  enum MemberEnumeration; // expected-note {{explicitly specialized declaration is here}} // expected-error {{ISO C++ forbids forward references to 'enum' types}}
};

}

namespace B {

template <>
class A::ClassTemplate<int>; // expected-warning {{class template specialization of 'ClassTemplate' outside namespace enclosing 'A' is a Microsoft extension}}

template <class T1>
class A::ClassTemplatePartial<T1, T1 *> {}; // expected-warning {{class template partial specialization of 'ClassTemplatePartial' outside namespace enclosing 'A' is a Microsoft extension}}

template <>
struct A::X<int>::MemberClass; // expected-warning {{member class specialization of 'MemberClass' outside namespace enclosing 'A' is a Microsoft extension}}

template <>
enum A::X<int>::MemberEnumeration; // expected-warning {{member enumeration specialization of 'MemberEnumeration' outside namespace enclosing 'A' is a Microsoft extension}} // expected-error {{ISO C++ forbids forward references to 'enum' types}}

}

