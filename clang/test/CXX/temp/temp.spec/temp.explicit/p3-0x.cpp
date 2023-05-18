// RUN: %clang_cc1 -std=c++11 -verify %s
// expected-no-diagnostics

// If the name declared in the explicit instantiation is an
// unqualified name, the explicit instantiation shall appear in the
// namespace where its template is declared or, if that namespace is
// inline (7.3.1), any namespace from its enclosing namespace set.

namespace has_inline_namespaces {
  inline namespace inner {
    template<class T> void f(T&) {}

    template<class T> 
    struct X0 {
      struct MemberClass {};

      void mem_func() {}

      template<typename U>
      struct MemberClassTemplate {};

      template<typename U>
      void mem_func_template(U&) {}

      static int value;
    };
  }

  template<typename T> int X0<T>::value = 17;

  struct X1 {};
  struct X2 {};

  template void f(X1&);
  template void f<X2>(X2&);

  template struct X0<X1>;

  template struct X0<X2>::MemberClass;

  template void X0<X2>::mem_func();

  template struct X0<X2>::MemberClassTemplate<X1>;

  template void X0<X2>::mem_func_template(X1&);

  template int X0<X2>::value;
}

struct X3;
struct X4;

template void has_inline_namespaces::f(X3&);
template void has_inline_namespaces::f<X4>(X4&);

template struct has_inline_namespaces::X0<X3>;

template struct has_inline_namespaces::X0<X4>::MemberClass;

template void has_inline_namespaces::X0<X4>::mem_func();

template 
struct has_inline_namespaces::X0<X4>::MemberClassTemplate<X3>;

template
void has_inline_namespaces::X0<X4>::mem_func_template(X3&);
