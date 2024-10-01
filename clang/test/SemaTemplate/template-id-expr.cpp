// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++03 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s
// PR5336
template<typename FromCl>
struct isa_impl_cl {
 template<class ToCl>
 static void isa(const FromCl &Val) { }
};

template<class X, class Y>
void isa(const Y &Val) {   return isa_impl_cl<Y>::template isa<X>(Val); }

class Value;
void f0(const Value &Val) { isa<Value>(Val); }

// Implicit template-ids.
template<typename T>
struct X0 {
  template<typename U>
  void f1();
  
  template<typename U>
  void f2(U) {
    f1<U>();
  }
};

void test_X0_int(X0<int> xi, float f) {
  xi.f2(f);
}

// Not template-id expressions, but they almost look like it.
template<typename F>
struct Y {
  Y(const F&);
};

template<int I>
struct X {
  X(int, int);
  void f() { 
    Y<X<I> >(X<I>(0, 0)); 
    Y<X<I> >(::X<I>(0, 0)); 
  }
};

template struct X<3>;

// 'template' as a disambiguator.
// PR7030
struct Y0 {
  template<typename U>
  void f1(U);

  template<typename U>
  static void f2(U);

  void f3(int); // expected-note 2{{declared as a non-template here}}

  static int f4(int);
  template<typename U>
  static void f4(U);

  template<typename U>
  void f() {
    Y0::template f1<U>(0);
    Y0::template f1(0); // expected-error {{a template argument list is expected after a name prefixed by the template keyword}}
    this->template f1(0); // expected-error {{a template argument list is expected after a name prefixed by the template keyword}}

    Y0::template f2<U>(0);
    Y0::template f2(0); // expected-error {{a template argument list is expected after a name prefixed by the template keyword}}

    Y0::template f3(0); // expected-error {{'f3' following the 'template' keyword does not refer to a template}}
    Y0::template f3(); // expected-error {{'f3' following the 'template' keyword does not refer to a template}}

    int x;
    x = Y0::f4(0);
    x = Y0::f4<int>(0); // expected-error {{assigning to 'int' from incompatible type 'void'}}
    x = Y0::template f4(0); // expected-error {{a template argument list is expected after a name prefixed by the template keyword}} expected-error {{assigning to 'int' from incompatible type 'void'}}

    x = this->f4(0);
    x = this->f4<int>(0); // expected-error {{assigning to 'int' from incompatible type 'void'}}
    x = this->template f4(0); // expected-error {{a template argument list is expected after a name prefixed by the template keyword}} expected-error {{assigning to 'int' from incompatible type 'void'}}
  }
};

template<typename U> void Y0
  ::template // expected-error {{expected unqualified-id}}
    f1(U) {}

// FIXME: error recovery is awful without this.
    ;

template<typename T>
struct Y1 {
  template<typename U>
  void f1(U);

  template<typename U>
  static void f2(U);

  void f3(int); // expected-note 4{{declared as a non-template here}}

  static int f4(int);
  template<typename U>
  static void f4(U);

  template<typename U>
  void f() {
    Y1::template f1<U>(0);
    Y1::template f1(0); // expected-error {{a template argument list is expected after a name prefixed by the template keyword}}
    this->template f1(0); // expected-error {{a template argument list is expected after a name prefixed by the template keyword}}

    Y1::template f2<U>(0);
    Y1::template f2(0); // expected-error {{a template argument list is expected after a name prefixed by the template keyword}}

    Y1::template f3(0); // expected-error {{'f3' following the 'template' keyword does not refer to a template}}
    Y1::template f3(); // expected-error {{'f3' following the 'template' keyword does not refer to a template}}

    int x;
    x = Y1::f4(0);
    x = Y1::f4<int>(0); // expected-error {{assigning to 'int' from incompatible type 'void'}}
    x = Y1::template f4(0); // expected-error {{a template argument list is expected after a name prefixed by the template keyword}} expected-error {{assigning to 'int' from incompatible type 'void'}}

    x = this->f4(0);
    x = this->f4<int>(0); // expected-error {{assigning to 'int' from incompatible type 'void'}}
    x = this->template f4(0); // expected-error {{a template argument list is expected after a name prefixed by the template keyword}} expected-error {{assigning to 'int' from incompatible type 'void'}}
  }
};

void use_Y1(Y1<int> y1) { y1.f<int>(); } // expected-note {{in instantiation of}}

template<typename T>
struct Y2 : Y1<T> {
  typedef ::Y1<T> Y1;

  template<typename U>
  void f(Y1 *p) {
    Y1::template f1<U>(0);
    Y1::template f1(0); // expected-error {{a template argument list is expected after a name prefixed by the template keyword}}
    p->template f1(0); // expected-error {{a template argument list is expected after a name prefixed by the template keyword}}

    Y1::template f2<U>(0);
    Y1::template f2(0); // expected-error {{a template argument list is expected after a name prefixed by the template keyword}}

    Y1::template f3(0); // expected-error {{'f3' following the 'template' keyword does not refer to a template}} expected-error {{a template argument list is expected after a name prefixed by the template keyword}}
    Y1::template f3(); // expected-error {{'f3' following the 'template' keyword does not refer to a template}} expected-error {{a template argument list is expected after a name prefixed by the template keyword}}

    int x;
    x = Y1::f4(0);
    x = Y1::f4<int>(0); // expected-error {{use 'template'}} expected-error {{assigning to 'int' from incompatible type 'void'}}
    x = Y1::template f4(0); // expected-error {{assigning to 'int' from incompatible type 'void'}} expected-error {{a template argument list is expected after a name prefixed by the template keyword}}

    x = p->f4(0);
    x = p->f4<int>(0); // expected-error {{assigning to 'int' from incompatible type 'void'}} expected-error {{use 'template'}}
    x = p->template f4(0); // expected-error {{assigning to 'int' from incompatible type 'void'}} expected-error {{a template argument list is expected after a name prefixed by the template keyword}}
  }
};

void use_Y2(Y2<int> y2) { y2.f<int>(0); } // expected-note {{in instantiation of}}

struct A {
  template<int I>
  struct B {
    static void b1(); // expected-note {{declared as a non-template here}}
  };
};

template<int I>
void f5() {
  A::template B<I>::template b1(); // expected-error {{'b1' following the 'template' keyword does not refer to a template}} expected-error {{a template argument list is expected after a name prefixed by the template keyword}}
}

template void f5<0>(); // expected-note {{in instantiation of function template specialization 'f5<0>' requested here}}

class C {};
template <template <typename> class D>
class E {
  template class D<C>;  // expected-error {{expected '<' after 'template'}}
  template<> class D<C>;  // expected-error {{cannot specialize a template template parameter}}
  friend class D<C>; // expected-error {{alias template 'D' cannot be referenced with the 'class' specifier}}
};
#if __cplusplus <= 199711L
// expected-warning@+2 {{extension}}
#endif
template<typename T> using D = int; // expected-note {{declared here}} 
E<D> ed; // expected-note {{instantiation of}}

namespace non_functions {

#if __cplusplus >= 201103L
namespace PR88832 {
template <typename T> struct O {
  static const T v = 0;
};

struct P {
  template <typename T> using I = typename O<T>::v; // #TypeAlias
};

struct Q {
  template <typename T> int foo() {
    return T::template I<int>;
    // expected-error@-1 {{'P::I' is expected to be a non-type template, but instantiated to a type alias template}}
    // expected-note@#TypeAlias {{type alias template declared here}}
  }
};

int bar() {
  return Q().foo<P>(); // expected-note-re {{function template specialization {{.*}} requested here}}
}

} // namespace PR88832
#endif

namespace PR63243 {

namespace std {
template <class T> struct add_pointer { // #add_pointer
};
} // namespace std

class A {};

int main() {
  std::__add_pointer<A>::type ptr;
  // expected-warning@-1 {{keyword '__add_pointer' will be made available as an identifier here}}
  // expected-error@-2 {{no template named '__add_pointer'}}
  // expected-note@#add_pointer {{'add_pointer' declared here}}
  // expected-error-re@-4 {{no type named 'type' in '{{.*}}std::add_pointer<{{.*}}A>'}}

  __add_pointer<A>::type ptr2;
  // expected-error@-1 {{no template named '__add_pointer'}}
  // expected-error-re@-2 {{no type named 'type' in '{{.*}}std::add_pointer<{{.*}}A>'}}
  // expected-note@#add_pointer {{'std::add_pointer' declared here}}
}

} // namespace PR63243

namespace PR48673 {

template <typename T> struct C {
  template <int TT> class Type {}; // #ClassTemplate
};

template <typename T1> struct A {

  template <typename T2>
  void foo(T2) {}

  void foo() {
    C<T1>::template Type<2>;
    // expected-error@-1 {{'C<float>::Type' is expected to be a non-type template, but instantiated to a class template}}}
    // expected-note@#ClassTemplate {{class template declared here}}

    foo(C<T1>::Type<2>); // expected-error {{expected expression}}

    foo(C<T1>::template Type<2>);
    // expected-error@-1 {{'C<float>::Type' is expected to be a non-type template, but instantiated to a class template}}
    // expected-note@#ClassTemplate {{class template declared here}}

    foo(C<T1>::template Type<2>());
    // expected-error@-1 {{'C<float>::Type' is expected to be a non-type template, but instantiated to a class template}}
    // expected-error@-2 {{called object type '<dependent type>' is not a function or function pointer}}
    // expected-note@#ClassTemplate {{class template declared here}}

    foo(typename C<T1>::template Type<2>());
  }
};

void test() {
  A<float>().foo(); // expected-note-re {{instantiation of member function {{.*}} requested here}}
}

} // namespace PR48673

}
