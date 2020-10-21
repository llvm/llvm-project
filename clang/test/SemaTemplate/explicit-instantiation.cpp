// RUN: %clang_cc1 -fsyntax-only -verify -fexceptions -fcxx-exceptions %s
// RUN: %clang_cc1 -fsyntax-only -verify -fexceptions -fcxx-exceptions -std=c++11 %s

template void *; // expected-error{{expected unqualified-id}}

template typedef void f0; // expected-error{{explicit instantiation of typedef}}

int v0; // expected-note{{refers here}}
template int v0; // expected-error{{does not refer}}

template<typename T>
struct X0 {
  static T value;
  
  T f0(T x) {
    return x + 1;  // expected-error{{invalid operands}}
  }
  T *f0(T *, T *) { return T(); } // expected-warning 0-1 {{expression which evaluates to zero treated as a null pointer constant of type 'int *'}} expected-error 0-1 {{cannot initialize return object of type 'int *' with an rvalue of type 'int'}}

  template <typename U> T f0(T, U) { return T(); } // expected-note-re {{candidate template ignored: could not match 'int (int, U){{( __attribute__\(\(thiscall\)\))?}}' against 'int (int){{( __attribute__\(\(thiscall\)\))?}} const'}} \
                                                   // expected-note {{candidate template ignored: could not match 'int' against 'int *'}}
};

template<typename T>
T X0<T>::value; // expected-error{{no matching constructor}}

template int X0<int>::value;

struct NotDefaultConstructible { // expected-note{{candidate constructor (the implicit copy constructor)}} expected-note 0-1 {{candidate constructor (the implicit move constructor)}}
  NotDefaultConstructible(int); // expected-note{{candidate constructor}}
};

template NotDefaultConstructible X0<NotDefaultConstructible>::value; // expected-note{{instantiation}}

template int X0<int>::f0(int);
template int* X0<int>::f0(int*, int*); // expected-note{{in instantiation of member function 'X0<int>::f0' requested here}}
template int X0<int>::f0(int, float);

template int X0<int>::f0(int) const; // expected-error{{does not refer}}
template int* X0<int>::f0(int*, float*); // expected-error{{does not refer}}

struct X1 { };
typedef int X1::*MemPtr;

template MemPtr X0<MemPtr>::f0(MemPtr); // expected-note{{requested here}}

struct X2 {
  int f0(int); // expected-note{{refers here}}
  
  template<typename T> T f1(T) { return T(); }
  template<typename T> T* f1(T*) { return 0; }

  template<typename T, typename U> void f2(T, U*) { } // expected-note{{candidate}}
  template<typename T, typename U> void f2(T*, U) { } // expected-note{{candidate}}
};

template int X2::f0(int); // expected-error{{not an instantiation}}

template int *X2::f1(int *); // okay

template void X2::f2(int *, int *); // expected-error{{ambiguous}}

template <typename T>
void print_type() {} // expected-note {{candidate template ignored: could not match 'void ()' against 'void (float *)'}}

template void print_type<int>();
template void print_type<float>();

template <typename T>
void print_type(T *) {} // expected-note {{candidate template ignored: could not match 'void (int *)' against 'void (float *)'}}

template void print_type(int*);
template void print_type<int>(float*); // expected-error{{does not refer}}

void print_type(double*);
template void print_type<double>(double*);

// PR5069
template<int I> void foo0 (int (&)[I + 1]) { }
template void foo0<2> (int (&)[3]);

namespace explicit_instantiation_after_implicit_instantiation {
  template <int I> struct X0 { static int x; };
  template <int I> int X0<I>::x;
  void test1() { (void)&X0<1>::x; }
  template struct X0<1>;
}

template<typename> struct X3 { };
inline template struct X3<int>; // expected-warning{{ignoring 'inline' keyword on explicit template instantiation}}
static template struct X3<float>; // expected-warning{{ignoring 'static' keyword on explicit template instantiation}}

namespace PR7622 {
  template<typename,typename=int>
  struct basic_streambuf;

  template<typename,typename>
  struct basic_streambuf{friend bob<>()}; // expected-error{{no template named 'bob'}} \
                                          // expected-error{{expected member name or ';' after declaration specifiers}}
  template struct basic_streambuf<int>;
}

// Test that we do not crash.
class TC1 {
  class TC2 {
    template
    void foo() { } // expected-error{{expected '<' after 'template'}}
   };
};

namespace PR8020 {
  template <typename T> struct X { X() {} };
  template<> struct X<int> { X(); };
  template X<int>::X() {}  // expected-error{{function cannot be defined in an explicit instantiation}}
}

namespace PR10086 {
  template void foobar(int i) {}  // expected-error{{function cannot be defined in an explicit instantiation}}
  int func() {
    foobar(5);
  }
}

namespace undefined_static_data_member {
  template<typename T> struct A {
    static int a; // expected-note {{here}}
    template<typename U> static int b; // expected-note {{here}} expected-warning 0+ {{extension}}
  };
  struct B {
    template<typename U> static int c; // expected-note {{here}} expected-warning 0+ {{extension}}
  };

  template int A<int>::a; // expected-error {{explicit instantiation of undefined static data member 'a' of class template 'undefined_static_data_member::A<int>'}}
  template int A<int>::b<int>; // expected-error {{explicit instantiation of undefined variable template 'undefined_static_data_member::A<int>::b<int>'}}
  template int B::c<int>; // expected-error {{explicit instantiation of undefined variable template 'undefined_static_data_member::B::c<int>'}}


  template<typename T> struct C {
    static int a;
    template<typename U> static int b; // expected-warning 0+ {{extension}}
  };
  struct D {
    template<typename U> static int c; // expected-warning 0+ {{extension}}
  };
  template<typename T> int C<T>::a;
  template<typename T> template<typename U> int C<T>::b; // expected-warning 0+ {{extension}}
  template<typename U> int D::c; // expected-warning 0+ {{extension}}

  template int C<int>::a;
  template int C<int>::b<int>;
  template int D::c<int>;
}

// expected-note@+1 3-4 {{explicit instantiation refers here}}
template <class T> void Foo(T i) throw(T) { throw i; }
// expected-error@+1 {{exception specification in explicit instantiation does not match instantiated one}}
template void Foo(int a) throw(char);
// expected-error@+1 {{exception specification in explicit instantiation does not match instantiated one}}
template void Foo(double a) throw();
// expected-error@+1 1 {{exception specification in explicit instantiation does not match instantiated one}}
template void Foo(long a) throw(long, char);
template void Foo(float a);
#if __cplusplus >= 201103L
// expected-error@+1 0-1 {{exception specification in explicit instantiation does not match instantiated one}}
template void Foo(double a) noexcept;
#endif

#if __cplusplus >= 201103L
namespace PR21942 {
template <int>
struct A {
  virtual void foo() final;
};

template <>
void A<0>::foo() {} // expected-note{{overridden virtual function is here}}

struct B : A<0> {
  virtual void foo() override; // expected-error{{declaration of 'foo' overrides a 'final' function}}
};
}

template<typename T> struct LambdaInDefaultMemberInitInExplicitInstantiation {
  int a = [this] { return a; }();
};
template struct LambdaInDefaultMemberInitInExplicitInstantiation<int>;
LambdaInDefaultMemberInitInExplicitInstantiation<float> x;
#endif
