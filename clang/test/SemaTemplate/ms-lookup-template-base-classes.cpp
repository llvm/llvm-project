// RUN: %clang_cc1 -fcxx-exceptions -fexceptions -std=c++1y -fms-compatibility -fno-spell-checking -fsyntax-only -verify %s
// RUN: %clang_cc1 -fcxx-exceptions -fexceptions -std=c++20 -fms-compatibility -fno-spell-checking -fsyntax-only -verify %s


template <class T>
class A {
public:
   void f(T a) { }// expected-note 2{{must qualify identifier to find this declaration in dependent base class}}
   void g();// expected-note 2{{must qualify identifier to find this declaration in dependent base class}}
};

template <class T>
class B : public A<T> {
public:
	void z(T a)
    {
       f(a); // expected-warning 2{{use of member 'f' found via unqualified lookup into dependent bases of class templates is a Microsoft extension}}
       g(); // expected-warning 2{{use of member 'g' found via unqualified lookup into dependent bases of class templates is a Microsoft extension}}
    }
};

template class B<int>; // expected-note {{requested here}}
template class B<char>; // expected-note {{requested here}}

void test()
{
    B<int> b;
    b.z(3);
}

struct A2 {
  template<class T> void f(T) {
    XX; //expected-error {{use of undeclared identifier 'XX'}}
    A2::XX; //expected-error {{no member named 'XX' in 'A2'}}
  }
};
template void A2::f(int);

template<class T0>
struct A3 {
  template<class T1> void f(T1) {
    XX; //expected-error {{use of undeclared identifier 'XX'}}
  }
};
template void A3<int>::f(int);

template<class T0>
struct A4 {
  void f(char) {
    XX; //expected-error {{use of undeclared identifier 'XX'}}
  }
};
template class A4<int>;


namespace lookup_dependent_bases_id_expr {

template<class T> class A {
public:
  int var;
};


template<class T>
class B : public A<T> {
public:
  void f() {
    var = 3; // expected-warning {{use of undeclared identifier 'var'; unqualified lookup into dependent bases of class template 'B' is a Microsoft extension}}
  }
};

template class B<int>;

}



namespace lookup_dependent_base_class_static_function {

template <class T>
class A {
public:
   static void static_func();// expected-note {{must qualify identifier to find this declaration in dependent base class}}
   void func();// expected-note {{must qualify identifier to find this declaration in dependent base class}}
};


template <class T>
class B : public A<T> {
public:
  static void z2(){
    static_func();  // expected-warning {{use of member 'static_func' found via unqualified lookup into dependent bases of class templates is a Microsoft extension}}
    func(); // expected-warning {{use of member 'func' found via unqualified lookup into dependent bases of class templates is a Microsoft extension}} expected-error {{call to non-static member function without an object argument}}
  }
};
template class B<int>; // expected-note {{requested here}}

} 



namespace lookup_dependent_base_class_default_argument {

template<class T>
class A {
public:
  static int f1(); // expected-note {{must qualify identifier to find this declaration in dependent base class}} 
  int f2(); // expected-note {{must qualify identifier to find this declaration in dependent base class}} 
};

template<class T>
class B : public A<T> {
public:
  void g1(int p = f1());// expected-warning {{use of member 'f1' found via unqualified lookup into dependent bases of class templates is a Microsoft extension}}
  void g2(int p = f2());// expected-warning {{use of member 'f2' found via unqualified lookup into dependent bases of class templates is a Microsoft extension}} expected-error {{call to non-static member function without an object argument}}
};

void foo()
{
	B<int> b;
	b.g1(); // expected-note {{required here}}
	b.g2(); // expected-note {{required here}}
}

}


namespace lookup_dependent_base_class_friend {

template <class T>
class B {
public:
  static void g();  // expected-note {{must qualify identifier to find this declaration in dependent base class}} 
};

template <class T>
class A : public B<T> {
public:
  friend void foo(A<T> p){
    g(); // expected-warning {{use of member 'g' found via unqualified lookup into dependent bases of class templates is a Microsoft extension}}
  }
};

int main2()
{
  A<int> a;
  foo(a); // expected-note {{requested here}}
}

}


namespace lookup_dependent_base_no_typo_correction {

class C {
public:
  int m_hWnd;
};

template <class T>
class A : public T {
public:
  void f(int hWnd) {
    m_hWnd = 1; // expected-warning {{use of undeclared identifier 'm_hWnd'; unqualified lookup into dependent bases of class template 'A' is a Microsoft extension}}
  }
};

template class A<C>;

}

namespace PR12701 {

class A {};
class B {};

template <class T>
class Base {
 public:
  bool base_fun(void* p) { return false; }  // expected-note {{must qualify identifier to find this declaration in dependent base class}}
  operator T*() const { return 0; }
};

template <class T>
class Container : public Base<T> {
 public:
  template <typename S>
  bool operator=(const Container<S>& rhs) {
    return base_fun(rhs);  // expected-warning {{use of member 'base_fun' found via unqualified lookup into dependent bases of class templates is a Microsoft extension}}
  }
};

void f() {
  Container<A> text_provider;
  Container<B> text_provider2;
  text_provider2 = text_provider;  // expected-note {{in instantiation of function template specialization}}
}

}  // namespace PR12701

namespace PR16014 {

struct A {
  int a;
  static int sa;
};
template <typename T> struct B : T {
  int     foo() { return a; }           // expected-warning {{lookup into dependent bases}}
  int    *bar() { return &a; }          // expected-warning {{lookup into dependent bases}}
  int     baz() { return T::a; }
  int T::*qux() { return &T::a; }
  static int T::*stuff() { return &T::a; }
  static int stuff1() { return T::sa; }
  static int *stuff2() { return &T::sa; }
  static int stuff3() { return sa; }    // expected-warning {{lookup into dependent bases}}
  static int *stuff4() { return &sa; }  // expected-warning {{lookup into dependent bases}}
};

template <typename T> struct C : T {
  int     foo() { return b; }      // expected-error {{no member named 'b' in 'PR16014::C<A>'}} expected-warning {{lookup into dependent bases}}
  int    *bar() { return &b; }     // expected-error {{no member named 'b' in 'PR16014::C<A>'}} expected-warning {{lookup into dependent bases}}
  int     baz() { return T::b; }   // expected-error {{no member named 'b' in 'PR16014::A'}}
  int T::*qux() { return &T::b; }  // expected-error {{no member named 'b' in 'PR16014::A'}}
  int T::*fuz() { return &U::a; }  // expected-error {{use of undeclared identifier 'U'}} \
  // expected-warning {{unqualified lookup into dependent bases of class template 'C'}}
};

template struct B<A>;
template struct C<A>;  // expected-note-re 1+ {{in instantiation of member function 'PR16014::C<PR16014::A>::{{.*}}' requested here}}

template <typename T> struct D : T {
  struct Inner {
    int foo() {
      // FIXME: MSVC can find this in D's base T!  Even worse, if ::sa exists,
      // clang will use it instead.
      return sa; // expected-error {{use of undeclared identifier 'sa'}}
    }
  };
};
template struct D<A>;

}

namespace PR19233 {
template <class T>
struct A : T {
  void foo() {
    ::undef(); // expected-error {{no member named 'undef' in the global namespace}}
  }
  void bar() {
    ::UndefClass::undef(); // expected-error {{no member named 'UndefClass' in the global namespace}}
  }
  void baz() {
    B::qux(); // expected-error {{use of undeclared identifier 'B'}} \
    // expected-warning {{unqualified lookup into dependent bases of class template 'A'}}
  }
};

struct B { void qux(); };
struct C : B { };
template struct A<C>; // No error!  B is a base of A<C>, and qux is available.

struct D { };
template struct A<D>; // expected-note {{in instantiation of member function 'PR19233::A<PR19233::D>::baz' requested here}}

}

namespace nonmethod_missing_this {
template <typename T> struct Base { int y = 42; };
template <typename T> struct Derived : Base<T> {
  int x = y; // expected-warning {{lookup into dependent bases}}
  auto foo(int j) -> decltype(y * j) { // expected-warning {{lookup into dependent bases}}
    return y * j; // expected-warning {{lookup into dependent bases}}
  }
  int bar() {
    return [&] { return y; }(); // expected-warning {{lookup into dependent bases}}
  }
};
template struct Derived<int>;
}

namespace typedef_in_base {
template <typename T> struct A { typedef T NameFromBase; };
template <typename T> struct B : A<T> {
  NameFromBase m; // expected-warning {{found via unqualified lookup into dependent bases}}
};
static_assert(sizeof(B<int>) == 4, "");
}

namespace struct_in_base {
template <typename T> struct A { struct NameFromBase {}; };
template <typename T> struct B : A<T> {
  NameFromBase m; // expected-warning {{found via unqualified lookup into dependent bases}}
};
static_assert(sizeof(B<int>) == 1, "");
}

namespace enum_in_base {
template <typename T> struct A { enum NameFromBase { X }; };
template <typename T> struct B : A<T> {
  NameFromBase m; // expected-warning {{found via unqualified lookup into dependent bases}}
};
static_assert(sizeof(B<int>) == sizeof(A<int>::NameFromBase), "");
}

namespace two_types_in_base {
template <typename T> struct A { typedef T NameFromBase; }; // expected-note {{member type 'int' found by ambiguous name lookup}}
template <typename T> struct B { struct NameFromBase { T m; }; }; // expected-note {{member type 'two_types_in_base::B<int>::NameFromBase' found by ambiguous name lookup}}
template <typename T> struct C : A<T>, B<T> {
  NameFromBase m; // expected-error {{member 'NameFromBase' found in multiple base classes of different types}} expected-warning {{use of member 'NameFromBase' found via unqualified lookup into dependent bases of class templates is a Microsoft extension}}
};
static_assert(sizeof(C<int>) != 0, ""); // expected-note {{in instantiation of template class 'two_types_in_base::C<int>' requested here}}
}

namespace type_and_decl_in_base {
template <typename T> struct A { typedef T NameFromBase; };
template <typename T> struct B { static const T NameFromBase = 42; };
template <typename T> struct C : A<T>, B<T> {
  NameFromBase m; // expected-error {{unknown type name 'NameFromBase'}}
};
}

namespace classify_type_from_base {
template <typename T> struct A { struct NameFromBase {}; };
template <typename T> struct B : A<T> {
  A<NameFromBase> m; // expected-warning {{found via unqualified lookup into dependent bases}}
};
}

namespace classify_nontype_from_base {
// MSVC does not do lookup of non-type declarations from dependent template base
// classes.  The extra lookup only applies to types.
template <typename T> struct A { void NameFromBase() {} };
template <void (*F)()> struct B { };
template <typename T> struct C : A<T> {
  B<C::NameFromBase> a; // correct
  B<NameFromBase> b; // expected-error {{use of undeclared identifier 'NameFromBase'}}
};
}

namespace template_in_base {
template <typename T> struct A {
  template <typename U> struct NameFromBase { U x; };
};
template <typename T> struct B : A<T> {
  // Correct form.
  typename B::template NameFromBase<T> m;
};
template <typename T> struct C : A<T> {
  // Incorrect form.
  NameFromBase<T> m; // expected-error {{no template named 'NameFromBase'}}
};
}

namespace type_in_inner_class_in_base {
template <typename T>
struct A {
  struct B { typedef T NameFromBase; };
};
template <typename T>
struct C : A<T>::B { NameFromBase m; }; // expected-error {{unknown type name 'NameFromBase'}}
}

namespace type_in_inner_template_class_in_base {
template <typename T>
struct A {
  template <typename U> struct B { typedef U InnerType; };
};
template <typename T>
struct C : A<T>::template B<T> {
  NameFromBase m; // expected-error {{unknown type name 'NameFromBase'}}
};
}

namespace have_nondependent_base {
template <typename T>
struct A {
  // Nothing, lookup should fail.
};
template <typename T>
struct B : A<T> { NameFromBase m; }; // expected-error {{unknown type name 'NameFromBase'}}
struct C : A<int> { NameFromBase m; }; // expected-error {{unknown type name 'NameFromBase'}}
}

namespace type_in_base_of_dependent_base {
struct A { typedef int NameFromBase; };
template <typename T>
struct B : A {};
template <typename T>
struct C : B<T> { NameFromBase m; }; // expected-warning {{use of member 'NameFromBase' found via unqualified lookup into dependent bases of class templates is a Microsoft extension}}
}

namespace type_in_second_dependent_base {
template <typename T>
struct A {};
template<typename T>
struct B { typedef T NameFromBase; };
template <typename T>
struct D : A<T>, B<T> { NameFromBase m; }; // expected-warning {{use of member 'NameFromBase' found via unqualified lookup into dependent bases of class templates is a Microsoft extension}}
}

namespace type_in_second_non_dependent_base {
struct A {};
struct B { typedef int NameFromBase; };
template<typename T>
struct C : A, B {};
template <typename T>
struct D : C<T> { NameFromBase m; }; // expected-warning {{use of member 'NameFromBase' found via unqualified lookup into dependent bases of class templates is a Microsoft extension}}
}

namespace type_in_virtual_base_of_dependent_base {
template <typename T>
struct A { typedef T NameFromBase; };
template <typename T>
struct B : virtual A<T> {};
template <typename T>
struct C : B<T>, virtual A<T> { NameFromBase m; }; // expected-warning {{use of member 'NameFromBase' found via unqualified lookup into dependent bases of class templates is a Microsoft extension}}
C<int> c;
}

namespace type_in_base_of_multiple_dependent_bases {
template <typename T>
struct A { typedef T NameFromBase; };
template <typename T>
struct B : public A<T> {};
template <typename T>
struct C : B<T>, public A<T> { NameFromBase m; }; // expected-warning {{use of member 'NameFromBase' found via unqualified lookup into dependent bases of class templates is a Microsoft extension}} expected-warning {{direct base 'A<int>' is inaccessible due to ambiguity:}}
C<int> c; // expected-note {{in instantiation of template class 'type_in_base_of_multiple_dependent_bases::C<int>' requested here}}
}

namespace type_in_dependent_base_of_non_dependent_type {
template<typename T> struct A { typedef int NameFromBase; };
template<typename T> struct B : A<T> {
  struct C;
  template<typename TT>
  struct D : C {
    NameFromBase m; // expected-warning {{use of member 'NameFromBase' found via unqualified lookup into dependent bases of class templates is a Microsoft extension}}
  };
  struct E : C {
    NameFromBase m; // expected-warning {{use of member 'NameFromBase' found via unqualified lookup into dependent bases of class templates is a Microsoft extension}}
  };
};
template<typename T> struct B<T>::C : B {
  NameFromBase m; // expected-warning {{use of member 'NameFromBase' found via unqualified lookup into dependent bases of class templates is a Microsoft extension}}
};
template<typename T> struct F : B<T>::C {
  NameFromBase m; // expected-error {{unknown type name 'NameFromBase'}}
};
}

namespace lookup_in_function_contexts {
template <typename T> struct A { typedef T NameFromBase; };
template <typename T>
struct B : A<T> {
  // expected-warning@+1 {{lookup into dependent bases}}
  static auto lateSpecifiedFunc() -> decltype(NameFromBase()) {
    return {};
  }

  static void memberFunc() {
    NameFromBase x; // expected-warning {{lookup into dependent bases}}
  }

  static void funcLocalClass() {
    struct X {
      NameFromBase x; // expected-warning {{lookup into dependent bases}}
    } y;
  }

  void localClassMethod() {
    struct X {
      void bar() {
        NameFromBase m; // expected-warning {{lookup into dependent bases}}
      }
    } x;
    x.bar();
  }

  static void funcLambda() {
    auto l = []() {
      NameFromBase x; // expected-warning {{lookup into dependent bases}}
    };
    l();
  }

  static constexpr int constexprFunc() {
    NameFromBase x = {}; // expected-warning {{lookup into dependent bases}}
    return sizeof(x);
  }

  static auto autoFunc() {
    NameFromBase x; // expected-warning {{lookup into dependent bases}}
    return x;
  }
};

// Force us to parse the methods.
template struct B<int>;
}

namespace function_template_deduction {
// Overloaded function templates.
template <int N> int f() { return N; }
template <typename T> int f() { return sizeof(T); }

// Dependent base class with type.
template <typename T>
struct A { typedef T NameFromBase; };
template <typename T>
struct B : A<T> {
  // expected-warning@+1 {{found via unqualified lookup into dependent bases}}
  int x = f<NameFromBase>();
};

// Dependent base class with enum.
template <typename T> struct C { enum { NameFromBase = 4 }; };
template <typename T> struct D : C<T> {
  // expected-warning@+1 {{use of undeclared identifier 'NameFromBase'; unqualified lookup into dependent bases}}
  int x = f<NameFromBase>();
};
}

namespace function_template_undef_impl {
template<class T>
void f() {
  Undef::staticMethod(); // expected-error {{use of undeclared identifier 'Undef'}}
  UndefVar.method(); // expected-error {{use of undeclared identifier 'UndefVar'}}
}
}

namespace PR20716 {
template <template <typename T> class A>
struct B : A<int>
{
  XXX x; // expected-error {{unknown type name}}
};

template <typename T>
struct C {};

template <typename T>
using D = C<T>;

template <typename T>
struct E : D<T>
{
  XXX x; // expected-error {{unknown type name}}
};
}

namespace PR23810 {
void f(int);
struct Base {
  void f(); // expected-note{{must qualify identifier to find this declaration in dependent base class}}
};
template <typename T> struct Template : T {
  void member() {
    f(); // expected-warning {{found via unqualified lookup into dependent bases}}
  }
};
void test() {
  Template<Base> x;
  x.member(); // expected-note{{requested here}}
};
}

namespace PR23823 {
// Don't delay lookup in SFINAE context.
template <typename T> decltype(g(T())) check(); // expected-note{{candidate template ignored: substitution failure [with T = int]: use of undeclared identifier 'g'}}
decltype(check<int>()) x; // expected-error{{no matching function for call to 'check'}}

void h();
template <typename T> decltype(h(T())) check2(); // expected-note{{candidate template ignored: substitution failure [with T = int]: no matching function for call to 'h'}}
decltype(check2<int>()) y; // expected-error{{no matching function for call to 'check2'}}
}

// We also allow unqualified lookup into bases in contexts where the we know the
// undeclared identifier *must* be a type, such as a new expression or catch
// parameter type.
template <typename T>
struct UseUnqualifiedTypeNames : T {
  void foo() {
    void *P = new TheType; // expected-warning {{unqualified lookup}} expected-error {{no type}}
    size_t x = __builtin_offsetof(TheType, f2); // expected-warning {{unqualified lookup}} expected-error {{no type}}
    try {
    } catch (TheType) { // expected-warning {{unqualified lookup}} expected-error {{no type}}
    }
    enum E : IntegerType { E0 = 42 }; // expected-warning {{unqualified lookup}} expected-error {{no type}}
    _Atomic(TheType) a; // expected-warning {{unqualified lookup}} expected-error {{no type}}
  }
  void out_of_line();
};
template <typename T>
void UseUnqualifiedTypeNames<T>::out_of_line() {
  void *p = new TheType; // expected-warning {{unqualified lookup}} expected-error {{no type}}
}
struct Base {
  typedef int IntegerType;
  struct TheType {
    int f1, f2;
  };
};
template struct UseUnqualifiedTypeNames<Base>;
struct BadBase { };
template struct UseUnqualifiedTypeNames<BadBase>; // expected-note-re 2 {{in instantiation {{.*}} requested here}}

namespace partial_template_lookup {

class Bar;
class Spare;

template <class T, class X = Bar>
class FooTemplated;

class FooBase {
public:
  typedef int BaseTypedef;
};

// Partial template spec (unused)
template <class T>
class FooTemplated<T, Spare> {};

// Partial template spec (used)
template <class T>
class FooTemplated<T, Bar> : public FooBase {};

// Full template spec
template <class T, class X>
class FooTemplated : public FooTemplated<T, Bar> {
public:
  BaseTypedef Member; // expected-warning {{unqualified lookup}}
};
}
