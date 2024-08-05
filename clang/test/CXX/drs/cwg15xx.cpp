// RUN: %clang_cc1 -std=c++98 -triple x86_64-unknown-unknown %s -verify=expected,cxx98 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-unknown %s -verify=expected,cxx11-20,since-cxx11,cxx11-14 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++14 -triple x86_64-unknown-unknown %s -verify=expected,cxx11-20,since-cxx11,cxx11-14,cxx14-17 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-unknown %s -verify=expected,cxx11-20,since-cxx11,since-cxx17 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-unknown %s -verify=expected,cxx11-20,since-cxx20,since-cxx11,since-cxx17 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++23 -triple x86_64-unknown-unknown %s -verify=expected,since-cxx23,since-cxx20,since-cxx11,since-cxx17 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++2c -triple x86_64-unknown-unknown %s -verify=expected,since-cxx23,since-cxx20,since-cxx11,since-cxx17 -fexceptions -fcxx-exceptions -pedantic-errors

#if __cplusplus == 199711L
#define static_assert(...) __extension__ _Static_assert(__VA_ARGS__)
// cxx98-error@-1 {{variadic macros are a C99 feature}}
#endif

namespace cwg1512 { // cwg1512: 4
  void f(char *p) {
    if (p > 0) {}
    // expected-error@-1 {{ordered comparison between pointer and zero ('char *' and 'int')}}
#if __cplusplus >= 201103L
    if (p > nullptr) {}
    // since-cxx11-error@-1 {{invalid operands to binary expression ('char *' and 'std::nullptr_t')}}
#endif
  }
  bool g(int **x, const int **y) {
    return x < y;
  }

  template<typename T> T val();

  template<typename A, typename B, typename C> void composite_pointer_type_is_base() {
    typedef __typeof(true ? val<A>() : val<B>()) type;
    typedef C type;

    typedef __typeof(val<A>() == val<B>()) cmp;
    typedef __typeof(val<A>() != val<B>()) cmp;
    typedef bool cmp;
  }

  template<typename A, typename B, typename C> void composite_pointer_type_is_ord() {
    composite_pointer_type_is_base<A, B, C>();

    typedef __typeof(val<A>() < val<B>()) cmp; // #cwg1512-lt 
    // since-cxx17-warning@#cwg1512-lt {{ordered comparison of function pointers ('int (*)() noexcept' and 'int (*)()')}}
    //   since-cxx17-note@#cwg1512-noexcept-1st {{in instantiation of function template specialization 'cwg1512::composite_pointer_type_is_ord<int (*)() noexcept, int (*)(), int (*)()>' requested here}}
    // since-cxx17-warning@#cwg1512-lt {{ordered comparison of function pointers ('int (*)()' and 'int (*)() noexcept')}}
    //   since-cxx17-note@#cwg1512-noexcept-2nd {{in instantiation of function template specialization 'cwg1512::composite_pointer_type_is_ord<int (*)(), int (*)() noexcept, int (*)()>' requested here}}
    typedef __typeof(val<A>() <= val<B>()) cmp;
    // since-cxx17-warning@-1 {{ordered comparison of function pointers ('int (*)() noexcept' and 'int (*)()')}}
    // since-cxx17-warning@-2 {{ordered comparison of function pointers ('int (*)()' and 'int (*)() noexcept')}}
    typedef __typeof(val<A>() > val<B>()) cmp;
    // since-cxx17-warning@-1 {{ordered comparison of function pointers ('int (*)() noexcept' and 'int (*)()')}}
    // since-cxx17-warning@-2 {{ordered comparison of function pointers ('int (*)()' and 'int (*)() noexcept')}}
    typedef __typeof(val<A>() >= val<B>()) cmp;
    // since-cxx17-warning@-1 {{ordered comparison of function pointers ('int (*)() noexcept' and 'int (*)()')}}
    // since-cxx17-warning@-2 {{ordered comparison of function pointers ('int (*)()' and 'int (*)() noexcept')}}
    typedef bool cmp;
  }

  template <typename A, typename B, typename C>
  void composite_pointer_type_is_unord(int = 0) {
    composite_pointer_type_is_base<A, B, C>();
  }
  template <typename A, typename B, typename C>
  void composite_pointer_type_is_unord(__typeof(val<A>() < val<B>()) * = 0);
  template <typename A, typename B, typename C>
  void composite_pointer_type_is_unord(__typeof(val<A>() <= val<B>()) * = 0);
  template <typename A, typename B, typename C>
  void composite_pointer_type_is_unord(__typeof(val<A>() > val<B>()) * = 0);
  template <typename A, typename B, typename C>
  void composite_pointer_type_is_unord(__typeof(val<A>() >= val<B>()) * = 0);

  // A call to this is ambiguous if a composite pointer type exists.
  template<typename A, typename B>
  void no_composite_pointer_type(__typeof((true ? val<A>() : val<B>()), void()) * = 0);
  template<typename A, typename B> void no_composite_pointer_type(int = 0);

  struct A {};
  struct B : A {};
  struct C {};

  void test() {
#if __cplusplus >= 201103L
    using nullptr_t = decltype(nullptr);
    composite_pointer_type_is_unord<nullptr_t, nullptr_t, nullptr_t>();
    no_composite_pointer_type<nullptr_t, int>();

    composite_pointer_type_is_unord<nullptr_t, const char**, const char**>();
    composite_pointer_type_is_unord<const char**, nullptr_t, const char**>();
#endif

    composite_pointer_type_is_ord<const int *, volatile void *, const volatile void*>();
    composite_pointer_type_is_ord<const void *, volatile int *, const volatile void*>();

    composite_pointer_type_is_ord<const A*, volatile B*, const volatile A*>();
    composite_pointer_type_is_ord<const B*, volatile A*, const volatile A*>();

    composite_pointer_type_is_unord<const int *A::*, volatile int *B::*, const volatile int *const B::*>();
    composite_pointer_type_is_unord<const int *B::*, volatile int *A::*, const volatile int *const B::*>();
    no_composite_pointer_type<int (A::*)(), int (C::*)()>();
    no_composite_pointer_type<const int (A::*)(), volatile int (C::*)()>();
    // since-cxx20-warning@-1 {{volatile-qualified return type 'volatile int' is deprecated}}

#if __cplusplus >= 201703L
    composite_pointer_type_is_ord<int (*)() noexcept, int (*)(), int (*)()>(); // #cwg1512-noexcept-1st
    composite_pointer_type_is_ord<int (*)(), int (*)() noexcept, int (*)()>(); // #cwg1512-noexcept-2nd
    composite_pointer_type_is_unord<int (A::*)() noexcept, int (A::*)(), int (A::*)()>();
    composite_pointer_type_is_unord<int (A::*)(), int (A::*)() noexcept, int (A::*)()>();
    // FIXME: This looks like a standard defect; these should probably all have type 'int (B::*)()'.
    composite_pointer_type_is_unord<int (B::*)(), int (A::*)() noexcept, int (B::*)()>();
    composite_pointer_type_is_unord<int (A::*)() noexcept, int (B::*)(), int (B::*)()>();
    composite_pointer_type_is_unord<int (B::*)() noexcept, int (A::*)(), int (B::*)()>();
    composite_pointer_type_is_unord<int (A::*)(), int (B::*)() noexcept, int (B::*)()>();

    // FIXME: It would be reasonable to permit these, with a common type of 'int (*const *)()'.
    no_composite_pointer_type<int (**)() noexcept, int (**)()>();
    no_composite_pointer_type<int (**)(), int (**)() noexcept>();

    // FIXME: It would be reasonable to permit these, with a common type of 'int (A::*)()'.
    no_composite_pointer_type<int (A::*)() const, int (A::*)()>();
    no_composite_pointer_type<int (A::*)(), int (A::*)() const>();

    // FIXME: It would be reasonable to permit these, with a common type of
    // 'int (A::*)() &' and 'int (A::*)() &&', respectively.
    no_composite_pointer_type<int (A::*)() &, int (A::*)()>();
    no_composite_pointer_type<int (A::*)(), int (A::*)() &>();
    no_composite_pointer_type<int (A::*)() &&, int (A::*)()>();
    no_composite_pointer_type<int (A::*)(), int (A::*)() &&>();

    no_composite_pointer_type<int (A::*)() &&, int (A::*)() &>();
    no_composite_pointer_type<int (A::*)() &, int (A::*)() &&>();

    no_composite_pointer_type<int (C::*)(), int (A::*)() noexcept>();
    no_composite_pointer_type<int (A::*)() noexcept, int (C::*)()>();
#endif
  }

#if __cplusplus >= 201103L
  template<typename T> struct Wrap { operator T(); }; // #cwg1512-Wrap
  void test_overload() {
    using nullptr_t = decltype(nullptr);
    void(Wrap<nullptr_t>() == Wrap<nullptr_t>());
    void(Wrap<nullptr_t>() != Wrap<nullptr_t>());
    void(Wrap<nullptr_t>() < Wrap<nullptr_t>());
    // since-cxx11-error@-1 {{invalid operands to binary expression ('Wrap<nullptr_t>' (aka 'Wrap<std::nullptr_t>') and 'Wrap<nullptr_t>' (aka 'Wrap<std::nullptr_t>'))}}
    void(Wrap<nullptr_t>() > Wrap<nullptr_t>());
    // since-cxx11-error@-1 {{invalid operands to binary expression ('Wrap<nullptr_t>' (aka 'Wrap<std::nullptr_t>') and 'Wrap<nullptr_t>' (aka 'Wrap<std::nullptr_t>'))}}
    void(Wrap<nullptr_t>() <= Wrap<nullptr_t>());
    // since-cxx11-error@-1 {{invalid operands to binary expression ('Wrap<nullptr_t>' (aka 'Wrap<std::nullptr_t>') and 'Wrap<nullptr_t>' (aka 'Wrap<std::nullptr_t>'))}}
    void(Wrap<nullptr_t>() >= Wrap<nullptr_t>());
    // since-cxx11-error@-1 {{invalid operands to binary expression ('Wrap<nullptr_t>' (aka 'Wrap<std::nullptr_t>') and 'Wrap<nullptr_t>' (aka 'Wrap<std::nullptr_t>'))}}

    // Under cwg1213, this is ill-formed: we select the builtin operator<(int*, int*)
    // but then only convert as far as 'nullptr_t', which we then can't convert to 'int*'.
    void(Wrap<nullptr_t>() == Wrap<int*>());
    void(Wrap<nullptr_t>() != Wrap<int*>());
    void(Wrap<nullptr_t>() < Wrap<int*>());
    // since-cxx11-error@-1 {{invalid operands to binary expression ('Wrap<nullptr_t>' (aka 'Wrap<std::nullptr_t>') and 'Wrap<int *>')}}
    //   since-cxx11-note@#cwg1512-Wrap {{first operand was implicitly converted to type 'std::nullptr_t'}}
    //   since-cxx11-note@#cwg1512-Wrap {{second operand was implicitly converted to type 'int *'}}
    void(Wrap<nullptr_t>() > Wrap<int*>());
    // since-cxx11-error@-1 {{invalid operands}}
    //   since-cxx11-note@#cwg1512-Wrap {{first operand was implicitly converted to type 'std::nullptr_t'}}
    //   since-cxx11-note@#cwg1512-Wrap{{second operand was implicitly converted to type 'int *'}}
    void(Wrap<nullptr_t>() <= Wrap<int*>());
    // since-cxx11-error@-1 {{invalid operands}}
    //   since-cxx11-note@#cwg1512-Wrap {{first operand was implicitly converted to type 'std::nullptr_t'}}
    //   since-cxx11-note@#cwg1512-Wrap {{second operand was implicitly converted to type 'int *'}}
    void(Wrap<nullptr_t>() >= Wrap<int*>());
    // since-cxx11-error@-1 {{invalid operands}}
    //   since-cxx11-note@#cwg1512-Wrap {{first operand was implicitly converted to type 'std::nullptr_t'}}
    //   since-cxx11-note@#cwg1512-Wrap {{second operand was implicitly converted to type 'int *'}}
  }
#endif
}

namespace cwg1514 { // cwg1514: 11
#if __cplusplus >= 201103L
  struct S {
    enum E : int {}; // #cwg1514-E
    enum E : int {};
    // since-cxx11-error@-1 {{redefinition of 'E'}}
    //   since-cxx11-note@#cwg1514-E {{previous definition is here}}
  };
  S::E se; // OK, complete type, not zero-width bitfield.

  // The behavior in other contexts is superseded by CWG1966.
#endif
}

namespace cwg1518 { // cwg1518: 4
#if __cplusplus >= 201103L
struct Z0 { // #cwg1518-Z0
  explicit Z0() = default; // #cwg1518-Z0-ctor
};
struct Z { // #cwg1518-Z
  explicit Z(); // #cwg1518-Z-ctor
  explicit Z(int); // #cwg1518-Z-int
  explicit Z(int, int); // #cwg1518-Z-int-int
};
template <class T> int Eat(T); // #cwg1518-Eat
Z0 a;
Z0 b{};
Z0 c = {};
// since-cxx11-error@-1 {{chosen constructor is explicit in copy-initialization}}
//   since-cxx11-note@#cwg1518-Z0-ctor {{explicit constructor declared here}}
int i = Eat<Z0>({});
// since-cxx11-error@-1 {{no matching function for call to 'Eat'}}
//   since-cxx11-note@#cwg1518-Eat {{candidate function template not viable: cannot convert initializer list argument to 'Z0'}}

Z c2 = {};
// since-cxx11-error@-1 {{chosen constructor is explicit in copy-initialization}}
//   since-cxx11-note@#cwg1518-Z-ctor {{explicit constructor declared here}}
int i2 = Eat<Z>({});
// since-cxx11-error@-1 {{no matching function for call to 'Eat'}}
//   since-cxx11-note@#cwg1518-Eat {{candidate function template not viable: cannot convert initializer list argument to 'Z'}}
Z a1 = 1;
// since-cxx11-error@-1 {{no viable conversion from 'int' to 'Z'}}
//   since-cxx11-note@#cwg1518-Z {{candidate constructor (the implicit copy constructor) not viable: no known conversion from 'int' to 'const Z &' for 1st argument}}
//   since-cxx11-note@#cwg1518-Z {{candidate constructor (the implicit move constructor) not viable: no known conversion from 'int' to 'Z &&' for 1st argument}}
//   since-cxx11-note@#cwg1518-Z-int {{explicit constructor is not a candidate}}
Z a3 = Z(1);
Z a2(1);
Z *p = new Z(1);
Z a4 = (Z)1;
Z a5 = static_cast<Z>(1);
Z a6 = {4, 3};
// since-cxx11-error@-1 {{chosen constructor is explicit in copy-initialization}}
//   since-cxx11-note@#cwg1518-Z-int-int {{explicit constructor declared here}}

struct UserProvidedBaseCtor { // #cwg1518-U
  UserProvidedBaseCtor() {}
};
struct DoesntInheritCtor : UserProvidedBaseCtor { // #cwg1518-D-U
  int x;
};
DoesntInheritCtor I{{}, 42};
// cxx11-14-error@-1 {{no matching constructor for initialization of 'DoesntInheritCtor'}}
//   cxx11-14-note@#cwg1518-D-U {{candidate constructor (the implicit copy constructor) not viable: requires 1 argument, but 2 were provided}}
//   cxx11-14-note@#cwg1518-D-U {{candidate constructor (the implicit move constructor) not viable: requires 1 argument, but 2 were provided}}
//   cxx11-14-note@#cwg1518-D-U {{candidate constructor (the implicit default constructor) not viable: requires 0 arguments, but 2 were provided}}

struct BaseCtor { BaseCtor() = default; }; // #cwg1518-BC
struct InheritsCtor : BaseCtor { // #cwg1518-I
  using BaseCtor::BaseCtor;      // #cwg1518-I-using
  int x;
};
InheritsCtor II = {{}, 42};
// since-cxx11-error@-1 {{no matching constructor for initialization of 'InheritsCtor'}}
//   since-cxx11-note@#cwg1518-BC {{candidate constructor (the implicit copy constructor) not viable: requires 1 argument, but 2 were provided}}
//   since-cxx11-note@#cwg1518-I-using {{constructor from base class 'BaseCtor' inherited here}}
//   since-cxx11-note@#cwg1518-BC {{candidate constructor (the implicit move constructor) not viable: requires 1 argument, but 2 were provided}}
//   since-cxx11-note@#cwg1518-I-using {{constructor from base class 'BaseCtor' inherited here}}
//   since-cxx11-note@#cwg1518-I {{candidate constructor (the implicit copy constructor) not viable: requires 1 argument, but 2 were provided}}
//   since-cxx11-note@#cwg1518-I {{candidate constructor (the implicit move constructor) not viable: requires 1 argument, but 2 were provided}}
//   since-cxx11-note@#cwg1518-I {{candidate constructor (the implicit default constructor) not viable: requires 0 arguments, but 2 were provided}}

namespace std_example {
  struct A {
    explicit A() = default; // #cwg1518-A
  };

  struct B : A {
    explicit B() = default; // #cwg1518-B
  };

  struct C {
    explicit C(); // #cwg1518-C
  };

  struct D : A {
    C c;
    explicit D() = default; // #cwg1518-D
  };

  template <typename T> void f() {
    T t; // ok
    T u{}; // ok
    T v = {}; // #cwg1518-v
    // since-cxx11-error@#cwg1518-v {{chosen constructor is explicit in copy-initialization}}
    //   since-cxx11-note@#cwg1518-f-A {{in instantiation of function template specialization 'cwg1518::std_example::f<cwg1518::std_example::A>' requested here}}
    //   since-cxx11-note@#cwg1518-A {{explicit constructor declared here}}
    // since-cxx11-error@#cwg1518-v {{chosen constructor is explicit in copy-initialization}}
    //   since-cxx11-note@#cwg1518-f-B {{in instantiation of function template specialization 'cwg1518::std_example::f<cwg1518::std_example::B>' requested here}}
    //   since-cxx11-note@#cwg1518-B {{explicit constructor declared here}}
    // since-cxx11-error@#cwg1518-v {{chosen constructor is explicit in copy-initialization}}
    //   since-cxx11-note@#cwg1518-f-C {{in instantiation of function template specialization 'cwg1518::std_example::f<cwg1518::std_example::C>' requested here}}
    //   since-cxx11-note@#cwg1518-C {{explicit constructor declared here}}
    // since-cxx11-error@#cwg1518-v {{chosen constructor is explicit in copy-initialization}}
    //   since-cxx11-note@#cwg1518-f-D {{in instantiation of function template specialization 'cwg1518::std_example::f<cwg1518::std_example::D>' requested here}}
    //   since-cxx11-note@#cwg1518-D {{explicit constructor declared here}}
  }
  template <typename T> void g() {
    void x(T t); // #cwg1518-x
    x({}); // #cwg1518-x-call
    // since-cxx11-error@#cwg1518-x-call {{chosen constructor is explicit in copy-initialization}}
    //   since-cxx11-note@#cwg1518-g-A {{in instantiation of function template specialization 'cwg1518::std_example::g<cwg1518::std_example::A>' requested here}}
    //   since-cxx11-note@#cwg1518-A {{explicit constructor declared here}}
    //   since-cxx11-note@#cwg1518-x {{passing argument to parameter 't' here}}
    // since-cxx11-error@#cwg1518-x-call {{chosen constructor is explicit in copy-initialization}}
    //   since-cxx11-note@#cwg1518-g-B {{in instantiation of function template specialization 'cwg1518::std_example::g<cwg1518::std_example::B>' requested here}}
    //   since-cxx11-note@#cwg1518-B {{explicit constructor declared here}}
    //   since-cxx11-note@#cwg1518-x {{passing argument to parameter 't' here}}
    // since-cxx11-error@#cwg1518-x-call {{chosen constructor is explicit in copy-initialization}}
    //   since-cxx11-note@#cwg1518-g-C {{in instantiation of function template specialization 'cwg1518::std_example::g<cwg1518::std_example::C>' requested here}}
    //   since-cxx11-note@#cwg1518-C {{explicit constructor declared here}}
    //   since-cxx11-note@#cwg1518-x {{passing argument to parameter 't' here}}
    // since-cxx11-error@#cwg1518-x-call {{chosen constructor is explicit in copy-initialization}}
    //   since-cxx11-note@#cwg1518-g-D {{in instantiation of function template specialization 'cwg1518::std_example::g<cwg1518::std_example::D>' requested here}}
    //   since-cxx11-note@#cwg1518-D {{explicit constructor declared here}}
    //   since-cxx11-note@#cwg1518-x {{passing argument to parameter 't' here}}
  }

  void test() {
    f<A>(); // #cwg1518-f-A
    f<B>(); // #cwg1518-f-B
    f<C>(); // #cwg1518-f-C
    f<D>(); // #cwg1518-f-D
    g<A>(); // #cwg1518-g-A
    g<B>(); // #cwg1518-g-B
    g<C>(); // #cwg1518-g-C
    g<D>(); // #cwg1518-g-D
  }
}
#endif // __cplusplus >= 201103L
}

namespace cwg1550 { // cwg1550: 3.4
  int f(bool b, int n) {
    return (b ? (throw 0) : n) + (b ? n : (throw 0));
  }
}

namespace cwg1558 { // cwg1558: 12
#if __cplusplus >= 201103L
  template<class T, class...> using first_of = T;
  template<class T> first_of<void, typename T::type> f(int); // #cwg1558-f 
  template<class T> void f(...) = delete; // #cwg1558-f-deleted

  struct X { typedef void type; };
  void test() {
    f<X>(0);
    f<int>(0);
    // since-cxx11-error@-1 {{call to deleted function 'f'}}
    //   since-cxx11-note@#cwg1558-f-deleted {{candidate function [with T = int] has been explicitly deleted}}
    //   since-cxx11-note@#cwg1558-f {{candidate template ignored: substitution failure [with T = int]: type 'int' cannot be used prior to '::' because it has no members}}
  }
#endif
}

namespace cwg1560 { // cwg1560: 3.5
  void f(bool b, int n) {
    (b ? throw 0 : n) = (b ? n : throw 0) = 0;
  }
  class X { X(const X&); };
  const X &get();
  const X &x = true ? get() : throw 0;
}

namespace cwg1563 { // cwg1563: yes
#if __cplusplus >= 201103L
  double bar(double) { return 0.0; }
  float bar(float) { return 0.0f; }

  using fun = double(double);
  fun &foo{bar}; // ok
#endif
}

namespace cwg1567 { // cwg1567: 3.3
#if __cplusplus >= 201103L
struct B;
struct A {
  A(const A&);
  A(const B&) = delete;
  A(A&&);
  A(B&&) = delete;
  A(int); // #cwg1567-A-int
};

struct B: A { // #cwg1567-B
  using A::A; // #cwg1567-using-A
  B(double); // #cwg1567-B-double
};

A a{0};
B b{1.0};
// Good, deleted converting ctors are not inherited as copy/move ctors
B b2{b};
B b3{B{1.0}};
// Good, copy/move ctors are not inherited
B b4{a};
// since-cxx11-error@-1 {{no matching constructor for initialization of 'B'}}
//   since-cxx11-note@#cwg1567-A-int {{candidate inherited constructor not viable: no known conversion from 'A' to 'int' for 1st argument}}
//   since-cxx11-note@#cwg1567-using-A {{constructor from base class 'A' inherited here}}
//   since-cxx11-note@#cwg1567-B {{candidate constructor (the implicit copy constructor) not viable: no known conversion from 'A' to 'const B' for 1st argument}}
//   since-cxx11-note@#cwg1567-B {{candidate constructor (the implicit move constructor) not viable: no known conversion from 'A' to 'B' for 1st argument}}
//   since-cxx11-note@#cwg1567-B-double {{candidate constructor not viable: no known conversion from 'A' to 'double' for 1st argument}}
B b5{A{0}};
// since-cxx11-error@-1 {{no matching constructor for initialization of 'B'}}
//   since-cxx11-note@#cwg1567-A-int {{candidate inherited constructor not viable: no known conversion from 'A' to 'int' for 1st argument}}
//   since-cxx11-note@#cwg1567-using-A {{constructor from base class 'A' inherited here}}
//   since-cxx11-note@#cwg1567-B {{candidate constructor (the implicit copy constructor) not viable: no known conversion from 'A' to 'const B' for 1st argument}}
//   since-cxx11-note@#cwg1567-B {{candidate constructor (the implicit move constructor) not viable: no known conversion from 'A' to 'B' for 1st argument}}
//   since-cxx11-note@#cwg1567-B-double {{candidate constructor not viable: no known conversion from 'A' to 'double' for 1st argument}}
#endif
}

namespace cwg1573 { // cwg1573: 3.9
#if __cplusplus >= 201103L
  // ellipsis is inherited (p0136r1 supersedes this part).
  struct A { A(); A(int, char, ...); };
  struct B : A { using A::A; };
  B b(1, 'x', 4.0, "hello"); // ok

  // inherited constructor is effectively constexpr if the user-written constructor would be
  struct C { C(); constexpr C(int) {} }; // #cwg1573-C
  struct D : C { using C::C; };
  constexpr D d = D(0); // ok
  struct E : C { using C::C; A a; }; // #cwg1573-E
  constexpr E e = E(0);
  // since-cxx11-error@-1 {{constexpr variable cannot have non-literal type 'const E'}}
  //   since-cxx11-note@#cwg1573-E {{'E' is not literal because it has data member 'a' of non-literal type 'A'}}

  // FIXME: This diagnostic is pretty bad; we should explain that the problem
  // is that F::c would be initialized by a non-constexpr constructor.
  struct F : C { using C::C; C c; }; // #cwg1573-F
  constexpr F f = F(0);
  // since-cxx11-error@-1 {{constexpr variable 'f' must be initialized by a constant expression}}
  //   cxx11-20-note@-2 {{constructor inherited from base class 'C' cannot be used in a constant expression; derived class cannot be implicitly initialized}}
  //   since-cxx23-note@-3 {{in implicit initialization for inherited constructor of 'F'}}
  //   since-cxx23-note@#cwg1573-F {{non-constexpr constructor 'C' cannot be used in a constant expression}}
  //   cxx11-20-note@#cwg1573-F {{declared here}}
  //   since-cxx23-note@#cwg1573-C {{declared here}}

  // inherited constructor is effectively deleted if the user-written constructor would be
  struct G { G(int); };
  struct H : G { using G::G; G g; }; // #cwg1573-H
  H h(0);
  // since-cxx11-error@-1 {{constructor inherited by 'H' from base class 'G' is implicitly deleted}}
  //   since-cxx11-note@#cwg1573-H {{constructor inherited by 'H' is implicitly deleted because field 'g' has no default constructor}}

  // deleted definition of constructor is inherited
  struct I { I(int) = delete; }; // #cwg1573-I
  struct J : I { using I::I; };
  J j(0);
  // since-cxx11-error@-1 {{call to deleted constructor of 'J'}}
  //   since-cxx11-note@#cwg1573-I {{'I' has been explicitly marked deleted here}}
#endif
}

#if __cplusplus >= 201103L
namespace std {
  typedef decltype(sizeof(int)) size_t;

  // libc++'s implementation
  template <class _E>
  class initializer_list
  {
    const _E* __begin_;
    size_t    __size_;

    initializer_list(const _E* __b, size_t __s)
    : __begin_(__b), __size_(__s) {}

  public:
    typedef _E        value_type;
    typedef const _E& reference;
    typedef const _E& const_reference;
    typedef size_t    size_type;

    typedef const _E* iterator;
    typedef const _E* const_iterator;

    initializer_list() : __begin_(nullptr), __size_(0) {}

    size_t    size()  const {return __size_;}
    const _E* begin() const {return __begin_;}
    const _E* end()   const {return __begin_ + __size_;}
  };

  template < class _T1, class _T2 > struct pair { _T2 second; };

  template<typename T> struct basic_string {
    basic_string(const T* x) {}
    ~basic_string() {};
  };
  typedef basic_string<char> string;

} // std
#endif

namespace cwg1579 { // cwg1579: 3.9
#if __cplusplus >= 201103L
template<class T>
struct GenericMoveOnly {
  GenericMoveOnly();
  template<class U> GenericMoveOnly(const GenericMoveOnly<U> &) = delete; // #cwg1579-deleted-U
  GenericMoveOnly(const int &) = delete; // #cwg1579-deleted-int
  template<class U> GenericMoveOnly(GenericMoveOnly<U> &&);
  GenericMoveOnly(int &&);
};

GenericMoveOnly<float> CWG1579_Eligible(GenericMoveOnly<char> CharMO) {
  int i;
  GenericMoveOnly<char> GMO;

  if (0)
    return i;
  else if (0)
    return GMO;
  else if (0)
    return ((GMO));
  else
    return CharMO;
}

GenericMoveOnly<char> GlobalMO;

GenericMoveOnly<float> CWG1579_Ineligible(int &AnInt,
                                          GenericMoveOnly<char> &CharMO) {
  static GenericMoveOnly<char> StaticMove;
  extern GenericMoveOnly<char> ExternMove;

  if (0)
    return AnInt;
    // since-cxx11-error@-1 {{conversion function from 'int' to 'GenericMoveOnly<float>' invokes a deleted function}}
    //   since-cxx11-note@#cwg1579-deleted-int {{'GenericMoveOnly' has been explicitly marked deleted here}}
  else if (0)
    return GlobalMO;
    // since-cxx11-error@-1 {{conversion function from 'GenericMoveOnly<char>' to 'GenericMoveOnly<float>' invokes a deleted function}}
    //   since-cxx11-note@#cwg1579-deleted-U {{'GenericMoveOnly<char>' has been explicitly marked deleted here}}
  else if (0)
    return StaticMove;
    // since-cxx11-error@-1 {{conversion function from 'GenericMoveOnly<char>' to 'GenericMoveOnly<float>' invokes a deleted function}}
    //   since-cxx11-note@#cwg1579-deleted-U {{'GenericMoveOnly<char>' has been explicitly marked deleted here}}
  else if (0)
    return ExternMove;
    // since-cxx11-error@-1 {{conversion function from 'GenericMoveOnly<char>' to 'GenericMoveOnly<float>' invokes a deleted function}}
    //   since-cxx11-note@#cwg1579-deleted-U {{'GenericMoveOnly<char>' has been explicitly marked deleted here}}
  else if (0)
    return AnInt;
    // since-cxx11-error@-1 {{conversion function from 'int' to 'GenericMoveOnly<float>' invokes a deleted function}}
    //   since-cxx11-note@#cwg1579-deleted-int {{'GenericMoveOnly' has been explicitly marked deleted here}}
  else
    return CharMO;
    // since-cxx11-error@-1 {{conversion function from 'GenericMoveOnly<char>' to 'GenericMoveOnly<float>' invokes a deleted function}}
    //   since-cxx11-note@#cwg1579-deleted-U {{'GenericMoveOnly<char>' has been explicitly marked deleted here}}
}

auto CWG1579_lambda_valid = [](GenericMoveOnly<float> mo) ->
  GenericMoveOnly<char> {
  return mo;
};

auto CWG1579_lambda_invalid = []() -> GenericMoveOnly<char> {
  static GenericMoveOnly<float> mo;
  return mo;
  // since-cxx11-error@-1 {{conversion function from 'GenericMoveOnly<float>' to 'GenericMoveOnly<char>' invokes a deleted function}}
  //   since-cxx11-note@#cwg1579-deleted-U {{'GenericMoveOnly<float>' has been explicitly marked deleted here}}
};
#endif
} // end namespace cwg1579

namespace cwg1584 { // cwg1584: 7 drafting 2015-05
// Deducing function types from cv-qualified types
template<typename T> void f(const T *); // #cwg1584-f
template<typename T> void g(T *, const T * = 0);
template<typename T> void h(T *) { T::error; }
// expected-error@-1 {{type 'void ()' cannot be used prior to '::' because it has no members}}
//   expected-note@#cwg1584-h {{in instantiation of function template specialization 'cwg1584::h<void ()>' requested here}}
template<typename T> void h(const T *);
void i() {
  f(&i);
  // expected-error@-1 {{no matching function for call to 'f'}}
  //   expected-note@#cwg1584-f {{candidate template ignored: could not match 'const T *' against 'void (*)()'}}
  g(&i);
  h(&i); // #cwg1584-h
}

template<typename T> struct tuple_size {
  static const bool is_primary = true;
};
template<typename T> struct tuple_size<T const> : tuple_size<T> {
  static const bool is_primary = false;
};

tuple_size<void()> t;
static_assert(tuple_size<void()>::is_primary, "");
static_assert(tuple_size<void()const>::is_primary, "");
} // namespace cwg1584

namespace cwg1589 {   // cwg1589: 3.7 c++11
#if __cplusplus >= 201103L
  // Ambiguous ranking of list-initialization sequences

  void f0(long, int=0);                 // Would makes selection of #0 ambiguous
  void f0(long);                        // #0
  void f0(std::initializer_list<int>);  // #00
  void g0() { f0({1L}); }               // chooses #00

  void f1(int, int=0);                    // Would make selection of #1 ambiguous
  void f1(int);                           // #1
  void f1(std::initializer_list<long>);   // #2
  void g1() { f1({42}); }                 // chooses #2

  void f2(std::pair<const char*, const char*>, int = 0); // Would makes selection of #3 ambiguous
  void f2(std::pair<const char*, const char*>); // #3
  void f2(std::initializer_list<std::string>);  // #4
  void g2() { f2({"foo","bar"}); }              // chooses #4

  namespace with_error {
    void f0(long);
    void f0(std::initializer_list<int>);          // #cwg1589-f0-ilist
    void f0(std::initializer_list<int>, int = 0); // #cwg1589-f0-ilist-int
    void g0() { f0({1L}); }
    // since-cxx11-error@-1 {{call to 'f0' is ambiguous}}
    //   since-cxx11-note@#cwg1589-f0-ilist {{candidate function}}
    //   since-cxx11-note@#cwg1589-f0-ilist-int {{candidate function}}

    void f1(int);
    void f1(std::initializer_list<long>);          // #cwg1589-f1-ilist
    void f1(std::initializer_list<long>, int = 0); // #cwg1589-f1-ilist-long
    void g1() { f1({42}); }
    // since-cxx11-error@-1 {{call to 'f1' is ambiguous}}
    //   since-cxx11-note@#cwg1589-f1-ilist {{candidate function}}
    //   since-cxx11-note@#cwg1589-f1-ilist-long {{candidate function}}

    void f2(std::pair<const char*, const char*>);
    void f2(std::initializer_list<std::string>);          // #cwg1589-f2-ilist
    void f2(std::initializer_list<std::string>, int = 0); // #cwg1589-f2-ilist-int
    void g2() { f2({"foo","bar"}); }
    // since-cxx11-error@-1 {{call to 'f2' is ambiguous}}
    //   since-cxx11-note@#cwg1589-f2-ilist {{candidate function}}
    //   since-cxx11-note@#cwg1589-f2-ilist-int {{candidate function}}
  }
#endif
} // cwg1589

namespace cwg1591 {  //cwg1591. Deducing array bound and element type from initializer list
#if __cplusplus >= 201103L
  template<class T, int N> int h(T const(&)[N]);
  int X = h({1,2,3});              // T deduced to int, N deduced to 3
  
  template<class T> int j(T const(&)[3]);
  int Y = j({42});                 // T deduced to int, array bound not considered

  struct Aggr { int i; int j; };
  template<int N> int k(Aggr const(&)[N]); // #cwg1591-k
  int Y0 = k({1,2,3});
  // since-cxx11-error@-1 {{no matching function for call to 'k'}}
  //   since-cxx11-note@#cwg1591-k {{candidate function [with N = 3] not viable: no known conversion from 'int' to 'const Aggr' for 1st argument}}
  int Z = k({{1},{2},{3}});        // OK, N deduced to 3

  template<int M, int N> int m(int const(&)[M][N]);
  int X0 = m({{1,2},{3,4}});        // M and N both deduced to 2

  template<class T, int N> int n(T const(&)[N], T);
  int X1 = n({{1},{2},{3}},Aggr()); // OK, T is Aggr, N is 3
  
  
  namespace check_multi_dim_arrays {
    template<class T, int N, int M, int O> int ***f(const T (&a)[N][M][O]); // #cwg1591-f-3
    template<class T, int N, int M> int **f(const T (&a)[N][M]); // #cwg1591-f-2
   
   template<class T, int N> int *f(const T (&a)[N]); // #cwg1591-f-1
    int ***p3 = f({  {  {1,2}, {3, 4}  }, {  {5,6}, {7, 8}  }, {  {9,10}, {11, 12}  } });
    int ***p33 = f({  {  {1,2}, {3, 4}  }, {  {5,6}, {7, 8}  }, {  {9,10}, {11, 12, 13}  } });
    // since-cxx11-error@-1 {{no matching function for call to 'f'}}
    //   since-cxx11-note@#cwg1591-f-2 {{candidate template ignored: couldn't infer template argument 'T'}}
    //   since-cxx11-note@#cwg1591-f-1 {{candidate template ignored: couldn't infer template argument 'T'}}
    //   since-cxx11-note@#cwg1591-f-3 {{candidate template ignored: deduced conflicting values for parameter 'O' (2 vs. 3)}}
    int **p2 = f({  {1,2,3}, {3, 4, 5}  });
    int **p22 = f({  {1,2}, {3, 4}  });
    int *p1 = f({1, 2, 3});
  }
  namespace check_multi_dim_arrays_rref {
    template<class T, int N, int M, int O> int ***g(T (&&a)[N][M][O]); // #cwg1591-g-3
    template<class T, int N, int M> int **g(T (&&a)[N][M]); // #cwg1591-g-2
   
    template<class T, int N> int *g(T (&&a)[N]); // #cwg1591-g-1
    int ***p3 = g({  {  {1,2}, {3, 4}  }, {  {5,6}, {7, 8}  }, {  {9,10}, {11, 12}  } });
    int ***p33 = g({  {  {1,2}, {3, 4}  }, {  {5,6}, {7, 8}  }, {  {9,10}, {11, 12, 13}  } });
    // since-cxx11-error@-1 {{no matching function for call to 'g'}}
    //   since-cxx11-note@#cwg1591-g-2 {{candidate template ignored: couldn't infer template argument 'T'}}
    //   since-cxx11-note@#cwg1591-g-1 {{candidate template ignored: couldn't infer template argument 'T'}}
    //   since-cxx11-note@#cwg1591-g-3 {{candidate template ignored: deduced conflicting values for parameter 'O' (2 vs. 3)}}
    int **p2 = g({  {1,2,3}, {3, 4, 5}  });
    int **p22 = g({  {1,2}, {3, 4}  });
    int *p1 = g({1, 2, 3});
  }
  
  namespace check_arrays_of_init_list {
    template<class T, int N> float *h(const std::initializer_list<T> (&)[N]);
    template<class T, int N> double *h(const T(&)[N]);
    double *p = h({1, 2, 3});
    float *fp = h({{1}, {1, 2}, {1, 2, 3}});
  }
  namespace core_reflector_28543 {
    
    template<class T, int N> int *i(T (&&)[N]);  // #1
    template<class T> char *i(std::initializer_list<T> &&);  // #2
    template<class T, int N, int M> int **i(T (&&)[N][M]); // #3 #cwg1591-i-2
    template<class T, int N> char **i(std::initializer_list<T> (&&)[N]); // #4 #cwg1591-i-1

    template<class T> short *i(T (&&)[2]);  // #5

    template<class T> using Arr = T[];
     
    char *pc = i({1, 2, 3}); // OK prefer #2 via 13.3.3.2 [over.ics.rank]
    char *pc2 = i({1, 2}); // #2 also 
    int *pi = i(Arr<int>{1, 2, 3}); // OK prefer #1

    void *pv1 = i({ {1, 2, 3}, {4, 5, 6} }); // ambiguous btw 3 & 4
    // since-cxx11-error@-1 {{call to 'i' is ambiguous}} 
    //   since-cxx11-note@#cwg1591-i-2 {{candidate function [with T = int, N = 2, M = 3]}}
    //   since-cxx11-note@#cwg1591-i-1 {{candidate function [with T = int, N = 2]}}
    char **pcc = i({ {1}, {2, 3} }); // OK #4

    short *ps = i(Arr<int>{1, 2});  // OK #5
  }
#endif
} // cwg1591
