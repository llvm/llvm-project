// RUN: %clang_cc1 -triple=x86_64-unknown-unknown -fsyntax-only -verify -std=c++11 %s
// RUN: %clang_cc1 -triple=x86_64-unknown-unknown -fsyntax-only -verify -std=c++20 %s
// RUN: %clang_cc1 -triple=x86_64-unknown-unknown -fsyntax-only -verify -std=c++2c %s

namespace std {
  typedef decltype(sizeof(int)) size_t;
  template <class _E> class initializer_list {
    const _E *__begin_;
    size_t __size_;

    constexpr initializer_list(const _E *__b, size_t __s)
        : __begin_(__b), __size_(__s) {}

  public:
    constexpr initializer_list() : __begin_(nullptr), __size_(0) {}
  };
} // namespace std

template <typename T>
struct Invalid { static_assert(false, "instantiated Invalid"); }; // #err-invalid

template <typename T>
int f(T a, Invalid<T> = {}); // #note-f

// sanity check
int e1 = f(0);
//expected-error@#err-invalid {{static assertion failed: instantiated Invalid}}
//expected-note@-2 {{in instantiation of default function argument expression for 'f<int>' required here}}
//expected-note@#note-f {{in instantiation of template class 'Invalid<int>' requested here}}
//expected-note@#note-f {{passing argument to parameter here}}

int f(int);
int ok1 = f(0);
int e4 = f((const int&)(ok1));

int f(int, int = 0);
int ok2 = f(0, 0);

int e2  = f(0L);
//expected-error@#err-invalid {{static assertion failed: instantiated Invalid}}
//expected-note@-2 {{in instantiation of default function argument expression for 'f<long>' required here}}
//expected-note@#note-f {{in instantiation of template class 'Invalid<long>' requested here}}
//expected-note@#note-f {{passing argument to parameter here}}

int f(long);
int ok3 = f(0L);

template <typename T>
struct Invalid2 { static_assert(false, "instantiated Invalid2"); }; // #err-qualifiers

template <typename T>
int ref(T a, Invalid2<T> = {}); // expected-note 2{{here}}
int ref(int&);
int ref1 = ref(ok3);
int ref2 = ref((const int&)ok3); // expected-note {{here}}
//expected-error@#err-qualifiers {{static assertion failed: instantiated Invalid2}}


template <typename T>
int f_alias(T a, Invalid<T> = {});
using Alias = int;
int f_alias(Alias);
int ok4 = f_alias(0);

#if __cplusplus >= 202002

struct Copyable {
  template <typename T>
  requires __is_constructible(Copyable, T)
  explicit Copyable(T op) noexcept; // #1
  Copyable(const Copyable&) noexcept = default; // #2
};
static_assert(__is_constructible(Copyable, const Copyable&));

struct ImplicitlyCopyable {
  template <typename T>
  requires __is_constructible(ImplicitlyCopyable, T)
  explicit ImplicitlyCopyable(T op) = delete; // #1
};
static_assert(__is_constructible(ImplicitlyCopyable, const ImplicitlyCopyable&));


struct Movable { // #Movable
  template <typename T>
  requires __is_constructible(Movable, T) // #err-self-constraint-1
  explicit Movable(T op) noexcept; // #Movable1
  Movable(Movable&&) noexcept = default; // #Movable2
};
static_assert(__is_constructible(Movable, Movable&&));
static_assert(__is_constructible(Movable, const Movable&));
// expected-error@-1 {{static assertion failed due to requirement '__is_constructible(Movable, const Movable &)'}} \
// expected-error@-1 {{call to implicitly-deleted copy constructor of 'Movable'}} \
// expected-note@#Movable  {{'Movable' defined here}} \
// expected-note@#Movable  {{candidate constructor (the implicit copy constructor) not viable: no known conversion from 'int' to 'const Movable' for 1st argument}} \
// expected-note@#Movable2  {{copy constructor is implicitly deleted because 'Movable' has a user-declared move constructor}} \
// expected-note@#Movable2  {{candidate constructor not viable: no known conversion from 'int' to 'Movable' for 1st argument}} \
// expected-note@#Movable1  {{candidate template ignored: constraints not satisfied [with T = int]}}


static_assert(__is_constructible(Movable, int));
// expected-error@-1 {{static assertion failed due to requirement '__is_constructible(Movable, int)'}} \
// expected-error@-1 {{no matching constructor for initialization of 'Movable'}} \
// expected-note@-1 2{{}}
// expected-error@#err-self-constraint-1{{satisfaction of constraint '__is_constructible(Movable, T)' depends on itself}}
// expected-note@#err-self-constraint-1 4{{}}
// expected-note@#Movable  {{'Movable' defined here}}

template <typename T>
struct Members {
    constexpr auto f(auto) {
        static_assert(false, "");
    }
    constexpr auto f(int) { return 1; }
    constexpr auto f(int) requires true { return 2; }

    constexpr auto g(auto) {
        static_assert(false, "instantiated member"); //#err-qualified-member
        return 0;
    }
    constexpr auto g(int) & { return 1; }

    static constexpr auto s(auto) {
        static_assert(false, "");
    }
    static constexpr auto s(int) {
        return 1;
    }
    static constexpr auto s(int) requires true {
        return 2;
    }
};

static_assert(Members<int[1]>{}.f(0) == 2);
static_assert(Members<int[2]>{}.g(0) == 0);
// expected-error@#err-qualified-member {{static assertion failed: instantiated member}} \
// expected-note@-1{{in instantiation of function template specialization 'Members<int[2]>::g<int>' }}
Members<int[3]> m1;
static_assert(m1.g(0) == 1);
static_assert(Members<int[3]>{}.s(0) == 2);


namespace ConstructorInit{
struct S {
  template <typename T>
  S(T&&) {}
};
struct Test {
  operator S() = delete;
};

static_assert(__is_constructible(S, Test));
}

namespace RefBinding {

template <typename> struct remove_reference;
template <typename _Tp> struct remove_reference<_Tp &> {
  using type = _Tp;
};
template <typename _Tp> remove_reference<_Tp>::type move(_Tp &&);
template <typename _Head> struct _Head_base {
  _Head_base(_Head &__h) : _M_head_impl(__h) {}
  template <typename _UHead> _Head_base(_UHead &&);
  _Head _M_head_impl;
};

template <typename _Elements> void forward_as_tuple(_Elements &&) {
  _Head_base<_Elements &&>(_Elements{});
}
struct StringRef {
  void operator[](const StringRef __k) { forward_as_tuple((move)(__k)); }
};

}

template <class> struct tuple {};
struct BonkersBananas {
  template <class T> operator T();
  template <class = void> explicit operator tuple<int>() = delete;
};
static_assert(!__is_constructible(tuple<int>, BonkersBananas));

namespace GH62096 {
template <typename T>
struct Oops {
  static_assert(sizeof(T) == 0); // #GH62096-err
  static constexpr bool value = true;
};

template <class OPERATOR>
concept Operator = Oops<OPERATOR>::value; // #GH62096-note1

template <Operator OP> void f(OP op); // // #GH62096-note2
void f(int);

void g(int n) { f(n); } // OK
void h(short n) { f(n); }
// expected-error@#GH62096-err {{static assertion failed due to requirement 'sizeof(short) == 0'}} \
// expected-note@-1{{while substituting deduced template arguments}} \
// expected-note@-1{{while checking constraint satisfaction for template}}
// expected-note@#GH62096-note1{{in instantiation}}
// expected-note@#GH62096-note1{{while substituting template arguments into constraint expression here}}
// expected-note@#GH62096-note2{{while substituting template arguments into constraint expression here}}
// expected-note@#GH62096-note2{{while checking the satisfaction of concept}}
// expected-note@#GH62096-err {{expression evaluates}}
}

#endif

template<typename ...Ts>
struct t1 {
};
struct t6 {
  template<typename T = int>
  operator t1<float>() {
    return {};
  }
};

void testT6() {
  t6 v6;
  v6.operator t1<float>();
}


using a = void(int &);
template <typename c> void d(c &);
void f(a);
template <class> void f(bool j) { f(&d<int>); }

struct InitListAreNotPerfect {
  InitListAreNotPerfect(int) = delete;
  template<class T>
  InitListAreNotPerfect(std::initializer_list<T>);
};
InitListAreNotPerfect InitListAreNotPerfect_test({0});
struct InitListAreNotPerfectCpy {
  InitListAreNotPerfectCpy();
  InitListAreNotPerfectCpy(const InitListAreNotPerfectCpy&);
  template <typename T> InitListAreNotPerfectCpy(std::initializer_list<T>);
};

InitListAreNotPerfectCpy InitListAreNotPerfectCpy_test({InitListAreNotPerfectCpy{}});

namespace PointerToMemFunc {
template <typename>
class A;
struct N {
  template <typename T>
  void f(T);
};
template <typename T>
struct E {
  template <class = A<int>>
  void g() = delete;
  void g(void (T::*)(char));
};
void f() {
  E<N> e;
  e.g(&N::f);
}
}

#if __cplusplus >= 201402
namespace PointerToMemData {
struct N {
  int field;
};
template <typename It, typename T>
struct B {
  B(It, T);
  template <typename It2>
  B(B<It2, T>);
};
template <typename T>
struct C {
  auto g() { return B<int, T>(0, T{}); }
};
void f() {
  using T = decltype(C<decltype(&N::field)>{}.g());
}

}

#endif

namespace GH147374 {

struct String {};
template <typename T> void operator+(T, String &&) = delete;

struct Bar {
    void operator+(String) const; // expected-note {{candidate function}}
    friend void operator+(Bar, String) {};  // expected-note {{candidate function}}
};

struct Baz {
    void operator+(String); // expected-note {{candidate function}}
    friend void operator+(Baz, String) {}; // expected-note {{candidate function}}
};

void test() {
    Bar a;
    String b;
    a + b;
    //expected-error@-1 {{use of overloaded operator '+' is ambiguous (with operand types 'Bar' and 'String')}}

    Baz z;
    z + b;
    //expected-error@-1 {{use of overloaded operator '+' is ambiguous (with operand types 'Baz' and 'String')}}
}

}
