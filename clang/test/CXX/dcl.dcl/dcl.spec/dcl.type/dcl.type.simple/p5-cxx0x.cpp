// RUN: %clang_cc1 -std=c++11 -verify %s

namespace std_example {

template<class T> struct A { ~A() = delete; }; // expected-note {{deleted here}}
template<class T> auto h() -> A<T>;
template<class T> auto i(T) -> T;
template<class T> auto f(T) -> decltype(i(h<T>())); // #1
template<class T> auto f(T) -> void; // #2
auto g() -> void {
  f(42); // ok, calls #2, since #1 is not viable.
}
template<class T> auto q(T) -> decltype((h<T>()));
void r() {
  // Deduction against q succeeds, but results in a temporary which can't be
  // destroyed.
  q(42); // expected-error {{attempt to use a deleted function}}
}

}

class PD {
  friend struct A;
  ~PD(); // expected-note 5{{here}}
public:
  typedef int n;
};
struct DD {
  ~DD() = delete; // expected-note 2{{here}}
  typedef int n;
};

PD pd();
DD dd();

struct A {
  decltype(pd()) s; // ok
  decltype(pd())::n n; // ok
  decltype(dd()) *p = new decltype(dd()); // ok
};
A a();

// Two errors here: one for the decltype, one for the variable.
decltype(
    pd(), // expected-error {{private destructor}}
    pd()) pd1; // expected-error {{private destructor}}
decltype(dd(), // expected-error {{deleted function}}
         dd()) dd1;
decltype(a(),
         dd()) dd2; // expected-error {{deleted function}}
decltype(
    pd(), // expected-error {{temporary of type 'PD' has private destructor}}
    0) pd2;

decltype(((13, ((dd())))))::n dd_parens; // ok
decltype(((((42)), pd())))::n pd_parens_comma; // ok

// Ensure parens aren't stripped from a decltype node.
extern decltype(pd()) pd_ref; // ok
decltype((pd_ref)) pd_ref3 = pd_ref; // ok, PD &
decltype(pd_ref) pd_ref2 = pd_ref; // expected-error {{private destructor}}

namespace libcxx_example {
  struct nat {
    nat() = delete;
    nat(const nat&) = delete;
    nat &operator=(const nat&) = delete;
    ~nat() = delete;
  };
  struct any {
    any(...);
  };

  template<typename T, typename U> struct is_same { static const bool value = false; };
  template<typename T> struct is_same<T, T> { static const bool value = true; };

  template<typename T> T declval();

  void swap(int &a, int &b);
  nat swap(any, any);

  template<typename T> struct swappable {
    typedef decltype(swap(declval<T&>(), declval<T&>())) type;
    static const bool value = !is_same<type, nat>::value;
    constexpr operator bool() const { return value; }
  };

  static_assert(swappable<int>(), "");
  static_assert(!swappable<const int>(), "");
}

namespace RequireCompleteType {
  template<int N, bool OK> struct S {
    static_assert(OK, "boom!"); // expected-error 2{{boom!}}
  };

  template<typename T> T make();
  template<int N, bool OK> S<N, OK> make();
  void consume(...);

  decltype(make<0, false>()) *p1; // ok
  decltype((make<1, false>())) *p2; // ok

  // A complete type is required here in order to detect an overloaded 'operator,'.
  decltype(123, make<2, false>()) *p3; // expected-note {{here}}

  decltype(consume(make<3, false>())) *p4; // expected-note {{here}}

  decltype(make<decltype(make<4, false>())>()) *p5; // ok
}

namespace Overload {
  DD operator+(PD &a, PD &b);
  decltype(pd()) *pd_ptr;
  decltype(*pd_ptr + *pd_ptr) *dd_ptr; // ok

  decltype(0, *pd_ptr) pd_ref2 = pd_ref; // ok
  DD operator,(int a, PD b);
  decltype(0, *pd_ptr) *dd_ptr2; // expected-error {{private destructor}}
}
