// RUN: %clang_cc1 -std=c++11 -verify %s

typedef int (*fp)(int);
int surrogate(int);
struct Incomplete;  // expected-note{{forward declaration of 'Incomplete'}} \
                    // expected-note {{forward declaration of 'Incomplete'}}

struct X {
  X() = default;  // expected-note{{candidate constructor not viable: requires 0 arguments, but 1 was provided}}
  X(const X&) = default;  // expected-note{{candidate constructor not viable: no known conversion from 'bool' to 'const X' for 1st argument}}
  X(bool b) __attribute__((enable_if(b, "chosen when 'b' is true")));  // expected-note{{candidate disabled: chosen when 'b' is true}}

  void f(int n) __attribute__((enable_if(n == 0, "chosen when 'n' is zero")));
  void f(int n) __attribute__((enable_if(n == 1, "chosen when 'n' is one")));  // expected-note{{member declaration nearly matches}} expected-note 2{{candidate disabled: chosen when 'n' is one}}

  void g(int n) __attribute__((enable_if(n == 0, "chosen when 'n' is zero")));  // expected-note{{candidate disabled: chosen when 'n' is zero}}

  void h(int n, int m = 0) __attribute__((enable_if(m == 0, "chosen when 'm' is zero")));  // expected-note{{candidate disabled: chosen when 'm' is zero}}

  static void s(int n) __attribute__((enable_if(n == 0, "chosen when 'n' is zero")));  // expected-note2{{candidate disabled: chosen when 'n' is zero}}

  void conflict(int n) __attribute__((enable_if(n+n == 10, "chosen when 'n' is five")));  // expected-note{{candidate function}}
  void conflict(int n) __attribute__((enable_if(n*2 == 10, "chosen when 'n' is five")));  // expected-note{{candidate function}}

  void hidden_by_argument_conversion(Incomplete n, int m = 0) __attribute__((enable_if(m == 10, "chosen when 'm' is ten")));
  Incomplete hidden_by_incomplete_return_value(int n = 0) __attribute__((enable_if(n == 10, "chosen when 'n' is ten"))); // expected-note{{'hidden_by_incomplete_return_value' declared here}}

  operator long() __attribute__((enable_if(true, "chosen on your platform")));
  operator int() __attribute__((enable_if(false, "chosen on other platform")));

  operator fp() __attribute__((enable_if(false, "never enabled"))) { return surrogate; }  // expected-note{{conversion candidate of type 'int (*)(int)'}}  // FIXME: the message is not displayed
};

void X::f(int n) __attribute__((enable_if(n == 0, "chosen when 'n' is zero")))  // expected-note{{member declaration nearly matches}} expected-note 2{{candidate disabled: chosen when 'n' is zero}}
{
}

void X::f(int n) __attribute__((enable_if(n == 2, "chosen when 'n' is two")))  // expected-error{{out-of-line definition of 'f' does not match any declaration in 'X'}}
{
}

X x1(true);
X x2(false); // expected-error{{no matching constructor for initialization of 'X'}}

__attribute__((deprecated)) constexpr int old() { return 0; }  // expected-note2{{'old' has been explicitly marked deprecated here}}
void deprec1(int i) __attribute__((enable_if(old() == 0, "chosen when old() is zero")));  // expected-warning{{'old' is deprecated}}
void deprec2(int i) __attribute__((enable_if(old() == 0, "chosen when old() is zero")));  // expected-warning{{'old' is deprecated}}

void overloaded(int);
void overloaded(long);

struct Int {
  constexpr Int(int i) : i(i) { }
  constexpr operator int() const { return i; }
  int i;
};

void default_argument(int n, int m = 0) __attribute__((enable_if(m == 0, "chosen when 'm' is zero")));  // expected-note{{candidate disabled: chosen when 'm' is zero}}
void default_argument_promotion(int n, int m = Int(0)) __attribute__((enable_if(m == 0, "chosen when 'm' is zero")));  // expected-note{{candidate disabled: chosen when 'm' is zero}}

struct Nothing { };
template<typename T> void typedep(T t) __attribute__((enable_if(t, "")));  // expected-note{{candidate disabled:}}  expected-error{{value of type 'Nothing' is not contextually convertible to 'bool'}}
template<int N> void valuedep() __attribute__((enable_if(N == 1, "")));

// FIXME: we skip potential constant expression evaluation on value dependent
// enable-if expressions
int not_constexpr();
template<int N> void valuedep() __attribute__((enable_if(N == not_constexpr(), "")));

template <typename T> void instantiationdep() __attribute__((enable_if(sizeof(sizeof(T)) != 0, "")));

void test() {
  X x;
  x.f(0);
  x.f(1);
  x.f(2);  // expected-error{{no matching member function for call to 'f'}}
  x.f(3);  // expected-error{{no matching member function for call to 'f'}}

  x.g(0);
  x.g(1);  // expected-error{{no matching member function for call to 'g'}}

  x.h(0);
  x.h(1, 2);  // expected-error{{no matching member function for call to 'h'}}

  x.s(0);
  x.s(1);  // expected-error{{no matching member function for call to 's'}}

  X::s(0);
  X::s(1);  // expected-error{{no matching member function for call to 's'}}

  x.conflict(5);  // expected-error{{call to member function 'conflict' is ambiguous}}

  x.hidden_by_argument_conversion(10);  // expected-error{{argument type 'Incomplete' is incomplete}}
  x.hidden_by_incomplete_return_value(10);  // expected-error{{calling 'hidden_by_incomplete_return_value' with incomplete return type 'Incomplete'}}

  deprec2(0);

  overloaded(x);

  default_argument(0);
  default_argument(1, 2);  // expected-error{{no matching function for call to 'default_argument'}}

  default_argument_promotion(0);
  default_argument_promotion(1, 2);  // expected-error{{no matching function for call to 'default_argument_promotion'}}

  int i = x(1);  // expected-error{{no matching function for call to object of type 'X'}}

  Nothing n;
  typedep(0);  // expected-error{{no matching function for call to 'typedep'}}
  typedep(1);
  typedep(n);  // expected-note{{in instantiation of function template specialization 'typedep<Nothing>' requested here}}
}

template <typename T> class C {
  void f() __attribute__((enable_if(T::expr == 0, ""))) {}
  void g() { f(); }
};

int fn3(bool b) __attribute__((enable_if(b, "")));
template <class T> void test3() {
  fn3(sizeof(T) == 1);
}

template <typename T>
struct Y {
  T h(int n, int m = 0) __attribute__((enable_if(m == 0, "chosen when 'm' is zero")));  // expected-note{{candidate disabled: chosen when 'm' is zero}}
};

void test4() {
  Y<int> y;

  int t0 = y.h(0);
  int t1 = y.h(1, 2);  // expected-error{{no matching member function for call to 'h'}}
}

// FIXME: issue an error (without instantiation) because ::h(T()) is not
// convertible to bool, because return types aren't overloadable.
void h(int);
template <typename T> void outer() {
  void local_function() __attribute__((enable_if(::h(T()), "")));
  local_function();
};

namespace PR20988 {
  struct Integer {
    Integer(int);
  };

  int fn1(const Integer &) __attribute__((enable_if(true, "")));
  template <class T> void test1() {
    int &expr = T::expr();
    fn1(expr);
  }

  int fn2(const Integer &) __attribute__((enable_if(false, "")));  // expected-note{{candidate disabled}}
  template <class T> void test2() {
    int &expr = T::expr();
    fn2(expr);  // expected-error{{no matching function for call to 'fn2'}}
  }

  int fn3(bool b) __attribute__((enable_if(b, "")));
  template <class T> void test3() {
    fn3(sizeof(T) == 1);
  }
}

namespace FnPtrs {
  int ovlFoo(int m) __attribute__((enable_if(m > 0, "")));
  int ovlFoo(int m);

  void test() {
    // Assignment gives us a different code path than declarations, and `&foo`
    // gives us a different code path than `foo`
    int (*p)(int) = ovlFoo;
    int (*p2)(int) = &ovlFoo;
    int (*a)(int);
    a = ovlFoo;
    a = &ovlFoo;
  }

  int ovlBar(int) __attribute__((enable_if(true, "")));
  int ovlBar(int m) __attribute__((enable_if(false, "")));
  void test2() {
    int (*p)(int) = ovlBar;
    int (*p2)(int) = &ovlBar;
    int (*a)(int);
    a = ovlBar;
    a = &ovlBar;
  }

  int ovlConflict(int m) __attribute__((enable_if(true, "")));
  int ovlConflict(int m) __attribute__((enable_if(1, "")));
  void test3() {
    int (*p)(int) = ovlConflict; // expected-error{{address of overloaded function 'ovlConflict' is ambiguous}} expected-note@191{{candidate function}} expected-note@192{{candidate function}}
    int (*p2)(int) = &ovlConflict; // expected-error{{address of overloaded function 'ovlConflict' is ambiguous}} expected-note@191{{candidate function}} expected-note@192{{candidate function}}
    int (*a)(int);
    a = ovlConflict; // expected-error{{assigning to 'int (*)(int)' from incompatible type '<overloaded function type>'}} expected-note@191{{candidate function}} expected-note@192{{candidate function}}
    a = &ovlConflict; // expected-error{{assigning to 'int (*)(int)' from incompatible type '<overloaded function type>'}} expected-note@191{{candidate function}} expected-note@192{{candidate function}}
  }

  template <typename T>
  T templated(T m) __attribute__((enable_if(true, ""))) { return T(); }
  template <typename T>
  T templated(T m) __attribute__((enable_if(false, ""))) { return T(); }
  void test4() {
    int (*p)(int) = templated<int>;
    int (*p2)(int) = &templated<int>;
    int (*a)(int);
    a = templated<int>;
    a = &templated<int>;
  }

  template <typename T>
  T templatedBar(T m) __attribute__((enable_if(m > 0, ""))) { return T(); }
  void test5() {
    int (*p)(int) = templatedBar<int>; // expected-error{{address of overloaded function 'templatedBar' does not match required type 'int (int)'}} expected-note@214{{candidate function made ineligible by enable_if}}
    int (*p2)(int) = &templatedBar<int>; // expected-error{{address of overloaded function 'templatedBar' does not match required type 'int (int)'}} expected-note@214{{candidate function made ineligible by enable_if}}
    int (*a)(int);
    a = templatedBar<int>; // expected-error{{assigning to 'int (*)(int)' from incompatible type '<overloaded function type>'}} expected-note@214{{candidate function made ineligible by enable_if}}
    a = &templatedBar<int>; // expected-error{{assigning to 'int (*)(int)' from incompatible type '<overloaded function type>'}} expected-note@214{{candidate function made ineligible by enable_if}}
  }

  template <typename T>
  T templatedConflict(T m) __attribute__((enable_if(false, ""))) { return T(); }
  template <typename T>
  T templatedConflict(T m) __attribute__((enable_if(true, ""))) { return T(); }
  template <typename T>
  T templatedConflict(T m) __attribute__((enable_if(1, ""))) { return T(); }
  void test6() {
    int (*p)(int) = templatedConflict<int>; // expected-error{{address of overloaded function 'templatedConflict' is ambiguous}} expected-note@224{{candidate function made ineligible by enable_if}} expected-note@226{{candidate function}} expected-note@228{{candidate function}}
    int (*p0)(int) = &templatedConflict<int>; // expected-error{{address of overloaded function 'templatedConflict' is ambiguous}} expected-note@224{{candidate function made ineligible by enable_if}} expected-note@226{{candidate function}} expected-note@228{{candidate function}}
    int (*a)(int);
    a = templatedConflict<int>; // expected-error{{assigning to 'int (*)(int)' from incompatible type '<overloaded function type>'}} expected-note@226{{candidate function}} expected-note@228{{candidate function}}
    a = &templatedConflict<int>; // expected-error{{assigning to 'int (*)(int)' from incompatible type '<overloaded function type>'}} expected-note@226{{candidate function}} expected-note@228{{candidate function}}
  }

  int ovlNoCandidate(int m) __attribute__((enable_if(false, "")));
  int ovlNoCandidate(int m) __attribute__((enable_if(0, "")));
  void test7() {
    int (*p)(int) = ovlNoCandidate; // expected-error{{address of overloaded function 'ovlNoCandidate' does not match required type}} expected-note@237{{made ineligible by enable_if}} expected-note@238{{made ineligible by enable_if}}
    int (*p2)(int) = &ovlNoCandidate; // expected-error{{address of overloaded function 'ovlNoCandidate' does not match required type}} expected-note@237{{made ineligible by enable_if}} expected-note@238{{made ineligible by enable_if}}
    int (*a)(int);
    a = ovlNoCandidate; // expected-error{{assigning to 'int (*)(int)' from incompatible type '<overloaded function type>'}} expected-note@237{{made ineligible by enable_if}} expected-note@238{{made ineligible by enable_if}}
    a = &ovlNoCandidate; // expected-error{{assigning to 'int (*)(int)' from incompatible type '<overloaded function type>'}} expected-note@237{{made ineligible by enable_if}} expected-note@238{{made ineligible by enable_if}}
  }

  int noOvlNoCandidate(int m) __attribute__((enable_if(false, "")));
  void test8() {
    int (*p)(int) = noOvlNoCandidate; // expected-error{{cannot take address of function 'noOvlNoCandidate' becuase it has one or more non-tautological enable_if conditions}}
    int (*p2)(int) = &noOvlNoCandidate; // expected-error{{cannot take address of function 'noOvlNoCandidate' becuase it has one or more non-tautological enable_if conditions}}
    int (*a)(int);
    a = noOvlNoCandidate; // expected-error{{cannot take address of function 'noOvlNoCandidate' becuase it has one or more non-tautological enable_if conditions}}
    a = &noOvlNoCandidate; // expected-error{{cannot take address of function 'noOvlNoCandidate' becuase it has one or more non-tautological enable_if conditions}}
  }
}
