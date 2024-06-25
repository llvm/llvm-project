// RUN: %clang_cc1 -std=c++2c -fsyntax-only -verify %s

struct S {
  void f() = delete("deleted (1)"); // expected-note {{explicitly marked deleted}}

  template <typename T>
  T g() = delete("deleted (2)"); // expected-note {{explicitly deleted}}
};

template <typename T>
struct TS {
  T f() = delete("deleted (3)"); // expected-note {{explicitly marked deleted}}

  template <typename U>
  T g(U) = delete("deleted (4)"); // expected-note {{explicitly deleted}}
};

void f() = delete("deleted (5)"); // expected-note {{explicitly deleted}}

template <typename T>
T g() = delete("deleted (6)"); // expected-note {{explicitly deleted}}

template <typename T, typename U>
struct TSp {
  T f() = delete("deleted (7)"); // expected-note 2 {{explicitly marked deleted}}
};

template <typename T>
struct TSp<T, int> {
  T f() = delete("deleted (8)"); // expected-note {{explicitly marked deleted}}
};

template <>
struct TSp<int, int> {
  int f() = delete("deleted (9)"); // expected-note {{explicitly marked deleted}}
};

void u1() = delete(L"\xFFFFFFFF"); // expected-error {{an unevaluated string literal cannot have an encoding prefix}} \
                                   // expected-error {{invalid escape sequence '\xFFFFFFFF' in an unevaluated string literal}}
void u2() = delete(u"\U000317FF"); // expected-error {{an unevaluated string literal cannot have an encoding prefix}}

void u3() = delete("Ω"); // expected-note {{explicitly deleted}}
void u4() = delete("\u1234"); // expected-note {{explicitly deleted}}

void u5() = delete("\x1ff"       // expected-error {{hex escape sequence out of range}} \
                                 // expected-error {{invalid escape sequence '\x1ff' in an unevaluated string literal}}
                     "0\x123"    // expected-error {{invalid escape sequence '\x123' in an unevaluated string literal}}
                     "fx\xfffff" // expected-error {{invalid escape sequence '\xfffff' in an unevaluated string literal}}
                     "goop");

void u6() = delete("\'\"\?\\\a\b\f\n\r\t\v"); // expected-note {{explicitly deleted}}
void u7() = delete("\xFF"); // expected-error {{invalid escape sequence '\xFF' in an unevaluated string literal}}
void u8() = delete("\123"); // expected-error {{invalid escape sequence '\123' in an unevaluated string literal}}
void u9() = delete("\pOh no, a Pascal string!"); // expected-warning {{unknown escape sequence '\p'}} \
                                                 // expected-error {{invalid escape sequence '\p' in an unevaluated string literal}}
// expected-note@+1 {{explicitly deleted}}
void u10() = delete(R"(a
\tb
c
)");

void u11() = delete("\u0080\u0081\u0082\u0083\u0099\u009A\u009B\u009C\u009D\u009E\u009F"); // expected-note {{explicitly deleted}}


//! Contains RTL/LTR marks
void u12() = delete("\u200Eabc\u200Fdef\u200Fgh"); // expected-note {{explicitly deleted}}

//! Contains ZWJ/regional indicators
void u13() = delete("🏳️‍🌈 🏴󠁧󠁢󠁥󠁮󠁧󠁿 🇪🇺"); // expected-note {{explicitly deleted}}

void h() {
  S{}.f(); // expected-error {{attempt to use a deleted function: deleted (1)}}
  S{}.g<int>(); // expected-error {{call to deleted member function 'g': deleted (2)}}
  TS<int>{}.f(); // expected-error {{attempt to use a deleted function: deleted (3)}}
  TS<int>{}.g<int>(0); // expected-error {{call to deleted member function 'g': deleted (4)}}
  f(); // expected-error {{call to deleted function 'f': deleted (5)}}
  g<int>(); // expected-error {{call to deleted function 'g': deleted (6)}}
  TSp<double, double>{}.f(); // expected-error {{attempt to use a deleted function: deleted (7)}}
  TSp<int, double>{}.f(); // expected-error {{attempt to use a deleted function: deleted (7)}}
  TSp<double, int>{}.f(); // expected-error {{attempt to use a deleted function: deleted (8)}}
  TSp<int, int>{}.f(); // expected-error {{attempt to use a deleted function: deleted (9)}}
  u3(); // expected-error {{call to deleted function 'u3': Ω}}
  u4(); // expected-error {{call to deleted function 'u4': ሴ}}
  u6(); // expected-error {{call to deleted function 'u6': '"?\<U+0007><U+0008>}}
  u10(); // expected-error {{call to deleted function 'u10': a\n\tb\nc\n}}
  u11(); // expected-error {{call to deleted function 'u11': <U+0080><U+0081><U+0082><U+0083><U+0099><U+009A><U+009B><U+009C><U+009D><U+009E><U+009F>}}
  u12(); // expected-error {{call to deleted function 'u12': ‎abc‏def‏gh}}
  u13(); // expected-error {{call to deleted function 'u13': 🏳️‍🌈 🏴󠁧󠁢󠁥󠁮󠁧󠁿 🇪🇺}}
}

struct C {
  C() = delete("deleted (C, Constructor)"); // expected-note {{explicitly marked deleted}}
  C(int) = delete("deleted (C, C(int))"); // expected-note {{explicitly marked deleted}}
  C(const C&) = delete("deleted (C, Copy Constructor)"); // expected-note {{explicitly marked deleted}}
  C(C&&) = delete("deleted (C, Move Constructor)"); // expected-note {{explicitly marked deleted}}
  C& operator=(const C&) = delete("deleted (C, Copy Assignment)"); // expected-note 2 {{explicitly deleted}}
  C& operator=(C&&) = delete("deleted (C, Move Assignment)"); // expected-note {{explicitly deleted}} expected-note {{not viable}}
  ~C() = delete("deleted (C, Destructor)"); // expected-note {{explicitly marked deleted}}
  void* operator new(__SIZE_TYPE__) = delete("deleted (C, New)"); // expected-note {{explicitly deleted}}
  void operator delete(void*) = delete("deleted (C, Delete)"); // expected-note {{explicitly marked deleted}}
};

template <typename T, typename U>
struct TC {
  TC() = delete("deleted (TC, Constructor)"); // expected-note {{explicitly marked deleted}}
  TC(int) = delete("deleted (TC, TC(int))"); // expected-note {{explicitly marked deleted}}
  TC(const TC&) = delete("deleted (TC, Copy Constructor)"); // expected-note {{explicitly marked deleted}}
  TC(TC&&) = delete("deleted (TC, Move Constructor)"); // expected-note {{explicitly marked deleted}}
  TC& operator=(const TC&) = delete("deleted (TC, Copy Assignment)"); // expected-note 2 {{explicitly deleted}}
  TC& operator=(TC&&) = delete("deleted (TC, Move Assignment)");  // expected-note {{explicitly deleted}} expected-note {{not viable}}
  ~TC() = delete("deleted (TC, Destructor)"); // expected-note {{explicitly marked deleted}}
  void* operator new(__SIZE_TYPE__) = delete("deleted (TC, New)"); // expected-note {{explicitly deleted}}
  void operator delete(void*) = delete("deleted (TC, Delete)"); // expected-note {{explicitly marked deleted}}
};

template <typename T>
struct TC<T, int> {
  TC() = delete("deleted (TC<T, int>, Constructor)"); // expected-note {{explicitly marked deleted}}
  TC(int) = delete("deleted (TC<T, int>, TC(int))"); // expected-note {{explicitly marked deleted}}
  TC(const TC&) = delete("deleted (TC<T, int>, Copy Constructor)"); // expected-note {{explicitly marked deleted}}
  TC(TC&&) = delete("deleted (TC<T, int>, Move Constructor)"); // expected-note {{explicitly marked deleted}}
  TC& operator=(const TC&) = delete("deleted (TC<T, int>, Copy Assignment)"); // expected-note 2 {{explicitly deleted}}
  TC& operator=(TC&&) = delete("deleted (TC<T, int>, Move Assignment)"); // expected-note {{explicitly deleted}} expected-note {{not viable}}
  ~TC() = delete("deleted (TC<T, int>, Destructor)"); // expected-note {{explicitly marked deleted}}
  void* operator new(__SIZE_TYPE__) = delete("deleted (TC<T, int>, New)"); // expected-note {{explicitly deleted}}
  void operator delete(void*) = delete("deleted (TC<T, int>, Delete)"); // expected-note {{explicitly marked deleted}}
};

template <>
struct TC<int, int> {
  TC() = delete("deleted (TC<int, int>, Constructor)"); // expected-note {{explicitly marked deleted}}
  TC(int) = delete("deleted (TC<int, int>, TC(int))"); // expected-note {{explicitly marked deleted}}
  TC(const TC&) = delete("deleted (TC<int, int>, Copy Constructor)"); // expected-note {{explicitly marked deleted}}
  TC(TC&&) = delete("deleted (TC<int, int>, Move Constructor)"); // expected-note {{explicitly marked deleted}}
  TC& operator=(const TC&) = delete("deleted (TC<int, int>, Copy Assignment)"); // expected-note 2 {{explicitly deleted}}
  TC& operator=(TC&&) = delete("deleted (TC<int, int>, Move Assignment)"); // expected-note {{explicitly deleted}} expected-note {{not viable}}
  ~TC() = delete("deleted (TC<int, int>, Destructor)"); // expected-note {{explicitly marked deleted}}
  void* operator new(__SIZE_TYPE__) = delete("deleted (TC<int, int>, New)"); // expected-note {{explicitly deleted}}
  void operator delete(void*) = delete("deleted (TC<int, int>, Delete)"); // expected-note {{explicitly marked deleted}}
};

void special_members(
  C& c1,
  C& c2,
  TC<double, double>& tc1,
  TC<double, double>& tc2,
  TC<double, int>& tc_int1,
  TC<double, int>& tc_int2,
  TC<int, int>& tc_int_int1,
  TC<int, int>& tc_int_int2
) {
  C{}; // expected-error {{call to deleted constructor of 'C': deleted (C, Constructor)}}
  C{c1}; // expected-error {{call to deleted constructor of 'C': deleted (C, Copy Constructor)}}
  C{static_cast<C&&>(c1)}; // expected-error {{call to deleted constructor of 'C': deleted (C, Move Constructor)}}
  c1 = c2; // expected-error {{overload resolution selected deleted operator '=': deleted (C, Copy Assignment)}}
  c1 = static_cast<C&&>(c2); // expected-error {{overload resolution selected deleted operator '=': deleted (C, Move Assignment)}}
  c1.~C(); // expected-error {{attempt to use a deleted function: deleted (C, Destructor)}}
  new C{}; // expected-error {{call to deleted function 'operator new': deleted (C, New)}}
  delete &c2; // expected-error {{attempt to use a deleted function: deleted (C, Delete)}}

  TC<double, double>{}; // expected-error {{call to deleted constructor of 'TC<double, double>': deleted (TC, Constructor)}}
  TC<double, double>{tc1}; // expected-error {{call to deleted constructor of 'TC<double, double>': deleted (TC, Copy Constructor)}}
  TC<double, double>{static_cast<TC<double, double>&&>(tc1)}; // expected-error {{call to deleted constructor of 'TC<double, double>': deleted (TC, Move Constructor)}}
  tc1 = tc2; // expected-error {{overload resolution selected deleted operator '=': deleted (TC, Copy Assignment)}}
  tc1 = static_cast<TC<double, double>&&>(tc2); // expected-error {{overload resolution selected deleted operator '=': deleted (TC, Move Assignment)}}
  tc1.~TC(); // expected-error {{attempt to use a deleted function: deleted (TC, Destructor)}}
  new TC<double, double>{}; // expected-error {{call to deleted function 'operator new': deleted (TC, New)}}
  delete &tc2; // expected-error {{attempt to use a deleted function: deleted (TC, Delete)}}

  TC<double, int>{}; // expected-error {{call to deleted constructor of 'TC<double, int>': deleted (TC<T, int>, Constructor)}}
  TC<double, int>{tc_int1}; // expected-error {{call to deleted constructor of 'TC<double, int>': deleted (TC<T, int>, Copy Constructor)}}
  TC<double, int>{static_cast<TC<double, int>&&>(tc_int1)}; // expected-error {{call to deleted constructor of 'TC<double, int>': deleted (TC<T, int>, Move Constructor)}}
  tc_int1 = tc_int2; // expected-error {{overload resolution selected deleted operator '=': deleted (TC<T, int>, Copy Assignment)}}
  tc_int1 = static_cast<TC<double, int>&&>(tc_int2); // expected-error {{overload resolution selected deleted operator '=': deleted (TC<T, int>, Move Assignment)}}
  tc_int1.~TC(); // expected-error {{attempt to use a deleted function: deleted (TC<T, int>, Destructor)}}
  new TC<double, int>{}; // expected-error {{call to deleted function 'operator new': deleted (TC<T, int>, New)}}
  delete &tc_int2; // expected-error {{attempt to use a deleted function: deleted (TC<T, int>, Delete)}}

  TC<int, int>{}; // expected-error {{call to deleted constructor of 'TC<int, int>': deleted (TC<int, int>, Constructor)}}
  TC<int, int>{tc_int_int1}; // expected-error {{call to deleted constructor of 'TC<int, int>': deleted (TC<int, int>, Copy Constructor)}}
  TC<int, int>{static_cast<TC<int, int>&&>(tc_int_int1)}; // expected-error {{call to deleted constructor of 'TC<int, int>': deleted (TC<int, int>, Move Constructor)}}
  tc_int_int1 = tc_int_int2; // expected-error {{overload resolution selected deleted operator '=': deleted (TC<int, int>, Copy Assignment)}}
  tc_int_int1 = static_cast<TC<int, int>&&>(tc_int_int2); // expected-error {{overload resolution selected deleted operator '=': deleted (TC<int, int>, Move Assignment)}}
  tc_int_int1.~TC(); // expected-error {{attempt to use a deleted function: deleted (TC<int, int>, Destructor)}}
  new TC<int, int>{}; // expected-error {{call to deleted function 'operator new': deleted (TC<int, int>, New)}}
  delete &tc_int_int2; // expected-error {{attempt to use a deleted function: deleted (TC<int, int>, Delete)}}
}

C conv1() { return 1; } // expected-error {{conversion function from 'int' to 'C' invokes a deleted function: deleted (C, C(int))}}
TC<double, double> conv2() { return 1; } // expected-error {{conversion function from 'int' to 'TC<double, double>' invokes a deleted function: deleted (TC, TC(int))}}
TC<double, int> conv3() { return 1; } // expected-error {{conversion function from 'int' to 'TC<double, int>' invokes a deleted function: deleted (TC<T, int>, TC(int))}}
TC<int, int> conv4() { return 1; } // expected-error {{conversion function from 'int' to 'TC<int, int>' invokes a deleted function: deleted (TC<int, int>, TC(int))}}

struct O {
  int x;
  int operator+() = delete("deleted (O, +)"); // expected-note {{explicitly deleted}}
  O* operator->() = delete("deleted (O, ->)"); // expected-note {{explicitly deleted}}
  int operator-(O) = delete("deleted (O, -)"); // expected-note {{explicitly deleted}}
  int operator[](O) = delete("deleted (O, [])"); // expected-note {{explicitly deleted}}
  int operator()(O) = delete("deleted (O, ())"); // expected-note {{explicitly marked deleted}} expected-note {{explicitly deleted}}
  explicit operator bool() = delete("deleted (O, operator bool)"); // expected-note {{explicitly marked deleted}} expected-note {{explicitly deleted}}
};

template <typename T, typename U>
struct TO {
  T x;
  T operator+() = delete("deleted (TO, +)"); // expected-note {{explicitly deleted}}
  T* operator->() = delete("deleted (TO, ->)"); // expected-note {{explicitly deleted}}
  T operator-(TO) = delete("deleted (TO, -)"); // expected-note {{explicitly deleted}}
  T operator[](TO) = delete("deleted (TO, [])"); // expected-note {{explicitly deleted}}
  T operator()(TO) = delete("deleted (TO, ())"); // expected-note {{explicitly marked deleted}} expected-note {{explicitly deleted}}
  explicit operator bool() = delete("deleted (TO, operator bool)"); // expected-note {{explicitly marked deleted}} expected-note {{explicitly deleted}}
};

template <typename T>
struct TO<T, int> {
  T x;
  T operator+() = delete("deleted (TO<T, int>, +)"); // expected-note {{explicitly deleted}}
  T* operator->() = delete("deleted (TO<T, int>, ->)"); // expected-note {{explicitly deleted}}
  T operator-(TO) = delete("deleted (TO<T, int>, -)"); // expected-note {{explicitly deleted}}
  T operator[](TO) = delete("deleted (TO<T, int>, [])"); // expected-note {{explicitly deleted}}
  T operator()(TO) = delete("deleted (TO<T, int>, ())"); // expected-note {{explicitly marked deleted}} expected-note {{explicitly deleted}}
  explicit operator bool() = delete("deleted (TO<T, int>, operator bool)"); // expected-note {{explicitly marked deleted}} expected-note {{explicitly deleted}}
};

template <>
struct TO<int, int> {
  int x;
  int operator+() = delete("deleted (TO<int, int>, +)"); // expected-note {{explicitly deleted}}
  int* operator->() = delete("deleted (TO<int, int>, ->)"); // expected-note {{explicitly deleted}}
  int operator-(TO) = delete("deleted (TO<int, int>, -)"); // expected-note {{explicitly deleted}}
  int operator[](TO) = delete("deleted (TO<int, int>, [])"); // expected-note {{explicitly deleted}}
  int operator()(TO) = delete("deleted (TO<int, int>, ())"); // expected-note {{explicitly marked deleted}} expected-note {{explicitly deleted}}
  explicit operator bool() = delete("deleted (TO<int, int>, operator bool)"); // expected-note {{explicitly marked deleted}} expected-note {{explicitly deleted}}
};

void operators() {
  O o;
  +o; // expected-error {{overload resolution selected deleted operator '+': deleted (O, +)}}
  o->x; // expected-error {{overload resolution selected deleted operator '->': deleted (O, ->)}}
  o - o; // expected-error {{overload resolution selected deleted operator '-': deleted (O, -)}}
  o[o]; // expected-error {{overload resolution selected deleted operator '[]': deleted (O, [])}}
  o(o); // expected-error {{call to deleted function call operator in type 'O': deleted (O, ())}} expected-error {{attempt to use a deleted function: deleted (O, ())}}
  if (o) {} // expected-error {{attempt to use a deleted function: deleted (O, operator bool)}}
  static_cast<bool>(o); // expected-error {{static_cast from 'O' to 'bool' uses deleted function: deleted (O, operator bool)}}

  TO<double, double> to;
  +to; // expected-error {{overload resolution selected deleted operator '+': deleted (TO, +)}}
  to->x; // expected-error {{overload resolution selected deleted operator '->': deleted (TO, ->)}}
  to - to; // expected-error {{overload resolution selected deleted operator '-': deleted (TO, -)}}
  to[to]; // expected-error {{overload resolution selected deleted operator '[]': deleted (TO, [])}}
  to(to); // expected-error {{call to deleted function call operator in type 'TO<double, double>': deleted (TO, ())}} expected-error {{attempt to use a deleted function: deleted (TO, ())}}
  if (to) {} // expected-error {{attempt to use a deleted function: deleted (TO, operator bool)}}
  static_cast<bool>(to); // expected-error {{static_cast from 'TO<double, double>' to 'bool' uses deleted function: deleted (TO, operator bool)}}

  TO<double, int> to_int;
  +to_int; // expected-error {{overload resolution selected deleted operator '+': deleted (TO<T, int>, +)}}
  to_int->x; // expected-error {{overload resolution selected deleted operator '->': deleted (TO<T, int>, ->)}}
  to_int - to_int; // expected-error {{overload resolution selected deleted operator '-': deleted (TO<T, int>, -)}}
  to_int[to_int]; // expected-error {{overload resolution selected deleted operator '[]': deleted (TO<T, int>, [])}}
  to_int(to_int); // expected-error {{call to deleted function call operator in type 'TO<double, int>': deleted (TO<T, int>, ())}} expected-error {{attempt to use a deleted function: deleted (TO<T, int>, ())}}
  if (to_int) {} // expected-error {{attempt to use a deleted function: deleted (TO<T, int>, operator bool)}}
  static_cast<bool>(to_int); // expected-error {{static_cast from 'TO<double, int>' to 'bool' uses deleted function: deleted (TO<T, int>, operator bool)}}

  TO<int, int> to_int_int;
  +to_int_int; // expected-error {{overload resolution selected deleted operator '+': deleted (TO<int, int>, +)}}
  to_int_int->x; // expected-error {{overload resolution selected deleted operator '->': deleted (TO<int, int>, ->)}}
  to_int_int - to_int_int; // expected-error {{overload resolution selected deleted operator '-': deleted (TO<int, int>, -)}}
  to_int_int[to_int_int]; // expected-error {{overload resolution selected deleted operator '[]': deleted (TO<int, int>, [])}}
  to_int_int(to_int_int); // expected-error {{call to deleted function call operator in type 'TO<int, int>': deleted (TO<int, int>, ())}} expected-error {{attempt to use a deleted function: deleted (TO<int, int>, ())}}
  if (to_int_int) {} // expected-error {{attempt to use a deleted function: deleted (TO<int, int>, operator bool)}}
  static_cast<bool>(to_int_int); // expected-error {{static_cast from 'TO<int, int>' to 'bool' uses deleted function: deleted (TO<int, int>, operator bool)}}
};
