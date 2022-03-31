// RUN: %clang_cc1 -fsyntax-only -Wcomma -std=c++11 -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wcomma -std=c++11 -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s

// RUN: %clang_cc1 -fsyntax-only -Wcomma -x c -std=c89 -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wcomma -x c -std=c99 -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wcomma -x c -std=c11 -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wcomma -x c -std=c17 -verify %s

// int returning function
int return_four(void) { return 5; }

// Test builtin operators
void test_builtin(void) {
  int x = 0, y = 0;
  for (; y < 10; x++, y++) {}
  for (; y < 10; ++x, y++) {}
  for (; y < 10; x++, ++y) {}
  for (; y < 10; ++x, ++y) {}
  for (; y < 10; x--, ++y) {}
  for (; y < 10; --x, ++y) {}
  for (; y < 10; x = 5, ++y) {}
  for (; y < 10; x *= 5, ++y) {}
  for (; y < 10; x /= 5, ++y) {}
  for (; y < 10; x %= 5, ++y) {}
  for (; y < 10; x += 5, ++y) {}
  for (; y < 10; x -= 5, ++y) {}
  for (; y < 10; x <<= 5, ++y) {}
  for (; y < 10; x >>= 5, ++y) {}
  for (; y < 10; x &= 5, ++y) {}
  for (; y < 10; x |= 5, ++y) {}
  for (; y < 10; x ^= 5, ++y) {}
}

// Test nested comma operators
void test_nested(void) {
  int x1, x2, x3;
  int y1, *y2 = 0, y3 = 5;

#if __STDC_VERSION >= 199901L
  for (int z1 = 5, z2 = 4, z3 = 3; x1 <4; ++x1) {}
#endif
}

// Confusing "," for "=="
void test_compare(void) {
  if (return_four(), 5) {}
  // expected-warning@-1{{comma operator}}
  // expected-note@-2{{cast expression to void}}
  // CHECK: fix-it:{{.*}}:{[[@LINE-3]]:7-[[@LINE-3]]:7}:"static_cast<void>("
  // CHECK: fix-it:{{.*}}:{[[@LINE-4]]:20-[[@LINE-4]]:20}:")"

  if (return_four() == 5) {}
}

// Confusing "," for "+"
int test_plus(void) {
  return return_four(), return_four();
  // expected-warning@-1{{comma operator}}
  // expected-note@-2{{cast expression to void}}
  // CHECK: fix-it:{{.*}}:{[[@LINE-3]]:10-[[@LINE-3]]:10}:"static_cast<void>("
  // CHECK: fix-it:{{.*}}:{[[@LINE-4]]:23-[[@LINE-4]]:23}:")"

  return return_four() + return_four();
}

// Be sure to look through parentheses
void test_parentheses(void) {
  int x, y;
  for (x = 0; return_four(), x;) {}
  // expected-warning@-1{{comma operator}}
  // expected-note@-2{{cast expression to void}}
  // CHECK: fix-it:{{.*}}:{[[@LINE-3]]:15-[[@LINE-3]]:15}:"static_cast<void>("
  // CHECK: fix-it:{{.*}}:{[[@LINE-4]]:28-[[@LINE-4]]:28}:")"

  for (x = 0; (return_four()), (x) ;) {}
  // expected-warning@-1{{comma operator}}
  // expected-note@-2{{cast expression to void}}
  // CHECK: fix-it:{{.*}}:{[[@LINE-3]]:15-[[@LINE-3]]:15}:"static_cast<void>("
  // CHECK: fix-it:{{.*}}:{[[@LINE-4]]:30-[[@LINE-4]]:30}:")"
}

void test_increment(void) {
  int x, y;
  ++x, ++y;
  // expected-warning@-1{{comma operator}}
  // expected-note@-2{{cast expression to void}}
  // CHECK: fix-it:{{.*}}:{[[@LINE-3]]:3-[[@LINE-3]]:3}:"static_cast<void>("
  // CHECK: fix-it:{{.*}}:{[[@LINE-4]]:6-[[@LINE-4]]:6}:")"
}

// Check for comma operator in conditions.
void test_conditions(int x) {
  x = (return_four(), x);
  // expected-warning@-1{{comma operator}}
  // expected-note@-2{{cast expression to void}}
  // CHECK: fix-it:{{.*}}:{[[@LINE-3]]:8-[[@LINE-3]]:8}:"static_cast<void>("
  // CHECK: fix-it:{{.*}}:{[[@LINE-4]]:21-[[@LINE-4]]:21}:")"

  int y = (return_four(), x);
  // expected-warning@-1{{comma operator}}
  // expected-note@-2{{cast expression to void}}
  // CHECK: fix-it:{{.*}}:{[[@LINE-3]]:12-[[@LINE-3]]:12}:"static_cast<void>("
  // CHECK: fix-it:{{.*}}:{[[@LINE-4]]:25-[[@LINE-4]]:25}:")"

  for (; return_four(), x;) {}
  // expected-warning@-1{{comma operator}}
  // expected-note@-2{{cast expression to void}}
  // CHECK: fix-it:{{.*}}:{[[@LINE-3]]:10-[[@LINE-3]]:10}:"static_cast<void>("
  // CHECK: fix-it:{{.*}}:{[[@LINE-4]]:23-[[@LINE-4]]:23}:")"

  while (return_four(), x) {}
  // expected-warning@-1{{comma operator}}
  // expected-note@-2{{cast expression to void}}
  // CHECK: fix-it:{{.*}}:{[[@LINE-3]]:10-[[@LINE-3]]:10}:"static_cast<void>("
  // CHECK: fix-it:{{.*}}:{[[@LINE-4]]:23-[[@LINE-4]]:23}:")"

  if (return_four(), x) {}
  // expected-warning@-1{{comma operator}}
  // expected-note@-2{{cast expression to void}}
  // CHECK: fix-it:{{.*}}:{[[@LINE-3]]:7-[[@LINE-3]]:7}:"static_cast<void>("
  // CHECK: fix-it:{{.*}}:{[[@LINE-4]]:20-[[@LINE-4]]:20}:")"

  do { } while (return_four(), x);
  // expected-warning@-1{{comma operator}}
  // expected-note@-2{{cast expression to void}}
  // CHECK: fix-it:{{.*}}:{[[@LINE-3]]:17-[[@LINE-3]]:17}:"static_cast<void>("
  // CHECK: fix-it:{{.*}}:{[[@LINE-4]]:30-[[@LINE-4]]:30}:")"
}

// Nested comma operator with fix-its.
void test_nested_fixits(void) {
  return_four(), return_four(), return_four(), return_four();
  // expected-warning@-1 3{{comma operator}}
  // expected-note@-2 3{{cast expression to void}}
  // CHECK: fix-it:{{.*}}:{[[@LINE-3]]:3-[[@LINE-3]]:3}:"static_cast<void>("
  // CHECK: fix-it:{{.*}}:{[[@LINE-4]]:16-[[@LINE-4]]:16}:")"
  // CHECK: fix-it:{{.*}}:{[[@LINE-5]]:18-[[@LINE-5]]:18}:"static_cast<void>("
  // CHECK: fix-it:{{.*}}:{[[@LINE-6]]:31-[[@LINE-6]]:31}:")"
  // CHECK: fix-it:{{.*}}:{[[@LINE-7]]:33-[[@LINE-7]]:33}:"static_cast<void>("
  // CHECK: fix-it:{{.*}}:{[[@LINE-8]]:46-[[@LINE-8]]:46}:")"
}

#ifdef __cplusplus
class S2 {
public:
  void advance();

  S2 operator++();
  S2 operator++(int);
  S2 operator--();
  S2 operator--(int);
  S2 operator=(int);
  S2 operator*=(int);
  S2 operator/=(int);
  S2 operator%=(int);
  S2 operator+=(int);
  S2 operator-=(int);
  S2 operator<<=(int);
  S2 operator>>=(int);
  S2 operator&=(int);
  S2 operator|=(int);
  S2 operator^=(int);
};

// Test overloaded operators
void test_overloaded_operator() {
  S2 x;
  int y;
  for (; y < 10; x++, y++) {}
  for (; y < 10; ++x, y++) {}
  for (; y < 10; x++, ++y) {}
  for (; y < 10; ++x, ++y) {}
  for (; y < 10; x--, ++y) {}
  for (; y < 10; --x, ++y) {}
  for (; y < 10; x = 5, ++y) {}
  for (; y < 10; x *= 5, ++y) {}
  for (; y < 10; x /= 5, ++y) {}
  for (; y < 10; x %= 5, ++y) {}
  for (; y < 10; x += 5, ++y) {}
  for (; y < 10; x -= 5, ++y) {}
  for (; y < 10; x <<= 5, ++y) {}
  for (; y < 10; x >>= 5, ++y) {}
  for (; y < 10; x &= 5, ++y) {}
  for (; y < 10; x |= 5, ++y) {}
  for (; y < 10; x ^= 5, ++y) {}
}

class Stream {
 public:
  Stream& operator<<(int);
} cout;

// Confusing "," for "<<"
void test_stream() {
  cout << 5 << return_four();
  cout << 5, return_four();
  // expected-warning@-1{{comma operator}}
  // expected-note@-2{{cast expression to void}}
  // CHECK: fix-it:{{.*}}:{[[@LINE-3]]:3-[[@LINE-3]]:3}:"static_cast<void>("
  // CHECK: fix-it:{{.*}}:{[[@LINE-4]]:12-[[@LINE-4]]:12}:")"
}

void Concat(int);
void Concat(int, int);

// Testing extra parentheses in function call
void test_overloaded_function() {
  Concat((return_four() , 5));
  // expected-warning@-1{{comma operator}}
  // expected-note@-2{{cast expression to void}}
  // CHECK: fix-it:{{.*}}:{[[@LINE-3]]:11-[[@LINE-3]]:11}:"static_cast<void>("
  // CHECK: fix-it:{{.*}}:{[[@LINE-4]]:24-[[@LINE-4]]:24}:")"

  Concat(return_four() , 5);
}

bool DoStuff();
class S9 {
public:
 bool Advance();
 bool More();
};

// Ignore comma operator in for-loop initializations and increments.
void test_for_loop() {
  int x, y;
  for (x = 0, y = 5; x < y; ++x) {}
  for (x = 0; x < 10; DoStuff(), ++x) {}
  for (S9 s; s.More(); s.Advance(), ++x) {}
}

// Ignore comma operator in templates.
namespace test_template {
template <bool T>
struct B { static const bool value = T; };

typedef B<true> true_type;
typedef B<false> false_type;

template <bool...>
struct bool_seq;

template <typename... xs>
class Foo {
  typedef bool_seq<((void)xs::value, true)...> all_true;
  typedef bool_seq<((void)xs::value, false)...> all_false;
  typedef bool_seq<xs::value...> seq;
};

const auto X = Foo<true_type>();
}

namespace test_mutex {
class Mutex {
 public:
  Mutex();
  ~Mutex();
};
class MutexLock {
public:
  MutexLock(Mutex &);
  MutexLock();
  ~MutexLock();
};
class BuiltinMutex {
  Mutex M;
};
Mutex StatusMutex;
bool Status;

bool get_status() {
  return (MutexLock(StatusMutex), Status);
  // expected-warning@-1{{comma operator}}
  // expected-note@-2{{cast expression to void}}
  // CHECK: fix-it:{{.*}}:{[[@LINE-3]]:11-[[@LINE-3]]:11}:"static_cast<void>("
  // CHECK: fix-it:{{.*}}:{[[@LINE-4]]:33-[[@LINE-4]]:33}:")"
  return (MutexLock(), Status);
  // expected-warning@-1{{comma operator}}
  // expected-note@-2{{cast expression to void}}
  // CHECK: fix-it:{{.*}}:{[[@LINE-3]]:11-[[@LINE-3]]:11}:"static_cast<void>("
  // CHECK: fix-it:{{.*}}:{[[@LINE-4]]:22-[[@LINE-4]]:22}:")"
  return (BuiltinMutex(), Status);
  // expected-warning@-1{{comma operator}}
  // expected-note@-2{{cast expression to void}}
  // CHECK: fix-it:{{.*}}:{[[@LINE-3]]:11-[[@LINE-3]]:11}:"static_cast<void>("
  // CHECK: fix-it:{{.*}}:{[[@LINE-4]]:25-[[@LINE-4]]:25}:")"
}
}

// PR39375 - test cast to void to silence warnings
template <typename T>
void test_dependent_cast() {
  (void)42, 0;
  static_cast<void>(42), 0;

  (void)T{}, 0;
  static_cast<void>(T{}), 0;
}
#endif  // ifdef __cplusplus
