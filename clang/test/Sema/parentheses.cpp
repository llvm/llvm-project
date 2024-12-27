// RUN: %clang_cc1 -Wparentheses -fsyntax-only -verify %s
// RUN: %clang_cc1 -Wparentheses -fsyntax-only -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s

bool someConditionFunc();

void conditional_op(int x, int y, bool b) {
  (void)(x + someConditionFunc() ? 1 : 2); // expected-warning {{operator '?:' has lower precedence than '+'}} \
                                           // expected-note {{place parentheses around the '+' expression to silence this warning}} \
                                           // expected-note {{place parentheses around the '?:' expression to evaluate it first}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:10-[[@LINE-3]]:10}:"("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:33-[[@LINE-4]]:33}:")"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-5]]:14-[[@LINE-5]]:14}:"("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-6]]:41-[[@LINE-6]]:41}:")"

  (void)(x - b ? 1 : 2); // expected-warning {{operator '?:' has lower precedence than '-'}} \
                         // expected-note {{place parentheses around the '-' expression to silence this warning}} \
                         // expected-note {{place parentheses around the '?:' expression to evaluate it first}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:10-[[@LINE-3]]:10}:"("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:15-[[@LINE-4]]:15}:")"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-5]]:14-[[@LINE-5]]:14}:"("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-6]]:23-[[@LINE-6]]:23}:")"

  (void)(x * (x == y) ? 1 : 2); // expected-warning {{operator '?:' has lower precedence than '*'}} \
                                // expected-note {{place parentheses around the '*' expression to silence this warning}} \
                                // expected-note {{place parentheses around the '?:' expression to evaluate it first}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:10-[[@LINE-3]]:10}:"("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:22-[[@LINE-4]]:22}:")"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-5]]:14-[[@LINE-5]]:14}:"("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-6]]:30-[[@LINE-6]]:30}:")"
}

class Stream {
public:
  operator int();
  Stream &operator<<(int);
  Stream &operator<<(const char*);
  Stream &operator>>(int);
  Stream &operator>>(const char*);
};

void f(Stream& s, bool b) {
  (void)(s << b ? "foo" : "bar"); // expected-warning {{operator '?:' has lower precedence than '<<'}} \
                                  // expected-note {{place parentheses around the '<<' expression to silence this warning}} \
                                  // expected-note {{place parentheses around the '?:' expression to evaluate it first}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:10-[[@LINE-3]]:10}:"("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:16-[[@LINE-4]]:16}:")"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-5]]:15-[[@LINE-5]]:15}:"("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-6]]:32-[[@LINE-6]]:32}:")"

  (void)(s << 5 == 1); // expected-warning {{overloaded operator << has higher precedence than comparison operator}} \
                       // expected-note {{place parentheses around the '<<' expression to silence this warning}} \
                       // expected-note {{place parentheses around comparison expression to evaluate it first}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:10-[[@LINE-3]]:10}:"("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:16-[[@LINE-4]]:16}:")"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-5]]:15-[[@LINE-5]]:15}:"("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-6]]:21-[[@LINE-6]]:21}:")"

  (void)(s >> 5 == 1); // expected-warning {{overloaded operator >> has higher precedence than comparison operator}} \
                       // expected-note {{place parentheses around the '>>' expression to silence this warning}} \
                       // expected-note {{place parentheses around comparison expression to evaluate it first}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:10-[[@LINE-3]]:10}:"("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:16-[[@LINE-4]]:16}:")"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-5]]:15-[[@LINE-5]]:15}:"("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-6]]:21-[[@LINE-6]]:21}:")"
}

struct S {
  operator int() { return 42; }
  friend S operator+(const S &lhs, bool) { return S(); }
};

void test(S *s, bool (S::*m_ptr)()) {
  (void)(*s + true ? "foo" : "bar"); // expected-warning {{operator '?:' has lower precedence than '+'}} \
                                     // expected-note {{place parentheses around the '+' expression to silence this warning}} \
                                     // expected-note {{place parentheses around the '?:' expression to evaluate it first}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:10-[[@LINE-3]]:10}:"("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:19-[[@LINE-4]]:19}:")"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-5]]:15-[[@LINE-5]]:15}:"("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-6]]:35-[[@LINE-6]]:35}:")"

  (void)((*s + true) ? "foo" : "bar"); // No warning.

  // Don't crash on unusual member call expressions.
  (void)((s->*m_ptr)() ? "foo" : "bar");
}

void test(int a, int b, int c) {
  (void)(a >> b + c); // expected-warning {{operator '>>' has lower precedence than '+'; '+' will be evaluated first}} \
                         expected-note {{place parentheses around the '+' expression to silence this warning}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:15-[[@LINE-2]]:15}:"("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:20-[[@LINE-3]]:20}:")"

  (void)(a - b << c); // expected-warning {{operator '<<' has lower precedence than '-'; '-' will be evaluated first}} \
                         expected-note {{place parentheses around the '-' expression to silence this warning}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:10-[[@LINE-2]]:10}:"("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:15-[[@LINE-3]]:15}:")"

  Stream() << b + c;
  Stream() >> b + c; // expected-warning {{operator '>>' has lower precedence than '+'; '+' will be evaluated first}} \
                        expected-note {{place parentheses around the '+' expression to silence this warning}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:15-[[@LINE-2]]:15}:"("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:20-[[@LINE-3]]:20}:")"
}

namespace PR15628 {
  struct BlockInputIter {
    void* operator++(int);
    void* operator--(int);
  };

  void test(BlockInputIter i) {
    (void)(i++ ? true : false); // no-warning
    (void)(i-- ? true : false); // no-warning
  }
}

namespace PR20735 {
  struct X {
    static int state;
    static int get();
    int get_num();
    int num;
  };
  namespace ns {
    int num = 0;
    int get();
  }
  void test(X x) {
    if (5 & x.get_num() != 0) {}
    // expected-warning@-1 {{& has lower precedence than !=; != will be evaluated first}}
    // expected-note@-2 {{place parentheses around the '!=' expression to silence this warning}}
    // expected-note@-3 {{place parentheses around the & expression to evaluate it first}}
    // CHECK: place parentheses around the '!=' expression to silence this warning
    // fix-it:"{{.*}}":{[[@LINE-5]]:13-[[@LINE-5]]:13}:"("
    // fix-it:"{{.*}}":{[[@LINE-6]]:29-[[@LINE-6]]:29}:")"
    // CHECK: place parentheses around the & expression to evaluate it first
    // fix-it:"{{.*}}":{[[@LINE-8]]:9-[[@LINE-8]]:9}:"("
    // fix-it:"{{.*}}":{[[@LINE-9]]:24-[[@LINE-9]]:24}:")"

    if (5 & x.num != 0) {}
    // expected-warning@-1 {{& has lower precedence than !=; != will be evaluated first}}
    // expected-note@-2 {{place parentheses around the '!=' expression to silence this warning}}
    // expected-note@-3 {{place parentheses around the & expression to evaluate it first}}
    // CHECK: place parentheses around the '!=' expression to silence this warning
    // fix-it:"{{.*}}":{[[@LINE-5]]:13-[[@LINE-5]]:13}:"("
    // fix-it:"{{.*}}":{[[@LINE-6]]:23-[[@LINE-6]]:23}:")"
    // CHECK: place parentheses around the & expression to evaluate it first
    // fix-it:"{{.*}}":{[[@LINE-8]]:9-[[@LINE-8]]:9}:"("
    // fix-it:"{{.*}}":{[[@LINE-9]]:18-[[@LINE-9]]:18}:")"

    if (5 & x.state != 0) {}
    // expected-warning@-1 {{& has lower precedence than !=; != will be evaluated first}}
    // expected-note@-2 {{place parentheses around the '!=' expression to silence this warning}}
    // expected-note@-3 {{place parentheses around the & expression to evaluate it first}}
    // CHECK: place parentheses around the '!=' expression to silence this warning
    // fix-it:"{{.*}}":{[[@LINE-5]]:13-[[@LINE-5]]:13}:"("
    // fix-it:"{{.*}}":{[[@LINE-6]]:25-[[@LINE-6]]:25}:")"
    // CHECK: place parentheses around the & expression to evaluate it first
    // fix-it:"{{.*}}":{[[@LINE-8]]:9-[[@LINE-8]]:9}:"("
    // fix-it:"{{.*}}":{[[@LINE-9]]:20-[[@LINE-9]]:20}:")"

    if (5 & x.get() != 0) {}
    // expected-warning@-1 {{& has lower precedence than !=; != will be evaluated first}}
    // expected-note@-2 {{place parentheses around the '!=' expression to silence this warning}}
    // expected-note@-3 {{place parentheses around the & expression to evaluate it first}}
    // CHECK: place parentheses around the '!=' expression to silence this warning
    // fix-it:"{{.*}}":{[[@LINE-5]]:13-[[@LINE-5]]:13}:"("
    // fix-it:"{{.*}}":{[[@LINE-6]]:25-[[@LINE-6]]:25}:")"
    // CHECK: place parentheses around the & expression to evaluate it first
    // fix-it:"{{.*}}":{[[@LINE-8]]:9-[[@LINE-8]]:9}:"("
    // fix-it:"{{.*}}":{[[@LINE-9]]:20-[[@LINE-9]]:20}:")"

    if (5 & X::state != 0) {}
    // expected-warning@-1 {{& has lower precedence than !=; != will be evaluated first}}
    // expected-note@-2 {{place parentheses around the '!=' expression to silence this warning}}
    // expected-note@-3 {{place parentheses around the & expression to evaluate it first}}
    // CHECK: place parentheses around the '!=' expression to silence this warning
    // fix-it:"{{.*}}":{[[@LINE-5]]:13-[[@LINE-5]]:13}:"("
    // fix-it:"{{.*}}":{[[@LINE-6]]:26-[[@LINE-6]]:26}:")"
    // CHECK: place parentheses around the & expression to evaluate it first
    // fix-it:"{{.*}}":{[[@LINE-8]]:9-[[@LINE-8]]:9}:"("
    // fix-it:"{{.*}}":{[[@LINE-9]]:21-[[@LINE-9]]:21}:")"

    if (5 & X::get() != 0) {}
    // expected-warning@-1 {{& has lower precedence than !=; != will be evaluated first}}
    // expected-note@-2 {{place parentheses around the '!=' expression to silence this warning}}
    // expected-note@-3 {{place parentheses around the & expression to evaluate it first}}
    // CHECK: place parentheses around the '!=' expression to silence this warning
    // fix-it:"{{.*}}":{[[@LINE-5]]:13-[[@LINE-5]]:13}:"("
    // fix-it:"{{.*}}":{[[@LINE-6]]:26-[[@LINE-6]]:26}:")"
    // CHECK: place parentheses around the & expression to evaluate it first
    // fix-it:"{{.*}}":{[[@LINE-8]]:9-[[@LINE-8]]:9}:"("
    // fix-it:"{{.*}}":{[[@LINE-9]]:21-[[@LINE-9]]:21}:")"

    if (5 & ns::get() != 0) {}
    // expected-warning@-1 {{& has lower precedence than !=; != will be evaluated first}}
    // expected-note@-2 {{place parentheses around the '!=' expression to silence this warning}}
    // expected-note@-3 {{place parentheses around the & expression to evaluate it first}}
    // CHECK: place parentheses around the '!=' expression to silence this warning
    // fix-it:"{{.*}}":{[[@LINE-5]]:13-[[@LINE-5]]:13}:"("
    // fix-it:"{{.*}}":{[[@LINE-6]]:27-[[@LINE-6]]:27}:")"
    // CHECK: place parentheses around the & expression to evaluate it first
    // fix-it:"{{.*}}":{[[@LINE-8]]:9-[[@LINE-8]]:9}:"("
    // fix-it:"{{.*}}":{[[@LINE-9]]:22-[[@LINE-9]]:22}:")"

    if (5 & ns::num != 0) {}
    // expected-warning@-1 {{& has lower precedence than !=; != will be evaluated first}}
    // expected-note@-2 {{place parentheses around the '!=' expression to silence this warning}}
    // expected-note@-3 {{place parentheses around the & expression to evaluate it first}}
    // CHECK: place parentheses around the '!=' expression to silence this warning
    // fix-it:"{{.*}}":{[[@LINE-5]]:13-[[@LINE-5]]:13}:"("
    // fix-it:"{{.*}}":{[[@LINE-6]]:25-[[@LINE-6]]:25}:")"
    // CHECK: place parentheses around the & expression to evaluate it first
    // fix-it:"{{.*}}":{[[@LINE-8]]:9-[[@LINE-8]]:9}:"("
    // fix-it:"{{.*}}":{[[@LINE-9]]:20-[[@LINE-9]]:20}:")"
  }
}

void consecutive_builtin_compare(int x, int y, int z) {
  (void)(x < y < z);  // expected-warning {{comparisons like 'X<=Y<=Z' don't have their mathematical meaning}}
  (void)(x < y > z);  // expected-warning {{comparisons like 'X<=Y<=Z' don't have their mathematical meaning}}
  (void)(x < y <= z); // expected-warning {{comparisons like 'X<=Y<=Z' don't have their mathematical meaning}}
  (void)(x <= y > z); // expected-warning {{comparisons like 'X<=Y<=Z' don't have their mathematical meaning}}
  (void)((x < y) < z);  // no-warning
  (void)((x < y) >= z); // no-warning

  // TODO: Also issue warning for consecutive comparisons that use overloaded comparison operators?
}
