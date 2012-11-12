// RUN: %clang_cc1 -Wparentheses -fsyntax-only -verify %s
// RUN: %clang_cc1 -Wparentheses -fixit %s -o - | %clang_cc1 -Wparentheses -Werror -

bool someConditionFunc();

void conditional_op(int x, int y, bool b) {
  (void)(x + someConditionFunc() ? 1 : 2); // expected-warning {{operator '?:' has lower precedence than '+'}} \
                                           // expected-note {{place parentheses around the '?:' expression to evaluate it first}} \
                                           // expected-note {{place parentheses around the '+' expression to silence this warning}}

  (void)(x - b ? 1 : 2); // expected-warning {{operator '?:' has lower precedence than '-'}} \
                         // expected-note {{place parentheses around the '?:' expression to evaluate it first}} \
                         // expected-note {{place parentheses around the '-' expression to silence this warning}}

  (void)(x * (x == y) ? 1 : 2); // expected-warning {{operator '?:' has lower precedence than '*'}} \
                                // expected-note {{place parentheses around the '?:' expression to evaluate it first}} \
                                // expected-note {{place parentheses around the '*' expression to silence this warning}}
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
                                  // expected-note {{place parentheses around the '?:' expression to evaluate it first}} \
                                  // expected-note {{place parentheses around the '<<' expression to silence this warning}}
}

struct S {
  operator int() { return 42; }
  friend S operator+(const S &lhs, bool) { return S(); }
};

void test(S *s, bool (S::*m_ptr)()) {
  (void)(*s + true ? "foo" : "bar"); // expected-warning {{operator '?:' has lower precedence than '+'}} \
                                     // expected-note {{place parentheses around the '?:' expression to evaluate it first}} \
                                     // expected-note {{place parentheses around the '+' expression to silence this warning}}

  (void)((*s + true) ? "foo" : "bar"); // No warning.

  // Don't crash on unusual member call expressions.
  (void)((s->*m_ptr)() ? "foo" : "bar");
}

void test(int a, int b, int c) {
  (void)(a >> b + c); // expected-warning {{operator '>>' has lower precedence than '+'; '+' will be evaluated first}} \
                         expected-note {{place parentheses around the '+' expression to silence this warning}}
  (void)(a - b << c); // expected-warning {{operator '<<' has lower precedence than '-'; '-' will be evaluated first}} \
                         expected-note {{place parentheses around the '-' expression to silence this warning}}
  Stream() << b + c;
  Stream() >> b + c; // expected-warning {{operator '>>' has lower precedence than '+'; '+' will be evaluated first}} \
                        expected-note {{place parentheses around the '+' expression to silence this warning}}
}
