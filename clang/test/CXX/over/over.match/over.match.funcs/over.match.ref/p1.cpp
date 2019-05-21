// RUN: %clang_cc1 %s -verify
// expected-no-diagnostics

namespace r360311_regression {
  struct string {};
  struct string_view {
    explicit operator string() const;
  };

  namespace ns {
    struct Base {};
    class Derived : public Base {};
    void f(string_view s, Base *c);
    void f(const string &s, Derived *c);
  } // namespace ns

  void g(string_view s) {
    ns::Derived d;
    f(s, &d);
  }
  } // namespace r360311_regression
