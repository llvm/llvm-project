// RUN: %clang_cc1 -std=c++23 -fsyntax-only -fcxx-exceptions -verify %s
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -fcxx-exceptions -verify %s
// RUN: %clang_cc1 -std=c++11 -fsyntax-only -fcxx-exceptions -verify %s
// expected-no-diagnostics

// Throwing
namespace test_throwing {
class Widget {
public:
  Widget(Widget &&);
  Widget(const Widget &) = delete;
};

void seven(Widget w) {
  throw w;
}
} // namespace test_throwing

// Non-constructor conversion
namespace test_non_constructor_conversion {
class Widget {};

struct To {
  operator Widget() const & = delete;
  operator Widget() &&;
};

Widget nine() {
  To t;
  return t;
}
} // namespace test_non_constructor_conversion

// By-value sinks
namespace test_by_value_sinks {
class Widget {
public:
  Widget();
  Widget(Widget &&);
  Widget(const Widget &) = delete;
};

struct Fowl {
  Fowl(Widget);
};

Fowl eleven() {
  Widget w;
  return w;
}
} // namespace test_by_value_sinks

// Slicing
namespace test_slicing {
class Base {
public:
  Base();
  Base(Base &&);
  Base(Base const &) = delete;
};

class Derived : public Base {};

Base thirteen() {
  Derived result;
  return result;
}
} // namespace test_slicing
