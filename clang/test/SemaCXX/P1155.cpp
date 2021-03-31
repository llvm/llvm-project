// RUN: %clang_cc1 -std=c++20 -fsyntax-only -fcxx-exceptions -verify=cxx20 %s
// RUN: %clang_cc1 -std=c++17 -fsyntax-only -fcxx-exceptions -verify=cxx11_14_17 %s
// RUN: %clang_cc1 -std=c++14 -fsyntax-only -fcxx-exceptions -verify=cxx11_14_17 %s
// RUN: %clang_cc1 -std=c++11 -fsyntax-only -fcxx-exceptions -verify=cxx11_14_17 %s
// cxx20-no-diagnostics

// Throwing
namespace test_throwing {
class Widget {
public:
  Widget(Widget &&);
  Widget(const Widget &) = delete;
};

void seven(Widget w) {
  throw w; // Clang already do this implicit move before -std=c++20
}
} // namespace test_throwing

// Non-constructor conversion
namespace test_non_constructor_conversion {
class Widget {};

struct To {
  operator Widget() const & = delete; // cxx11_14_17-note {{'operator Widget' has been explicitly marked deleted here}}
  operator Widget() &&;
};

Widget nine() {
  To t;
  return t; // cxx11_14_17-error {{conversion function from 'test_non_constructor_conversion::To' to 'test_non_constructor_conversion::Widget' invokes a deleted function}}
}
} // namespace test_non_constructor_conversion

// By-value sinks
namespace test_by_value_sinks {
class Widget {
public:
  Widget();
  Widget(Widget &&);
  Widget(const Widget &) = delete; // cxx11_14_17-note {{'Widget' has been explicitly marked deleted here}}
};

struct Fowl {
  Fowl(Widget); // cxx11_14_17-note {{passing argument to parameter here}}
};

Fowl eleven() {
  Widget w;
  return w; // cxx11_14_17-error {{call to deleted constructor of 'test_by_value_sinks::Widget'}}
}
} // namespace test_by_value_sinks

// Slicing
namespace test_slicing {
class Base {
public:
  Base();
  Base(Base &&);
  Base(Base const &) = delete; // cxx11_14_17-note {{'Base' has been explicitly marked deleted here}}
};

class Derived : public Base {};

Base thirteen() {
  Derived result;
  return result; // cxx11_14_17-error {{call to deleted constructor of 'test_slicing::Base'}}
}
} // namespace test_slicing
