// RUN: %clang_cc1 -std=c++2c -fcontracts -fsyntax-only -verify %s

// This file exercises the declaration paths which can lead to a function
// contract specifier. Contract predicates are parsed but not retained yet.

template <typename>
concept Always = true;

int declaration(int value) pre(value > 0) post(result: value >= 0);

int definition(int value) pre(value > 0) post(result: value >= 0) {
  return value;
}

auto trailing_return(int value) -> int pre(value > 0) post(value >= 0) {
  return value;
}

decltype(auto) reference_return(int &value) pre(value > 0) post(value > 0) {
  return (value);
}

int with_defaults(int value = 1) noexcept pre(value > 0) {
  return value;
}

template <typename T>
auto constrained(T value) -> T requires Always<T> pre(value > T{}) {
  return value;
}

int first(int value) pre(value > 0),
    second(int value) post(result: value > 0);

struct Widget {
  int value;

  Widget(int input) pre(input > 0);
  ~Widget() pre(true);

  static int static_member(int input) noexcept pre(input > 0) {
    return input;
  }

  friend int inspect(const Widget &, int input) pre(input > 0);

  explicit operator bool() const pre(this->value >= 0);
  int operator()(int input) const & noexcept pre(input > 0) post(input >= 0);

  virtual int pure(int input) const pre(input > 0) = 0;

  template <typename T>
  T member_template(T input) requires Always<T> pre(input > T{}) {
    return input;
  }
};

Widget::Widget(int input) pre(input > 0) : value(input) {}
Widget::~Widget() pre(true) {}

namespace API {
struct Service {
  int limit;
  int call(int input) const;
};
} // namespace API

int API::Service::call(int input) const
    pre(this->limit >= 0 && input > 0) post(input > 0) {
  return input;
}

int object pre(true); // expected-error {{contract specifiers can only be applied to function declarations}}
int (*function_pointer)(int) pre(true); // expected-error {{contract specifiers can only be applied to function declarations}}
extern int (&function_reference)(int) pre(true); // expected-error {{contract specifiers can only be applied to function declarations}}

typedef int FunctionTypedef(int) pre(true); // expected-error {{contract specifiers can only be applied to function declarations}}
