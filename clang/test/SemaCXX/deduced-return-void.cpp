// RUN: %clang_cc1 -std=c++14 -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s

// Check that we don't get any extra warning for "return" without an
// expression, in a function that might have been intended to return
// void all along.
decltype(h1) h1() { // expected-error {{use of undeclared identifier 'h1'}}
  return;
}

namespace JustAuto {
int i;
auto f1() { }
auto f2() { return; }
auto f3() { return void(); }
auto f4() {
  return i;
  return; // expected-error {{'auto' in return type deduced as 'void' here but deduced as 'int' in earlier return statement}}
}
auto f5() {
  return i;
  return void(); // expected-error {{'auto' in return type deduced as 'void' here but deduced as 'int' in earlier return statement}}
}

auto l1 = []() { };
auto l2 = []() { return; };
auto l3 = []() { return void(); };
auto l4 = []() {
  return i;
  return; // expected-error {{return type 'void' must match previous return type 'int' when lambda expression has unspecified explicit return type}}
};
auto l5 = []() {
  return i;
  return void(); // expected-error {{return type 'void' must match previous return type 'int' when lambda expression has unspecified explicit return type}}
};

} // namespace JustAuto

namespace DecltypeAuto {
int i;
decltype(auto) f1() { }
decltype(auto) f2() { return; }
decltype(auto) f3() { return void(); }
decltype(auto) f4() {
  return i;
  return; // expected-error {{'decltype(auto)' in return type deduced as 'void' here but deduced as 'int' in earlier return statement}}
}
decltype(auto) f5() {
  return i;
  return void(); // expected-error {{'decltype(auto)' in return type deduced as 'void' here but deduced as 'int' in earlier return statement}}
}

auto l1 = []() -> decltype(auto) { };
auto l2 = []() -> decltype(auto) { return; };
auto l3 = []() -> decltype(auto) { return void(); };
auto l4 = []() -> decltype(auto) {
  return i;
  return; // expected-error {{'decltype(auto)' in return type deduced as 'void' here but deduced as 'int' in earlier return statement}}
};
auto l5 = []() -> decltype(auto) {
  return i;
  return void(); // expected-error {{'decltype(auto)' in return type deduced as 'void' here but deduced as 'int' in earlier return statement}}
};

} // namespace DecltypeAuto

namespace AutoPtr {
int i;
auto *f1() { } // expected-error {{cannot deduce return type 'auto *' for function with no return statements}}
auto *f2() {
  return; // expected-error {{cannot deduce return type 'auto *' from omitted return expression}}
}
auto *f3() {
  return void(); // expected-error {{cannot deduce return type 'auto *' from returned value of type 'void'}}
}
auto *f4() {
  return &i;
  return; // expected-error {{cannot deduce return type 'auto *' from omitted return expression}}
}
auto *f5() {
  return &i;
  return void(); // expected-error {{cannot deduce return type 'auto *' from returned value of type 'void'}}
}

auto l1 = []() -> auto* { }; // expected-error {{cannot deduce return type 'auto *' for function with no return statements}}
auto l2 = []() -> auto* {
  return; // expected-error {{cannot deduce return type 'auto *' from omitted return expression}}
};
auto l3 = []() -> auto* {
  return void(); // expected-error {{cannot deduce return type 'auto *' from returned value of type 'void'}}
};
auto l4 = []() -> auto* {
  return &i;
  return; // expected-error {{cannot deduce return type 'auto *' from omitted return expression}}
};
auto l5 = []() -> auto* {
  return &i;
  return void(); // expected-error {{cannot deduce return type 'auto *' from returned value of type 'void'}}
};
} // namespace AutoPtr

namespace AutoRef {
int i;
auto& f1() { // expected-error {{cannot deduce return type 'auto &' for function with no return statements}}
}
auto& f2() {
  return; // expected-error {{cannot deduce return type 'auto &' from omitted return expression}}
}
auto& f3() {
  return void(); // expected-error@-1 {{cannot form a reference to 'void'}}
}
auto& f4() {
  return i;
  return; // expected-error {{cannot deduce return type 'auto &' from omitted return expression}}
}
auto& f5() {
  return i;
  return void(); // expected-error@-2 {{cannot form a reference to 'void'}}
}
auto& f6() { return 42; } // expected-error {{non-const lvalue reference to type 'int' cannot bind to a temporary of type 'int'}}

auto l1 = []() -> auto& { }; // expected-error {{cannot deduce return type 'auto &' for function with no return statements}}
auto l2 = []() -> auto& {
  return; // expected-error {{cannot deduce return type 'auto &' from omitted return expression}}
};
auto l3 = []() -> auto& { // expected-error {{cannot form a reference to 'void'}}
  return void();
};
auto l4 = []() -> auto& {
  return i;
  return; // expected-error {{cannot deduce return type 'auto &' from omitted return expression}}
};
auto l5 = []() -> auto& { // expected-error {{cannot form a reference to 'void'}}
  return i;
  return void();
};
auto l6 = []() -> auto& {
  return 42; // expected-error {{non-const lvalue reference to type 'int' cannot bind to a temporary of type 'int'}}
};
} // namespace AutoRef
