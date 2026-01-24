// RUN: %clang_cc1 -fsyntax-only -verify -std=c++2b %s

// https://github.com/llvm/llvm-project/issues/177741

struct S {
  static int operator()(this S) { return 0; }
  // expected-error@-1 {{an explicit object parameter cannot appear in a static function}}
  // expected-note@-2 {{candidate function not viable}}
};

void foo() {
  S s{};
  s(0);
  // expected-error@-1 {{no matching function for call to object of type 'S'}}
}
