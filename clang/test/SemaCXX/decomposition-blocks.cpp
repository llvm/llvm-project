// RUN: %clang_cc1 -std=c++17 -fsyntax-only -verify %s -fblocks

struct S {
  int i : 1;
  int j;
};

void run(void (^)());
void test() {
  auto [i, j] = S{-1, 42}; // expected-note {{'i' declared here}}
  run(^{
    (void)i; // expected-error {{reference to local binding 'i' declared in enclosing function 'test'}}
  });
}
