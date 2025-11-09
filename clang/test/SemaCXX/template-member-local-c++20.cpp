// RUN: %clang_cc1 -fsyntax-only -std=c++20 -verify %s

template<typename T>
struct Outer {
  void member() {
    struct Local { // expected-error {{templates cannot be declared inside of a local class}}
      void baz(auto x) {}
    };
    (void)sizeof(Local);
  }
};

int main() {
  Outer<int> o;
  o.member();
}
