// RUN: %clang_cc1 -fsyntax-only -verify %s

template<typename T>
struct Outer {
  void member() {
    struct Local {
      void baz(auto x) {} // expected-error {{'auto' not allowed in function prototype}}
    };
    (void)sizeof(Local);
  }
};

int main() {
  Outer<int> o;
  o.member();
}
