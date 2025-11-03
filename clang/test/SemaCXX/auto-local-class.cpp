// RUN: %clang_cc1 %std_cxx98-14 -fsyntax-only -verify=expected,precxx17 %s
// RUN: %clang_cc1 %std_cxx17- -fsyntax-only -verify=expected,cxx17 %s

// Check that 'auto' in a local class function parameter
// produces the correct diagnostic depending on the language standard:
//  * Before C++17 → "'auto' not allowed in function prototype"
//  * C++17 and later → "templates cannot be declared inside of a local class"

int main() {
  struct A {
    void foo(auto x) {} // precxx17-error {{'auto' not allowed in function prototype}} \
                        // cxx17-error {{templates cannot be declared inside of a local class}}
  };
}
