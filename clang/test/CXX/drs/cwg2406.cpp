// RUN: %clang_cc1 -std=c++98 %s -fexceptions -fcxx-exceptions -pedantic-errors -verify=expected,cxx98-14
// RUN: %clang_cc1 -std=c++11 %s -fexceptions -fcxx-exceptions -pedantic-errors -verify=expected,cxx98-14
// RUN: %clang_cc1 -std=c++14 %s -fexceptions -fcxx-exceptions -pedantic-errors -verify=expected,cxx98-14
// RUN: %clang_cc1 -std=c++17 %s -fexceptions -fcxx-exceptions -pedantic-errors -verify=expected,since-cxx17
// RUN: %clang_cc1 -std=c++20 %s -fexceptions -fcxx-exceptions -pedantic-errors -verify=expected,since-cxx17
// RUN: %clang_cc1 -std=c++23 %s -fexceptions -fcxx-exceptions -pedantic-errors -verify=expected,since-cxx17
// RUN: %clang_cc1 -std=c++2c %s -fexceptions -fcxx-exceptions -pedantic-errors -verify=expected,since-cxx17

// cxx98-14-no-diagnostics

namespace cwg2406 { // cwg2406: 5
#if __cplusplus >= 201703L
void fallthrough(int n) {
  void g(), h(), i();
  switch (n) {
  case 1:
  case 2:
    g();
    [[fallthrough]];
  case 3: // warning on fallthrough discouraged
    do {
      [[fallthrough]];
      // since-cxx17-error@-1 {{fallthrough annotation does not directly precede switch label}}
    } while (false);
  case 6:
    do {
      [[fallthrough]];
      // since-cxx17-error@-1 {{fallthrough annotation does not directly precede switch label}}
    } while (n);
  case 7:
    while (false) {
      [[fallthrough]];
      // since-cxx17-error@-1 {{fallthrough annotation does not directly precede switch label}}
    }
  case 5:
    h();
  case 4: // implementation may warn on fallthrough
    i();
    [[fallthrough]];
    // since-cxx17-error@-1 {{fallthrough annotation does not directly precede switch label}}
  }
}
#endif
} // namespace cwg2406
