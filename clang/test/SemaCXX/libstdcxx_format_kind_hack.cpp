// RUN: %clang_cc1 -fsyntax-only -std=c++23 -verify %s
// RUN: %clang_cc1 -E -std=c++23 %s -o %t.ii
// RUN: echo '// expected-no-diagnostics' >> %t.ii
// RUN: %clang_cc1 -fsyntax-only -std=c++23 -verify %t.ii

// expected-no-diagnostics

// Primary variable template std::format_kind is defined as followed since
// libstdc++ 15.1, which triggers compilation error introduced by GH134522.
// This file tests the workaround.

#define __GLIBCXX__ 20250513

namespace std {
  template<typename _Rg>
    constexpr auto format_kind =
    __primary_template_not_defined(
      format_kind<_Rg>
    );
}
