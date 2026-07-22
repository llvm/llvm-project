// '__GLIBCXX__' defined:
//
// RUN: %clang_cc1 -fsyntax-only -std=c++23 -verify %s -DDEFINE
// RUN: %clang_cc1 -E -std=c++23 %s -o %t.ii -DDEFINE
// RUN: echo '// expected-no-diagnostics' >> %t.ii
// RUN: FileCheck --input-file=%t.ii %s
// RUN: %clang_cc1 -fsyntax-only -std=c++23 -verify %t.ii

// Version set via pragma:
//
// RUN: %clang_cc1 -fsyntax-only -std=c++23 -verify %s
// RUN: %clang_cc1 -E -std=c++23 %s -o %t.ii
// RUN: echo '// expected-no-diagnostics' >> %t.ii
// RUN: FileCheck --input-file=%t.ii %s
// RUN: %clang_cc1 -fsyntax-only -std=c++23 -verify %t.ii

// expected-no-diagnostics

// Primary variable template std::format_kind is defined as followed since
// libstdc++ 15.1, which triggers compilation error introduced by GH134522.
// This file tests the workaround.
//
// Since the workaround relies on '__GLIBCXX__' being defined, we emit a pragma
// that ensures '__GLIBCXX__' is defined if the user first preprocesses the file
// with '-E' before passing the output of that back to Clang.

// CHECK: #pragma clang glibcxx_version 20250513
#ifdef DEFINE
#   define __GLIBCXX__ 20250513
#else
#   pragma clang glibcxx_version 20250513
#endif

namespace std {
  template<typename _Rg>
    constexpr auto format_kind =
    __primary_template_not_defined(
      format_kind<_Rg>
    );
}
