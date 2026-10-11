// Check that we accept the program if '__GLIBCXX__' is defined:
// RUN: %clang_cc1 -fsyntax-only -std=c++23 -verify %s -DDEFINE_GLIBCXX

// Check that we preserve the value of __GLIBCXX__ via a pragma when preprocessing:
// RUN: %clang_cc1 -E -std=c++23 %s -o %t.ii -DDEFINE_GLIBCXX
// RUN: FileCheck --input-file=%t.ii %s

// Check that the preprocessed file compiles with no diagnostics:
// RUN: echo '// expected-no-diagnostics' >> %t.ii
// RUN: %clang_cc1 -fsyntax-only -std=c++23 -verify %t.ii

// Check that we accept the program if the pragma is present:
// RUN: %clang_cc1 -fsyntax-only -std=c++23 -verify %s -DUSE_PRAGMA

// Check that we preserve the pragma when preprocessing:
// RUN: %clang_cc1 -E -std=c++23 %s -o %t.ii -DUSE_PRAGMA
// RUN: FileCheck --input-file=%t.ii %s

// Check that the preprocessed file compiles with no diagnostics:
// RUN: echo '// expected-no-diagnostics' >> %t.ii
// RUN: %clang_cc1 -fsyntax-only -std=c++23 -verify %t.ii

// Irrespective of whether we used the pragma directly or defined __GLIBCXX__,
// the preprocessed output should contain the pragma:
// CHECK: #pragma clang __set_pp_state __GLIBCXX__ 20250513

// expected-no-diagnostics

// Primary variable template std::format_kind is defined as followed since
// libstdc++ 15.1, which triggers compilation error introduced by GH134522.
// This file tests the workaround.
//
// Since the workaround relies on '__GLIBCXX__' being defined, we emit a pragma
// that ensures '__GLIBCXX__' is defined if the user first preprocesses the file
// with '-E' before passing the output of that back to Clang.

#ifdef DEFINE_GLIBCXX
#   define __GLIBCXX__ 20250513
#endif

#ifdef USE_PRAGMA
#   pragma clang __set_pp_state __GLIBCXX__ 20250513
#endif

namespace std {
  template<typename _Rg>
    constexpr auto format_kind =
    __primary_template_not_defined(
      format_kind<_Rg>
    );
}
