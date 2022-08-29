// RUN: %clang_cc1 -std=c99 -E %s -o - | FileCheck --check-prefix=CHECK-NONE %s

// RUN: %clang_cc1 -std=gnu89 -E %s -o - \
// RUN:     | FileCheck --check-prefix=CHECK-GNU-KEYWORDS %s
// RUN: %clang_cc1 -std=c99 -fgnu-keywords -E %s -o - \
// RUN:     | FileCheck --check-prefix=CHECK-GNU-KEYWORDS %s
// RUN: %clang_cc1 -std=gnu89 -fno-gnu-keywords -E %s -o - \
// RUN:     | FileCheck --check-prefix=CHECK-NONE %s

// RUN: %clang_cc1 -std=c99 -fms-extensions -fms-compatibility -E %s -o - \
// RUN:     | FileCheck --check-prefix=CHECK-MS-KEYWORDS %s
// RUN: %clang_cc1 -std=c99 -fdeclspec -E %s -o - \
// RUN:     | FileCheck --check-prefix=CHECK-DECLSPEC-KEYWORD %s
// RUN: %clang_cc1 -std=c99 -fms-extensions -fno-declspec -E %s -o - \
// RUN:     | FileCheck --check-prefix=CHECK-MS-KEYWORDS-WITHOUT-DECLSPEC %s

// RUN: %clang_cc1 -std=c99 -DC99 -fsyntax-only %s
// RUN: %clang_cc1 -std=c2x -DC99 -DC2x -fsyntax-only %s

// RUN: %clang_cc1 -fsyntax-only -std=c89 -DFutureKeyword -Wc2x-compat -Wc99-compat -verify=c89 %s

#define IS_KEYWORD(NAME) _Static_assert(!__is_identifier(NAME), #NAME)
#define NOT_KEYWORD(NAME) _Static_assert(__is_identifier(NAME), #NAME)

#if defined(C99)
#define C99_KEYWORD(NAME)  IS_KEYWORD(NAME)
#else
#define C99_KEYWORD(NAME)  NOT_KEYWORD(NAME)
#endif

#if defined(C2x)
#define C2x_KEYWORD(NAME)  IS_KEYWORD(NAME)
#else
#define C2x_KEYWORD(NAME)  NOT_KEYWORD(NAME)
#endif

// C99 Keywords.
C99_KEYWORD(restrict);
C99_KEYWORD(inline);

// C2x Keywords.
C2x_KEYWORD(bool);
C2x_KEYWORD(true);
C2x_KEYWORD(false);
C2x_KEYWORD(static_assert);
C2x_KEYWORD(typeof);
C2x_KEYWORD(thread_local);
C2x_KEYWORD(alignas);
C2x_KEYWORD(alignof);

void f() {
// CHECK-NONE: int asm
// CHECK-GNU-KEYWORDS: asm ("ret" : :)
#if __is_identifier(asm)
  int asm;
#else
  asm ("ret" : :);
#endif
}

// CHECK-NONE: no_ms_wchar
// CHECK-MS-KEYWORDS: has_ms_wchar
// CHECK-MS-KEYWORDS-WITHOUT-DECLSPEC: has_ms_wchar
#if __is_identifier(__wchar_t)
void no_ms_wchar();
#else
void has_ms_wchar();
#endif

// CHECK-NONE: no_declspec
// CHECK-MS-KEYWORDS: has_declspec
// CHECK-MS-KEYWORDS-WITHOUT-DECLSPEC: no_declspec
// CHECK-DECLSPEC-KEYWORD: has_declspec
#if __is_identifier(__declspec)
void no_declspec();
#else
void has_declspec();
#endif

// CHECK-NONE: no_static_assert
// CHECK-GNU-KEYWORDS: no_static_assert
// CHECK-MS-KEYWORDS: has_static_assert
// CHECK-MS-KEYWORDS-WITHOUT-DECLSPEC: no_static_assert
#if __is_identifier(static_assert)
void no_static_assert();
#else
void has_static_assert();
#endif

#ifdef FutureKeyword

  int restrict; // c89-warning {{'restrict' is a keyword in C99}}
  int inline;  // c89-warning {{'inline' is a keyword in C99}}

  int bool; // c89-warning {{'bool' is a keyword in C2x}}
  char true; // c89-warning {{'true' is a keyword in C2x}}
  char false; // c89-warning {{'false' is a keyword in C2x}}
  float alignof; // c89-warning {{'alignof' is a keyword in C2x}}
  int typeof; // c89-warning {{'typeof' is a keyword in C2x}}
  int alignas; // c89-warning {{'alignas' is a keyword in C2x}}
  int static_assert; // c89-warning {{'static_assert' is a keyword in C2x}}

#endif
