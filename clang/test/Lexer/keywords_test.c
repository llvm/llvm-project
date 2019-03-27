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
