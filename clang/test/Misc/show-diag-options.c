// RUN: %clang_cc1 -fsyntax-only %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix=BASE
// RUN: %clang_cc1 -fsyntax-only -fdiagnostics-show-option %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix=OPTION
// RUN: %clang_cc1 -fsyntax-only -fdiagnostics-show-option -Werror %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix=OPTION_ERROR
// RUN: %clang_cc1 -fsyntax-only -std=c89 -pedantic -fdiagnostics-show-option %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix=OPTION_PEDANTIC
// RUN: %clang_cc1 -fsyntax-only -fdiagnostics-show-category id %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix=CATEGORY_ID
// RUN: %clang_cc1 -fsyntax-only -fdiagnostics-show-category name %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix=CATEGORY_NAME
// RUN: %clang_cc1 -fsyntax-only -fdiagnostics-show-option -fdiagnostics-show-category name -Werror %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix=OPTION_ERROR_CATEGORY

void test(int x, int y) {
  if (x = y) ++x;
  // BASE: {{.*}}: warning: {{[a-z ]+$}}
  // OPTION: {{.*}}: warning: {{[a-z ]+}} [-Wparentheses]
  // OPTION_ERROR: {{.*}}: error: {{[a-z ]+}} [-Werror,-Wparentheses]
  // CATEGORY_ID: {{.*}}: warning: {{[a-z ]+}} [2]
  // CATEGORY_NAME: {{.*}}: warning: {{[a-z ]+}} [Semantic Issue]
  // OPTION_ERROR_CATEGORY: {{.*}}: error: {{[a-z ]+}} [-Werror,-Wparentheses,Semantic Issue]

  // Leverage the fact that all these '//'s get warned about in C89 pedantic.
  // OPTION_PEDANTIC: {{.*}}: warning: {{[/a-z ]+}} [-Wcomment]
}
