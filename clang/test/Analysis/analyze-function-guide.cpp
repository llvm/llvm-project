int fizzbuzz(int x, bool y) {
  return x + y;
}

// C++ but not uses parentheses in the '-analyze-function' option.
//
// RUN: %clang_analyze_cc1 -analyzer-checker=core \
// RUN:   -analyze-function='missing_fn' -x c++ \
// RUN:   -triple x86_64-pc-linux-gnu 2>&1 %s \
// RUN: | FileCheck %s -check-prefix=CHECK-CXX
//
// CHECK-CXX:      Every top-level function was skipped.
// CHECK-CXX-NEXT: Pass the -analyzer-display-progress for tracking which functions are analyzed.
// CHECK-CXX-NEXT: For analyzing C++ code you need to pass the function parameter list: -analyze-function="foobar(int, _Bool)"

// C but uses parentheses in the '-analyze-function' option.
//
// RUN: %clang_analyze_cc1 -analyzer-checker=core \
// RUN:   -analyze-function='missing_fn()' -x c -Dbool=_Bool \
// RUN:   -triple x86_64-pc-linux-gnu 2>&1 %s \
// RUN: | FileCheck %s -check-prefix=CHECK-C
//
// CHECK-C:      Every top-level function was skipped.
// CHECK-C-NEXT: Pass the -analyzer-display-progress for tracking which functions are analyzed.
// CHECK-C-NEXT: For analyzing C code you shouldn't pass the function parameter list, only the name of the function: -analyze-function=foobar

// The user passed the '-analyzer-display-progress' option, we don't need to advocate it.
//
// RUN: %clang_analyze_cc1 -analyzer-checker=core \
// RUN:   -analyze-function=missing_fn \
// RUN:   -analyzer-display-progress -x c -Dbool=_Bool \
// RUN:   -triple x86_64-pc-linux-gnu 2>&1 %s \
// RUN: | FileCheck %s -check-prefix=CHECK-DONT-ADVOCATE-DISPLAY-PROGRESS
//
// CHECK-DONT-ADVOCATE-DISPLAY-PROGRESS:     Every top-level function was skipped.
// CHECK-DONT-ADVOCATE-DISPLAY-PROGRESS-NOT: Pass the -analyzer-display-progress

// The user passed the '-analyze-function' option but that doesn't mach to any declaration.
//
// RUN: %clang_analyze_cc1 -analyzer-checker=core \
// RUN:   -analyze-function='missing_fn()' -x c++ \
// RUN:   -triple x86_64-pc-linux-gnu 2>&1 %s \
// RUN: | FileCheck %s -check-prefix=CHECK-ADVOCATE-DISPLAY-PROGRESS
//
// CHECK-ADVOCATE-DISPLAY-PROGRESS:      Every top-level function was skipped.
// CHECK-ADVOCATE-DISPLAY-PROGRESS-NEXT: Pass the -analyzer-display-progress for tracking which functions are analyzed.
// CHECK-ADVOCATE-DISPLAY-PROGRESS-NOT:  For analyzing

// Same as the previous but syntax mode only.
// FIXME: This should have empty standard output.
//
// RUN: %clang_analyze_cc1 -analyzer-checker=core -analyzer-config ipa=none \
// RUN:   -analyze-function='fizzbuzz(int, _Bool)' -x c++ \
// RUN:   -triple x86_64-pc-linux-gnu 2>&1 %s \
// RUN: | FileCheck %s -check-prefix=CHECK-EMPTY3 --allow-empty
//
// FIXME: This should have empty standard output.
// CHECK-EMPTY3:      Every top-level function was skipped.
// CHECK-EMPTY3-NEXT: Pass the -analyzer-display-progress for tracking which functions are analyzed.
