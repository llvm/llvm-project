// RUN: not %clang_cc1 -std=c++11 -fsyntax-only %s -fconstexpr-backtrace-limit=0 -fconstexpr-depth=4 -fno-caret-diagnostics 2>&1 | FileCheck %s -check-prefix=TEST1
// TEST1: constant expression
// TEST1-NEXT: exceeded maximum depth of 4
// TEST1-NEXT: in call to 'recurse(2)'
// TEST1-NEXT: in call to 'recurse(3)'
// TEST1-NEXT: in call to 'recurse(4)'
// TEST1-NEXT: in call to 'recurse(5)'

// RUN: not %clang_cc1 -std=c++11 -fsyntax-only %s -fconstexpr-backtrace-limit=2 -fconstexpr-depth=4 -fno-caret-diagnostics 2>&1 | FileCheck %s -check-prefix=TEST2
// TEST2: constant expression
// TEST2-NEXT: exceeded maximum depth of 4
// TEST2-NEXT: in call to 'recurse(2)'
// TEST2-NEXT: skipping 2 calls
// TEST2-NEXT: in call to 'recurse(5)'

// RUN: not %clang_cc1 -std=c++11 -fsyntax-only %s -fconstexpr-backtrace-limit=2 -fconstexpr-depth=8 -fno-caret-diagnostics 2>&1 | FileCheck %s -check-prefix=TEST3
// TEST3: constant expression
// TEST3-NEXT: reinterpret_cast
// TEST3-NEXT: in call to 'recurse(0)'
// TEST3-NEXT: skipping 4 calls
// TEST3-NEXT: in call to 'recurse(5)'

// RUN: not %clang_cc1 -std=c++11 -fsyntax-only %s -fconstexpr-backtrace-limit=8 -fconstexpr-depth=8 -fno-caret-diagnostics 2>&1 | FileCheck %s -check-prefix=TEST4
// TEST4: constant expression
// TEST4-NEXT: reinterpret_cast
// TEST4-NEXT: in call to 'recurse(0)'
// TEST4-NEXT: in call to 'recurse(1)'
// TEST4-NEXT: in call to 'recurse(2)'
// TEST4-NEXT: in call to 'recurse(3)'
// TEST4-NEXT: in call to 'recurse(4)'
// TEST4-NEXT: in call to 'recurse(5)'

constexpr int recurse(int n) { return n ? recurse(n-1) : *(int*)n; }
static_assert(recurse(5), "");
