// RUN: not clang-tidy %s \
// RUN:     -checks="-*,bugprone-suspicious-semicolon" -- -DERROR 2>&1 \
// RUN:   | FileCheck %s -check-prefix=CHECK-ERROR \
// RUN:       -implicit-check-not="{{warning|error}}:"
// RUN: not clang-tidy %s \
// RUN:     -checks="-*,bugprone-suspicious-semicolon,clang-diagnostic*" \
// RUN:    -- -DWERROR -Wno-everything -Werror=unused-variable 2>&1 \
// RUN:   | FileCheck %s -check-prefix=CHECK-WERROR \
// RUN:       -implicit-check-not="{{warning|error}}:"

// Note: This test verifies that, the checker does not emit any warning for
//       files that do not compile.

bool g();

void f() {
  if (g());
  // CHECK-WERROR: :[[@LINE-1]]:11: warning: potentially unintended semicolon [bugprone-suspicious-semicolon]
#if ERROR
  int a
  // CHECK-ERROR: :[[@LINE-1]]:8: error: expected ';' at end of declaration [clang-diagnostic-error]
#elif WERROR
  int a;
  // CHECK-WERROR: :[[@LINE-1]]:7: error: unused variable 'a' [clang-diagnostic-unused-variable]
#else
#error "One of ERROR or WERROR should be defined.
#endif
}
