// This should warn by default.
// RUN: %clang_cc1 %s 2>&1 | grep "warning:"

// This should not emit anything.
// RUN: %clang_cc1 %s -w 2>&1 | not grep diagnostic
// RUN: %clang_cc1 %s -Wno-#warnings 2>&1 | not grep diagnostic

// -Werror can map all warnings to error.
// RUN: %clang_cc1 %s -Werror 2>&1 | grep "error:"

// -Werror can map this one warning to error.
// RUN: %clang_cc1 %s -Werror=#warnings 2>&1 | grep "error:"

// -Wno-error= overrides -Werror.  rdar://3158301
// RUN: %clang_cc1 %s -Werror -Wno-error=#warnings 2>&1 | grep "warning:"

// -Wno-error overrides -Werror.  PR4715
// RUN: %clang_cc1 %s -Werror -Wno-error 2>&1 | grep "warning:"

#warning foo


