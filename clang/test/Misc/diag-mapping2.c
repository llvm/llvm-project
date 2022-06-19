// This should warn by default.
// RUN: %clang_cc1 %s 2>&1 | grep "warning: foo"

// This should not emit anything.
// RUN: %clang_cc1 %s -w 2>&1 | not grep diagnostic
// RUN: %clang_cc1 %s -Wno-#warnings 2>&1 | not grep diagnostic
// RUN: %clang_cc1 %s -Wno-cpp 2>&1 | not grep diagnostic

// -Werror can map all warnings to error.
// RUN: not %clang_cc1 %s -Werror 2>&1 | grep "error: foo"

// -Werror can map this one warning to error.
// RUN: not %clang_cc1 %s -Werror=#warnings 2>&1 | grep "error: foo"
// RUN: not %clang_cc1 %s -Werror=#warnings -W#warnings 2>&1 | grep "error: foo"

// -Wno-error= overrides -Werror. -Wno-error applies to a subsequent warning of the same name.
// RUN: %clang_cc1 %s -Werror -Wno-error=#warnings 2>&1 | grep "warning: foo"
// RUN: %clang_cc1 %s -Werror -Wno-error=#warnings -W#warnings 2>&1 | grep "warning: foo"

// -Wno-error overrides -Werror.  PR4715
// RUN: %clang_cc1 %s -Werror -Wno-error 2>&1 | grep "warning: foo"

#warning foo

