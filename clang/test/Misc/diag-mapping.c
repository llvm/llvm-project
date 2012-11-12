// This should warn by default.
// RUN: %clang_cc1 %s 2>&1 | grep "warning:"
// This should not emit anything.
// RUN: %clang_cc1 %s -Wno-extra-tokens 2>&1 | not grep diagnostic

// -Werror can map all warnings to error.
// RUN: %clang_cc1 %s -Werror 2>&1 | grep "error:"

// -Werror can map this one warning to error.
// RUN: %clang_cc1 %s -Werror=extra-tokens 2>&1 | grep "error:"

// Mapping unrelated diags to errors doesn't affect this one.
// RUN: %clang_cc1 %s -Werror=trigraphs 2>&1 | grep "warning:"

// This should stay a warning with -pedantic.
// RUN: %clang_cc1 %s -pedantic 2>&1 | grep "warning:"

// This should emit an error with -pedantic-errors.
// RUN: %clang_cc1 %s -pedantic-errors 2>&1 | grep "error:"

// This should emit a warning, because -Wfoo overrides -pedantic*.
// RUN: %clang_cc1 %s -pedantic-errors -Wextra-tokens 2>&1 | grep "warning:"

// This should emit nothing, because -Wno-extra-tokens overrides -pedantic*
// RUN: %clang_cc1 %s -pedantic-errors -Wno-extra-tokens 2>&1 | not grep diagnostic

#ifdef foo
#endif bad // extension!

int x;
