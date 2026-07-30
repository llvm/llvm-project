// RUN: %clang_cc1 -E %s -Wall -verify
// RUN: %clang_cc1 -Eonly %s -Wall -verify
// RUN: %clang -M -Wall %s -Xclang -verify
// RUN: %clang -E -frewrite-includes %s -Wall -Xclang -verify
// RUN: %clang -E -dD -dM %s -Wall -Xclang -verify
// expected-no-diagnostics

#pragma GCC visibility push (default)
#pragma weak
#pragma this_pragma_does_not_exist
