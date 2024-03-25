// RUN: %clang_cc1 -verify=ref,both %s -fms-extensions
// RUN: %clang_cc1 -verify=expected,both %s -fexperimental-new-constant-interpreter -fms-extensions

// ref-no-diagnostics
// expected-no-diagnostics

/// Used to assert because the two parameters to _rotl do not have the same type.
static_assert(_rotl(0x01, 5) == 32);
