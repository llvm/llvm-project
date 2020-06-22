// Test that a [no_undeclared_headers] module can include builtin headers, even
// if these have been "claimed" by a different module that wraps these builtin
// headers. libc++ does this, for example.
//
// The test inputs used here replicates the relationship between libc++ and
// glibc. When modularizing glibc, [no_undeclared_headers] must be used to
// prevent glibc from including the libc++ versions of the C standard library
// headers.

// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules-cache-path=%t -fmodules -fimplicit-module-maps -I %S/Inputs/no-undeclared-includes-builtins/libcxx -I %S/Inputs/no-undeclared-includes-builtins/glibc %s
// expected-no-diagnostics

#include <stddef.h>
