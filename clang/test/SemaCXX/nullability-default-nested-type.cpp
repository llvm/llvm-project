// Regression test: -fnullability-default injection must NOT emit the
// warn_nullability_inferred_on_nested_type diagnostic (with its bogus
// nested-position _Nonnull fixit). That diagnostic is reserved for the
// genuine `#pragma clang assume_nonnull` inference path, which must still
// emit it.

// Default-injection path: no nested-type inference warning.
// RUN: %clang_cc1 -fsyntax-only -fnullability-default=nonnull -verify=default %s
// RUN: %clang_cc1 -fsyntax-only -fnullability-default=nullable -verify=default %s
// Pragma path: nested-type inference warning is preserved.
// RUN: %clang_cc1 -fsyntax-only -verify=pragma %s -DPRAGMA

#ifdef PRAGMA
#pragma clang assume_nonnull begin

// pragma-warning@+1 {{inferring '_Nonnull' for pointer type within reference is deprecated}}
extern int *&ref;
// pragma-warning@+1 {{inferring '_Nonnull' for pointer type within array is deprecated}}
extern int *arr[2];

#pragma clang assume_nonnull end
#else
// default-no-diagnostics

// Same declarators under -fnullability-default must not warn about
// inferring within a nested (array/reference) chunk.
extern int *&ref;
extern int *arr[2];
#endif
