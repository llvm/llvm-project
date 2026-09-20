// RUN: %clang_cc1 -triple powerpc-apple-macosx10.4.0 -verify -fsyntax-only %s
// expected-no-diagnostics
extern __typeof(+(_Bool)0) should_be_int;
extern int should_be_int;
