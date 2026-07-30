// RUN: %clang_cc1 -E -verify %s
// expected-no-diagnostics

#if 0

// Shouldn't get warnings here.
??( ??)

// Should not get an error here.
` ` ` `
#endif
