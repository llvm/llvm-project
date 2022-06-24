// The standard grammar allows an function-body to use any declarator, including
// a non-function declarator. This creates an ambiguity where a
// simple-declaration is misparsed as a function-definition.
// FIXME: eliminate this false parse.
// XFAIL: *

// RUN: clang-pseudo -grammar=%cxx-bnf-file -source=%s --print-forest | FileCheck %s
void (*s)(){};
// CHECK-NOT:      function-definition
// CHECK:          init-declarator := declarator initializer
// CHECK-NOT:      function-definition
