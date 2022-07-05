// The standard grammar allows an init-list with any declarator, including
// a function declarator. This creates an ambiguity where a function-definition
// is misparsed as a simple-declaration.
// FIXME: eliminate this false parse.
// XFAIL: *

// RUN: clang-pseudo -grammar=cxx -source=%s --print-forest | FileCheck %s
void s(){};
// CHECK-NOT:      simple-declaration
// CHECK:          function-definition := decl-specifier-seq declarator
// function-body CHECK-NOT:      simple-declaration
