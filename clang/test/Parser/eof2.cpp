// RUN: not %clang_cc1 %s -fsyntax-only 2>&1 | FileCheck %s

// CHECK: error: expected expression
// CHECK: error: expected '>'
// CHECK: error: expected member name or ';' after declaration specifiers
// CHECK: error: expected '}'
// CHECK: note: to match this '{'
// CHECK: error: expected ';' after class
// CHECK: 5 errors generated.

// Do not add anything to the end of this file.  This requires the whitespace
// plus EOF after the '<' token.

template <typename T>
class a {
  a< 
