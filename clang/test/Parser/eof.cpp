// RUN: not %clang_cc1 %s -fsyntax-only 2>&1 | FileCheck %s

// CHECK: error: expected '<' after 'template'
// CHECK: error: expected '}'
// CHECK: note: to match this '{'
// CHECK: error: expected ';' after class
// CHECK: error: anonymous structs and classes must be class members
// CHECK: 4 errors generated.

// Do not add anything to the end of this file.  This requires the whitespace
// plus EOF after the template keyword.

class { template     
