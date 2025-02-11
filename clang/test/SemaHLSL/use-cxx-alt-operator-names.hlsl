// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library %s -ast-dump | FileCheck %s

// CHECK-LABEL: and
void and() {}

// CHECK-LABEL: and_eq
void and_eq() {}

// CHECK-LABEL: bitand
void bitand() {}

// CHECK-LABEL: bitor
void bitor() {}

// CHECK-LABEL: compl
void compl() {}

// CHECK-LABEL: not
void not() {}

// CHECK-LABEL: not_eq
void not_eq() {}

// CHECK-LABEL: or
void or() {}

// CHECK-LABEL: or_eq
void or_eq() {}

// CHECK-LABEL: xor
void xor() {}

// CHECK-LABEL: xor_eq
void xor_eq() {}
