
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s  \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s

// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN: spirv-unknown-vulkan-compute %s \
// RUN: -emit-llvm -disable-llvm-passes -o - | FileCheck %s

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
