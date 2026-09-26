// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.2-vertex -emit-llvm -finclude-default-header -disable-llvm-passes -fdx-semantic-signature-packing-mode=prefix-stable -o - %s | FileCheck %s --check-prefix=WIDE
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.2-vertex -emit-llvm -finclude-default-header -disable-llvm-passes -fdx-semantic-signature-packing-mode=optimized -o - %s | FileCheck %s --check-prefix=WIDE
// RUN: %clang_dxc -T vs_6_2 -fcgl -enable-16bit-types -pack-prefix-stable %s | FileCheck %s --check-prefix=NATIVE
// RUN: %clang_dxc -T vs_6_2 -fcgl -enable-16bit-types -pack-optimized %s | FileCheck %s --check-prefix=NATIVE

struct Output {
  half a : A;
  float b : B;
  half c : C;
};

Output main() {
  Output output = {1, 2, 3};
  return output;
}

// Without native 16-bit types, half is float and all elements share a row.
// WIDE-DAG: !{i32 0, !"A", i32 9, i32 0, !{{[0-9]+}}, i32 0, i32 1, i8 1, i32 0, i8 0, i8 0, i8 0, i32 0}
// WIDE-DAG: !{i32 1, !"B", i32 9, i32 0, !{{[0-9]+}}, i32 0, i32 1, i8 1, i32 0, i8 1, i8 0, i8 0, i32 0}
// WIDE-DAG: !{i32 2, !"C", i32 9, i32 0, !{{[0-9]+}}, i32 0, i32 1, i8 1, i32 0, i8 2, i8 0, i8 0, i32 0}

// Native 16-bit and 32-bit components must occupy different rows in both modes.
// NATIVE-DAG: !{i32 0, !"A", i32 8, i32 0, !{{[0-9]+}}, i32 0, i32 1, i8 1, i32 0, i8 0, i8 0, i8 0, i32 0}
// NATIVE-DAG: !{i32 1, !"B", i32 9, i32 0, !{{[0-9]+}}, i32 0, i32 1, i8 1, i32 1, i8 0, i8 0, i8 0, i32 0}
// NATIVE-DAG: !{i32 2, !"C", i32 8, i32 0, !{{[0-9]+}}, i32 0, i32 1, i8 1, i32 0, i8 1, i8 0, i8 0, i32 0}
