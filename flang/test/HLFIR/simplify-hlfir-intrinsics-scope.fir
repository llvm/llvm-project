// Test the scope option of --simplify-hlfir-intrinsics.
// RUN: fir-opt %s --simplify-hlfir-intrinsics | FileCheck %s --check-prefixes=ALL,SUM-SIMPLIFY,OTHER-SIMPLIFY
// RUN: fir-opt %s --simplify-hlfir-intrinsics=scope=all | FileCheck %s --check-prefixes=ALL,SUM-SIMPLIFY,OTHER-SIMPLIFY
// RUN: fir-opt %s --simplify-hlfir-intrinsics=scope=only-sum | FileCheck %s --check-prefixes=ALL,SUM-SIMPLIFY,OTHER-KEEP
// RUN: fir-opt %s --simplify-hlfir-intrinsics=scope=except-sum | FileCheck %s --check-prefixes=ALL,SUM-KEEP,OTHER-SIMPLIFY
// RUN: fir-opt %s --simplify-hlfir-intrinsics=scope=none | FileCheck %s --check-prefixes=ALL,SUM-KEEP,OTHER-KEEP

func.func @sum_real(%arg0: !fir.box<!fir.array<?xf32>>) -> f32 {
  %res = hlfir.sum %arg0 : (!fir.box<!fir.array<?xf32>>) -> f32
  return %res : f32
}
// ALL-LABEL: func.func @sum_real(
// SUM-SIMPLIFY-NOT: hlfir.sum
// SUM-SIMPLIFY:     fir.do_loop
// SUM-SIMPLIFY-NOT: hlfir.sum
// SUM-SIMPLIFY:     arith.addf
// SUM-SIMPLIFY-NOT: hlfir.sum
// SUM-KEEP-NOT:     fir.do_loop
// SUM-KEEP:         hlfir.sum %{{.*}} : (!fir.box<!fir.array<?xf32>>) -> f32
// SUM-KEEP-NOT:     fir.do_loop
// ALL:              return

func.func @product_real(%arg0: !fir.box<!fir.array<?xf32>>) -> f32 {
  %res = hlfir.product %arg0 : (!fir.box<!fir.array<?xf32>>) -> f32
  return %res : f32
}
// ALL-LABEL: func.func @product_real(
// OTHER-SIMPLIFY-NOT: hlfir.product
// OTHER-SIMPLIFY:     fir.do_loop
// OTHER-SIMPLIFY-NOT: hlfir.product
// OTHER-SIMPLIFY:     arith.mulf
// OTHER-SIMPLIFY-NOT: hlfir.product
// OTHER-KEEP-NOT:     fir.do_loop
// OTHER-KEEP:         hlfir.product %{{.*}} : (!fir.box<!fir.array<?xf32>>) -> f32
// OTHER-KEEP-NOT:     fir.do_loop
// ALL:                return
