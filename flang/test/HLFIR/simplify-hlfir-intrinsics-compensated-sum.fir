// Test SUM inlining and pattern selection independently.
// RUN: fir-opt %s --simplify-hlfir-intrinsics | FileCheck %s --check-prefixes=ALL,SUM-SIMPLIFY,OTHER-SIMPLIFY
// RUN: fir-opt %s --simplify-hlfir-intrinsics="only-sum=false compensated-sum=false" | FileCheck %s --check-prefixes=ALL,SUM-SIMPLIFY,OTHER-SIMPLIFY
// RUN: fir-opt %s --simplify-hlfir-intrinsics="only-sum=true compensated-sum=false" | FileCheck %s --check-prefixes=ALL,SUM-SIMPLIFY,OTHER-KEEP
// RUN: fir-opt %s --simplify-hlfir-intrinsics="only-sum=false compensated-sum=true" | FileCheck %s --check-prefixes=ALL,SUM-KEEP,OTHER-SIMPLIFY
// RUN: fir-opt %s --simplify-hlfir-intrinsics="only-sum=true compensated-sum=true" | FileCheck %s --check-prefixes=ALL,SUM-KEEP,OTHER-KEEP

func.func @sum_integer(%arg0: !fir.box<!fir.array<?xi32>>) -> i32 {
  %res = hlfir.sum %arg0 : (!fir.box<!fir.array<?xi32>>) -> i32
  return %res : i32
}
// ALL-LABEL: func.func @sum_integer(
// ALL-NOT:   hlfir.sum
// ALL:       fir.do_loop
// ALL-NOT:   hlfir.sum
// ALL:       arith.addi
// ALL-NOT:   hlfir.sum
// ALL:       return

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

func.func @sum_complex_dim(%arg0: !fir.box<!fir.array<?x3xcomplex<f64>>>) -> !hlfir.expr<3xcomplex<f64>> {
  %dim = arith.constant 1 : i32
  %res = hlfir.sum %arg0 dim %dim : (!fir.box<!fir.array<?x3xcomplex<f64>>>, i32) -> !hlfir.expr<3xcomplex<f64>>
  return %res : !hlfir.expr<3xcomplex<f64>>
}
// ALL-LABEL: func.func @sum_complex_dim(
// SUM-SIMPLIFY-NOT: hlfir.sum
// SUM-SIMPLIFY:     hlfir.elemental
// SUM-SIMPLIFY-NOT: hlfir.sum
// SUM-SIMPLIFY:     fir.do_loop
// SUM-SIMPLIFY-NOT: hlfir.sum
// SUM-SIMPLIFY:     fir.addc
// SUM-SIMPLIFY-NOT: hlfir.sum
// SUM-KEEP-NOT:     hlfir.elemental
// SUM-KEEP-NOT:     fir.do_loop
// SUM-KEEP:         hlfir.sum
// SUM-KEEP-NOT:     hlfir.elemental
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
