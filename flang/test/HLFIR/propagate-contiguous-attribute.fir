// RUN: fir-opt --propagate-fortran-attrs --split-input-file %s | FileCheck %s

//subroutine test1(x, n, k1, k2)
//  real :: x(n)
//  x(1:k1) = x(k1+1:k2)
//end subroutine test1

// CHECK-LABEL:   func.func @_QPtest1(
func.func @_QPtest1(%arg0: !fir.ref<!fir.array<?xf32>> {fir.bindc_name = "x"}, %arg1: !fir.ref<i32> {fir.bindc_name = "n"}, %arg2: !fir.ref<i32> {fir.bindc_name = "k1"}, %arg3: !fir.ref<i32> {fir.bindc_name = "k2"}) {
  %c1 = arith.constant 1 : index
  %c1_i32 = arith.constant 1 : i32
  %c0 = arith.constant 0 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg2 dummy_scope %0 {uniq_name = "_QFtest1Ek1"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %2:2 = hlfir.declare %arg3 dummy_scope %0 {uniq_name = "_QFtest1Ek2"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %3:2 = hlfir.declare %arg1 dummy_scope %0 {uniq_name = "_QFtest1En"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %4 = fir.load %3#0 : !fir.ref<i32>
  %5 = fir.convert %4 : (i32) -> index
  %6 = arith.cmpi sgt, %5, %c0 : index
  %7 = arith.select %6, %5, %c0 : index
  %8 = fir.shape %7 : (index) -> !fir.shape<1>
  %9:2 = hlfir.declare %arg0(%8) dummy_scope %0 {uniq_name = "_QFtest1Ex"} : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>)
  %10 = fir.load %1#0 : !fir.ref<i32>
  %11 = arith.addi %10, %c1_i32 overflow<nsw> : i32
  %12 = fir.load %2#0 : !fir.ref<i32>
  %13 = fir.convert %11 : (i32) -> index
  %14 = fir.convert %12 : (i32) -> index
  %15 = arith.subi %14, %13 : index
  %16 = arith.addi %15, %c1 : index
  %17 = arith.cmpi sgt, %16, %c0 : index
  %18 = arith.select %17, %16, %c0 : index
  %19 = fir.shape %18 : (index) -> !fir.shape<1>
// CHECK: hlfir.designate{{.*}}{fortran_attrs = #fir.var_attrs<contiguous>} : (!fir.box<!fir.array<?xf32>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
  %20 = hlfir.designate %9#0 (%13:%14:%c1)  shape %19 : (!fir.box<!fir.array<?xf32>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
  %21 = fir.convert %10 : (i32) -> index
  %22 = arith.cmpi sgt, %21, %c0 : index
  %23 = arith.select %22, %21, %c0 : index
  %24 = fir.shape %23 : (index) -> !fir.shape<1>
// CHECK: hlfir.designate{{.*}}{fortran_attrs = #fir.var_attrs<contiguous>} : (!fir.box<!fir.array<?xf32>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
  %25 = hlfir.designate %9#0 (%c1:%21:%c1)  shape %24 : (!fir.box<!fir.array<?xf32>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
  hlfir.assign %20 to %25 : !fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>
  return
}

// -----

//subroutine test2(x, n, k1, k2)
//  real, contiguous :: x(:)
//  x(1:k1) = x(k1+1:k2)
//end subroutine test2

// CHECK-LABEL:   func.func @_QPtest2(
func.func @_QPtest2(%arg0: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x", fir.contiguous}, %arg1: !fir.ref<i32> {fir.bindc_name = "n"}, %arg2: !fir.ref<i32> {fir.bindc_name = "k1"}, %arg3: !fir.ref<i32> {fir.bindc_name = "k2"}) {
  %c1_i32 = arith.constant 1 : i32
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg2 dummy_scope %0 {uniq_name = "_QFtest2Ek1"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %2:2 = hlfir.declare %arg3 dummy_scope %0 {uniq_name = "_QFtest2Ek2"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %3:2 = hlfir.declare %arg1 dummy_scope %0 {uniq_name = "_QFtest2En"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %4 = fir.box_addr %arg0 : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
  %5:3 = fir.box_dims %arg0, %c0 : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
  %6 = fir.shape_shift %c1, %5#1 : (index, index) -> !fir.shapeshift<1>
  %7:2 = hlfir.declare %4(%6) dummy_scope %0 {fortran_attrs = #fir.var_attrs<contiguous>, uniq_name = "_QFtest2Ex"} : (!fir.ref<!fir.array<?xf32>>, !fir.shapeshift<1>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>)
  %8 = fir.load %1#0 : !fir.ref<i32>
  %9 = arith.addi %8, %c1_i32 overflow<nsw> : i32
  %10 = fir.load %2#0 : !fir.ref<i32>
  %11 = fir.convert %9 : (i32) -> index
  %12 = fir.convert %10 : (i32) -> index
  %13 = arith.subi %12, %11 : index
  %14 = arith.addi %13, %c1 : index
  %15 = arith.cmpi sgt, %14, %c0 : index
  %16 = arith.select %15, %14, %c0 : index
  %17 = fir.shape %16 : (index) -> !fir.shape<1>
// CHECK: hlfir.designate{{.*}}{fortran_attrs = #fir.var_attrs<contiguous>} : (!fir.box<!fir.array<?xf32>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
  %18 = hlfir.designate %7#0 (%11:%12:%c1)  shape %17 : (!fir.box<!fir.array<?xf32>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
  %19 = fir.convert %8 : (i32) -> index
  %20 = arith.cmpi sgt, %19, %c0 : index
  %21 = arith.select %20, %19, %c0 : index
  %22 = fir.shape %21 : (index) -> !fir.shape<1>
// CHECK: hlfir.designate{{.*}}{fortran_attrs = #fir.var_attrs<contiguous>} : (!fir.box<!fir.array<?xf32>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
  %23 = hlfir.designate %7#0 (%c1:%19:%c1)  shape %22 : (!fir.box<!fir.array<?xf32>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
  hlfir.assign %18 to %23 : !fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>
  return
}

// -----

//subroutine test3
//  real :: x(10:100)
//  call inner(x)
//contains
//  subroutine inner(x) ! manually inlined
//    real :: x(:)
//    x(10:20) = x(21:31)
//  end subroutine inner
//end subroutine test3

// CHECK-LABEL:   func.func @_QPtest3() {
func.func @_QPtest3() {
  %c91 = arith.constant 91 : index
  %c10 = arith.constant 10 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.alloca !fir.array<91xf32> {bindc_name = "x", uniq_name = "_QFtest3Ex"}
  %2 = fir.shape_shift %c10, %c91 : (index, index) -> !fir.shapeshift<1>
  %3:2 = hlfir.declare %1(%2) {uniq_name = "_QFtest3Ex"} : (!fir.ref<!fir.array<91xf32>>, !fir.shapeshift<1>) -> (!fir.box<!fir.array<91xf32>>, !fir.ref<!fir.array<91xf32>>)
  %4 = fir.convert %3#0 : (!fir.box<!fir.array<91xf32>>) -> !fir.box<!fir.array<?xf32>>
  // inner() code:
  %c20 = arith.constant 20 : index
  %c11 = arith.constant 11 : index
  %c1 = arith.constant 1 : index
  %c31 = arith.constant 31 : index
  %c21 = arith.constant 21 : index
  %100 = fir.dummy_scope : !fir.dscope
  %101:2 = hlfir.declare %4 dummy_scope %100 {uniq_name = "_QFtest3FinnerEx"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
  %102 = fir.shape %c11 : (index) -> !fir.shape<1>
// CHECK: hlfir.designate{{.*}}{fortran_attrs = #fir.var_attrs<contiguous>} : (!fir.box<!fir.array<?xf32>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<11xf32>>
  %103 = hlfir.designate %101#0 (%c21:%c31:%c1)  shape %102 : (!fir.box<!fir.array<?xf32>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<11xf32>>
// CHECK: hlfir.designate{{.*}}{fortran_attrs = #fir.var_attrs<contiguous>} : (!fir.box<!fir.array<?xf32>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<11xf32>>
  %104 = hlfir.designate %101#0 (%c10:%c20:%c1)  shape %102 : (!fir.box<!fir.array<?xf32>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<11xf32>>
  hlfir.assign %103 to %104 : !fir.box<!fir.array<11xf32>>, !fir.box<!fir.array<11xf32>>
  return
}

// -----

//subroutine test4
//  real :: x(100)
//  call inner(x)
//contains
//  subroutine inner(x) ! manually inlined
//    real :: x(:)
//    x(10:20) = x(21:31)
//  end subroutine inner
//end subroutine test4

// CHECK-LABEL:   func.func @_QPtest4() {
func.func @_QPtest4() {
  %c100 = arith.constant 100 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.alloca !fir.array<100xf32> {bindc_name = "x", uniq_name = "_QFtest4Ex"}
  %2 = fir.shape %c100 : (index) -> !fir.shape<1>
  %3:2 = hlfir.declare %1(%2) {uniq_name = "_QFtest4Ex"} : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100xf32>>)
  %4 = fir.embox %3#0(%2) : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<100xf32>>
  %5 = fir.convert %4 : (!fir.box<!fir.array<100xf32>>) -> !fir.box<!fir.array<?xf32>>
  // inner() code:
  %c20 = arith.constant 20 : index
  %c10 = arith.constant 10 : index
  %c11 = arith.constant 11 : index
  %c1 = arith.constant 1 : index
  %c31 = arith.constant 31 : index
  %c21 = arith.constant 21 : index
  %100 = fir.dummy_scope : !fir.dscope
// CHECK: hlfir.declare{{.*}}{fortran_attrs = #fir.var_attrs<contiguous>, uniq_name = "_QFtest4FinnerEx"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
  %101:2 = hlfir.declare %5 dummy_scope %100 {uniq_name = "_QFtest4FinnerEx"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
  %102 = fir.shape %c11 : (index) -> !fir.shape<1>
// CHECK: hlfir.designate{{.*}}{fortran_attrs = #fir.var_attrs<contiguous>} : (!fir.box<!fir.array<?xf32>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<11xf32>>
  %103 = hlfir.designate %101#0 (%c21:%c31:%c1)  shape %102 : (!fir.box<!fir.array<?xf32>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<11xf32>>
// CHECK: hlfir.designate{{.*}}{fortran_attrs = #fir.var_attrs<contiguous>} : (!fir.box<!fir.array<?xf32>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<11xf32>>
  %104 = hlfir.designate %101#0 (%c10:%c20:%c1)  shape %102 : (!fir.box<!fir.array<?xf32>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<11xf32>>
  hlfir.assign %103 to %104 : !fir.box<!fir.array<11xf32>>, !fir.box<!fir.array<11xf32>>
  return
}

// -----

//subroutine test5
//  real :: x(100)
//  call inner(x(1:50))
//contains
//  subroutine inner(x) ! manually inlined
//    real :: x(:)
//    x(10:20) = x(21:31)
//  end subroutine inner
//end subroutine test5

// CHECK-LABEL:   func.func @_QPtest5() {
func.func @_QPtest5() {
  %c50 = arith.constant 50 : index
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.alloca !fir.array<100xf32> {bindc_name = "x", uniq_name = "_QFtest5Ex"}
  %2 = fir.shape %c100 : (index) -> !fir.shape<1>
  %3:2 = hlfir.declare %1(%2) {uniq_name = "_QFtest5Ex"} : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100xf32>>)
  %4 = fir.shape %c50 : (index) -> !fir.shape<1>
  %5 = hlfir.designate %3#0 (%c1:%c50:%c1)  shape %4 : (!fir.ref<!fir.array<100xf32>>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<50xf32>>
  %6 = fir.embox %5(%4) : (!fir.ref<!fir.array<50xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<50xf32>>
  %7 = fir.convert %6 : (!fir.box<!fir.array<50xf32>>) -> !fir.box<!fir.array<?xf32>>
  // inner() code:
  %c20 = arith.constant 20 : index
  %c10 = arith.constant 10 : index
  %c11 = arith.constant 11 : index
  %c31 = arith.constant 31 : index
  %c21 = arith.constant 21 : index
  %100 = fir.dummy_scope : !fir.dscope
// CHECK: hlfir.declare{{.*}}{fortran_attrs = #fir.var_attrs<contiguous>, uniq_name = "_QFtest5FinnerEx"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
  %101:2 = hlfir.declare %7 dummy_scope %100 {uniq_name = "_QFtest5FinnerEx"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
  %102 = fir.shape %c11 : (index) -> !fir.shape<1>
// CHECK: hlfir.designate{{.*}}{fortran_attrs = #fir.var_attrs<contiguous>} : (!fir.box<!fir.array<?xf32>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<11xf32>>
  %103 = hlfir.designate %101#0 (%c21:%c31:%c1)  shape %102 : (!fir.box<!fir.array<?xf32>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<11xf32>>
// CHECK: hlfir.designate{{.*}}{fortran_attrs = #fir.var_attrs<contiguous>} : (!fir.box<!fir.array<?xf32>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<11xf32>>
  %104 = hlfir.designate %101#0 (%c10:%c20:%c1)  shape %102 : (!fir.box<!fir.array<?xf32>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<11xf32>>
  hlfir.assign %103 to %104 : !fir.box<!fir.array<11xf32>>, !fir.box<!fir.array<11xf32>>
  return
}

// -----

//subroutine test6(x,k)
//  real, contiguous :: x(:)
//  call inner(x(1:k))
//contains
//  subroutine inner(x) ! manually inlined
//    real :: x(:)
//    x(10:20) = x(21:31)
//  end subroutine inner
//end subroutine test6

// CHECK-LABEL:   func.func @_QPtest6(
func.func @_QPtest6(%arg0: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x", fir.contiguous}, %arg1: !fir.ref<i32> {fir.bindc_name = "k"}) {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg1 dummy_scope %0 {uniq_name = "_QFtest6Ek"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %2 = fir.box_addr %arg0 : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
  %3:3 = fir.box_dims %arg0, %c0 : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
  %4 = fir.shape_shift %c1, %3#1 : (index, index) -> !fir.shapeshift<1>
  %5:2 = hlfir.declare %2(%4) dummy_scope %0 {fortran_attrs = #fir.var_attrs<contiguous>, uniq_name = "_QFtest6Ex"} : (!fir.ref<!fir.array<?xf32>>, !fir.shapeshift<1>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>)
  %6 = fir.load %1#0 : !fir.ref<i32>
  %7 = fir.convert %6 : (i32) -> index
  %8 = arith.cmpi sgt, %7, %c0 : index
  %9 = arith.select %8, %7, %c0 : index
  %10 = fir.shape %9 : (index) -> !fir.shape<1>
  // hlfir.designate manually replaced with fir.rebox:
  %slice = fir.slice %c1, %7, %c1 : (index, index, index) -> !fir.slice<1>
  %11 = fir.rebox %5#0 [%slice] : (!fir.box<!fir.array<?xf32>>, !fir.slice<1>) -> !fir.box<!fir.array<?xf32>>
  // inner() code:
  %c20 = arith.constant 20 : index
  %c10 = arith.constant 10 : index
  %c11 = arith.constant 11 : index
  %c31 = arith.constant 31 : index
  %c21 = arith.constant 21 : index
  %100 = fir.dummy_scope : !fir.dscope
// CHECK: hlfir.declare{{.*}}{fortran_attrs = #fir.var_attrs<contiguous>, uniq_name = "_QFtest6FinnerEx"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
  %101:2 = hlfir.declare %11 dummy_scope %100 {uniq_name = "_QFtest6FinnerEx"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
  %102 = fir.shape %c11 : (index) -> !fir.shape<1>
// CHECK: hlfir.designate{{.*}}{fortran_attrs = #fir.var_attrs<contiguous>} : (!fir.box<!fir.array<?xf32>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<11xf32>>
  %103 = hlfir.designate %101#0 (%c21:%c31:%c1)  shape %102 : (!fir.box<!fir.array<?xf32>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<11xf32>>
// CHECK: hlfir.designate{{.*}}{fortran_attrs = #fir.var_attrs<contiguous>} : (!fir.box<!fir.array<?xf32>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<11xf32>>
  %104 = hlfir.designate %101#0 (%c10:%c20:%c1)  shape %102 : (!fir.box<!fir.array<?xf32>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<11xf32>>
  hlfir.assign %103 to %104 : !fir.box<!fir.array<11xf32>>, !fir.box<!fir.array<11xf32>>
  return
}

// -----

// Make sure contiguous attribute is not set on hlfir.declare:
//subroutine test7()
//  real, pointer :: x(:)
//end subroutine test7

func.func @_QPtest7() {
  %c0 = arith.constant 0 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.alloca !fir.box<!fir.ptr<!fir.array<?xf32>>> {bindc_name = "x", uniq_name = "_QFtest7Ex"}
  %2 = fir.zero_bits !fir.ptr<!fir.array<?xf32>>
  %3 = fir.shape %c0 : (index) -> !fir.shape<1>
  %4 = fir.embox %2(%3) : (!fir.ptr<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
  fir.store %4 to %1 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  %5:2 = hlfir.declare %1 {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtest7Ex"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>)
  return
}
// CHECK-LABEL:   func.func @_QPtest7(
// CHECK-NOT: contiguous
