// Test that the redundant fir.[un]pack_array operations
// are optimized away, when the source is statically known
// to be contiguous.
// RUN: fir-opt --optimize-array-repacking %s | FileCheck %s

// FIR is produced by compiling the sources with -mllvm -inline-all.
// module inner
// contains
//   subroutine inner_repack1(x)
//     real :: x(:)
//   end subroutine inner_repack1
//   subroutine inner_repack2(x)
//     real :: x(:,:)
//   end subroutine inner_repack2
// end module inner

// subroutine test_explicit_shape_cst(x)
//   real :: x(100)
//   call repack(x(1:50))
// contains
//   subroutine repack(x)
//     real :: x(:)
//   end subroutine repack
// end subroutine test_explicit_shape_cst
//
// CHECK-LABEL:   func.func @_QPtest_explicit_shape_cst(
// CHECK-NOT: fir.{{.*}}pack_array
func.func @_QPtest_explicit_shape_cst(%arg0: !fir.ref<!fir.array<100xf32>> {fir.bindc_name = "x"}) {
  %c50 = arith.constant 50 : index
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.shape %c100 : (index) -> !fir.shape<1>
  %2 = fir.declare %arg0(%1) dummy_scope %0 {uniq_name = "_QFtest_explicit_shape_cstEx"} : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<100xf32>>
  %3 = fir.shape %c50 : (index) -> !fir.shape<1>
  %4 = fir.array_coor %2(%1) %c1 : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
  %5 = fir.convert %4 : (!fir.ref<f32>) -> !fir.ref<!fir.array<50xf32>>
  %6 = fir.embox %5(%3) : (!fir.ref<!fir.array<50xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<50xf32>>
  %7 = fir.convert %6 : (!fir.box<!fir.array<50xf32>>) -> !fir.box<!fir.array<?xf32>>
  %8 = fir.dummy_scope : !fir.dscope
  %9 = fir.pack_array %7 heap whole : (!fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>>
  %10 = fir.declare %9 dummy_scope %8 {uniq_name = "_QFtest_explicit_shape_cstFrepackEx"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> !fir.box<!fir.array<?xf32>>
  fir.unpack_array %9 to %7 heap : !fir.box<!fir.array<?xf32>>
  return
}

// subroutine test_explicit_shape_var(x, n, l, u)
//   integer :: n, l, u
//   real :: x(n)
//   call repack(x(l:u))
// contains
//   subroutine repack(x)
//     real :: x(:)
//   end subroutine repack
// end subroutine test_explicit_shape_var
//
// CHECK-LABEL:   func.func @_QPtest_explicit_shape_var(
// CHECK-NOT: fir.{{.*}}pack_array
func.func @_QPtest_explicit_shape_var(%arg0: !fir.ref<!fir.array<?xf32>> {fir.bindc_name = "x"}, %arg1: !fir.ref<i32> {fir.bindc_name = "n"}, %arg2: !fir.ref<i32> {fir.bindc_name = "l"}, %arg3: !fir.ref<i32> {fir.bindc_name = "u"}) {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.declare %arg2 dummy_scope %0 {uniq_name = "_QFtest_explicit_shape_varEl"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %2 = fir.declare %arg1 dummy_scope %0 {uniq_name = "_QFtest_explicit_shape_varEn"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %3 = fir.declare %arg3 dummy_scope %0 {uniq_name = "_QFtest_explicit_shape_varEu"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %4 = fir.load %2 : !fir.ref<i32>
  %5 = fir.convert %4 : (i32) -> index
  %6 = arith.cmpi sgt, %5, %c0 : index
  %7 = arith.select %6, %5, %c0 : index
  %8 = fir.shape %7 : (index) -> !fir.shape<1>
  %9 = fir.declare %arg0(%8) dummy_scope %0 {uniq_name = "_QFtest_explicit_shape_varEx"} : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<?xf32>>
  %10 = fir.load %1 : !fir.ref<i32>
  %11 = fir.load %3 : !fir.ref<i32>
  %12 = fir.convert %10 : (i32) -> index
  %13 = fir.convert %11 : (i32) -> index
  %14 = fir.slice %12, %13, %c1 : (index, index, index) -> !fir.slice<1>
  %15 = fir.embox %9(%8) [%14] : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.box<!fir.array<?xf32>>
  %16 = fir.dummy_scope : !fir.dscope
  %17 = fir.pack_array %15 heap whole : (!fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>>
  %18 = fir.declare %17 dummy_scope %16 {uniq_name = "_QFtest_explicit_shape_varFrepackEx"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> !fir.box<!fir.array<?xf32>>
  fir.unpack_array %17 to %15 heap : !fir.box<!fir.array<?xf32>>
  return
}

// subroutine test_assumed_size_cst(x)
//   real :: x(*)
//   call repack(x(1:50))
// contains
//   subroutine repack(x)
//     real :: x(:)
//   end subroutine repack
// end subroutine test_assumed_size_cst
//
// CHECK-LABEL:   func.func @_QPtest_assumed_size_cst(
// CHECK-NOT: fir.{{.*}}pack_array
func.func @_QPtest_assumed_size_cst(%arg0: !fir.ref<!fir.array<?xf32>> {fir.bindc_name = "x"}) {
  %c50 = arith.constant 50 : index
  %c1 = arith.constant 1 : index
  %c-1 = arith.constant -1 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.shape %c-1 : (index) -> !fir.shape<1>
  %2 = fir.declare %arg0(%1) dummy_scope %0 {uniq_name = "_QFtest_assumed_size_cstEx"} : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<?xf32>>
  %3 = fir.shape %c50 : (index) -> !fir.shape<1>
  %4 = fir.array_coor %2(%1) %c1 : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
  %5 = fir.convert %4 : (!fir.ref<f32>) -> !fir.ref<!fir.array<50xf32>>
  %6 = fir.embox %5(%3) : (!fir.ref<!fir.array<50xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<50xf32>>
  %7 = fir.convert %6 : (!fir.box<!fir.array<50xf32>>) -> !fir.box<!fir.array<?xf32>>
  %8 = fir.dummy_scope : !fir.dscope
  %9 = fir.pack_array %7 heap whole : (!fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>>
  %10 = fir.declare %9 dummy_scope %8 {uniq_name = "_QFtest_assumed_size_cstFrepackEx"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> !fir.box<!fir.array<?xf32>>
  fir.unpack_array %9 to %7 heap : !fir.box<!fir.array<?xf32>>
  return
}

// subroutine test_assumed_size_var(x, l, u)
//   integer :: l, u
//   real :: x(*)
//   call repack(x(l:u))
// contains
//   subroutine repack(x)
//     real :: x(:)
//   end subroutine repack
// end subroutine test_assumed_size_var
//
// CHECK-LABEL:   func.func @_QPtest_assumed_size_var(
// CHECK-NOT: fir.{{.*}}pack_array
func.func @_QPtest_assumed_size_var(%arg0: !fir.ref<!fir.array<?xf32>> {fir.bindc_name = "x"}, %arg1: !fir.ref<i32> {fir.bindc_name = "l"}, %arg2: !fir.ref<i32> {fir.bindc_name = "u"}) {
  %c1 = arith.constant 1 : index
  %c-1 = arith.constant -1 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.declare %arg1 dummy_scope %0 {uniq_name = "_QFtest_assumed_size_varEl"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %2 = fir.declare %arg2 dummy_scope %0 {uniq_name = "_QFtest_assumed_size_varEu"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %3 = fir.shape %c-1 : (index) -> !fir.shape<1>
  %4 = fir.declare %arg0(%3) dummy_scope %0 {uniq_name = "_QFtest_assumed_size_varEx"} : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<?xf32>>
  %5 = fir.load %1 : !fir.ref<i32>
  %6 = fir.load %2 : !fir.ref<i32>
  %7 = fir.convert %5 : (i32) -> index
  %8 = fir.convert %6 : (i32) -> index
  %9 = fir.slice %7, %8, %c1 : (index, index, index) -> !fir.slice<1>
  %10 = fir.embox %4(%3) [%9] : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.box<!fir.array<?xf32>>
  %11 = fir.dummy_scope : !fir.dscope
  %12 = fir.pack_array %10 heap whole : (!fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>>
  %13 = fir.declare %12 dummy_scope %11 {uniq_name = "_QFtest_assumed_size_varFrepackEx"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> !fir.box<!fir.array<?xf32>>
  fir.unpack_array %12 to %10 heap : !fir.box<!fir.array<?xf32>>
  return
}

// subroutine test_allocatable_cst(x)
//   real, allocatable :: x(:)
//   call repack(x(10:50))
// contains
//   subroutine repack(x)
//     real :: x(:)
//   end subroutine repack
// end subroutine test_allocatable_cst
//
// CHECK-LABEL:   func.func @_QPtest_allocatable_cst(
// CHECK-NOT: fir.{{.*}}pack_array
func.func @_QPtest_allocatable_cst(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>> {fir.bindc_name = "x"}) {
  %c0 = arith.constant 0 : index
  %c41 = arith.constant 41 : index
  %c10 = arith.constant 10 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.declare %arg0 dummy_scope %0 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest_allocatable_cstEx"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.dscope) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  %2 = fir.load %1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  %3 = fir.shape %c41 : (index) -> !fir.shape<1>
  %4 = fir.box_addr %2 : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
  %5:3 = fir.box_dims %2, %c0 : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> (index, index, index)
  %6 = fir.shape_shift %5#0, %5#1 : (index, index) -> !fir.shapeshift<1>
  %7 = fir.array_coor %4(%6) %c10 : (!fir.heap<!fir.array<?xf32>>, !fir.shapeshift<1>, index) -> !fir.ref<f32>
  %8 = fir.convert %7 : (!fir.ref<f32>) -> !fir.ref<!fir.array<41xf32>>
  %9 = fir.embox %8(%3) : (!fir.ref<!fir.array<41xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<41xf32>>
  %10 = fir.convert %9 : (!fir.box<!fir.array<41xf32>>) -> !fir.box<!fir.array<?xf32>>
  %11 = fir.dummy_scope : !fir.dscope
  %12 = fir.pack_array %10 heap whole : (!fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>>
  %13 = fir.declare %12 dummy_scope %11 {uniq_name = "_QFtest_allocatable_cstFrepackEx"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> !fir.box<!fir.array<?xf32>>
  fir.unpack_array %12 to %10 heap : !fir.box<!fir.array<?xf32>>
  return
}

// subroutine test_allocatable_var(x, l, u)
//   integer :: l, u
//   real, allocatable :: x(:)
//   call repack(x(l:u))
// contains
//   subroutine repack(x)
//     real :: x(:)
//   end subroutine repack
// end subroutine test_allocatable_var
//
// CHECK-LABEL:   func.func @_QPtest_allocatable_var(
// CHECK-NOT: fir.{{.*}}pack_array
func.func @_QPtest_allocatable_var(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>> {fir.bindc_name = "x"}, %arg1: !fir.ref<i32> {fir.bindc_name = "l"}, %arg2: !fir.ref<i32> {fir.bindc_name = "u"}) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.declare %arg1 dummy_scope %0 {uniq_name = "_QFtest_allocatable_varEl"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %2 = fir.declare %arg2 dummy_scope %0 {uniq_name = "_QFtest_allocatable_varEu"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %3 = fir.declare %arg0 dummy_scope %0 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest_allocatable_varEx"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.dscope) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  %4 = fir.load %3 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  %5 = fir.load %1 : !fir.ref<i32>
  %6 = fir.load %2 : !fir.ref<i32>
  %7 = fir.convert %5 : (i32) -> index
  %8 = fir.convert %6 : (i32) -> index
  %9 = fir.box_addr %4 : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
  %10:3 = fir.box_dims %4, %c0 : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> (index, index, index)
  %11 = fir.shape_shift %10#0, %10#1 : (index, index) -> !fir.shapeshift<1>
  %12 = fir.slice %7, %8, %c1 : (index, index, index) -> !fir.slice<1>
  %13 = fir.embox %9(%11) [%12] : (!fir.heap<!fir.array<?xf32>>, !fir.shapeshift<1>, !fir.slice<1>) -> !fir.box<!fir.array<?xf32>>
  %14 = fir.dummy_scope : !fir.dscope
  %15 = fir.pack_array %13 heap whole : (!fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>>
  %16 = fir.declare %15 dummy_scope %14 {uniq_name = "_QFtest_allocatable_varFrepackEx"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> !fir.box<!fir.array<?xf32>>
  fir.unpack_array %15 to %13 heap : !fir.box<!fir.array<?xf32>>
  return
}

// subroutine test_allocatable_full(x)
//   real, allocatable :: x(:,:)
//   call repack(x(:,:))
// contains
//   subroutine repack(x)
//     real :: x(:,:)
//   end subroutine repack
// end subroutine test_allocatable_full
//
// CHECK-LABEL:   func.func @_QPtest_allocatable_full(
// CHECK-NOT: fir.{{.*}}pack_array
func.func @_QPtest_allocatable_full(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>> {fir.bindc_name = "x"}) {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.declare %arg0 dummy_scope %0 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest_allocatable_fullEx"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>, !fir.dscope) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>
  %2 = fir.load %1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>
  %3:3 = fir.box_dims %2, %c0 : (!fir.box<!fir.heap<!fir.array<?x?xf32>>>, index) -> (index, index, index)
  %4:3 = fir.box_dims %2, %c1 : (!fir.box<!fir.heap<!fir.array<?x?xf32>>>, index) -> (index, index, index)
  %5 = arith.addi %3#0, %3#1 : index
  %6 = arith.subi %5, %c1 : index
  %7 = arith.addi %4#0, %4#1 : index
  %8 = arith.subi %7, %c1 : index
  %9 = fir.box_addr %2 : (!fir.box<!fir.heap<!fir.array<?x?xf32>>>) -> !fir.heap<!fir.array<?x?xf32>>
  %10 = fir.shape_shift %3#0, %3#1, %4#0, %4#1 : (index, index, index, index) -> !fir.shapeshift<2>
  %11 = fir.slice %3#0, %6, %c1, %4#0, %8, %c1 : (index, index, index, index, index, index) -> !fir.slice<2>
  %12 = fir.embox %9(%10) [%11] : (!fir.heap<!fir.array<?x?xf32>>, !fir.shapeshift<2>, !fir.slice<2>) -> !fir.box<!fir.array<?x?xf32>>
  %13 = fir.dummy_scope : !fir.dscope
  %14 = fir.pack_array %12 heap innermost : (!fir.box<!fir.array<?x?xf32>>) -> !fir.box<!fir.array<?x?xf32>>
  %15 = fir.declare %14 dummy_scope %13 {uniq_name = "_QFtest_allocatable_fullFrepackEx"} : (!fir.box<!fir.array<?x?xf32>>, !fir.dscope) -> !fir.box<!fir.array<?x?xf32>>
  fir.unpack_array %14 to %12 heap : !fir.box<!fir.array<?x?xf32>>
  return
}

// subroutine test_explicit_shape_cst_chain(x)
//   real :: x(100)
//   call repack(x(1:50))
// contains
//   subroutine repack(x)
//     use inner
//     real :: x(:)
//     call inner_repack1(x)
//   end subroutine repack
// end subroutine test_explicit_shape_cst_chain
//
// CHECK-LABEL:   func.func @_QPtest_explicit_shape_cst_chain(
// CHECK-NOT: fir.{{.*}}pack_array
func.func @_QPtest_explicit_shape_cst_chain(%arg0: !fir.ref<!fir.array<100xf32>> {fir.bindc_name = "x"}) {
  %c50 = arith.constant 50 : index
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.shape %c100 : (index) -> !fir.shape<1>
  %2 = fir.declare %arg0(%1) dummy_scope %0 {uniq_name = "_QFtest_explicit_shape_cst_chainEx"} : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<100xf32>>
  %3 = fir.shape %c50 : (index) -> !fir.shape<1>
  %4 = fir.array_coor %2(%1) %c1 : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
  %5 = fir.convert %4 : (!fir.ref<f32>) -> !fir.ref<!fir.array<50xf32>>
  %6 = fir.embox %5(%3) : (!fir.ref<!fir.array<50xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<50xf32>>
  %7 = fir.convert %6 : (!fir.box<!fir.array<50xf32>>) -> !fir.box<!fir.array<?xf32>>
  %8 = fir.dummy_scope : !fir.dscope
  %9 = fir.pack_array %7 heap whole : (!fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>>
  %10 = fir.declare %9 dummy_scope %8 {uniq_name = "_QFtest_explicit_shape_cst_chainFrepackEx"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> !fir.box<!fir.array<?xf32>>
  %11 = fir.rebox %10 : (!fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>>
  %12 = fir.dummy_scope : !fir.dscope
  %13 = fir.pack_array %11 heap whole : (!fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>>
  %14 = fir.declare %13 dummy_scope %12 {uniq_name = "_QMinnerFinner_repack1Ex"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> !fir.box<!fir.array<?xf32>>
  fir.unpack_array %13 to %11 heap : !fir.box<!fir.array<?xf32>>
  fir.unpack_array %9 to %7 heap : !fir.box<!fir.array<?xf32>>
  return
}

// subroutine test_explicit_shape_var_chain(x, n, l, u)
//   integer :: n, l, u
//   real :: x(n)
//   call repack(x(l:u))
// contains
//   subroutine repack(x)
//     use inner
//     real :: x(:)
//     call inner_repack1(x)
//   end subroutine repack
// end subroutine test_explicit_shape_var_chain
//
// CHECK-LABEL:   func.func @_QPtest_explicit_shape_var_chain(
// CHECK-NOT: fir.{{.*}}pack_array
func.func @_QPtest_explicit_shape_var_chain(%arg0: !fir.ref<!fir.array<?xf32>> {fir.bindc_name = "x"}, %arg1: !fir.ref<i32> {fir.bindc_name = "n"}, %arg2: !fir.ref<i32> {fir.bindc_name = "l"}, %arg3: !fir.ref<i32> {fir.bindc_name = "u"}) {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.declare %arg2 dummy_scope %0 {uniq_name = "_QFtest_explicit_shape_var_chainEl"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %2 = fir.declare %arg1 dummy_scope %0 {uniq_name = "_QFtest_explicit_shape_var_chainEn"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %3 = fir.declare %arg3 dummy_scope %0 {uniq_name = "_QFtest_explicit_shape_var_chainEu"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %4 = fir.load %2 : !fir.ref<i32>
  %5 = fir.convert %4 : (i32) -> index
  %6 = arith.cmpi sgt, %5, %c0 : index
  %7 = arith.select %6, %5, %c0 : index
  %8 = fir.shape %7 : (index) -> !fir.shape<1>
  %9 = fir.declare %arg0(%8) dummy_scope %0 {uniq_name = "_QFtest_explicit_shape_var_chainEx"} : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<?xf32>>
  %10 = fir.load %1 : !fir.ref<i32>
  %11 = fir.load %3 : !fir.ref<i32>
  %12 = fir.convert %10 : (i32) -> index
  %13 = fir.convert %11 : (i32) -> index
  %14 = fir.slice %12, %13, %c1 : (index, index, index) -> !fir.slice<1>
  %15 = fir.embox %9(%8) [%14] : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.box<!fir.array<?xf32>>
  %16 = fir.dummy_scope : !fir.dscope
  %17 = fir.pack_array %15 heap whole : (!fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>>
  %18 = fir.declare %17 dummy_scope %16 {uniq_name = "_QFtest_explicit_shape_var_chainFrepackEx"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> !fir.box<!fir.array<?xf32>>
  %19 = fir.rebox %18 : (!fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>>
  %20 = fir.dummy_scope : !fir.dscope
  %21 = fir.pack_array %19 heap whole : (!fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>>
  %22 = fir.declare %21 dummy_scope %20 {uniq_name = "_QMinnerFinner_repack1Ex"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> !fir.box<!fir.array<?xf32>>
  fir.unpack_array %21 to %19 heap : !fir.box<!fir.array<?xf32>>
  fir.unpack_array %17 to %15 heap : !fir.box<!fir.array<?xf32>>
  return
}

// subroutine test_assumed_size_cst_chain(x)
//   real :: x(*)
//   call repack(x(1:50))
// contains
//   subroutine repack(x)
//     use inner
//     real :: x(:)
//     call inner_repack1(x)
//   end subroutine repack
// end subroutine test_assumed_size_cst_chain
//
// CHECK-LABEL:   func.func @_QPtest_assumed_size_cst_chain(
// CHECK-NOT: fir.{{.*}}pack_array
func.func @_QPtest_assumed_size_cst_chain(%arg0: !fir.ref<!fir.array<?xf32>> {fir.bindc_name = "x"}) {
  %c50 = arith.constant 50 : index
  %c1 = arith.constant 1 : index
  %c-1 = arith.constant -1 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.shape %c-1 : (index) -> !fir.shape<1>
  %2 = fir.declare %arg0(%1) dummy_scope %0 {uniq_name = "_QFtest_assumed_size_cst_chainEx"} : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<?xf32>>
  %3 = fir.shape %c50 : (index) -> !fir.shape<1>
  %4 = fir.array_coor %2(%1) %c1 : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
  %5 = fir.convert %4 : (!fir.ref<f32>) -> !fir.ref<!fir.array<50xf32>>
  %6 = fir.embox %5(%3) : (!fir.ref<!fir.array<50xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<50xf32>>
  %7 = fir.convert %6 : (!fir.box<!fir.array<50xf32>>) -> !fir.box<!fir.array<?xf32>>
  %8 = fir.dummy_scope : !fir.dscope
  %9 = fir.pack_array %7 heap whole : (!fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>>
  %10 = fir.declare %9 dummy_scope %8 {uniq_name = "_QFtest_assumed_size_cst_chainFrepackEx"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> !fir.box<!fir.array<?xf32>>
  %11 = fir.rebox %10 : (!fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>>
  %12 = fir.dummy_scope : !fir.dscope
  %13 = fir.pack_array %11 heap whole : (!fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>>
  %14 = fir.declare %13 dummy_scope %12 {uniq_name = "_QMinnerFinner_repack1Ex"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> !fir.box<!fir.array<?xf32>>
  fir.unpack_array %13 to %11 heap : !fir.box<!fir.array<?xf32>>
  fir.unpack_array %9 to %7 heap : !fir.box<!fir.array<?xf32>>
  return
}

// subroutine test_assumed_size_var_chain(x, l, u)
//   integer :: l, u
//   real :: x(*)
//   call repack(x(l:u))
// contains
//   subroutine repack(x)
//     use inner
//     real :: x(:)
//     call inner_repack1(x)
//   end subroutine repack
// end subroutine test_assumed_size_var_chain
//
// CHECK-LABEL:   func.func @_QPtest_assumed_size_var_chain(
// CHECK-NOT: fir.{{.*}}pack_array
func.func @_QPtest_assumed_size_var_chain(%arg0: !fir.ref<!fir.array<?xf32>> {fir.bindc_name = "x"}, %arg1: !fir.ref<i32> {fir.bindc_name = "l"}, %arg2: !fir.ref<i32> {fir.bindc_name = "u"}) {
  %c1 = arith.constant 1 : index
  %c-1 = arith.constant -1 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.declare %arg1 dummy_scope %0 {uniq_name = "_QFtest_assumed_size_var_chainEl"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %2 = fir.declare %arg2 dummy_scope %0 {uniq_name = "_QFtest_assumed_size_var_chainEu"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %3 = fir.shape %c-1 : (index) -> !fir.shape<1>
  %4 = fir.declare %arg0(%3) dummy_scope %0 {uniq_name = "_QFtest_assumed_size_var_chainEx"} : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<?xf32>>
  %5 = fir.load %1 : !fir.ref<i32>
  %6 = fir.load %2 : !fir.ref<i32>
  %7 = fir.convert %5 : (i32) -> index
  %8 = fir.convert %6 : (i32) -> index
  %9 = fir.slice %7, %8, %c1 : (index, index, index) -> !fir.slice<1>
  %10 = fir.embox %4(%3) [%9] : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.box<!fir.array<?xf32>>
  %11 = fir.dummy_scope : !fir.dscope
  %12 = fir.pack_array %10 heap whole : (!fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>>
  %13 = fir.declare %12 dummy_scope %11 {uniq_name = "_QFtest_assumed_size_var_chainFrepackEx"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> !fir.box<!fir.array<?xf32>>
  %14 = fir.rebox %13 : (!fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>>
  %15 = fir.dummy_scope : !fir.dscope
  %16 = fir.pack_array %14 heap whole : (!fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>>
  %17 = fir.declare %16 dummy_scope %15 {uniq_name = "_QMinnerFinner_repack1Ex"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> !fir.box<!fir.array<?xf32>>
  fir.unpack_array %16 to %14 heap : !fir.box<!fir.array<?xf32>>
  fir.unpack_array %12 to %10 heap : !fir.box<!fir.array<?xf32>>
  return
}

// subroutine test_allocatable_cst_chain(x)
//   real, allocatable :: x(:)
//   call repack(x(10:50))
// contains
//   subroutine repack(x)
//     use inner
//     real :: x(:)
//     call inner_repack1(x)
//   end subroutine repack
// end subroutine test_allocatable_cst_chain
//
// CHECK-LABEL:   func.func @_QPtest_allocatable_cst_chain(
// CHECK-NOT: fir.{{.*}}pack_array
func.func @_QPtest_allocatable_cst_chain(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>> {fir.bindc_name = "x"}) {
  %c0 = arith.constant 0 : index
  %c41 = arith.constant 41 : index
  %c10 = arith.constant 10 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.declare %arg0 dummy_scope %0 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest_allocatable_cst_chainEx"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.dscope) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  %2 = fir.load %1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  %3 = fir.shape %c41 : (index) -> !fir.shape<1>
  %4 = fir.box_addr %2 : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
  %5:3 = fir.box_dims %2, %c0 : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> (index, index, index)
  %6 = fir.shape_shift %5#0, %5#1 : (index, index) -> !fir.shapeshift<1>
  %7 = fir.array_coor %4(%6) %c10 : (!fir.heap<!fir.array<?xf32>>, !fir.shapeshift<1>, index) -> !fir.ref<f32>
  %8 = fir.convert %7 : (!fir.ref<f32>) -> !fir.ref<!fir.array<41xf32>>
  %9 = fir.embox %8(%3) : (!fir.ref<!fir.array<41xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<41xf32>>
  %10 = fir.convert %9 : (!fir.box<!fir.array<41xf32>>) -> !fir.box<!fir.array<?xf32>>
  %11 = fir.dummy_scope : !fir.dscope
  %12 = fir.pack_array %10 heap whole : (!fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>>
  %13 = fir.declare %12 dummy_scope %11 {uniq_name = "_QFtest_allocatable_cst_chainFrepackEx"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> !fir.box<!fir.array<?xf32>>
  %14 = fir.rebox %13 : (!fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>>
  %15 = fir.dummy_scope : !fir.dscope
  %16 = fir.pack_array %14 heap whole : (!fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>>
  %17 = fir.declare %16 dummy_scope %15 {uniq_name = "_QMinnerFinner_repack1Ex"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> !fir.box<!fir.array<?xf32>>
  fir.unpack_array %16 to %14 heap : !fir.box<!fir.array<?xf32>>
  fir.unpack_array %12 to %10 heap : !fir.box<!fir.array<?xf32>>
  return
}

// subroutine test_allocatable_var_chain(x, l, u)
//   integer :: l, u
//   real, allocatable :: x(:)
//   call repack(x(l:u))
// contains
//   subroutine repack(x)
//     use inner
//     real :: x(:)
//     call inner_repack1(x)
//   end subroutine repack
// end subroutine test_allocatable_var_chain
//
// CHECK-LABEL:   func.func @_QPtest_allocatable_var_chain(
// CHECK-NOT: fir.{{.*}}pack_array
func.func @_QPtest_allocatable_var_chain(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>> {fir.bindc_name = "x"}, %arg1: !fir.ref<i32> {fir.bindc_name = "l"}, %arg2: !fir.ref<i32> {fir.bindc_name = "u"}) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.declare %arg1 dummy_scope %0 {uniq_name = "_QFtest_allocatable_var_chainEl"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %2 = fir.declare %arg2 dummy_scope %0 {uniq_name = "_QFtest_allocatable_var_chainEu"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %3 = fir.declare %arg0 dummy_scope %0 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest_allocatable_var_chainEx"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.dscope) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  %4 = fir.load %3 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  %5 = fir.load %1 : !fir.ref<i32>
  %6 = fir.load %2 : !fir.ref<i32>
  %7 = fir.convert %5 : (i32) -> index
  %8 = fir.convert %6 : (i32) -> index
  %9 = fir.box_addr %4 : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
  %10:3 = fir.box_dims %4, %c0 : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> (index, index, index)
  %11 = fir.shape_shift %10#0, %10#1 : (index, index) -> !fir.shapeshift<1>
  %12 = fir.slice %7, %8, %c1 : (index, index, index) -> !fir.slice<1>
  %13 = fir.embox %9(%11) [%12] : (!fir.heap<!fir.array<?xf32>>, !fir.shapeshift<1>, !fir.slice<1>) -> !fir.box<!fir.array<?xf32>>
  %14 = fir.dummy_scope : !fir.dscope
  %15 = fir.pack_array %13 heap whole : (!fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>>
  %16 = fir.declare %15 dummy_scope %14 {uniq_name = "_QFtest_allocatable_var_chainFrepackEx"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> !fir.box<!fir.array<?xf32>>
  %17 = fir.rebox %16 : (!fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>>
  %18 = fir.dummy_scope : !fir.dscope
  %19 = fir.pack_array %17 heap whole : (!fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>>
  %20 = fir.declare %19 dummy_scope %18 {uniq_name = "_QMinnerFinner_repack1Ex"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> !fir.box<!fir.array<?xf32>>
  fir.unpack_array %19 to %17 heap : !fir.box<!fir.array<?xf32>>
  fir.unpack_array %15 to %13 heap : !fir.box<!fir.array<?xf32>>
  return
}

// subroutine test_allocatable_full_chain(x)
//   real, allocatable :: x(:,:)
//   call repack(x(:,:))
// contains
//   subroutine repack(x)
//     use inner
//     real :: x(:,:)
//     call inner_repack2(x)
//   end subroutine repack
// end subroutine test_allocatable_full_chain
//
// CHECK-LABEL:   func.func @_QPtest_allocatable_full_chain(
// CHECK-NOT: fir.{{.*}}pack_array
func.func @_QPtest_allocatable_full_chain(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>> {fir.bindc_name = "x"}) {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.declare %arg0 dummy_scope %0 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest_allocatable_full_chainEx"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>, !fir.dscope) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>
  %2 = fir.load %1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>
  %3:3 = fir.box_dims %2, %c0 : (!fir.box<!fir.heap<!fir.array<?x?xf32>>>, index) -> (index, index, index)
  %4:3 = fir.box_dims %2, %c1 : (!fir.box<!fir.heap<!fir.array<?x?xf32>>>, index) -> (index, index, index)
  %5 = arith.addi %3#0, %3#1 : index
  %6 = arith.subi %5, %c1 : index
  %7 = arith.addi %4#0, %4#1 : index
  %8 = arith.subi %7, %c1 : index
  %9 = fir.box_addr %2 : (!fir.box<!fir.heap<!fir.array<?x?xf32>>>) -> !fir.heap<!fir.array<?x?xf32>>
  %10 = fir.shape_shift %3#0, %3#1, %4#0, %4#1 : (index, index, index, index) -> !fir.shapeshift<2>
  %11 = fir.slice %3#0, %6, %c1, %4#0, %8, %c1 : (index, index, index, index, index, index) -> !fir.slice<2>
  %12 = fir.embox %9(%10) [%11] : (!fir.heap<!fir.array<?x?xf32>>, !fir.shapeshift<2>, !fir.slice<2>) -> !fir.box<!fir.array<?x?xf32>>
  %13 = fir.dummy_scope : !fir.dscope
  %14 = fir.pack_array %12 heap innermost : (!fir.box<!fir.array<?x?xf32>>) -> !fir.box<!fir.array<?x?xf32>>
  %15 = fir.declare %14 dummy_scope %13 {uniq_name = "_QFtest_allocatable_full_chainFrepackEx"} : (!fir.box<!fir.array<?x?xf32>>, !fir.dscope) -> !fir.box<!fir.array<?x?xf32>>
  %16 = fir.rebox %15 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.box<!fir.array<?x?xf32>>
  %17 = fir.dummy_scope : !fir.dscope
  %18 = fir.pack_array %16 heap innermost : (!fir.box<!fir.array<?x?xf32>>) -> !fir.box<!fir.array<?x?xf32>>
  %19 = fir.declare %18 dummy_scope %17 {uniq_name = "_QMinnerFinner_repack2Ex"} : (!fir.box<!fir.array<?x?xf32>>, !fir.dscope) -> !fir.box<!fir.array<?x?xf32>>
  fir.unpack_array %18 to %16 heap : !fir.box<!fir.array<?x?xf32>>
  fir.unpack_array %14 to %12 heap : !fir.box<!fir.array<?x?xf32>>
  return
}

// TODO: if both fir.pack_array have the same property,
// then the second one is redundant, because the first
// repack makes 'x' contiguous.
// subroutine neg_test_assumed_shape(x)
//   real :: x(:)
//   call repack(x(1:50))
// contains
//   subroutine repack(x)
//     real :: x(:)
//   end subroutine repack
// end subroutine neg_test_assumed_shape
//
// CHECK-LABEL:   func.func @_QPneg_test_assumed_shape(
// CHECK: fir.pack_array
// CHECK: fir.pack_array
// CHECK: fir.unpack_array
// CHECK: fir.unpack_array
func.func @_QPneg_test_assumed_shape(%arg0: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x"}) {
  %c50 = arith.constant 50 : index
  %c1 = arith.constant 1 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.pack_array %arg0 heap whole : (!fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>>
  %2 = fir.declare %1 dummy_scope %0 {uniq_name = "_QFneg_test_assumed_shapeEx"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> !fir.box<!fir.array<?xf32>>
  %3 = fir.rebox %2 : (!fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>>
  %4 = fir.slice %c1, %c50, %c1 : (index, index, index) -> !fir.slice<1>
  %5 = fir.rebox %3 [%4] : (!fir.box<!fir.array<?xf32>>, !fir.slice<1>) -> !fir.box<!fir.array<50xf32>>
  %6 = fir.convert %5 : (!fir.box<!fir.array<50xf32>>) -> !fir.box<!fir.array<?xf32>>
  %7 = fir.dummy_scope : !fir.dscope
  %8 = fir.pack_array %6 heap whole : (!fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>>
  %9 = fir.declare %8 dummy_scope %7 {uniq_name = "_QFneg_test_assumed_shapeFrepackEx"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> !fir.box<!fir.array<?xf32>>
  fir.unpack_array %8 to %6 heap : !fir.box<!fir.array<?xf32>>
  fir.unpack_array %1 to %arg0 heap : !fir.box<!fir.array<?xf32>>
  return
}

// subroutine neg_test_non_contig_slice_cst(x)
//   real :: x(100)
//   call repack(x(1:50:2))
// contains
//   subroutine repack(x)
//     real :: x(:)
//   end subroutine repack
// end subroutine neg_test_non_contig_slice_cst
//
// CHECK-LABEL:   func.func @_QPneg_test_non_contig_slice_cst(
// CHECK: fir.pack_array
// CHECK: fir.unpack_array
func.func @_QPneg_test_non_contig_slice_cst(%arg0: !fir.ref<!fir.array<100xf32>> {fir.bindc_name = "x"}) {
  %c2 = arith.constant 2 : index
  %c50 = arith.constant 50 : index
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.shape %c100 : (index) -> !fir.shape<1>
  %2 = fir.declare %arg0(%1) dummy_scope %0 {uniq_name = "_QFneg_test_non_contig_slice_cstEx"} : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<100xf32>>
  %3 = fir.slice %c1, %c50, %c2 : (index, index, index) -> !fir.slice<1>
  %4 = fir.embox %2(%1) [%3] : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.box<!fir.array<25xf32>>
  %5 = fir.convert %4 : (!fir.box<!fir.array<25xf32>>) -> !fir.box<!fir.array<?xf32>>
  %6 = fir.dummy_scope : !fir.dscope
  %7 = fir.pack_array %5 heap whole : (!fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>>
  %8 = fir.declare %7 dummy_scope %6 {uniq_name = "_QFneg_test_non_contig_slice_cstFrepackEx"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> !fir.box<!fir.array<?xf32>>
  fir.unpack_array %7 to %5 heap : !fir.box<!fir.array<?xf32>>
  return
}

// subroutine neg_test_non_contig_slice_var(x, s)
//   integer :: s
//   real :: x(100)
//   call repack(x(1:50:s))
// contains
//   subroutine repack(x)
//     real :: x(:)
//   end subroutine repack
// end subroutine neg_test_non_contig_slice_var
//
// CHECK-LABEL:   func.func @_QPneg_test_non_contig_slice_var(
// CHECK: fir.pack_array
// CHECK: fir.unpack_array
func.func @_QPneg_test_non_contig_slice_var(%arg0: !fir.ref<!fir.array<100xf32>> {fir.bindc_name = "x"}, %arg1: !fir.ref<i32> {fir.bindc_name = "s"}) {
  %c50 = arith.constant 50 : index
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.declare %arg1 dummy_scope %0 {uniq_name = "_QFneg_test_non_contig_slice_varEs"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %2 = fir.shape %c100 : (index) -> !fir.shape<1>
  %3 = fir.declare %arg0(%2) dummy_scope %0 {uniq_name = "_QFneg_test_non_contig_slice_varEx"} : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<100xf32>>
  %4 = fir.load %1 : !fir.ref<i32>
  %5 = fir.convert %4 : (i32) -> index
  %6 = fir.slice %c1, %c50, %5 : (index, index, index) -> !fir.slice<1>
  %7 = fir.embox %3(%2) [%6] : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.box<!fir.array<?xf32>>
  %8 = fir.dummy_scope : !fir.dscope
  %9 = fir.pack_array %7 heap whole : (!fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>>
  %10 = fir.declare %9 dummy_scope %8 {uniq_name = "_QFneg_test_non_contig_slice_varFrepackEx"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> !fir.box<!fir.array<?xf32>>
  fir.unpack_array %9 to %7 heap : !fir.box<!fir.array<?xf32>>
  return
}

// subroutine neg_test_pointer(x)
//   real, pointer :: x(:)
//   call repack(x(1:50))
// contains
//   subroutine repack(x)
//     real :: x(:)
//   end subroutine repack
// end subroutine neg_test_pointer
//
// CHECK-LABEL:   func.func @_QPneg_test_pointer(
// CHECK: fir.pack_array
// CHECK: fir.unpack_array
func.func @_QPneg_test_pointer(%arg0: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>> {fir.bindc_name = "x"}) {
  %c0 = arith.constant 0 : index
  %c50 = arith.constant 50 : index
  %c1 = arith.constant 1 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.declare %arg0 dummy_scope %0 {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFneg_test_pointerEx"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>, !fir.dscope) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  %2 = fir.load %1 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  %3:3 = fir.box_dims %2, %c0 : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, index) -> (index, index, index)
  %4 = fir.shift %3#0 : (index) -> !fir.shift<1>
  %5 = fir.slice %c1, %c50, %c1 : (index, index, index) -> !fir.slice<1>
  %6 = fir.rebox %2(%4) [%5] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, !fir.shift<1>, !fir.slice<1>) -> !fir.box<!fir.array<50xf32>>
  %7 = fir.convert %6 : (!fir.box<!fir.array<50xf32>>) -> !fir.box<!fir.array<?xf32>>
  %8 = fir.dummy_scope : !fir.dscope
  %9 = fir.pack_array %7 heap whole : (!fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>>
  %10 = fir.declare %9 dummy_scope %8 {uniq_name = "_QFneg_test_pointerFrepackEx"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> !fir.box<!fir.array<?xf32>>
  fir.unpack_array %9 to %7 heap : !fir.box<!fir.array<?xf32>>
  return
}

// Test a long chain of fir.pack_array operations.
// The rewriter used to use a down-top traversal that optimized
// fir.pack_array operations starting from the innermost one.
// The rewriter did not converge in 10 (default) iterations
// causing the pass to report a failure.
// A top-down traversal should fix this an allow optimizing
// all the repackings.
// CHECK-LABEL:   func.func @test_long_chain(
// CHECK-NOT: fir.pack_array
// CHECK-NOT: fir.unpack_array
func.func @test_long_chain(%pred: i1) {
  %c10 = arith.constant 10 : index
  %3 = fir.dummy_scope : !fir.dscope
  %4 = fir.address_of(@aaa) : !fir.ref<!fir.array<10x10xi32>>
  %5 = fir.shape %c10, %c10 : (index, index) -> !fir.shape<2>
  %6 = fir.declare %4(%5) {uniq_name = "aaa"} : (!fir.ref<!fir.array<10x10xi32>>, !fir.shape<2>) -> !fir.ref<!fir.array<10x10xi32>>
  %9 = fir.embox %6(%5) : (!fir.ref<!fir.array<10x10xi32>>, !fir.shape<2>) -> !fir.box<!fir.array<10x10xi32>>
  %10 = fir.convert %9 : (!fir.box<!fir.array<10x10xi32>>) -> !fir.box<!fir.array<?x?xi32>>
  %11 = fir.dummy_scope : !fir.dscope
  %12 = fir.pack_array %10 heap innermost : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<!fir.array<?x?xi32>>
  %13 = fir.declare %12 dummy_scope %11 {uniq_name = "aaa"} : (!fir.box<!fir.array<?x?xi32>>, !fir.dscope) -> !fir.box<!fir.array<?x?xi32>>
  %14 = fir.rebox %13 : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<!fir.array<?x?xi32>>
  cf.cond_br %pred, ^bb17, ^bb1
^bb1:  // pred: ^bb0
  %20 = fir.dummy_scope : !fir.dscope
  %21 = fir.pack_array %14 heap innermost : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<!fir.array<?x?xi32>>
  %22 = fir.declare %21 dummy_scope %20 {uniq_name = "aaa"} : (!fir.box<!fir.array<?x?xi32>>, !fir.dscope) -> !fir.box<!fir.array<?x?xi32>>
  %23 = fir.rebox %22 : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<!fir.array<?x?xi32>>
  %28 = fir.dummy_scope : !fir.dscope
  %29 = fir.pack_array %23 heap innermost : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<!fir.array<?x?xi32>>
  %30 = fir.declare %29 dummy_scope %28 {uniq_name = "aaa"} : (!fir.box<!fir.array<?x?xi32>>, !fir.dscope) -> !fir.box<!fir.array<?x?xi32>>
  %31 = fir.rebox %30 : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<!fir.array<?x?xi32>>
  cf.cond_br %pred, ^bb16, ^bb2
^bb2:  // pred: ^bb1
  %37 = fir.dummy_scope : !fir.dscope
  %38 = fir.pack_array %31 heap innermost : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<!fir.array<?x?xi32>>
  %39 = fir.declare %38 dummy_scope %37 {uniq_name = "aaa"} : (!fir.box<!fir.array<?x?xi32>>, !fir.dscope) -> !fir.box<!fir.array<?x?xi32>>
  %40 = fir.rebox %39 : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<!fir.array<?x?xi32>>
  %45 = fir.dummy_scope : !fir.dscope
  %46 = fir.pack_array %40 heap innermost : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<!fir.array<?x?xi32>>
  %47 = fir.declare %46 dummy_scope %45 {uniq_name = "aaa"} : (!fir.box<!fir.array<?x?xi32>>, !fir.dscope) -> !fir.box<!fir.array<?x?xi32>>
  %48 = fir.rebox %47 : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<!fir.array<?x?xi32>>
  cf.cond_br %pred, ^bb15, ^bb3
^bb3:  // pred: ^bb2
  %54 = fir.dummy_scope : !fir.dscope
  %55 = fir.pack_array %48 heap innermost : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<!fir.array<?x?xi32>>
  %56 = fir.declare %55 dummy_scope %54 {uniq_name = "aaa"} : (!fir.box<!fir.array<?x?xi32>>, !fir.dscope) -> !fir.box<!fir.array<?x?xi32>>
  %57 = fir.rebox %56 : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<!fir.array<?x?xi32>>
  %62 = fir.dummy_scope : !fir.dscope
  %63 = fir.pack_array %57 heap innermost : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<!fir.array<?x?xi32>>
  %64 = fir.declare %63 dummy_scope %62 {uniq_name = "aaa"} : (!fir.box<!fir.array<?x?xi32>>, !fir.dscope) -> !fir.box<!fir.array<?x?xi32>>
  %65 = fir.rebox %64 : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<!fir.array<?x?xi32>>
  cf.cond_br %pred, ^bb14, ^bb4
^bb4:  // pred: ^bb3
  %71 = fir.dummy_scope : !fir.dscope
  %72 = fir.pack_array %65 heap innermost : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<!fir.array<?x?xi32>>
  %73 = fir.declare %72 dummy_scope %71 {uniq_name = "aaa"} : (!fir.box<!fir.array<?x?xi32>>, !fir.dscope) -> !fir.box<!fir.array<?x?xi32>>
  %74 = fir.rebox %73 : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<!fir.array<?x?xi32>>
  %79 = fir.dummy_scope : !fir.dscope
  %80 = fir.pack_array %74 heap innermost : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<!fir.array<?x?xi32>>
  %81 = fir.declare %80 dummy_scope %79 {uniq_name = "aaa"} : (!fir.box<!fir.array<?x?xi32>>, !fir.dscope) -> !fir.box<!fir.array<?x?xi32>>
  %82 = fir.rebox %81 : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<!fir.array<?x?xi32>>
  cf.cond_br %pred, ^bb13, ^bb5
^bb5:  // pred: ^bb4
  %88 = fir.dummy_scope : !fir.dscope
  %89 = fir.pack_array %82 heap innermost : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<!fir.array<?x?xi32>>
  %90 = fir.declare %89 dummy_scope %88 {uniq_name = "aaa"} : (!fir.box<!fir.array<?x?xi32>>, !fir.dscope) -> !fir.box<!fir.array<?x?xi32>>
  %91 = fir.rebox %90 : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<!fir.array<?x?xi32>>
  %96 = fir.dummy_scope : !fir.dscope
  %97 = fir.pack_array %91 heap innermost : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<!fir.array<?x?xi32>>
  %98 = fir.declare %97 dummy_scope %96 {uniq_name = "aaa"} : (!fir.box<!fir.array<?x?xi32>>, !fir.dscope) -> !fir.box<!fir.array<?x?xi32>>
  %99 = fir.rebox %98 : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<!fir.array<?x?xi32>>
  cf.cond_br %pred, ^bb12, ^bb6
^bb6:  // pred: ^bb5
  %105 = fir.dummy_scope : !fir.dscope
  %106 = fir.pack_array %99 heap innermost : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<!fir.array<?x?xi32>>
  %107 = fir.declare %106 dummy_scope %105 {uniq_name = "aaa"} : (!fir.box<!fir.array<?x?xi32>>, !fir.dscope) -> !fir.box<!fir.array<?x?xi32>>
  %108 = fir.rebox %107 : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<!fir.array<?x?xi32>>
  %113 = fir.dummy_scope : !fir.dscope
  %114 = fir.pack_array %108 heap innermost : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<!fir.array<?x?xi32>>
  %115 = fir.declare %114 dummy_scope %113 {uniq_name = "aaa"} : (!fir.box<!fir.array<?x?xi32>>, !fir.dscope) -> !fir.box<!fir.array<?x?xi32>>
  %116 = fir.rebox %115 : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<!fir.array<?x?xi32>>
  cf.cond_br %pred, ^bb11, ^bb7
^bb7:  // pred: ^bb6
  %122 = fir.dummy_scope : !fir.dscope
  %123 = fir.pack_array %116 heap innermost : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<!fir.array<?x?xi32>>
  %124 = fir.declare %123 dummy_scope %122 {uniq_name = "aaa"} : (!fir.box<!fir.array<?x?xi32>>, !fir.dscope) -> !fir.box<!fir.array<?x?xi32>>
  %125 = fir.rebox %124 : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<!fir.array<?x?xi32>>
  %130 = fir.dummy_scope : !fir.dscope
  %131 = fir.pack_array %125 heap innermost : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<!fir.array<?x?xi32>>
  %132 = fir.declare %131 dummy_scope %130 {uniq_name = "aaa"} : (!fir.box<!fir.array<?x?xi32>>, !fir.dscope) -> !fir.box<!fir.array<?x?xi32>>
  %133 = fir.rebox %132 : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<!fir.array<?x?xi32>>
  cf.cond_br %pred, ^bb9, ^bb8
^bb8:  // pred: ^bb7
  %139 = fir.dummy_scope : !fir.dscope
  %140 = fir.pack_array %133 heap innermost : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<!fir.array<?x?xi32>>
  fir.unpack_array %140 to %133 heap : !fir.box<!fir.array<?x?xi32>>
  cf.br ^bb9
^bb9:  // 2 preds: ^bb7, ^bb8
  fir.unpack_array %131 to %125 heap : !fir.box<!fir.array<?x?xi32>>
  cf.br ^bb10
^bb10:  // pred: ^bb9
  fir.unpack_array %123 to %116 heap : !fir.box<!fir.array<?x?xi32>>
  cf.br ^bb11
^bb11:  // 2 preds: ^bb6, ^bb10
  fir.unpack_array %114 to %108 heap : !fir.box<!fir.array<?x?xi32>>
  fir.unpack_array %106 to %99 heap : !fir.box<!fir.array<?x?xi32>>
  cf.br ^bb12
^bb12:  // 2 preds: ^bb5, ^bb11
  fir.unpack_array %97 to %91 heap : !fir.box<!fir.array<?x?xi32>>
  fir.unpack_array %89 to %82 heap : !fir.box<!fir.array<?x?xi32>>
  cf.br ^bb13
^bb13:  // 2 preds: ^bb4, ^bb12
  fir.unpack_array %80 to %74 heap : !fir.box<!fir.array<?x?xi32>>
  fir.unpack_array %72 to %65 heap : !fir.box<!fir.array<?x?xi32>>
  cf.br ^bb14
^bb14:  // 2 preds: ^bb3, ^bb13
  fir.unpack_array %63 to %57 heap : !fir.box<!fir.array<?x?xi32>>
  fir.unpack_array %55 to %48 heap : !fir.box<!fir.array<?x?xi32>>
  cf.br ^bb15
^bb15:  // 2 preds: ^bb2, ^bb14
  fir.unpack_array %46 to %40 heap : !fir.box<!fir.array<?x?xi32>>
  fir.unpack_array %38 to %31 heap : !fir.box<!fir.array<?x?xi32>>
  cf.br ^bb16
^bb16:  // 2 preds: ^bb1, ^bb15
  fir.unpack_array %29 to %23 heap : !fir.box<!fir.array<?x?xi32>>
  fir.unpack_array %21 to %14 heap : !fir.box<!fir.array<?x?xi32>>
  cf.br ^bb17
^bb17:  // 2 preds: ^bb0, ^bb16
  fir.unpack_array %12 to %10 heap : !fir.box<!fir.array<?x?xi32>>
  return
}
