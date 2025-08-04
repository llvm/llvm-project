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
