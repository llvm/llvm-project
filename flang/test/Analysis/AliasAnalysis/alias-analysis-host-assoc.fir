// Test alias analysis queries for host associated accesses.
// RUN: fir-opt %s --test-fir-alias-analysis -split-input-file --mlir-disable-threading 2>&1 | FileCheck %s

// subroutine test1
//   integer :: x(10)
// contains
//   subroutine inner(y)
//     integer :: y(10)
//     x(1) = y(1)
//   end subroutine inner
// end subroutine test1

// F18 15.5.2.13 (4):
// CHECK: test1_y(1)#0 <-> test1_x(1)#0: NoAlias
func.func @_QFtest1Pinner(%arg0: !fir.ref<!fir.array<10xi32>> {fir.bindc_name = "y"}, %arg1: !fir.ref<tuple<!fir.box<!fir.array<10xi32>>>> {fir.host_assoc}) attributes {fir.internal_proc} {
  %c0_i32 = arith.constant 0 : i32
  %0 = fir.coordinate_of %arg1, %c0_i32 : (!fir.ref<tuple<!fir.box<!fir.array<10xi32>>>>, i32) -> !fir.ref<!fir.box<!fir.array<10xi32>>>
  %1 = fir.load %0 : !fir.ref<!fir.box<!fir.array<10xi32>>>
  %2 = fir.box_addr %1 : (!fir.box<!fir.array<10xi32>>) -> !fir.ref<!fir.array<10xi32>>
  %c0 = arith.constant 0 : index
  %3:3 = fir.box_dims %1, %c0 : (!fir.box<!fir.array<10xi32>>, index) -> (index, index, index)
  %4 = fir.shape %3#1 : (index) -> !fir.shape<1>
  %5:2 = hlfir.declare %2(%4) {fortran_attrs = #fir.var_attrs<host_assoc>, uniq_name = "_QFtest1Ex"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
  %c10 = arith.constant 10 : index
  %6 = fir.shape %c10 : (index) -> !fir.shape<1>
  %7:2 = hlfir.declare %arg0(%6) {uniq_name = "_QFtest1FinnerEy"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
  %c1 = arith.constant 1 : index
  %8 = hlfir.designate %7#0 (%c1) {test.ptr = "test1_y(1)"} : (!fir.ref<!fir.array<10xi32>>, index) -> !fir.ref<i32>
  %9 = fir.load %8 : !fir.ref<i32>
  %c1_0 = arith.constant 1 : index
  %10 = hlfir.designate %5#0 (%c1_0) {test.ptr = "test1_x(1)"} : (!fir.ref<!fir.array<10xi32>>, index) -> !fir.ref<i32>
  hlfir.assign %9 to %10 : i32, !fir.ref<i32>
  return
}

// -----

// subroutine test2
//   integer, target :: x(10)
//   call inner(x)
// contains
//   subroutine inner(y)
//     integer, pointer, intent(in) :: y(:)
//     x(1) = y(1)
//   end subroutine inner
// end subroutine test2

// F18 15.5.2.13 (4) (a):
// CHECK: test2_y(1)#0 <-> test2_x(1)#0: MayAlias
func.func @_QFtest2Pinner(%arg0: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>> {fir.bindc_name = "y"}, %arg1: !fir.ref<tuple<!fir.box<!fir.array<10xi32>>>> {fir.host_assoc}) attributes {fir.internal_proc} {
  %c0_i32 = arith.constant 0 : i32
  %0 = fir.coordinate_of %arg1, %c0_i32 : (!fir.ref<tuple<!fir.box<!fir.array<10xi32>>>>, i32) -> !fir.ref<!fir.box<!fir.array<10xi32>>>
  %1 = fir.load %0 : !fir.ref<!fir.box<!fir.array<10xi32>>>
  %2 = fir.box_addr %1 : (!fir.box<!fir.array<10xi32>>) -> !fir.ref<!fir.array<10xi32>>
  %c0 = arith.constant 0 : index
  %3:3 = fir.box_dims %1, %c0 : (!fir.box<!fir.array<10xi32>>, index) -> (index, index, index)
  %4 = fir.shape %3#1 : (index) -> !fir.shape<1>
  %5:2 = hlfir.declare %2(%4) {fortran_attrs = #fir.var_attrs<target, host_assoc>, uniq_name = "_QFtest2Ex"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
  %6:2 = hlfir.declare %arg0 {fortran_attrs = #fir.var_attrs<intent_in, pointer>, uniq_name = "_QFtest2FinnerEy"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>)
  %7 = fir.load %6#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
  %c1 = arith.constant 1 : index
  %8 = hlfir.designate %7 (%c1) {test.ptr = "test2_y(1)"} : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, index) -> !fir.ref<i32>
  %9 = fir.load %8 : !fir.ref<i32>
  %c1_0 = arith.constant 1 : index
  %10 = hlfir.designate %5#0 (%c1_0) {test.ptr = "test2_x(1)"} : (!fir.ref<!fir.array<10xi32>>, index) -> !fir.ref<i32>
  hlfir.assign %9 to %10 : i32, !fir.ref<i32>
  return
}

// -----

// subroutine test3
//   integer :: x(10)
//   call inner(x)
// contains
//   subroutine inner(y)
//     integer, target :: y(:)
//     x(1) = y(1)
//   end subroutine inner
// end subroutine test3

// F18 15.5.2.13 (4) (b):
// CHECK: test3_y(1)#0 <-> test3_x(1)#0: MayAlias
func.func @_QFtest3Pinner(%arg0: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "y", fir.target}, %arg1: !fir.ref<tuple<!fir.box<!fir.array<10xi32>>>> {fir.host_assoc}) attributes {fir.internal_proc} {
  %c0_i32 = arith.constant 0 : i32
  %0 = fir.coordinate_of %arg1, %c0_i32 : (!fir.ref<tuple<!fir.box<!fir.array<10xi32>>>>, i32) -> !fir.ref<!fir.box<!fir.array<10xi32>>>
  %1 = fir.load %0 : !fir.ref<!fir.box<!fir.array<10xi32>>>
  %2 = fir.box_addr %1 : (!fir.box<!fir.array<10xi32>>) -> !fir.ref<!fir.array<10xi32>>
  %c0 = arith.constant 0 : index
  %3:3 = fir.box_dims %1, %c0 : (!fir.box<!fir.array<10xi32>>, index) -> (index, index, index)
  %4 = fir.shape %3#1 : (index) -> !fir.shape<1>
  %5:2 = hlfir.declare %2(%4) {fortran_attrs = #fir.var_attrs<host_assoc>, uniq_name = "_QFtest3Ex"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
  %6:2 = hlfir.declare %arg0 {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFtest3FinnerEy"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
  %c1 = arith.constant 1 : index
  %7 = hlfir.designate %6#0 (%c1) {test.ptr = "test3_y(1)"} : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
  %8 = fir.load %7 : !fir.ref<i32>
  %c1_0 = arith.constant 1 : index
  %9 = hlfir.designate %5#0 (%c1_0) {test.ptr = "test3_x(1)"} : (!fir.ref<!fir.array<10xi32>>, index) -> !fir.ref<i32>
  hlfir.assign %8 to %9 : i32, !fir.ref<i32>
  return
}

// -----

// subroutine test4
//   integer :: n(10), m(10)
//   equivalence (n, m)
// contains
//   subroutine inner()
//     n(1) = m(1)
//   end subroutine inner
// end subroutine test4

// CHECK: test4_m(1)#0 <-> test4_n(1)#0: MayAlias
func.func @_QFtest4Pinner(%arg0: !fir.ref<tuple<!fir.box<!fir.array<10xi32>>, !fir.box<!fir.array<10xi32>>>> {fir.host_assoc}) attributes {fir.internal_proc} {
  %c0_i32 = arith.constant 0 : i32
  %0 = fir.coordinate_of %arg0, %c0_i32 : (!fir.ref<tuple<!fir.box<!fir.array<10xi32>>, !fir.box<!fir.array<10xi32>>>>, i32) -> !fir.ref<!fir.box<!fir.array<10xi32>>>
  %1 = fir.load %0 : !fir.ref<!fir.box<!fir.array<10xi32>>>
  %2 = fir.box_addr %1 : (!fir.box<!fir.array<10xi32>>) -> !fir.ref<!fir.array<10xi32>>
  %c0 = arith.constant 0 : index
  %3:3 = fir.box_dims %1, %c0 : (!fir.box<!fir.array<10xi32>>, index) -> (index, index, index)
  %4 = fir.shape %3#1 : (index) -> !fir.shape<1>
  %5:2 = hlfir.declare %2(%4) {fortran_attrs = #fir.var_attrs<host_assoc>, uniq_name = "_QFtest4En"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
  %c1_i32 = arith.constant 1 : i32
  %6 = fir.coordinate_of %arg0, %c1_i32 : (!fir.ref<tuple<!fir.box<!fir.array<10xi32>>, !fir.box<!fir.array<10xi32>>>>, i32) -> !fir.ref<!fir.box<!fir.array<10xi32>>>
  %7 = fir.load %6 : !fir.ref<!fir.box<!fir.array<10xi32>>>
  %8 = fir.box_addr %7 : (!fir.box<!fir.array<10xi32>>) -> !fir.ref<!fir.array<10xi32>>
  %c0_0 = arith.constant 0 : index
  %9:3 = fir.box_dims %7, %c0_0 : (!fir.box<!fir.array<10xi32>>, index) -> (index, index, index)
  %10 = fir.shape %9#1 : (index) -> !fir.shape<1>
  %11:2 = hlfir.declare %8(%10) {fortran_attrs = #fir.var_attrs<host_assoc>, uniq_name = "_QFtest4Em"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
  %12 = fir.alloca !fir.array<40xi8> {uniq_name = "_QFtest4Em"}
  %c1 = arith.constant 1 : index
  %13 = hlfir.designate %11#0 (%c1) {test.ptr = "test4_m(1)"} : (!fir.ref<!fir.array<10xi32>>, index) -> !fir.ref<i32>
  %14 = fir.load %13 : !fir.ref<i32>
  %c1_1 = arith.constant 1 : index
  %15 = hlfir.designate %5#0 (%c1_1) {test.ptr = "test4_n(1)"} : (!fir.ref<!fir.array<10xi32>>, index) -> !fir.ref<i32>
  hlfir.assign %14 to %15 : i32, !fir.ref<i32>
  return
}

// -----

// subroutine test5
//   integer, target :: x(10)
// contains
//   subroutine inner(y)
//     integer :: y(10)
//     x(1) = y(1)
//   end subroutine inner
// end subroutine test5

// F18 15.5.2.13 (4):
// CHECK: test5_y(1)#0 <-> test5_x(1)#0: NoAlias
func.func @_QFtest5Pinner(%arg0: !fir.ref<!fir.array<10xi32>> {fir.bindc_name = "y"}, %arg1: !fir.ref<tuple<!fir.box<!fir.array<10xi32>>>> {fir.host_assoc}) attributes {fir.internal_proc} {
  %c0_i32 = arith.constant 0 : i32
  %0 = fir.coordinate_of %arg1, %c0_i32 : (!fir.ref<tuple<!fir.box<!fir.array<10xi32>>>>, i32) -> !fir.ref<!fir.box<!fir.array<10xi32>>>
  %1 = fir.load %0 : !fir.ref<!fir.box<!fir.array<10xi32>>>
  %2 = fir.box_addr %1 : (!fir.box<!fir.array<10xi32>>) -> !fir.ref<!fir.array<10xi32>>
  %c0 = arith.constant 0 : index
  %3:3 = fir.box_dims %1, %c0 : (!fir.box<!fir.array<10xi32>>, index) -> (index, index, index)
  %4 = fir.shape %3#1 : (index) -> !fir.shape<1>
  %5:2 = hlfir.declare %2(%4) {fortran_attrs = #fir.var_attrs<target, host_assoc>, uniq_name = "_QFtest5Ex"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
  %c10 = arith.constant 10 : index
  %6 = fir.shape %c10 : (index) -> !fir.shape<1>
  %7:2 = hlfir.declare %arg0(%6) {uniq_name = "_QFtest5FinnerEy"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
  %c1 = arith.constant 1 : index
  %8 = hlfir.designate %7#0 (%c1) {test.ptr = "test5_y(1)"} : (!fir.ref<!fir.array<10xi32>>, index) -> !fir.ref<i32>
  %9 = fir.load %8 : !fir.ref<i32>
  %c1_0 = arith.constant 1 : index
  %10 = hlfir.designate %5#0 (%c1_0) {test.ptr = "test5_x(1)"} : (!fir.ref<!fir.array<10xi32>>, index) -> !fir.ref<i32>
  hlfir.assign %9 to %10 : i32, !fir.ref<i32>
  return
}

// -----

// subroutine test6
//   integer, pointer :: x(:)
// contains
//   subroutine inner(y)
//     integer :: y(10)
//     x(1) = y(1)
//   end subroutine inner
// end subroutine test6

// F18 15.5.2.13 (4):
// FIXME: 'x' is classified as Indirect access leading to a conservative reply:
// CHECK: test6_y(1)#0 <-> test6_x(1)#0: MayAlias
func.func @_QFtest6Pinner(%arg0: !fir.ref<!fir.array<10xi32>> {fir.bindc_name = "y"}, %arg1: !fir.ref<tuple<!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>>> {fir.host_assoc}) attributes {fir.internal_proc} {
  %c0_i32 = arith.constant 0 : i32
  %0 = fir.coordinate_of %arg1, %c0_i32 : (!fir.ref<tuple<!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>>>, i32) -> !fir.llvm_ptr<!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>>
  %1 = fir.load %0 : !fir.llvm_ptr<!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>>
  %2:2 = hlfir.declare %1 {fortran_attrs = #fir.var_attrs<pointer, host_assoc>, uniq_name = "_QFtest6Ex"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>)
  %c10 = arith.constant 10 : index
  %3 = fir.shape %c10 : (index) -> !fir.shape<1>
  %4:2 = hlfir.declare %arg0(%3) {uniq_name = "_QFtest6FinnerEy"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
  %c1 = arith.constant 1 : index
  %5 = hlfir.designate %4#0 (%c1) {test.ptr = "test6_y(1)"} : (!fir.ref<!fir.array<10xi32>>, index) -> !fir.ref<i32>
  %6 = fir.load %5 : !fir.ref<i32>
  %7 = fir.load %2#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
  %c1_0 = arith.constant 1 : index
  %8 = hlfir.designate %7 (%c1_0) {test.ptr = "test6_x(1)"} : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, index) -> !fir.ref<i32>
  hlfir.assign %6 to %8 : i32, !fir.ref<i32>
  return
}

// -----

// module globals
//   integer :: g(10)
// end module globals
// subroutine test7
//   use globals
//   integer :: x(10)
// contains
//   subroutine inner()
//     x(1) = g(1)
//   end subroutine inner
// end subroutine test7

// CHECK: test7_g(1)#0 <-> test7_x(1)#0: NoAlias
func.func @_QFtest7Pinner(%arg0: !fir.ref<tuple<!fir.box<!fir.array<10xi32>>>> {fir.host_assoc}) attributes {fir.internal_proc} {
  %0 = fir.address_of(@_QMglobalsEg) : !fir.ref<!fir.array<10xi32>>
  %c10 = arith.constant 10 : index
  %1 = fir.shape %c10 : (index) -> !fir.shape<1>
  %2:2 = hlfir.declare %0(%1) {uniq_name = "_QMglobalsEg"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
  %c0_i32 = arith.constant 0 : i32
  %3 = fir.coordinate_of %arg0, %c0_i32 : (!fir.ref<tuple<!fir.box<!fir.array<10xi32>>>>, i32) -> !fir.ref<!fir.box<!fir.array<10xi32>>>
  %4 = fir.load %3 : !fir.ref<!fir.box<!fir.array<10xi32>>>
  %5 = fir.box_addr %4 : (!fir.box<!fir.array<10xi32>>) -> !fir.ref<!fir.array<10xi32>>
  %c0 = arith.constant 0 : index
  %6:3 = fir.box_dims %4, %c0 : (!fir.box<!fir.array<10xi32>>, index) -> (index, index, index)
  %7 = fir.shape %6#1 : (index) -> !fir.shape<1>
  %8:2 = hlfir.declare %5(%7) {fortran_attrs = #fir.var_attrs<host_assoc>, uniq_name = "_QFtest7Ex"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
  %c1 = arith.constant 1 : index
  %9 = hlfir.designate %2#0 (%c1) {test.ptr = "test7_g(1)"} : (!fir.ref<!fir.array<10xi32>>, index) -> !fir.ref<i32>
  %10 = fir.load %9 : !fir.ref<i32>
  %c1_0 = arith.constant 1 : index
  %11 = hlfir.designate %8#0 (%c1_0) {test.ptr = "test7_x(1)"}  : (!fir.ref<!fir.array<10xi32>>, index) -> !fir.ref<i32>
  hlfir.assign %10 to %11 : i32, !fir.ref<i32>
  return
}
fir.global @_QMglobalsEg : !fir.array<10xi32> {
  %0 = fir.undefined !fir.array<10xi32>
  fir.has_value %0 : !fir.array<10xi32>
}

// -----

// module globals
//   integer, target :: g(10)
// end module globals
// subroutine test8
//   use globals
//   integer :: x(10)
// contains
//   subroutine inner()
//     x(1) = g(1)
//   end subroutine inner
// end subroutine test8

// CHECK: test8_g(1)#0 <-> test8_x(1)#0: NoAlias
func.func @_QFtest8Pinner(%arg0: !fir.ref<tuple<!fir.box<!fir.array<10xi32>>>> {fir.host_assoc}) attributes {fir.internal_proc} {
  %0 = fir.address_of(@_QMglobalsEg) : !fir.ref<!fir.array<10xi32>>
  %c10 = arith.constant 10 : index
  %1 = fir.shape %c10 : (index) -> !fir.shape<1>
  %2:2 = hlfir.declare %0(%1) {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QMglobalsEg"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
  %c0_i32 = arith.constant 0 : i32
  %3 = fir.coordinate_of %arg0, %c0_i32 : (!fir.ref<tuple<!fir.box<!fir.array<10xi32>>>>, i32) -> !fir.ref<!fir.box<!fir.array<10xi32>>>
  %4 = fir.load %3 : !fir.ref<!fir.box<!fir.array<10xi32>>>
  %5 = fir.box_addr %4 : (!fir.box<!fir.array<10xi32>>) -> !fir.ref<!fir.array<10xi32>>
  %c0 = arith.constant 0 : index
  %6:3 = fir.box_dims %4, %c0 : (!fir.box<!fir.array<10xi32>>, index) -> (index, index, index)
  %7 = fir.shape %6#1 : (index) -> !fir.shape<1>
  %8:2 = hlfir.declare %5(%7) {fortran_attrs = #fir.var_attrs<host_assoc>, uniq_name = "_QFtest8Ex"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
  %c1 = arith.constant 1 : index
  %9 = hlfir.designate %2#0 (%c1) {test.ptr = "test8_g(1)"} : (!fir.ref<!fir.array<10xi32>>, index) -> !fir.ref<i32>
  %10 = fir.load %9 : !fir.ref<i32>
  %c1_0 = arith.constant 1 : index
  %11 = hlfir.designate %8#0 (%c1_0) {test.ptr = "test8_x(1)"} : (!fir.ref<!fir.array<10xi32>>, index) -> !fir.ref<i32>
  hlfir.assign %10 to %11 : i32, !fir.ref<i32>
  return
}

// -----

// module globals
//   integer, pointer :: g(:)
// end module globals
// subroutine test9
//   use globals
//   integer :: x(10)
// contains
//   subroutine inner()
//     x(1) = g(1)
//   end subroutine inner
// end subroutine test9

// CHECK: test9_g(1)#0 <-> test9_x(1)#0: NoAlias
func.func @_QFtest9Pinner(%arg0: !fir.ref<tuple<!fir.box<!fir.array<10xi32>>>> {fir.host_assoc}) attributes {fir.internal_proc} {
  %0 = fir.address_of(@_QMglobalsEg) : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
  %1:2 = hlfir.declare %0 {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QMglobalsEg"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>)
  %c0_i32 = arith.constant 0 : i32
  %2 = fir.coordinate_of %arg0, %c0_i32 : (!fir.ref<tuple<!fir.box<!fir.array<10xi32>>>>, i32) -> !fir.ref<!fir.box<!fir.array<10xi32>>>
  %3 = fir.load %2 : !fir.ref<!fir.box<!fir.array<10xi32>>>
  %4 = fir.box_addr %3 : (!fir.box<!fir.array<10xi32>>) -> !fir.ref<!fir.array<10xi32>>
  %c0 = arith.constant 0 : index
  %5:3 = fir.box_dims %3, %c0 : (!fir.box<!fir.array<10xi32>>, index) -> (index, index, index)
  %6 = fir.shape %5#1 : (index) -> !fir.shape<1>
  %7:2 = hlfir.declare %4(%6) {fortran_attrs = #fir.var_attrs<host_assoc>, uniq_name = "_QFtest9Ex"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
  %8 = fir.load %1#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
  %c1 = arith.constant 1 : index
  %9 = hlfir.designate %8 (%c1) {test.ptr = "test9_g(1)"} : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, index) -> !fir.ref<i32>
  %10 = fir.load %9 : !fir.ref<i32>
  %c1_0 = arith.constant 1 : index
  %11 = hlfir.designate %7#0 (%c1_0) {test.ptr = "test9_x(1)"} : (!fir.ref<!fir.array<10xi32>>, index) -> !fir.ref<i32>
  hlfir.assign %10 to %11 : i32, !fir.ref<i32>
  return
}
fir.global @_QMglobalsEg : !fir.box<!fir.ptr<!fir.array<?xi32>>> {
  %0 = fir.zero_bits !fir.ptr<!fir.array<?xi32>>
  %c0 = arith.constant 0 : index
  %1 = fir.shape %c0 : (index) -> !fir.shape<1>
  %2 = fir.embox %0(%1) : (!fir.ptr<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>>
  fir.has_value %2 : !fir.box<!fir.ptr<!fir.array<?xi32>>>
}

// -----

// module globals
//   integer, pointer :: g(:)
// end module globals
// subroutine test10
//   use globals
//   integer, target :: x(10)
// contains
//   subroutine inner()
//     x(1) = g(1)
//   end subroutine inner
// end subroutine test10

// CHECK: test10_g(1)#0 <-> test10_x(1)#0: MayAlias
func.func @_QFtest10Pinner(%arg0: !fir.ref<tuple<!fir.box<!fir.array<10xi32>>>> {fir.host_assoc}) attributes {fir.internal_proc} {
  %0 = fir.address_of(@_QMglobalsEg) : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
  %1:2 = hlfir.declare %0 {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QMglobalsEg"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>)
  %c0_i32 = arith.constant 0 : i32
  %2 = fir.coordinate_of %arg0, %c0_i32 : (!fir.ref<tuple<!fir.box<!fir.array<10xi32>>>>, i32) -> !fir.ref<!fir.box<!fir.array<10xi32>>>
  %3 = fir.load %2 : !fir.ref<!fir.box<!fir.array<10xi32>>>
  %4 = fir.box_addr %3 : (!fir.box<!fir.array<10xi32>>) -> !fir.ref<!fir.array<10xi32>>
  %c0 = arith.constant 0 : index
  %5:3 = fir.box_dims %3, %c0 : (!fir.box<!fir.array<10xi32>>, index) -> (index, index, index)
  %6 = fir.shape %5#1 : (index) -> !fir.shape<1>
  %7:2 = hlfir.declare %4(%6) {fortran_attrs = #fir.var_attrs<target, host_assoc>, uniq_name = "_QFtest10Ex"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
  %8 = fir.load %1#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
  %c1 = arith.constant 1 : index
  %9 = hlfir.designate %8 (%c1) {test.ptr = "test10_g(1)"} : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, index) -> !fir.ref<i32>
  %10 = fir.load %9 : !fir.ref<i32>
  %c1_0 = arith.constant 1 : index
  %11 = hlfir.designate %7#0 (%c1_0) {test.ptr = "test10_x(1)"} : (!fir.ref<!fir.array<10xi32>>, index) -> !fir.ref<i32>
  hlfir.assign %10 to %11 : i32, !fir.ref<i32>
  return
}
fir.global @_QMglobalsEg : !fir.box<!fir.ptr<!fir.array<?xi32>>> {
  %0 = fir.zero_bits !fir.ptr<!fir.array<?xi32>>
  %c0 = arith.constant 0 : index
  %1 = fir.shape %c0 : (index) -> !fir.shape<1>
  %2 = fir.embox %0(%1) : (!fir.ptr<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>>
  fir.has_value %2 : !fir.box<!fir.ptr<!fir.array<?xi32>>>
}

// -----

// module globals
//   integer, target :: g(10)
// end module globals
// subroutine test11
//   use globals
//   integer, pointer :: x(:)
// contains
//   subroutine inner()
//     x(1) = g(1)
//   end subroutine inner
// end subroutine test11

// CHECK: test11_g(1)#0 <-> test11_x(1)#0: MayAlias
func.func @_QFtest11Pinner(%arg0: !fir.ref<tuple<!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>>> {fir.host_assoc}) attributes {fir.internal_proc} {
  %0 = fir.address_of(@_QMglobalsEg) : !fir.ref<!fir.array<10xi32>>
  %c10 = arith.constant 10 : index
  %1 = fir.shape %c10 : (index) -> !fir.shape<1>
  %2:2 = hlfir.declare %0(%1) {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QMglobalsEg"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
  %c0_i32 = arith.constant 0 : i32
  %3 = fir.coordinate_of %arg0, %c0_i32 : (!fir.ref<tuple<!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>>>, i32) -> !fir.llvm_ptr<!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>>
  %4 = fir.load %3 : !fir.llvm_ptr<!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>>
  %5:2 = hlfir.declare %4 {fortran_attrs = #fir.var_attrs<pointer, host_assoc>, uniq_name = "_QFtest11Ex"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>)
  %c1 = arith.constant 1 : index
  %6 = hlfir.designate %2#0 (%c1) {test.ptr = "test11_g(1)"} : (!fir.ref<!fir.array<10xi32>>, index) -> !fir.ref<i32>
  %7 = fir.load %6 : !fir.ref<i32>
  %8 = fir.load %5#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
  %c1_0 = arith.constant 1 : index
  %9 = hlfir.designate %8 (%c1_0) {test.ptr = "test11_x(1)"} : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, index) -> !fir.ref<i32>
  hlfir.assign %7 to %9 : i32, !fir.ref<i32>
  return
}
fir.global @_QMglobalsEg target : !fir.array<10xi32> {
  %0 = fir.undefined !fir.array<10xi32>
  fir.has_value %0 : !fir.array<10xi32>
}
