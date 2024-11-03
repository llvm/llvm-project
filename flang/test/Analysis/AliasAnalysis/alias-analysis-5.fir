// Use --mlir-disable-threading so that the AA queries are serialized
// as well as its diagnostic output.
// RUN: fir-opt %s -pass-pipeline='builtin.module(func.func(test-fir-alias-analysis))' -split-input-file --mlir-disable-threading 2>&1 | FileCheck %s

// CHECK-LABEL: Testing : "_QPtest1"
// CHECK: arraya(ii)#0 <-> arrayc(ii)#0: NoAlias

// Generated from 
// subroutine test1(arrayA, arrayB, arrayC, N)
//   integer, dimension(:) :: arrayA, arrayB, arrayC
//   integer N
//   do ii = 1, N
//     arrayC(ii) = arrayB(ii) + arrayA(ii)
//   end do
// end subroutine

func.func @_QPtest1(%arg0: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "arraya"}, %arg1: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "arrayb"}, %arg2: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "arrayc"}, %arg3: !fir.ref<i32> {fir.bindc_name = "n"}) {
  %c1 = arith.constant 1 : index
  %c1_i64 = arith.constant 1 : i64
  %0 = fir.alloca i32 {bindc_name = "ii", uniq_name = "_QFtest1Eii"}
  %1 = fir.load %arg3 : !fir.ref<i32>
  %2 = fir.convert %1 : (i32) -> index
  %3 = fir.convert %c1 : (index) -> i32
  %4:2 = fir.do_loop %arg4 = %c1 to %2 step %c1 iter_args(%arg5 = %3) -> (index, i32) {
    fir.store %arg5 to %0 : !fir.ref<i32>
    %5 = fir.load %0 : !fir.ref<i32>
    %6 = fir.convert %5 : (i32) -> i64
    %7 = arith.subi %6, %c1_i64 : i64
    %8 = fir.coordinate_of %arg1, %7 : (!fir.box<!fir.array<?xi32>>, i64) -> !fir.ref<i32>
    %9 = fir.load %8 : !fir.ref<i32>
    %10 = fir.coordinate_of %arg0, %7 {test.ptr = "arraya(ii)"} : (!fir.box<!fir.array<?xi32>>, i64) -> !fir.ref<i32>
    %11 = fir.load %10 : !fir.ref<i32>
    %12 = arith.addi %9, %11 : i32
    %13 = fir.coordinate_of %arg2, %7 {test.ptr = "arrayc(ii)"} : (!fir.box<!fir.array<?xi32>>, i64) -> !fir.ref<i32>
    fir.store %12 to %13 : !fir.ref<i32>
    %14 = arith.addi %arg4, %c1 : index
    %15 = fir.load %0 : !fir.ref<i32>
    %16 = arith.addi %15, %3 : i32
    fir.result %14, %16 : index, i32
  }
  fir.store %4#1 to %0 : !fir.ref<i32>
  return
}

// -----

// CHECK-LABEL: Testing : "_QPtest3"
// CHECK: arraya(ii)#0 <-> arrayc(ii)#0: MayAlias

// subroutine test3(arrayA, arrayB, arrayC, N)
//   integer, dimension(:), target :: arrayA
//   integer, dimension(:) :: arrayB
//   integer, dimension(:), pointer :: arrayC
//   do ii = 1, N
//     arrayC(ii) = arrayB(ii) + arrayA(ii)
//   end do
// end subroutine

func.func @_QPtest3(%arg0: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "arraya", fir.target}, %arg1: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "arrayb"}, %arg2: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>> {fir.bindc_name = "arrayc"}, %arg3: !fir.ref<i32> {fir.bindc_name = "n"}) {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c1_i64 = arith.constant 1 : i64
  %0 = fir.alloca i32 {bindc_name = "ii", uniq_name = "_QFtest3Eii"}
  %1 = fir.load %arg3 : !fir.ref<i32>
  %2 = fir.convert %1 : (i32) -> index
  %3 = fir.convert %c1 : (index) -> i32
  %4:2 = fir.do_loop %arg4 = %c1 to %2 step %c1 iter_args(%arg5 = %3) -> (index, i32) {
    fir.store %arg5 to %0 : !fir.ref<i32>
    %5 = fir.load %0 : !fir.ref<i32>
    %6 = fir.convert %5 : (i32) -> i64
    %7 = arith.subi %6, %c1_i64 : i64
    %8 = fir.coordinate_of %arg1, %7 : (!fir.box<!fir.array<?xi32>>, i64) -> !fir.ref<i32>
    %9 = fir.load %8 : !fir.ref<i32>
    %10 = fir.coordinate_of %arg0, %7 {test.ptr = "arraya(ii)"} : (!fir.box<!fir.array<?xi32>>, i64) -> !fir.ref<i32>
    %11 = fir.load %10 : !fir.ref<i32>
    %12 = arith.addi %9, %11 : i32
    %13 = fir.load %arg2 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
    %14:3 = fir.box_dims %13, %c0 : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, index) -> (index, index, index)
    %15 = fir.convert %14#0 : (index) -> i64
    %16 = arith.subi %6, %15 : i64
    %17 = fir.coordinate_of %13, %16 {test.ptr = "arrayc(ii)"} : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, i64) -> !fir.ref<i32>
    fir.store %12 to %17 : !fir.ref<i32>
    %18 = arith.addi %arg4, %c1 : index
    %19 = fir.load %0 : !fir.ref<i32>
    %20 = arith.addi %19, %3 : i32
    fir.result %18, %20 : index, i32
  }
  fir.store %4#1 to %0 : !fir.ref<i32>
  return
}

// -----

// CHECK-LABEL: Testing : "_QMtest4Ptest"
// CHECK: arraya(ii)#0 <-> arrayc(ii)#0: NoAlias

// module test4
//   integer, dimension(:), allocatable :: arrayA
//   integer, dimension(:), allocatable :: arrayB
//   integer, dimension(:), allocatable :: arrayC
//   integer :: N
// contains
// subroutine test
//   do ii = 1, N
//     arrayC(ii) = arrayB(ii) + arrayA(ii)
//   end do
// end subroutine
// endmodule

fir.global @_QMtest4Earraya : !fir.box<!fir.heap<!fir.array<?xi32>>> 
fir.global @_QMtest4Earrayb : !fir.box<!fir.heap<!fir.array<?xi32>>> 
fir.global @_QMtest4Earrayc : !fir.box<!fir.heap<!fir.array<?xi32>>> 
fir.global @_QMtest4En : i32

func.func @_QMtest4Ptest() {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = fir.address_of(@_QMtest4Earraya) : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  %1 = fir.address_of(@_QMtest4Earrayb) : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  %2 = fir.address_of(@_QMtest4Earrayc) : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  %3 = fir.address_of(@_QMtest4En) : !fir.ref<i32>
  %4 = fir.alloca i32 {bindc_name = "ii", uniq_name = "_QMtest4FtestEii"}
  %5 = fir.load %3 : !fir.ref<i32>
  %6 = fir.convert %5 : (i32) -> index
  %7 = fir.convert %c1 : (index) -> i32
  %8:2 = fir.do_loop %arg0 = %c1 to %6 step %c1 iter_args(%arg1 = %7) -> (index, i32) {
    fir.store %arg1 to %4 : !fir.ref<i32>
    %9 = fir.load %1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
    %10:3 = fir.box_dims %9, %c0 : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
    %11 = fir.box_addr %9 : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
    %12 = fir.load %4 : !fir.ref<i32>
    %13 = fir.convert %12 : (i32) -> i64
    %14 = fir.convert %10#0 : (index) -> i64
    %15 = arith.subi %13, %14 : i64
    %16 = fir.coordinate_of %11, %15 : (!fir.heap<!fir.array<?xi32>>, i64) -> !fir.ref<i32>
    %17 = fir.load %16 : !fir.ref<i32>
    %18 = fir.load %0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
    %19:3 = fir.box_dims %18, %c0 : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
    %20 = fir.box_addr %18 : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
    %21 = fir.convert %19#0 : (index) -> i64
    %22 = arith.subi %13, %21 : i64
    %23 = fir.coordinate_of %20, %22 {test.ptr = "arraya(ii)"} : (!fir.heap<!fir.array<?xi32>>, i64) -> !fir.ref<i32>
    %24 = fir.load %23 : !fir.ref<i32>
    %25 = arith.addi %17, %24 : i32
    %26 = fir.load %2 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
    %27:3 = fir.box_dims %26, %c0 : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
    %28 = fir.box_addr %26 : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
    %29 = fir.convert %27#0 : (index) -> i64
    %30 = arith.subi %13, %29 : i64
    %31 = fir.coordinate_of %28, %30 {test.ptr = "arrayc(ii)"} : (!fir.heap<!fir.array<?xi32>>, i64) -> !fir.ref<i32>
    fir.store %25 to %31 : !fir.ref<i32>
    %32 = arith.addi %arg0, %c1 : index
    %33 = fir.load %4 : !fir.ref<i32>
    %34 = arith.addi %33, %7 : i32
    fir.result %32, %34 : index, i32
  }
  fir.store %8#1 to %4 : !fir.ref<i32>
  return
}

// -----

// CHECK-LABEL: Testing : "_QMtest5Ptest"
// CHECK: arraya(ii)#0 <-> arrayc(ii)#0: NoAlias

// module test5
//   integer, dimension(:), allocatable :: arrayA
//   integer, dimension(:), allocatable :: arrayB
//   integer, dimension(:), pointer :: arrayC
//   integer :: N
// contains
// subroutine test
//   do ii = 1, N
//     arrayC(ii) = arrayB(ii) + arrayA(ii)
//   end do
// end subroutine
// endmodule

fir.global @_QMtest5Earraya : !fir.box<!fir.heap<!fir.array<?xi32>>> 
fir.global @_QMtest5Earrayb : !fir.box<!fir.heap<!fir.array<?xi32>>>
fir.global @_QMtest5Earrayc : !fir.box<!fir.ptr<!fir.array<?xi32>>>
fir.global @_QMtest5En : i32

func.func @_QMtest5Ptest() {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = fir.address_of(@_QMtest5Earraya) : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  %1 = fir.address_of(@_QMtest5Earrayb) : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  %2 = fir.address_of(@_QMtest5Earrayc) : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
  %3 = fir.address_of(@_QMtest5En) : !fir.ref<i32>
  %4 = fir.alloca i32 {bindc_name = "ii", uniq_name = "_QMtest5FtestEii"}
  %5 = fir.load %3 : !fir.ref<i32>
  %6 = fir.convert %5 : (i32) -> index
  %7 = fir.convert %c1 : (index) -> i32
  %8:2 = fir.do_loop %arg0 = %c1 to %6 step %c1 iter_args(%arg1 = %7) -> (index, i32) {
    fir.store %arg1 to %4 : !fir.ref<i32>
    %9 = fir.load %1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
    %10:3 = fir.box_dims %9, %c0 : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
    %11 = fir.box_addr %9 : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
    %12 = fir.load %4 : !fir.ref<i32>
    %13 = fir.convert %12 : (i32) -> i64
    %14 = fir.convert %10#0 : (index) -> i64
    %15 = arith.subi %13, %14 : i64
    %16 = fir.coordinate_of %11, %15 : (!fir.heap<!fir.array<?xi32>>, i64) -> !fir.ref<i32>
    %17 = fir.load %16 : !fir.ref<i32>
    %18 = fir.load %0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
    %19:3 = fir.box_dims %18, %c0 : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
    %20 = fir.box_addr %18 : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
    %21 = fir.convert %19#0 : (index) -> i64
    %22 = arith.subi %13, %21 : i64
    %23 = fir.coordinate_of %20, %22 {test.ptr = "arraya(ii)"} : (!fir.heap<!fir.array<?xi32>>, i64) -> !fir.ref<i32>
    %24 = fir.load %23 : !fir.ref<i32>
    %25 = arith.addi %17, %24 : i32
    %26 = fir.load %2 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
    %27:3 = fir.box_dims %26, %c0 : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, index) -> (index, index, index)
    %28 = fir.convert %27#0 : (index) -> i64
    %29 = arith.subi %13, %28 : i64
    %30 = fir.coordinate_of %26, %29 {test.ptr = "arrayc(ii)"}  : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, i64) -> !fir.ref<i32>
    fir.store %25 to %30 : !fir.ref<i32>
    %31 = arith.addi %arg0, %c1 : index
    %32 = fir.load %4 : !fir.ref<i32>
    %33 = arith.addi %32, %7 : i32
    fir.result %31, %33 : index, i32
  }
  fir.store %8#1 to %4 : !fir.ref<i32>
  return
}

// -----

// CHECK-LABEL: Testing : "_QMtest6Ptest"
// CHECK: arraya(ii)#0 <-> arrayc(ii)#0: MayAlias

// module test6
//   integer, dimension(:), allocatable, target :: arrayA
//   integer, dimension(:), allocatable :: arrayB
//   integer, dimension(:), pointer :: arrayC
//   integer :: N
// contains
// subroutine test
//   do ii = 1, N
//     arrayC(ii) = arrayB(ii) + arrayA(ii)
//   end do
// end subroutine
// endmodule

fir.global @_QMtest6Earraya target : !fir.box<!fir.heap<!fir.array<?xi32>>> 
fir.global @_QMtest6Earrayb : !fir.box<!fir.heap<!fir.array<?xi32>>>
fir.global @_QMtest6Earrayc : !fir.box<!fir.ptr<!fir.array<?xi32>>>
fir.global @_QMtest6En : i32

func.func @_QMtest6Ptest() {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = fir.address_of(@_QMtest6Earraya) : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  %1 = fir.address_of(@_QMtest6Earrayb) : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  %2 = fir.address_of(@_QMtest6Earrayc) : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
  %3 = fir.address_of(@_QMtest6En) : !fir.ref<i32>
  %4 = fir.alloca i32 {bindc_name = "ii", uniq_name = "_QMtest6FtestEii"}
  %5 = fir.load %3 : !fir.ref<i32>
  %6 = fir.convert %5 : (i32) -> index
  %7 = fir.convert %c1 : (index) -> i32
  %8:2 = fir.do_loop %arg0 = %c1 to %6 step %c1 iter_args(%arg1 = %7) -> (index, i32) {
    fir.store %arg1 to %4 : !fir.ref<i32>
    %9 = fir.load %1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
    %10:3 = fir.box_dims %9, %c0 : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
    %11 = fir.box_addr %9 : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
    %12 = fir.load %4 : !fir.ref<i32>
    %13 = fir.convert %12 : (i32) -> i64
    %14 = fir.convert %10#0 : (index) -> i64
    %15 = arith.subi %13, %14 : i64
    %16 = fir.coordinate_of %11, %15 : (!fir.heap<!fir.array<?xi32>>, i64) -> !fir.ref<i32>
    %17 = fir.load %16 : !fir.ref<i32>
    %18 = fir.load %0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
    %19:3 = fir.box_dims %18, %c0 : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
    %20 = fir.box_addr %18 : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
    %21 = fir.convert %19#0 : (index) -> i64
    %22 = arith.subi %13, %21 : i64
    %23 = fir.coordinate_of %20, %22 {test.ptr = "arraya(ii)"} : (!fir.heap<!fir.array<?xi32>>, i64) -> !fir.ref<i32>
    %24 = fir.load %23 : !fir.ref<i32>
    %25 = arith.addi %17, %24 : i32
    %26 = fir.load %2 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
    %27:3 = fir.box_dims %26, %c0 : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, index) -> (index, index, index)
    %28 = fir.convert %27#0 : (index) -> i64
    %29 = arith.subi %13, %28 : i64
    %30 = fir.coordinate_of %26, %29 {test.ptr = "arrayc(ii)"} : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, i64) -> !fir.ref<i32>
    fir.store %25 to %30 : !fir.ref<i32>
    %31 = arith.addi %arg0, %c1 : index
    %32 = fir.load %4 : !fir.ref<i32>
    %33 = arith.addi %32, %7 : i32
    fir.result %31, %33 : index, i32
  }
  fir.store %8#1 to %4 : !fir.ref<i32>
  return
}

// -----

// CHECK-LABEL: Testing : "_QMtest7Ptest"
// CHECK: arraya(ii)#0 <-> arrayc(ii)#0: MayAlias

// module test7
//   integer, dimension(1000), target :: arrayA
//   integer, dimension(:), allocatable :: arrayB
//   integer, dimension(:), pointer :: arrayC
//   integer :: N
// contains
// subroutine test
//   do ii = 1, N
//     arrayC(ii) = arrayB(ii) + arrayA(ii)
//   end do
// end subroutine
// endmodule

fir.global @_QMtest7Earraya target : !fir.array<1000xi32> 
fir.global @_QMtest7Earrayb : !fir.box<!fir.heap<!fir.array<?xi32>>> 
fir.global @_QMtest7Earrayc : !fir.box<!fir.ptr<!fir.array<?xi32>>> 
fir.global @_QMtest7En : i32 

func.func @_QMtest7Ptest() {
  %c1 = arith.constant 1 : index
  %c1_i64 = arith.constant 1 : i64
  %c0 = arith.constant 0 : index
  %0 = fir.address_of(@_QMtest7Earraya) : !fir.ref<!fir.array<1000xi32>>
  %1 = fir.address_of(@_QMtest7Earrayb) : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  %2 = fir.address_of(@_QMtest7Earrayc) : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
  %3 = fir.address_of(@_QMtest7En) : !fir.ref<i32>
  %4 = fir.alloca i32 {bindc_name = "ii", uniq_name = "_QMtest7FtestEii"}
  %5 = fir.load %3 : !fir.ref<i32>
  %6 = fir.convert %5 : (i32) -> index
  %7 = fir.convert %c1 : (index) -> i32
  %8:2 = fir.do_loop %arg0 = %c1 to %6 step %c1 iter_args(%arg1 = %7) -> (index, i32) {
    fir.store %arg1 to %4 : !fir.ref<i32>
    %9 = fir.load %1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
    %10:3 = fir.box_dims %9, %c0 : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
    %11 = fir.box_addr %9 : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
    %12 = fir.load %4 : !fir.ref<i32>
    %13 = fir.convert %12 : (i32) -> i64
    %14 = fir.convert %10#0 : (index) -> i64
    %15 = arith.subi %13, %14 : i64
    %16 = fir.coordinate_of %11, %15 : (!fir.heap<!fir.array<?xi32>>, i64) -> !fir.ref<i32>
    %17 = fir.load %16 : !fir.ref<i32>
    %18 = arith.subi %13, %c1_i64 : i64
    %19 = fir.coordinate_of %0, %18 {test.ptr = "arraya(ii)"} : (!fir.ref<!fir.array<1000xi32>>, i64) -> !fir.ref<i32>
    %20 = fir.load %19 : !fir.ref<i32>
    %21 = arith.addi %17, %20 : i32
    %22 = fir.load %2 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
    %23:3 = fir.box_dims %22, %c0 : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, index) -> (index, index, index)
    %24 = fir.convert %23#0 : (index) -> i64
    %25 = arith.subi %13, %24 : i64
    %26 = fir.coordinate_of %22, %25 {test.ptr = "arrayc(ii)"} : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, i64) -> !fir.ref<i32>
    fir.store %21 to %26 : !fir.ref<i32>
    %27 = arith.addi %arg0, %c1 : index
    %28 = fir.load %4 : !fir.ref<i32>
    %29 = arith.addi %28, %7 : i32
    fir.result %27, %29 : index, i32
  }
  fir.store %8#1 to %4 : !fir.ref<i32>
  return
}

// -----

// CHECK-LABEL: Testing : "_QMtest8Ptest"
// CHECK: arraya(ii)#0 <-> arrayc(ii)#0: NoAlias

// module test8
//   integer, dimension(1000) :: arrayA
//   integer, dimension(:), allocatable :: arrayB
//   integer, dimension(:), pointer :: arrayC
//   integer :: N
// contains
// subroutine test
//   do ii = 1, N
//     arrayC(ii) = arrayB(ii) + arrayA(ii)
//   end do
// end subroutine
// endmodule

fir.global @_QMtest8Earraya : !fir.array<1000xi32> 
fir.global @_QMtest8Earrayb : !fir.box<!fir.heap<!fir.array<?xi32>>> 
fir.global @_QMtest8Earrayc : !fir.box<!fir.ptr<!fir.array<?xi32>>> 
fir.global @_QMtest8En : i32 

func.func @_QMtest8Ptest() {
  %c1 = arith.constant 1 : index
  %c1_i64 = arith.constant 1 : i64
  %c0 = arith.constant 0 : index
  %0 = fir.address_of(@_QMtest8Earraya) : !fir.ref<!fir.array<1000xi32>>
  %1 = fir.address_of(@_QMtest8Earrayb) : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  %2 = fir.address_of(@_QMtest8Earrayc) : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
  %3 = fir.address_of(@_QMtest8En) : !fir.ref<i32>
  %4 = fir.alloca i32 {bindc_name = "ii", uniq_name = "_QMtest8FtestEii"}
  %5 = fir.load %3 : !fir.ref<i32>
  %6 = fir.convert %5 : (i32) -> index
  %7 = fir.convert %c1 : (index) -> i32
  %8:2 = fir.do_loop %arg0 = %c1 to %6 step %c1 iter_args(%arg1 = %7) -> (index, i32) {
    fir.store %arg1 to %4 : !fir.ref<i32>
    %9 = fir.load %1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
    %10:3 = fir.box_dims %9, %c0 : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
    %11 = fir.box_addr %9 : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
    %12 = fir.load %4 : !fir.ref<i32>
    %13 = fir.convert %12 : (i32) -> i64
    %14 = fir.convert %10#0 : (index) -> i64
    %15 = arith.subi %13, %14 : i64
    %16 = fir.coordinate_of %11, %15 : (!fir.heap<!fir.array<?xi32>>, i64) -> !fir.ref<i32>
    %17 = fir.load %16 : !fir.ref<i32>
    %18 = arith.subi %13, %c1_i64 : i64
    %19 = fir.coordinate_of %0, %18 {test.ptr = "arraya(ii)"} : (!fir.ref<!fir.array<1000xi32>>, i64) -> !fir.ref<i32>
    %20 = fir.load %19 : !fir.ref<i32>
    %21 = arith.addi %17, %20 : i32
    %22 = fir.load %2 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
    %23:3 = fir.box_dims %22, %c0 : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, index) -> (index, index, index)
    %24 = fir.convert %23#0 : (index) -> i64
    %25 = arith.subi %13, %24 : i64
    %26 = fir.coordinate_of %22, %25 {test.ptr = "arrayc(ii)"} : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, i64) -> !fir.ref<i32>
    fir.store %21 to %26 : !fir.ref<i32>
    %27 = arith.addi %arg0, %c1 : index
    %28 = fir.load %4 : !fir.ref<i32>
    %29 = arith.addi %28, %7 : i32
    fir.result %27, %29 : index, i32
  }
  fir.store %8#1 to %4 : !fir.ref<i32>
  return
}

// -----

// CHECK-LABEL: Testing : "_QMtest9Ptest"
// CHECK: arraya(ii)#0 <-> arrayc(ii)#0: NoAlias

// module test9
//   integer, dimension(:), allocatable :: arrayB
//   integer, dimension(:), pointer :: arrayC
//   integer :: N
// contains
// subroutine test(arrayA)
//   integer, dimension(:) :: arrayA
//   do ii = 1, N
//     arrayC(ii) = arrayB(ii) + arrayA(ii)
//   end do
// end subroutine
// endmodule

fir.global @_QMtest9Earrayb : !fir.box<!fir.heap<!fir.array<?xi32>>> 
fir.global @_QMtest9Earrayc : !fir.box<!fir.ptr<!fir.array<?xi32>>>
fir.global @_QMtest9En : i32

func.func @_QMtest9Ptest(%arg0: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "arraya"}) {
  %c1 = arith.constant 1 : index
  %c1_i64 = arith.constant 1 : i64
  %c0 = arith.constant 0 : index
  %0 = fir.address_of(@_QMtest9Earrayb) : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  %1 = fir.address_of(@_QMtest9Earrayc) : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
  %2 = fir.address_of(@_QMtest9En) : !fir.ref<i32>
  %3 = fir.alloca i32 {bindc_name = "ii", uniq_name = "_QMtest9FtestEii"}
  %4 = fir.load %2 : !fir.ref<i32>
  %5 = fir.convert %4 : (i32) -> index
  %6 = fir.convert %c1 : (index) -> i32
  %7:2 = fir.do_loop %arg1 = %c1 to %5 step %c1 iter_args(%arg2 = %6) -> (index, i32) {
    fir.store %arg2 to %3 : !fir.ref<i32>
    %8 = fir.load %0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
    %9:3 = fir.box_dims %8, %c0 : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
    %10 = fir.box_addr %8 : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
    %11 = fir.load %3 : !fir.ref<i32>
    %12 = fir.convert %11 : (i32) -> i64
    %13 = fir.convert %9#0 : (index) -> i64
    %14 = arith.subi %12, %13 : i64
    %15 = fir.coordinate_of %10, %14 : (!fir.heap<!fir.array<?xi32>>, i64) -> !fir.ref<i32>
    %16 = fir.load %15 : !fir.ref<i32>
    %17 = arith.subi %12, %c1_i64 : i64
    %18 = fir.coordinate_of %arg0, %17 {test.ptr = "arraya(ii)"} : (!fir.box<!fir.array<?xi32>>, i64) -> !fir.ref<i32>
    %19 = fir.load %18 : !fir.ref<i32>
    %20 = arith.addi %16, %19 : i32
    %21 = fir.load %1 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
    %22:3 = fir.box_dims %21, %c0 : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, index) -> (index, index, index)
    %23 = fir.convert %22#0 : (index) -> i64
    %24 = arith.subi %12, %23 : i64
    %25 = fir.coordinate_of %21, %24 {test.ptr = "arrayc(ii)"} : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, i64) -> !fir.ref<i32>
    fir.store %20 to %25 : !fir.ref<i32>
    %26 = arith.addi %arg1, %c1 : index
    %27 = fir.load %3 : !fir.ref<i32>
    %28 = arith.addi %27, %6 : i32
    fir.result %26, %28 : index, i32
  }
  fir.store %7#1 to %3 : !fir.ref<i32>
  return
}

// -----

// CHECK-LABEL: Testing : "_QMtest10Ptest"
// CHECK: arraya(ii)#0 <-> arrayc(ii)#0: MayAlias

// module test10
//   integer, dimension(:), allocatable :: arrayB
//   integer, dimension(:), pointer :: arrayC
//   integer :: N
// contains
// subroutine test(arrayA)
//   integer, dimension(:), target :: arrayA
//   do ii = 1, N
//     arrayC(ii) = arrayB(ii) + arrayA(ii)
//   end do
// end subroutine
// endmodule

fir.global @_QMtest10Earrayb : !fir.box<!fir.heap<!fir.array<?xi32>>> 
fir.global @_QMtest10Earrayc : !fir.box<!fir.ptr<!fir.array<?xi32>>> 
fir.global @_QMtest10En : i32

func.func @_QMtest10Ptest(%arg0: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "arraya", fir.target}) {
  %c1 = arith.constant 1 : index
  %c1_i64 = arith.constant 1 : i64
  %c0 = arith.constant 0 : index
  %0 = fir.address_of(@_QMtest10Earrayb) : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  %1 = fir.address_of(@_QMtest10Earrayc) : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
  %2 = fir.address_of(@_QMtest10En) : !fir.ref<i32>
  %3 = fir.alloca i32 {bindc_name = "ii", uniq_name = "_QMtest10FtestEii"}
  %4 = fir.load %2 : !fir.ref<i32>
  %5 = fir.convert %4 : (i32) -> index
  %6 = fir.convert %c1 : (index) -> i32
  %7:2 = fir.do_loop %arg1 = %c1 to %5 step %c1 iter_args(%arg2 = %6) -> (index, i32) {
    fir.store %arg2 to %3 : !fir.ref<i32>
    %8 = fir.load %0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
    %9:3 = fir.box_dims %8, %c0 : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
    %10 = fir.box_addr %8 : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
    %11 = fir.load %3 : !fir.ref<i32>
    %12 = fir.convert %11 : (i32) -> i64
    %13 = fir.convert %9#0 : (index) -> i64
    %14 = arith.subi %12, %13 : i64
    %15 = fir.coordinate_of %10, %14 : (!fir.heap<!fir.array<?xi32>>, i64) -> !fir.ref<i32>
    %16 = fir.load %15 : !fir.ref<i32>
    %17 = arith.subi %12, %c1_i64 : i64
    %18 = fir.coordinate_of %arg0, %17 {test.ptr = "arraya(ii)"} : (!fir.box<!fir.array<?xi32>>, i64) -> !fir.ref<i32>
    %19 = fir.load %18 : !fir.ref<i32>
    %20 = arith.addi %16, %19 : i32
    %21 = fir.load %1 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
    %22:3 = fir.box_dims %21, %c0 : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, index) -> (index, index, index)
    %23 = fir.convert %22#0 : (index) -> i64
    %24 = arith.subi %12, %23 : i64
    %25 = fir.coordinate_of %21, %24 {test.ptr = "arrayc(ii)"} : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, i64) -> !fir.ref<i32>
    fir.store %20 to %25 : !fir.ref<i32>
    %26 = arith.addi %arg1, %c1 : index
    %27 = fir.load %3 : !fir.ref<i32>
    %28 = arith.addi %27, %6 : i32
    fir.result %26, %28 : index, i32
  }
  fir.store %7#1 to %3 : !fir.ref<i32>
  return
}

// -----

// CHECK-LABEL: Testing : "_QMtest11Ptest"
// CHECK: arraya(ii)#0 <-> arrayc(ii)#0: MayAlias

// module test11
//   integer, dimension(:), allocatable :: arrayB
//   integer, dimension(1000), target :: arrayC
//   integer :: N
// contains
// subroutine test(arrayA)
//   integer, dimension(:), target :: arrayA
//   do ii = 1, N
//     arrayC(ii) = arrayB(ii) + arrayA(ii)
//   end do
// end subroutine
// endmodule

fir.global @_QMtest11Earrayb : !fir.box<!fir.heap<!fir.array<?xi32>>> 
fir.global @_QMtest11Earrayc target : !fir.array<1000xi32> 
fir.global @_QMtest11En : i32 

func.func @_QMtest11Ptest(%arg0: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "arraya", fir.target}) {
  %c1 = arith.constant 1 : index
  %c1_i64 = arith.constant 1 : i64
  %c0 = arith.constant 0 : index
  %0 = fir.address_of(@_QMtest11Earrayb) : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  %1 = fir.address_of(@_QMtest11Earrayc) : !fir.ref<!fir.array<1000xi32>>
  %2 = fir.address_of(@_QMtest11En) : !fir.ref<i32>
  %3 = fir.alloca i32 {bindc_name = "ii", uniq_name = "_QMtest11FtestEii"}
  %4 = fir.load %2 : !fir.ref<i32>
  %5 = fir.convert %4 : (i32) -> index
  %6 = fir.convert %c1 : (index) -> i32
  %7:2 = fir.do_loop %arg1 = %c1 to %5 step %c1 iter_args(%arg2 = %6) -> (index, i32) {
    fir.store %arg2 to %3 : !fir.ref<i32>
    %8 = fir.load %0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
    %9:3 = fir.box_dims %8, %c0 : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
    %10 = fir.box_addr %8 : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
    %11 = fir.load %3 : !fir.ref<i32>
    %12 = fir.convert %11 : (i32) -> i64
    %13 = fir.convert %9#0 : (index) -> i64
    %14 = arith.subi %12, %13 : i64
    %15 = fir.coordinate_of %10, %14 : (!fir.heap<!fir.array<?xi32>>, i64) -> !fir.ref<i32>
    %16 = fir.load %15 : !fir.ref<i32>
    %17 = arith.subi %12, %c1_i64 : i64
    %18 = fir.coordinate_of %arg0, %17 {test.ptr = "arraya(ii)"} : (!fir.box<!fir.array<?xi32>>, i64) -> !fir.ref<i32>
    %19 = fir.load %18 : !fir.ref<i32>
    %20 = arith.addi %16, %19 : i32
    %21 = fir.coordinate_of %1, %17 {test.ptr = "arrayc(ii)"} : (!fir.ref<!fir.array<1000xi32>>, i64) -> !fir.ref<i32>
    fir.store %20 to %21 : !fir.ref<i32>
    %22 = arith.addi %arg1, %c1 : index
    %23 = fir.load %3 : !fir.ref<i32>
    %24 = arith.addi %23, %6 : i32
    fir.result %22, %24 : index, i32
  }
  fir.store %7#1 to %3 : !fir.ref<i32>
  return
}


// -----

// CHECK-LABEL: Testing : "_QMtest12Ptest"
// CHECK: arraya(ii)#0 <-> arrayc(ii)#0: NoAlias

// module test12
//   integer, dimension(:), allocatable :: arrayB
//   integer, dimension(1000) :: arrayC
//   integer :: N
// contains
// subroutine test(arrayA)
//   integer, dimension(:), target :: arrayA
//   do ii = 1, N
//     arrayC(ii) = arrayB(ii) + arrayA(ii)
//   end do
// end subroutine
// endmodule

fir.global @_QMtest12Earrayb : !fir.box<!fir.heap<!fir.array<?xi32>>>
fir.global @_QMtest12Earrayc : !fir.array<1000xi32> 
fir.global @_QMtest12En : i32

func.func @_QMtest12Ptest(%arg0: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "arraya", fir.target}) {
  %c1 = arith.constant 1 : index
  %c1_i64 = arith.constant 1 : i64
  %c0 = arith.constant 0 : index
  %0 = fir.address_of(@_QMtest12Earrayb) : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  %1 = fir.address_of(@_QMtest12Earrayc) : !fir.ref<!fir.array<1000xi32>>
  %2 = fir.address_of(@_QMtest12En) : !fir.ref<i32>
  %3 = fir.alloca i32 {bindc_name = "ii", uniq_name = "_QMtest12FtestEii"}
  %4 = fir.load %2 : !fir.ref<i32>
  %5 = fir.convert %4 : (i32) -> index
  %6 = fir.convert %c1 : (index) -> i32
  %7:2 = fir.do_loop %arg1 = %c1 to %5 step %c1 iter_args(%arg2 = %6) -> (index, i32) {
    fir.store %arg2 to %3 : !fir.ref<i32>
    %8 = fir.load %0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
    %9:3 = fir.box_dims %8, %c0 : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
    %10 = fir.box_addr %8 : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
    %11 = fir.load %3 : !fir.ref<i32>
    %12 = fir.convert %11 : (i32) -> i64
    %13 = fir.convert %9#0 : (index) -> i64
    %14 = arith.subi %12, %13 : i64
    %15 = fir.coordinate_of %10, %14 : (!fir.heap<!fir.array<?xi32>>, i64) -> !fir.ref<i32>
    %16 = fir.load %15 : !fir.ref<i32>
    %17 = arith.subi %12, %c1_i64 : i64
    %18 = fir.coordinate_of %arg0, %17 {test.ptr = "arraya(ii)"} : (!fir.box<!fir.array<?xi32>>, i64) -> !fir.ref<i32>
    %19 = fir.load %18 : !fir.ref<i32>
    %20 = arith.addi %16, %19 : i32
    %21 = fir.coordinate_of %1, %17 {test.ptr = "arrayc(ii)"} : (!fir.ref<!fir.array<1000xi32>>, i64) -> !fir.ref<i32>
    fir.store %20 to %21 : !fir.ref<i32>
    %22 = arith.addi %arg1, %c1 : index
    %23 = fir.load %3 : !fir.ref<i32>
    %24 = arith.addi %23, %6 : i32
    fir.result %22, %24 : index, i32
  }
  fir.store %7#1 to %3 : !fir.ref<i32>
  return
}

