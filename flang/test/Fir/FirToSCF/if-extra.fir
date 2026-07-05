// Test conversions of fir.if (extra coverage vs upstream `if.fir`).
//
// RUN: fir-opt %s --fir-to-scf --allow-unregistered-dialect | FileCheck %s

// Two sided if conversion
//  subroutine double(x)
//    integer :: x
//    if (x .lt. 0) then
//      x = 0
//    else
//      x = 1
//    endif
//  end

// CHECK-LABEL: func.func @double
// CHECK:       [[COND:%[0-9]+]] = arith.cmpi
// CHECK-NEXT:  scf.if [[COND]] {
// CHECK-NEXT:  fir.store
// CHECK-NEXT:  } else {
// CHECK-NEXT:  fir.store
// CHECK-NEXT:  }

func.func @double(%arg0: !fir.ref<i32> {fir.bindc_name = "x"}) {
  %c1_i32 = arith.constant 1 : i32
  %c0_i32 = arith.constant 0 : i32
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.declare %arg0 dummy_scope %0 {uniq_name = "double"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %2 = fir.load %1 : !fir.ref<i32>
  %3 = arith.cmpi slt, %2, %c0_i32 : i32
  fir.if %3 {
    fir.store %c0_i32 to %1 : !fir.ref<i32>
  } else {
    fir.store %c1_i32 to %1 : !fir.ref<i32>
  }
  return
}

// One sided if conversion
// subroutine single(x)
//   integer :: x
//   if (x .gt. 0) x = 0
// end

// CHECK-LABEL: func.func @single
// CHECK:       [[COND:%[0-9]+]] = arith.cmpi
// CHECK-NEXT:  scf.if [[COND]] {
// CHECK-NEXT:  fir.store
// CHECK-NEXT:  } else {
// CHECK-NEXT:  }

func.func @single(%arg0: !fir.ref<i32> {fir.bindc_name = "x"}) {
  %c0_i32 = arith.constant 0 : i32
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.declare %arg0 dummy_scope %0 {uniq_name = "single"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %2 = fir.load %1 : !fir.ref<i32>
  %3 = arith.cmpi sgt, %2, %c0_i32 : i32
  fir.if %3 {
    fir.store %c0_i32 to %1 : !fir.ref<i32>
  } else {
  }
  return
}

// Result if conversion (written by hand)

// CHECK-LABEL: func.func @if_result
// CHECK:       [[COND:%[0-9]+]] = arith.cmpi
// CHECK-NEXT:  [[RES:%[0-9]+]] = scf.if [[COND]] -> (i32) {
// CHECK-NEXT:    [[CON5:%.+]] = arith.constant 5 : i32
// CHECK-NEXT:    scf.yield [[CON5]]
// CHECK-NEXT:  } else {
// CHECK-NEXT:    [[CON3:%.+]] = arith.constant 3 : i32
// CHECK-NEXT:    scf.yield [[CON3]]
// CHECK-NEXT:  }
// CHECK-NEXT:  return [[RES]] : i32

func.func @if_result(%arg0: !fir.ref<i32> {fir.bindc_name = "c"}) -> i32 {
  %c0 = arith.constant 0 : i32
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.declare %arg0 dummy_scope %0 {uniq_name = "c"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %2 = fir.load %1 : !fir.ref<i32>
  %3 = arith.cmpi sgt, %2, %c0 : i32
  %4 = fir.if %3 -> i32 {
    %c5 = arith.constant 5 : i32
    fir.result %c5 : i32
  } else {
    %c3 = arith.constant 3 : i32
    fir.result %c3 : i32
  }
  return %4 : i32
}

// CHECK-LABEL: func.func @if_result_multi
// CHECK:       [[COND:%[0-9]+]] = arith.cmpi
// CHECK-NEXT:  [[IF:%[0-9]+]]:2 = scf.if [[COND]] -> (i32, i32) {
// CHECK-NEXT:    [[CON5:%.+]] = arith.constant 5 : i32
// CHECK-NEXT:    scf.yield [[CON5]]
// CHECK-NEXT:  } else {
// CHECK-NEXT:    [[CON3:%.+]] = arith.constant 3 : i32
// CHECK-NEXT:    scf.yield [[CON3]]
// CHECK-NEXT:  }
// CHECK-NEXT:  [[RES:%[0-9]+]] = arith.addi [[IF]]#0, [[IF]]#1
// CHECK-NEXT:  return [[RES]] : i32

func.func @if_result_multi(%arg0: !fir.ref<i32> {fir.bindc_name = "c"}) -> i32 {
  %c0 = arith.constant 0 : i32
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.declare %arg0 dummy_scope %0 {uniq_name = "c"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %2 = fir.load %1 : !fir.ref<i32>
  %3 = arith.cmpi sgt, %2, %c0 : i32
  %4:2 = fir.if %3 -> (i32, i32) {
    %c5 = arith.constant 5 : i32
    fir.result %c5, %c5 : i32, i32
  } else {
    %c3 = arith.constant 3 : i32
    fir.result %c3, %c3 : i32, i32
  }
  %5 = arith.addi %4#0, %4#1 : i32
  return %5 : i32
}


