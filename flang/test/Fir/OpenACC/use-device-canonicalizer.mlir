// RUN: fir-opt %s --acc-use-device-canonicalizer -split-input-file | FileCheck %s

// -----

// Test hoisting of load/box_addr/convert pattern out of acc.host_data with function call
func.func @test_host_data_hoisting_function_call(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>) {
  // CHECK: %[[LOADED:.*]] = fir.load %arg0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>
  // CHECK: %[[ADDR:.*]] = fir.box_addr %[[LOADED]] : (!fir.box<!fir.heap<!fir.array<?xf64>>>) -> !fir.heap<!fir.array<?xf64>>
  // CHECK: %[[DEV_PTR:.*]] = acc.use_device varPtr(%[[ADDR]] : !fir.heap<!fir.array<?xf64>>) varType(!fir.box<!fir.heap<!fir.array<?xf64>>>) -> !fir.heap<!fir.array<?xf64>>
  // CHECK: acc.host_data dataOperands(%[[DEV_PTR]]
  // CHECK: %[[EMBOX:.*]] = fir.embox %[[DEV_PTR]]
  // CHECK: %[[ALLOCA:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xf64>>>
  // CHECK: fir.store %[[EMBOX]] to %[[ALLOCA]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>
  // CHECK: %[[LOAD2:.*]] = fir.load %[[ALLOCA]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>
  // CHECK: %[[DEV_PTRADDR:.*]] = fir.box_addr %[[LOAD2]] : (!fir.box<!fir.heap<!fir.array<?xf64>>>
  // CHECK: %[[CONV:.*]] = fir.convert %[[DEV_PTRADDR]] : (!fir.heap<!fir.array<?xf64>>) -> !fir.ref<!fir.array<?xf64>>
  // CHECK: fir.call @_QMmPvadd(%[[CONV]]
  %dev_ptr = acc.use_device varPtr(%arg0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>> {name = "a"}
  acc.host_data dataOperands(%dev_ptr : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>) {
    %loaded = fir.load %dev_ptr : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>
    %addr = fir.box_addr %loaded : (!fir.box<!fir.heap<!fir.array<?xf64>>>) -> !fir.heap<!fir.array<?xf64>>
    %conv = fir.convert %addr : (!fir.heap<!fir.array<?xf64>>) -> !fir.ref<!fir.array<?xf64>>
    fir.call @_QMmPvadd(%conv, %conv) : (!fir.ref<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>) -> ()
    acc.terminator
  }
  return
}

// -----

// Test hoisting of load/box_addr/convert pattern out of acc.host_data with load operation
func.func @test_host_data_hoisting_load(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>) {
  // CHECK: %[[LOADED:.*]] = fir.load %arg0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>
  // CHECK: %[[ADDR:.*]] = fir.box_addr %[[LOADED]] : (!fir.box<!fir.heap<!fir.array<?xf64>>>) -> !fir.heap<!fir.array<?xf64>>
  // CHECK: %[[DEV_PTR:.*]] = acc.use_device varPtr(%[[ADDR]] :
  // CHECK: acc.host_data dataOperands(%[[DEV_PTR]]
  // CHECK: %[[EMBOX:.*]] = fir.embox %[[DEV_PTR]]
  // CHECK: %[[ALLOCA:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xf64>>>
  // CHECK: fir.store %[[EMBOX]] to %[[ALLOCA]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>
  // CHECK: %[[LOAD2:.*]] = fir.load %[[ALLOCA]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>
  // CHECK: %[[DEV_PTRADDR:.*]] = fir.box_addr %[[LOAD2]] : (!fir.box<!fir.heap<!fir.array<?xf64>>>
  // CHECK: %[[CONV:.*]] = fir.convert %[[DEV_PTRADDR]] : (!fir.heap<!fir.array<?xf64>>) -> !fir.ref<!fir.array<?xf64>>
  // CHECK: %[[VAL:.*]] = fir.load %[[CONV]]
  %dev_ptr = acc.use_device varPtr(%arg0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>> {name = "a"}
  acc.host_data dataOperands(%dev_ptr : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>) {
    %loaded = fir.load %dev_ptr : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>
    %addr = fir.box_addr %loaded : (!fir.box<!fir.heap<!fir.array<?xf64>>>) -> !fir.heap<!fir.array<?xf64>>
    %conv = fir.convert %addr : (!fir.heap<!fir.array<?xf64>>) -> !fir.ref<!fir.array<?xf64>>
    %val = fir.load %conv : !fir.ref<!fir.array<?xf64>>
    fir.call @foo(%val) : (!fir.array<?xf64>) -> ()
    acc.terminator
  }
  return
}

// -----

// Test hoisting for pointer attributes: load/box_addr hoisted, remove additional
// unused use_device clause for a different variable
func.func @test_host_data_hoisting_ref_to_box() {
  %1 = fir.alloca !fir.box<!fir.ptr<i32>> {bindc_name = "ptr", uniq_name = "_QFEptr"}
  // CHECK: %[[ALLOCA:.*]] = fir.alloca !fir.box<!fir.ptr<i32>> {bindc_name = "ptr", uniq_name = "_QFEptr"}
  %4 = fir.declare %1 {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFEptr"} : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> !fir.ref<!fir.box<!fir.ptr<i32>>>
  // CHECK: %[[DECLARE:.*]] = fir.declare %[[ALLOCA]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFEptr"} : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> !fir.ref<!fir.box<!fir.ptr<i32>>>
  // Second pointer variable (unused in host_data region)
  %ptr2_alloca = fir.alloca !fir.box<!fir.ptr<i32>> {bindc_name = "ptr2", uniq_name = "_QFEptr2"}
  %ptr2_decl = fir.declare %ptr2_alloca {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFEptr2"} : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> !fir.ref<!fir.box<!fir.ptr<i32>>>
  %5 = fir.address_of(@_QFEtgt) : !fir.ref<i32>
  %6 = fir.declare %5 {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFEtgt"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %8 = fir.embox %6 : (!fir.ref<i32>) -> !fir.box<!fir.ptr<i32>>
  fir.store %8 to %4 : !fir.ref<!fir.box<!fir.ptr<i32>>>
  fir.store %8 to %ptr2_decl : !fir.ref<!fir.box<!fir.ptr<i32>>>
  // CHECK: %[[LOAD:.*]] = fir.load %[[DECLARE]]
  // CHECK: %[[BOXADDR:.*]] = fir.box_addr %[[LOAD]]
  // CHECK: %[[DEV_PTR:.*]] = acc.use_device varPtr(%[[BOXADDR]] : !fir.ptr<i32>) varType(!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32> {name = "ptr"}
  // CHECK: acc.host_data dataOperands(%[[DEV_PTR]] : !fir.ptr<i32>) {
  // CHECK: %[[EMBOX:.*]] = fir.embox %[[DEV_PTR]] : (!fir.ptr<i32>) -> !fir.box<!fir.ptr<i32>>
  // CHECK: %[[ALLOCA2:.*]] = fir.alloca !fir.box<!fir.ptr<i32>>
  // CHECK: fir.store %[[EMBOX]] to %[[ALLOCA2]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
  // CHECK: %[[LOAD2:.*]] = fir.load %[[ALLOCA2]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
  // CHECK: %[[DEV_PTRADDR:.*]] = fir.box_addr %[[LOAD2]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
  // CHECK: %[[CONV:.*]] = fir.convert %[[DEV_PTRADDR]] : (!fir.ptr<i32>) -> i64
  // CHECK: fir.call @foo(%[[CONV]]) : (i64) -> ()
  %9 = acc.use_device varPtr(%4 : !fir.ref<!fir.box<!fir.ptr<i32>>>) -> !fir.ref<!fir.box<!fir.ptr<i32>>> {name = "ptr"}
  // This use_device clause is for a different variable (ptr2) and has no uses - should be removed
  %12 = acc.use_device varPtr(%ptr2_decl : !fir.ref<!fir.box<!fir.ptr<i32>>>) -> !fir.ref<!fir.box<!fir.ptr<i32>>> {name = "ptr2"}
  acc.host_data dataOperands(%9, %12 : !fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.box<!fir.ptr<i32>>>) {
    %14 = fir.load %9 : !fir.ref<!fir.box<!fir.ptr<i32>>>
    %15 = fir.box_addr %14 : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
    %16 = fir.convert %15 : (!fir.ptr<i32>) -> i64
    fir.call @foo(%16) : (i64) -> ()
    acc.terminator
  }
  return
}

