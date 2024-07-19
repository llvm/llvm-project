// Use --mlir-disable-threading so that the AA queries are serialized
// as well as its diagnostic output.
// RUN: fir-opt %s -pass-pipeline='builtin.module(func.func(test-fir-alias-analysis))' -split-input-file --mlir-disable-threading 2>&1 | FileCheck %s

// CHECK-LABEL: Testing : "_QFPtest"

// p1.addr and p2.addr result from 2 different allocas
// They cannot physically alias
// CHECK-DAG: p1.addr#0 <-> p2.addr#0: NoAlias

// p1.addr and p2.addr could both be wrapped inside boxes
// CHECK-DAG: p1.addr#0 <-> boxp1.addr#0: MayAlias
// CHECK-DAG: p2.addr#0 <-> boxp1.addr#0: MayAlias

// TODO: To really see aliasing, we should be looking at a load of p1.addr
// p1.addr is just a local address holding the address to the data 
// CHECK-DAG: p1.addr#0 <-> arg2.addr#0: NoAlias
// CHECK-DAG: p2.addr#0 <-> arg2.addr#0: NoAlias

// p1.addr and p2.addr are the result of an allocation
// They cannot physically alias with an argument
// CHECK-DAG: p1.addr#0 <-> func.region0#0: NoAlias
// CHECK-DAG: p2.addr#0 <-> func.region0#0: NoAlias
// CHECK-DAG: p1.addr#0 <-> func.region0#1: NoAlias
// CHECK-DAG: p2.addr#0 <-> func.region0#1: NoAlias
// CHECK-DAG: p1.addr#0 <-> func.region0#2: NoAlias
// CHECK-DAG: p2.addr#0 <-> func.region0#2: NoAlias

// All arguments are either pointers or targets
// A pointer in a box may alias with both
// CHECK-DAG: boxp1.addr#0 <-> func.region0#0: MayAlias
// CHECK-DAG: boxp1.addr#0 <-> func.region0#1: MayAlias
// CHECK-DAG: boxp1.addr#0 <-> func.region0#2: MayAlias

// A target dummy may alias with another target
// CHECK-DAG: func.region0#0 <-> func.region0#1: MayAlias

// arg2 is a reference to a pointer. Modifying arg2 could
// modify a target with a pointer component
// CHECK-DAG: arg2.load#0 <-> func.region0#0: MayAlias
// CHECK-DAG: arg2.load#0 <-> func.region0#1: MayAlias

// However, the address wrapped by arg2, can alias with any target or
// pointer arguments
// CHECK-DAG: arg2.addr#0 <-> func.region0#0: MayAlias
// CHECK-DAG: arg2.addr#0 <-> func.region0#1: MayAlias
// CHECK-DAG: arg2.load#0 <-> arg2.addr#0: MustAlias
// CHECK-DAG: boxp1.addr#0 <-> arg2.addr#0: MayAlias

func.func @_QFPtest(%arg0: !fir.ref<f32> {fir.bindc_name = "v1", fir.target}, %arg1: !fir.ref<f32> {fir.bindc_name = "v2", fir.target}, %arg2: !fir.ref<!fir.box<!fir.ptr<f32>>> ) attributes {test.ptr = "func"} {

  %1 = fir.alloca !fir.ptr<f32> {test.ptr = "p1.addr"}
  %2 = fir.zero_bits !fir.ptr<f32>
  fir.store %2 to %1 : !fir.ref<!fir.ptr<f32>>

  %4 = fir.alloca !fir.ptr<f32> {test.ptr = "p2.addr"}
  fir.store %2 to %4 : !fir.ref<!fir.ptr<f32>>

  %5 = fir.convert %arg0 : (!fir.ref<f32>) -> !fir.ptr<f32>
  fir.store %5 to %1 : !fir.ref<!fir.ptr<f32>>

  %6 = fir.convert %arg1 : (!fir.ref<f32>) -> !fir.ptr<f32>
  fir.store %6 to %4 : !fir.ref<!fir.ptr<f32>>

  %0 = fir.alloca !fir.box<!fir.ptr<f32>> {bindc_name = "p1", uniq_name = "_QFtestEp1"}
  %7 = fir.load %1 : !fir.ref<!fir.ptr<f32>>
  %8 = fir.embox %7 : (!fir.ptr<f32>) -> !fir.box<!fir.ptr<f32>>
  fir.store %8 to %0 : !fir.ref<!fir.box<!fir.ptr<f32>>>

  %3 = fir.alloca !fir.box<!fir.ptr<f32>> {bindc_name = "p2", uniq_name = "_QFtestEp2"}
  %9 = fir.load %4 : !fir.ref<!fir.ptr<f32>>
  %10 = fir.embox %9 : (!fir.ptr<f32>) -> !fir.box<!fir.ptr<f32>>
  fir.store %10 to %3 : !fir.ref<!fir.box<!fir.ptr<f32>>>

  %11 = fir.load %0 : !fir.ref<!fir.box<!fir.ptr<f32>>>
  %12 = fir.box_addr %11 {test.ptr = "boxp1.addr"} : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
  fir.store %12 to %1 : !fir.ref<!fir.ptr<f32>>

  %13 = fir.load %3 : !fir.ref<!fir.box<!fir.ptr<f32>>>
  %14 = fir.box_addr %13 : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
  fir.store %14 to %4 : !fir.ref<!fir.ptr<f32>>

  %15 = fir.load %arg2 {test.ptr = "arg2.load"} : !fir.ref<!fir.box<!fir.ptr<f32>>>
  %16 = fir.box_addr %15 {test.ptr = "arg2.addr"} : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
  return
}

// -----

// CHECK-LABEL: Testing : "_QFPtest2"

// subroutine test2(v1,p1,p2)
//   real, target :: v1
//   real, pointer :: p1, p2
//   ...
// end subroutine

// Direct access to dummy POINTER references can modify other dummy POINTER references
// CHECK-DAG: func.region0#1 <-> func.region0#2: MayAlias

// They can also modify targets that have pointer components
// CHECK-DAG: arg1.load#0 <-> func.region0#0: MayAlias
// CHECK-DAG: arg2.load#0 <-> func.region0#0: MayAlias

func.func @_QFPtest2(%arg0: !fir.ref<f32> {fir.bindc_name = "v1", fir.target}, %arg1: !fir.ref<!fir.box<!fir.ptr<f32>>>, %arg2: !fir.ref<!fir.box<!fir.ptr<f32>>> ) attributes {test.ptr = "func"} {
  %0 = fir.load %arg1 {test.ptr = "arg1.load"} : !fir.ref<!fir.box<!fir.ptr<f32>>>
  %1 = fir.load %arg2 {test.ptr = "arg2.load"} : !fir.ref<!fir.box<!fir.ptr<f32>>>
  return
}

// -----

// CHECK-LABEL: Testing : "_QFPtest3"

// module pointers
//   real, pointer :: p
// end module
//
// program main
//   use pointers
//   real, target :: var1 = 1, var2 =2
//   p => var1
//
//   call test3(p)
//
// contains
//   subroutine test3(p1)
//     real, pointer :: p1
//     p1 => var2
//     print *, p
//   end subroutine
// end

// The global pointer p may alias with the dummy argument p1
// but not with the dummy arg1 which is just a regular dummy
// CHECK-DAG: p#0 <-> func.region0#0: MayAlias
// CHECK-DAG: p#0 <-> func.region0#1: NoAlias

// p could be pointing to var2 
// var2, being a target, could also be passed as argument arg0 

// This was the wrong question to ask. We are asking if the address of box _QMpointersEp can 
// alias with the wrapped scalar _QFEvar2. We meant box_addr of _QMpointersEp 
// CHECK-DAG: p#0 <-> box.addr#0: NoAlias

// Handling gracefully the difference between !fir.ref<!fir.box<>> and !fir.box<> 
// CHECK-DAG: box.addr#0 <-> func.region0#0: NoAlias

// var2, although it is a target, cannot alias with p
// A modification of p would only make them point to a new target but not modify it
// CHECK-DAG: var2#0 <-> p#0: NoAlias
// It can alias with p1, if p1 is a pointer component
// CHECK-DAG: arg0.load#0 <-> var2#0: MayAlias
// It is the same as box.addr
// CHECK-DAG: var2#0 <-> box.addr#0: MustAlias

// A global may not alias with a dummy
// CHECK-DAG: var2#0 <-> func.region0#1: NoAlias

// A pointer may only alias with a target but arg1 is a regular dummy
// CHECK-DAG: box.addr#0 <-> func.region0#1: NoAlias

// Dummy argument do not alias
// CHECK-DAG: func.region0#0 <-> func.region0#1: NoAlias

fir.global @_QMpointersEp : !fir.box<!fir.ptr<f32>> {
  %0 = fir.zero_bits !fir.ptr<f32>
  %1 = fir.embox %0 : (!fir.ptr<f32>) -> !fir.box<!fir.ptr<f32>>
  fir.has_value %1 : !fir.box<!fir.ptr<f32>>
}

fir.global internal @_QFEvar2 target : f32 {
  %cst = arith.constant 2.000000e+00 : f32
  fir.has_value %cst : f32
}

func.func @_QFPtest3(%arg0: !fir.ref<!fir.box<!fir.ptr<f32>>> {fir.bindc_name = "p1"}, %arg1: !fir.ref<f32>) attributes {test.ptr = "func"} {
  %3 = fir.load %arg0 {test.ptr = "arg0.load"}: !fir.ref<!fir.box<!fir.ptr<f32>>>
  %4 = fir.address_of(@_QFEvar2) {test.ptr = "var2"} : !fir.ref<f32>
  %5 = fir.address_of(@_QMpointersEp) {test.ptr = "p"} : !fir.ref<!fir.box<!fir.ptr<f32>>>
  %6 = fir.embox %4 : (!fir.ref<f32>) -> !fir.box<!fir.ptr<f32>>
  %13 = fir.box_addr %6 {test.ptr = "box.addr"} : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
  return
}

// -----

// CHECK-LABEL: Testing : "_QFPtest4"

// Same as test3 but check that the alias analysis can follow (hl)fir.declare
// operations

// CHECK-DAG: p#0 <-> func.region0#0: MayAlias
// CHECK-DAG: p_fir#0 <-> func.region0#0: MayAlias
// CHECK-DAG: p_hlfir#0 <-> func.region0#0: MayAlias
// CHECK-DAG: p_hlfir#1 <-> func.region0#0: MayAlias

// CHECK-DAG: p#0 <-> func.region0#1: NoAlias
// CHECK-DAG: p_fir#0 <-> func.region0#1: NoAlias
// CHECK-DAG: p_hlfir#0 <-> func.region0#1: NoAlias
// CHECK-DAG: p_hlfir#1 <-> func.region0#1: NoAlias

// CHECK-DAG: var2#0 <-> p#0: NoAlias
// CHECK-DAG: var2#0 <-> p_fir#0: NoAlias
// CHECK-DAG: var2#0 <-> p_hlfir#0: NoAlias
// CHECK-DAG: var2#0 <-> p_hlfir#1: NoAlias
// CHECK-DAG: var2_fir#0 <-> p#0: NoAlias
// CHECK-DAG: var2_fir#0 <-> p_fir#0: NoAlias
// CHECK-DAG: var2_fir#0 <-> p_hlfir#0: NoAlias
// CHECK-DAG: var2_fir#0 <-> p_hlfir#1: NoAlias
// CHECK-DAG: var2_hlfir#0 <-> p#0: NoAlias
// CHECK-DAG: var2_hlfir#0 <-> p_fir#0: NoAlias
// CHECK-DAG: var2_hlfir#0 <-> p_hlfir#0: NoAlias
// CHECK-DAG: var2_hlfir#0 <-> p_hlfir#1: NoAlias
// CHECK-DAG: var2_hlfir#1 <-> p#0: NoAlias
// CHECK-DAG: var2_hlfir#1 <-> p_fir#0: NoAlias
// CHECK-DAG: var2_hlfir#1 <-> p_hlfir#0: NoAlias
// CHECK-DAG: var2_hlfir#1 <-> p_hlfir#1: NoAlias

// The data cannot alias with the box references
// CHECK-DAG: var2#0 <-> func.region0#0: NoAlias
// CHECK-DAG: var2_fir#0 <-> func.region0#0: NoAlias
// CHECK-DAG: var2_hlfir#0 <-> func.region0#0: NoAlias
// CHECK-DAG: var2_hlfir#1 <-> func.region0#0: NoAlias

// But it can alias with the box's own data
// CHECK-DAG: arg0.load#0 <-> var2#0: MayAlias
// CHECK-DAG: arg0.load#0 <-> var2_fir#0: MayAlias
// CHECK-DAG: arg0.load#0 <-> var2_hlfir#0: MayAlias
// CHECK-DAG: arg0.load#0 <-> var2_hlfir#1: MayAlias

// CHECK-DAG: var2#0 <-> box.addr#0: MustAlias
// CHECK-DAG: var2#0 <-> box.addr_fir#0: MustAlias
// CHECK-DAG: var2#0 <-> box.addr_hlfir#0: MustAlias
// CHECK-DAG: var2#0 <-> box.addr_hlfir#1: MustAlias
// CHECK-DAG: var2_fir#0 <-> box.addr#0: MustAlias
// CHECK-DAG: var2_fir#0 <-> box.addr_fir#0: MustAlias
// CHECK-DAG: var2_fir#0 <-> box.addr_hlfir#0: MustAlias
// CHECK-DAG: var2_fir#0 <-> box.addr_hlfir#1: MustAlias
// CHECK-DAG: var2_hlfir#0 <-> box.addr#0: MustAlias
// CHECK-DAG: var2_hlfir#0 <-> box.addr_fir#0: MustAlias
// CHECK-DAG: var2_hlfir#0 <-> box.addr_hlfir#0: MustAlias
// CHECK-DAG: var2_hlfir#0 <-> box.addr_hlfir#1: MustAlias
// CHECK-DAG: var2_hlfir#1 <-> box.addr#0: MustAlias
// CHECK-DAG: var2_hlfir#1 <-> box.addr_fir#0: MustAlias
// CHECK-DAG: var2_hlfir#1 <-> box.addr_hlfir#0: MustAlias
// CHECK-DAG: var2_hlfir#1 <-> box.addr_hlfir#1: MustAlias

// CHECK-DAG: var2#0 <-> func.region0#1: NoAlias
// CHECK-DAG: var2_fir#0 <-> func.region0#1: NoAlias
// CHECK-DAG: var2_hlfir#0 <-> func.region0#1: NoAlias
// CHECK-DAG: var2_hlfir#1 <-> func.region0#1: NoAlias

// CHECK-DAG: func.region0#0 <-> func.region0#1: NoAlias

fir.global @_QMpointersEp : !fir.box<!fir.ptr<f32>> {
  %0 = fir.zero_bits !fir.ptr<f32>
  %1 = fir.embox %0 : (!fir.ptr<f32>) -> !fir.box<!fir.ptr<f32>>
  fir.has_value %1 : !fir.box<!fir.ptr<f32>>
}

fir.global internal @_QFEvar2 target : f32 {
  %cst = arith.constant 2.000000e+00 : f32
  fir.has_value %cst : f32
}

func.func @_QFPtest4(%arg0: !fir.ref<!fir.box<!fir.ptr<f32>>> {fir.bindc_name = "p1"}, %arg1: !fir.ref<f32>) attributes {test.ptr = "func"} {
  %3 = fir.load %arg0 {test.ptr = "arg0.load"} : !fir.ref<!fir.box<!fir.ptr<f32>>>
  %4 = fir.address_of(@_QFEvar2) {test.ptr = "var2"} : !fir.ref<f32>
  %fir_decl_var2 = fir.declare %4 {uniq_name = "var2_fir", test.ptr = "var2_fir"}: (!fir.ref<f32>) -> !fir.ref<f32>
  %hlfir_decl_var2:2 = hlfir.declare %4 {uniq_name = "var2_hlfir", test.ptr = "var2_hlfir"}: (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  %5 = fir.address_of(@_QMpointersEp) {test.ptr = "p"} : !fir.ref<!fir.box<!fir.ptr<f32>>>
  %fir_decl_p = fir.declare %5 {uniq_name = "p_fir", test.ptr = "p_fir"}: (!fir.ref<!fir.box<!fir.ptr<f32>>>) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
  %hlfir_decl_p:2 = hlfir.declare %5 {uniq_name = "p_hlfir", test.ptr = "p_hlfir"}: (!fir.ref<!fir.box<!fir.ptr<f32>>>) -> (!fir.ref<!fir.box<!fir.ptr<f32>>>, !fir.ref<!fir.box<!fir.ptr<f32>>>)
  %13 = fir.convert %4 {test.ptr = "box.addr"} : (!fir.ref<f32>) -> !fir.ptr<f32>
  %fir_decl_convert = fir.declare %13 {uniq_name = "box_addr_fir", test.ptr = "box.addr_fir"}: (!fir.ptr<f32>) -> !fir.ptr<f32>
  %hlfir_decl_convert:2 = hlfir.declare %13 {uniq_name = "box_addr_hlfir", test.ptr = "box.addr_hlfir"}: (!fir.ptr<f32>) -> (!fir.ptr<f32>, !fir.ptr<f32>)
  return
}
