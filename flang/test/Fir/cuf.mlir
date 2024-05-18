// RUN: fir-opt --split-input-file %s | fir-opt --split-input-file | FileCheck %s

// Simple round trip test of operations.

func.func @_QPsub1() {
  %0 = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>> {bindc_name = "a", uniq_name = "_QFsub1Ea"}
  %4:2 = hlfir.declare %0 {cuda_attr = #fir.cuda<device>, fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFsub1Ea"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>)
  %11 = fir.convert %4#1 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<none>>
  %13 = fir.cuda_allocate %11 : !fir.ref<!fir.box<none>> {cuda_attr = #fir.cuda<device>} -> i32
  %14 = fir.cuda_deallocate %11 : !fir.ref<!fir.box<none>> {cuda_attr = #fir.cuda<device>} -> i32
  return
}

// CHECK: fir.cuda_allocate %{{.*}} : !fir.ref<!fir.box<none>> {cuda_attr = #fir.cuda<device>} -> i32
// CHECK: fir.cuda_deallocate %{{.*}} : !fir.ref<!fir.box<none>> {cuda_attr = #fir.cuda<device>} -> i32

// -----

func.func @_QPsub1() {
  %0 = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>> {bindc_name = "a", uniq_name = "_QFsub1Ea"}
  %1 = fir.alloca i32
  %4:2 = hlfir.declare %0 {cuda_attr = #fir.cuda<device>, fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFsub1Ea"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>)
  %11 = fir.convert %4#1 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<none>>
  %s = fir.load %1 : !fir.ref<i32>
  %13 = fir.cuda_allocate %11 : !fir.ref<!fir.box<none>> stream(%s : i32) {cuda_attr = #fir.cuda<device>} -> i32
  return
}

// CHECK: fir.cuda_allocate %{{.*}} : !fir.ref<!fir.box<none>> stream(%{{.*}} : i32) {cuda_attr = #fir.cuda<device>} -> i32

// -----

func.func @_QPsub1() {
  %0 = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>> {bindc_name = "a", uniq_name = "_QFsub1Ea"}
  %1 = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>> {bindc_name = "b", uniq_name = "_QFsub1Eb"}
  %4:2 = hlfir.declare %0 {cuda_attr = #fir.cuda<device>, fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFsub1Ea"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>)
  %5:2 = hlfir.declare %1 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFsub1Ea"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>)
  %11 = fir.convert %4#1 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<none>>
  %12 = fir.convert %5#1 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<none>>
  %13 = fir.cuda_allocate %11 : !fir.ref<!fir.box<none>> source(%12 : !fir.ref<!fir.box<none>>) {cuda_attr = #fir.cuda<device>} -> i32
  return
}

// CHECK: fir.cuda_allocate %{{.*}} : !fir.ref<!fir.box<none>> source(%{{.*}} : !fir.ref<!fir.box<none>>) {cuda_attr = #fir.cuda<device>} -> i32

// -----

func.func @_QPsub1() {
  %0 = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>> {bindc_name = "a", uniq_name = "_QFsub1Ea"}
  %pinned = fir.alloca i1
  %4:2 = hlfir.declare %0 {cuda_attr = #fir.cuda<device>, fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFsub1Ea"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>)
  %11 = fir.convert %4#1 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<none>>
  %13 = fir.cuda_allocate %11 : !fir.ref<!fir.box<none>> pinned(%pinned : !fir.ref<i1>) {cuda_attr = #fir.cuda<device>} -> i32
  return
}

// CHECK: fir.cuda_allocate %{{.*}} : !fir.ref<!fir.box<none>> pinned(%{{.*}} : !fir.ref<i1>) {cuda_attr = #fir.cuda<device>} -> i32

// -----

func.func @_QPsub1() {
  %0 = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>> {bindc_name = "a", uniq_name = "_QFsub1Ea"}
  %4:2 = hlfir.declare %0 {cuda_attr = #fir.cuda<device>, fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFsub1Ea"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>)
  %c100 = arith.constant 100 : index
  %7 = fir.alloca !fir.char<1,100> {bindc_name = "msg", uniq_name = "_QFsub1Emsg"}
  %8:2 = hlfir.declare %7 typeparams %c100 {uniq_name = "_QFsub1Emsg"} : (!fir.ref<!fir.char<1,100>>, index) -> (!fir.ref<!fir.char<1,100>>, !fir.ref<!fir.char<1,100>>)
  %9 = fir.embox %8#1 : (!fir.ref<!fir.char<1,100>>) -> !fir.box<!fir.char<1,100>>
  %11 = fir.convert %4#1 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<none>>
  %16 = fir.convert %9 : (!fir.box<!fir.char<1,100>>) -> !fir.box<none>
  %13 = fir.cuda_allocate %11 : !fir.ref<!fir.box<none>> errmsg(%16 : !fir.box<none>) {cuda_attr = #fir.cuda<device>, hasStat} -> i32
  %14 = fir.cuda_deallocate %11 : !fir.ref<!fir.box<none>> errmsg(%16 : !fir.box<none>) {cuda_attr = #fir.cuda<device>, hasStat} -> i32
  return
}

// CHECK: fir.cuda_allocate %{{.*}} : !fir.ref<!fir.box<none>> errmsg(%{{.*}} : !fir.box<none>) {cuda_attr = #fir.cuda<device>, hasStat} -> i32
// CHECK: fir.cuda_deallocate %{{.*}} : !fir.ref<!fir.box<none>> errmsg(%{{.*}} : !fir.box<none>) {cuda_attr = #fir.cuda<device>, hasStat} -> i32
