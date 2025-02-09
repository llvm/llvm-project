// RUN: fir-opt --split-input-file %s | fir-opt --split-input-file | FileCheck %s

// Simple round trip test of operations.

func.func @_QPsub1() {
  %0 = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>> {bindc_name = "a", uniq_name = "_QFsub1Ea"}
  %4:2 = hlfir.declare %0 {data_attr = #cuf.cuda<device>, fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFsub1Ea"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>)
  %11 = fir.convert %4#1 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<none>>
  %13 = cuf.allocate %11 : !fir.ref<!fir.box<none>> {data_attr = #cuf.cuda<device>} -> i32
  %14 = cuf.deallocate %11 : !fir.ref<!fir.box<none>> {data_attr = #cuf.cuda<device>} -> i32
  return
}

// CHECK: cuf.allocate %{{.*}} : !fir.ref<!fir.box<none>> {data_attr = #cuf.cuda<device>} -> i32
// CHECK: cuf.deallocate %{{.*}} : !fir.ref<!fir.box<none>> {data_attr = #cuf.cuda<device>} -> i32

// -----

func.func @_QPsub1() {
  %0 = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>> {bindc_name = "a", uniq_name = "_QFsub1Ea"}
  %1 = fir.alloca i32
  %4:2 = hlfir.declare %0 {data_attr = #cuf.cuda<device>, fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFsub1Ea"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>)
  %11 = fir.convert %4#1 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<none>>
  %s = fir.load %1 : !fir.ref<i32>
  %13 = cuf.allocate %11 : !fir.ref<!fir.box<none>> stream(%s : i32) {data_attr = #cuf.cuda<device>} -> i32
  return
}

// CHECK: cuf.allocate %{{.*}} : !fir.ref<!fir.box<none>> stream(%{{.*}} : i32) {data_attr = #cuf.cuda<device>} -> i32

// -----

func.func @_QPsub1() {
  %0 = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>> {bindc_name = "a", uniq_name = "_QFsub1Ea"}
  %1 = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>> {bindc_name = "b", uniq_name = "_QFsub1Eb"}
  %4:2 = hlfir.declare %0 {data_attr = #cuf.cuda<device>, fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFsub1Ea"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>)
  %5:2 = hlfir.declare %1 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFsub1Ea"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>)
  %11 = fir.convert %4#1 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<none>>
  %12 = fir.convert %5#1 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<none>>
  %13 = cuf.allocate %11 : !fir.ref<!fir.box<none>> source(%12 : !fir.ref<!fir.box<none>>) {data_attr = #cuf.cuda<device>} -> i32
  return
}

// CHECK: cuf.allocate %{{.*}} : !fir.ref<!fir.box<none>> source(%{{.*}} : !fir.ref<!fir.box<none>>) {data_attr = #cuf.cuda<device>} -> i32

// -----

func.func @_QPsub1() {
  %0 = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>> {bindc_name = "a", uniq_name = "_QFsub1Ea"}
  %pinned = fir.alloca i1
  %4:2 = hlfir.declare %0 {data_attr = #cuf.cuda<device>, fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFsub1Ea"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>)
  %11 = fir.convert %4#1 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<none>>
  %13 = cuf.allocate %11 : !fir.ref<!fir.box<none>> pinned(%pinned : !fir.ref<i1>) {data_attr = #cuf.cuda<device>} -> i32
  return
}

// CHECK: cuf.allocate %{{.*}} : !fir.ref<!fir.box<none>> pinned(%{{.*}} : !fir.ref<i1>) {data_attr = #cuf.cuda<device>} -> i32

// -----

func.func @_QPsub1() {
  %0 = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>> {bindc_name = "a", uniq_name = "_QFsub1Ea"}
  %4:2 = hlfir.declare %0 {data_attr = #cuf.cuda<device>, fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFsub1Ea"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>)
  %c100 = arith.constant 100 : index
  %7 = fir.alloca !fir.char<1,100> {bindc_name = "msg", uniq_name = "_QFsub1Emsg"}
  %8:2 = hlfir.declare %7 typeparams %c100 {uniq_name = "_QFsub1Emsg"} : (!fir.ref<!fir.char<1,100>>, index) -> (!fir.ref<!fir.char<1,100>>, !fir.ref<!fir.char<1,100>>)
  %9 = fir.embox %8#1 : (!fir.ref<!fir.char<1,100>>) -> !fir.box<!fir.char<1,100>>
  %11 = fir.convert %4#1 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<none>>
  %16 = fir.convert %9 : (!fir.box<!fir.char<1,100>>) -> !fir.box<none>
  %13 = cuf.allocate %11 : !fir.ref<!fir.box<none>> errmsg(%16 : !fir.box<none>) {data_attr = #cuf.cuda<device>, hasStat} -> i32
  %14 = cuf.deallocate %11 : !fir.ref<!fir.box<none>> errmsg(%16 : !fir.box<none>) {data_attr = #cuf.cuda<device>, hasStat} -> i32
  return
}

// CHECK: cuf.allocate %{{.*}} : !fir.ref<!fir.box<none>> errmsg(%{{.*}} : !fir.box<none>) {data_attr = #cuf.cuda<device>, hasStat} -> i32
// CHECK: cuf.deallocate %{{.*}} : !fir.ref<!fir.box<none>> errmsg(%{{.*}} : !fir.box<none>) {data_attr = #cuf.cuda<device>, hasStat} -> i32

// -----

func.func @_QPsub1() {
  %0 = cuf.alloc f32 {bindc_name = "r", data_attr = #cuf.cuda<device>, uniq_name = "_QFsub1Er"} -> !fir.ref<f32>
  cuf.free %0 : !fir.ref<f32> {data_attr = #cuf.cuda<device>}
  return
}

// CHECK: cuf.alloc
// CHECK: cuf.free

