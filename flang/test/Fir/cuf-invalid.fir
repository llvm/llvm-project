// RUN: fir-opt -split-input-file -verify-diagnostics %s

func.func @_QPsub1() {
  %0 = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>> {bindc_name = "a", uniq_name = "_QFsub1Ea"}
  %1 = fir.alloca i32
  %pinned = fir.alloca i1
  %4:2 = hlfir.declare %0 {data_attr = #cuf.cuda<device>, fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFsub1Ea"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>)
  %11 = fir.convert %4#1 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<none>>
  %s = fir.load %1 : !fir.ref<i32>
  // expected-error@+1{{'cuf.allocate' op pinned and stream cannot appears at the same time}}
  %13 = cuf.allocate %11 : !fir.ref<!fir.box<none>> stream(%s : i32) pinned(%pinned : !fir.ref<i1>) {data_attr = #cuf.cuda<device>} -> i32
  return
}

// -----

func.func @_QPsub1() {
  %1 = fir.alloca i32
  // expected-error@+1{{'cuf.allocate' op expect box to be a reference to a class or box type value}}
  %2 = cuf.allocate %1 : !fir.ref<i32> {data_attr = #cuf.cuda<device>} -> i32
  return
}

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
  // expected-error@+1{{'cuf.allocate' op expect stat attribute when errmsg is provided}}
  %13 = cuf.allocate %11 : !fir.ref<!fir.box<none>> errmsg(%16 : !fir.box<none>) {data_attr = #cuf.cuda<device>} -> i32
  return
}

// -----

func.func @_QPsub1() {
  %0 = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>> {bindc_name = "a", uniq_name = "_QFsub1Ea"}
  %4:2 = hlfir.declare %0 {data_attr = #cuf.cuda<device>, fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFsub1Ea"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>)
  %1 = fir.alloca i32
  %11 = fir.convert %4#1 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<none>>
  // expected-error@+1{{'cuf.allocate' op expect errmsg to be a reference to/or a box type value}}
  %13 = cuf.allocate %11 : !fir.ref<!fir.box<none>> errmsg(%1 : !fir.ref<i32>) {data_attr = #cuf.cuda<device>, hasStat} -> i32
  return
}

// -----

func.func @_QPsub1() {
  %1 = fir.alloca i32
  // expected-error@+1{{'cuf.deallocate' op expect box to be a reference to class or box type value}}
  %2 = cuf.deallocate %1 : !fir.ref<i32> {data_attr = #cuf.cuda<device>} -> i32
  return
}

// -----

func.func @_QPsub1() {
  %0 = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>> {bindc_name = "a", uniq_name = "_QFsub1Ea"}
  %4:2 = hlfir.declare %0 {data_attr = #cuf.cuda<device>, fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFsub1Ea"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>)
  %1 = fir.alloca i32
  %11 = fir.convert %4#1 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<none>>
  // expected-error@+1{{'cuf.deallocate' op expect errmsg to be a reference to/or a box type value}}
  %13 = cuf.deallocate %11 : !fir.ref<!fir.box<none>> errmsg(%1 : !fir.ref<i32>) {data_attr = #cuf.cuda<device>, hasStat} -> i32
  return
}

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
  // expected-error@+1{{'cuf.deallocate' op expect stat attribute when errmsg is provided}}
  %13 = cuf.deallocate %11 : !fir.ref<!fir.box<none>> errmsg(%16 : !fir.box<none>) {data_attr = #cuf.cuda<device>} -> i32
  return
}

// -----

func.func @_QPsub1() {
  // expected-error@+1{{'cuf.alloc' op expect device, managed or unified cuda attribute}}
  %0 = cuf.alloc f32 {bindc_name = "r", data_attr = #cuf.cuda<pinned>, uniq_name = "_QFsub1Er"} -> !fir.ref<f32>
  cuf.free %0 : !fir.ref<f32> {data_attr = #cuf.cuda<constant>}
  return
}

// -----

func.func @_QPsub1() {
  %0 = cuf.alloc f32 {bindc_name = "r", data_attr = #cuf.cuda<device>, uniq_name = "_QFsub1Er"} -> !fir.ref<f32>
  // expected-error@+1{{'cuf.free' op expect device, managed or unified cuda attribute}}
  cuf.free %0 : !fir.ref<f32> {data_attr = #cuf.cuda<constant>}
  return
}
