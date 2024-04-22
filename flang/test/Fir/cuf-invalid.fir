// RUN: fir-opt -split-input-file -verify-diagnostics %s

func.func @_QPsub1() {
  %0 = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>> {bindc_name = "a", uniq_name = "_QFsub1Ea"}
  %1 = fir.alloca i32
  %pinned = fir.alloca i1
  %4:2 = hlfir.declare %0 {cuda_attr = #fir.cuda<device>, fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFsub1Ea"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>)
  %11 = fir.convert %4#1 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<none>>
  %s = fir.load %1 : !fir.ref<i32>
  // expected-error@+1{{'fir.cuda_allocate' op pinned and stream cannot appears at the same time}}
  %13 = fir.cuda_allocate %11 : !fir.ref<!fir.box<none>> stream(%s : i32) pinned(%pinned : !fir.ref<i1>) {cuda_attr = #fir.cuda<device>} -> i32
  return
}

// -----

func.func @_QPsub1() {
  %1 = fir.alloca i32
  // expected-error@+1{{'fir.cuda_allocate' op expect box to be a reference to a class or box type value}}
  %2 = fir.cuda_allocate %1 : !fir.ref<i32> {cuda_attr = #fir.cuda<device>} -> i32
  return
}

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
  // expected-error@+1{{'fir.cuda_allocate' op expect stat attribute when errmsg is provided}}
  %13 = fir.cuda_allocate %11 : !fir.ref<!fir.box<none>> errmsg(%16 : !fir.box<none>) {cuda_attr = #fir.cuda<device>} -> i32
  return
}

// -----

func.func @_QPsub1() {
  %0 = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>> {bindc_name = "a", uniq_name = "_QFsub1Ea"}
  %4:2 = hlfir.declare %0 {cuda_attr = #fir.cuda<device>, fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFsub1Ea"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>)
  %1 = fir.alloca i32
  %11 = fir.convert %4#1 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<none>>
  // expected-error@+1{{'fir.cuda_allocate' op expect errmsg to be a reference to/or a box type value}}
  %13 = fir.cuda_allocate %11 : !fir.ref<!fir.box<none>> errmsg(%1 : !fir.ref<i32>) {cuda_attr = #fir.cuda<device>, hasStat} -> i32
  return
}

// -----

func.func @_QPsub1() {
  %1 = fir.alloca i32
  // expected-error@+1{{'fir.cuda_deallocate' op expect box to be a reference to class or box type value}}
  %2 = fir.cuda_deallocate %1 : !fir.ref<i32> {cuda_attr = #fir.cuda<device>} -> i32
  return
}

// -----

func.func @_QPsub1() {
  %0 = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>> {bindc_name = "a", uniq_name = "_QFsub1Ea"}
  %4:2 = hlfir.declare %0 {cuda_attr = #fir.cuda<device>, fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFsub1Ea"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>)
  %1 = fir.alloca i32
  %11 = fir.convert %4#1 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<none>>
  // expected-error@+1{{'fir.cuda_deallocate' op expect errmsg to be a reference to/or a box type value}}
  %13 = fir.cuda_deallocate %11 : !fir.ref<!fir.box<none>> errmsg(%1 : !fir.ref<i32>) {cuda_attr = #fir.cuda<device>, hasStat} -> i32
  return
}

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
  // expected-error@+1{{'fir.cuda_deallocate' op expect stat attribute when errmsg is provided}}
  %13 = fir.cuda_deallocate %11 : !fir.ref<!fir.box<none>> errmsg(%16 : !fir.box<none>) {cuda_attr = #fir.cuda<device>} -> i32
  return
}
