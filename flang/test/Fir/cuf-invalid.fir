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
  %0 = cuf.alloc f32 {bindc_name = "r", data_attr = #cuf.cuda<device>, uniq_name = "_QFsub1Er"} -> !fir.ref<f32>
  // expected-error@+1{{'cuf.free' op expect device, managed, pinned or unified cuda attribute}}
  cuf.free %0 : !fir.ref<f32> {data_attr = #cuf.cuda<constant>}
  return
}

// -----

func.func @_QPsub1(%arg0: !fir.ref<!fir.array<?xf32>> {cuf.data_attr = #cuf.cuda<device>, fir.bindc_name = "adev"}, %arg1: !fir.ref<!fir.array<?xf32>> {fir.bindc_name = "ahost"}, %arg2: !fir.ref<i32> {fir.bindc_name = "n"}, %arg3: !fir.ref<i32> {fir.bindc_name = "m"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg2 dummy_scope %0 {uniq_name = "_QFsub1En"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %2:2 = hlfir.declare %arg3 dummy_scope %0 {uniq_name = "_QFsub1Em"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %3 = fir.load %1#0 : !fir.ref<i32>
  %4 = fir.load %2#0 : !fir.ref<i32>
  %5 = arith.muli %3, %4 : i32
  %6 = fir.convert %5 : (i32) -> i64
  %7 = fir.convert %6 : (i64) -> index
  %c0 = arith.constant 0 : index
  %8 = arith.cmpi sgt, %7, %c0 : index
  %9 = arith.select %8, %7, %c0 : index
  %10 = fir.shape %9 : (index) -> !fir.shape<1>
  %11:2 = hlfir.declare %arg0(%10) dummy_scope %0 {data_attr = #cuf.cuda<device>, uniq_name = "_QFsub1Eadev"} : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>)
  %12 = fir.load %1#0 : !fir.ref<i32>
  %13 = fir.load %2#0 : !fir.ref<i32>
  %14 = arith.muli %12, %13 : i32
  %15 = fir.convert %14 : (i32) -> i64
  %16 = fir.convert %15 : (i64) -> index
  %c0_0 = arith.constant 0 : index
  %17 = arith.cmpi sgt, %16, %c0_0 : index
  %18 = arith.select %17, %16, %c0_0 : index
  %19 = fir.shape %18 : (index) -> !fir.shape<1>
  %20:2 = hlfir.declare %arg1(%19) dummy_scope %0 {uniq_name = "_QFsub1Eahost"} : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>)
  // expected-error@+1{{'cuf.data_transfer' op shape can only be specified on data transfer with references}}
  cuf.data_transfer %20#0 to %11#0, %19 : !fir.shape<1> {transfer_kind = #cuf.cuda_transfer<host_device>} : !fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>
  return
}
