// RUN: fir-opt --cuf-convert %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>} {

func.func @_QPsub1() {
  %0 = cuf.alloc !fir.box<!fir.heap<!fir.array<?xi32>>> {bindc_name = "adev", data_attr = #cuf.cuda<device>, uniq_name = "_QFsub1Eadev"} -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  %4:2 = hlfir.declare %0 {data_attr = #cuf.cuda<device>, fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFsub1Eadev"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
  %5 = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>> {bindc_name = "ahost", uniq_name = "_QFsub1Eahost"}
  %9:2 = hlfir.declare %5 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFsub1Eahost"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
  cuf.data_transfer %4#0 to %9#0 {transfer_kind = #cuf.cuda_transfer<device_host>} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  cuf.free %4#1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {data_attr = #cuf.cuda<device>}
  return
}

// CHECK-LABEL: func.func @_QPsub1()
// CHECK: %[[ADEV:.*]]:2 = hlfir.declare %{{.*}} {data_attr = #cuf.cuda<device>, fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFsub1Eadev"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
// CHECK: %[[AHOST:.*]]:2 = hlfir.declare %{{.*}} {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFsub1Eahost"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
// CHECK: %[[AHOST_BOX:.*]] = fir.convert %[[AHOST]]#0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
// CHECK: %[[ADEV_BOX:.*]] = fir.convert %[[ADEV]]#0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
// CHECK: fir.call @_FortranACUFDataTransferDescDesc(%[[AHOST_BOX]], %[[ADEV_BOX]], %c1{{.*}}, %{{.*}}, %{{.*}}) : (!fir.ref<!fir.box<none>>, !fir.ref<!fir.box<none>>, i32, !fir.ref<i8>, i32) -> none

func.func @_QPsub2() {
  %0 = cuf.alloc !fir.box<!fir.heap<!fir.array<?xi32>>> {bindc_name = "adev", data_attr = #cuf.cuda<device>, uniq_name = "_QFsub2Eadev"} -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  %4:2 = hlfir.declare %0 {data_attr = #cuf.cuda<device>, fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFsub2Eadev"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
  %c2_i32 = arith.constant 2 : i32
  cuf.data_transfer %c2_i32 to %4#0 {transfer_kind = #cuf.cuda_transfer<host_device>} : i32, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  cuf.free %4#1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {data_attr = #cuf.cuda<device>}
  return
}

// CHECK-LABEL: func.func @_QPsub2()
// CHECK: %[[TEMP_BOX:.*]] = fir.alloca !fir.box<i32>
// CHECK: %[[TEMP:.*]] = fir.alloca i32
// CHECK: %[[ADEV:.*]]:2 = hlfir.declare %{{.*}} {data_attr = #cuf.cuda<device>, fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFsub2Eadev"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
// CHECK: %[[C2:.*]] = arith.constant 2 : i32
// CHECK: fir.store %[[C2]] to %[[TEMP]] : !fir.ref<i32>
// CHECK: %[[EMBOX:.*]] = fir.embox %[[TEMP]] : (!fir.ref<i32>) -> !fir.box<i32>
// CHECK: fir.store %[[EMBOX]] to %[[TEMP_BOX]] : !fir.ref<!fir.box<i32>>
// CHECK: %[[ADEV_BOX:.*]] = fir.convert %[[ADEV]]#0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
// CHECK: %[[TEMP_CONV:.*]] = fir.convert %[[TEMP_BOX]] : (!fir.ref<!fir.box<i32>>) -> !fir.ref<!fir.box<none>>
// CHECK: fir.call @_FortranACUFDataTransferDescDesc(%[[ADEV_BOX]], %[[TEMP_CONV]], %c0{{.*}}, %{{.*}}, %{{.*}}) : (!fir.ref<!fir.box<none>>, !fir.ref<!fir.box<none>>, i32, !fir.ref<i8>, i32) -> none

func.func @_QPsub3() {
  %0 = cuf.alloc !fir.box<!fir.heap<!fir.array<?xi32>>> {bindc_name = "adev", data_attr = #cuf.cuda<device>, uniq_name = "_QFsub3Eadev"} -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  %4:2 = hlfir.declare %0 {data_attr = #cuf.cuda<device>, fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFsub3Eadev"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
  %5 = fir.alloca i32 {bindc_name = "v", uniq_name = "_QFsub3Ev"}
  %6:2 = hlfir.declare %5 {uniq_name = "_QFsub3Ev"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  cuf.data_transfer %6#0 to %4#0 {transfer_kind = #cuf.cuda_transfer<host_device>} : !fir.ref<i32>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  cuf.free %4#1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {data_attr = #cuf.cuda<device>}
  return
}

// CHECK-LABEL: func.func @_QPsub3()
// CHECK: %[[TEMP_BOX:.*]] = fir.alloca !fir.box<i32>
// CHECK: %[[ADEV:.*]]:2 = hlfir.declare %{{.*}} {data_attr = #cuf.cuda<device>, fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFsub3Eadev"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
// CHECK: %[[V:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFsub3Ev"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK: %[[EMBOX:.*]] = fir.embox %[[V]]#0 : (!fir.ref<i32>) -> !fir.box<i32>
// CHECK: fir.store %[[EMBOX]] to %[[TEMP_BOX]] : !fir.ref<!fir.box<i32>>
// CHECK: %[[ADEV_BOX:.*]] = fir.convert %[[ADEV]]#0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
// CHECK: %[[V_CONV:.*]] = fir.convert %[[TEMP_BOX]] : (!fir.ref<!fir.box<i32>>) -> !fir.ref<!fir.box<none>>
// CHECK: fir.call @_FortranACUFDataTransferDescDesc(%[[ADEV_BOX]], %[[V_CONV]], %c0{{.*}}, %{{.*}}, %{{.*}}) : (!fir.ref<!fir.box<none>>, !fir.ref<!fir.box<none>>, i32, !fir.ref<i8>, i32) -> none
  
func.func @_QPsub4() {
  %0 = cuf.alloc !fir.box<!fir.heap<!fir.array<?xi32>>> {bindc_name = "adev", data_attr = #cuf.cuda<device>, uniq_name = "_QFsub4Eadev"} -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  %4:2 = hlfir.declare %0 {data_attr = #cuf.cuda<device>, fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFsub4Eadev"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
  %c10 = arith.constant 10 : index
  %5 = fir.alloca !fir.array<10xi32> {bindc_name = "ahost", uniq_name = "_QFsub4Eahost"}
  %6 = fir.shape %c10 : (index) -> !fir.shape<1>
  %7:2 = hlfir.declare %5(%6) {uniq_name = "_QFsub4Eahost"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
  cuf.data_transfer %7#0 to %4#0 {transfer_kind = #cuf.cuda_transfer<host_device>} : !fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  cuf.data_transfer %4#0 to %7#0 {transfer_kind = #cuf.cuda_transfer<device_host>} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.array<10xi32>>
  cuf.free %4#1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {data_attr = #cuf.cuda<device>}
  return
}
// CHECK-LABEL: func.func @_QPsub4()
// CHECK: %[[TEMP_BOX:.*]] = fir.alloca !fir.box<!fir.array<10xi32>>
// CHECK: %[[ADEV:.*]]:2 = hlfir.declare %{{.*}} {data_attr = #cuf.cuda<device>, fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFsub4Eadev"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
// CHECK: %[[AHOST:.*]]:2 = hlfir.declare %{{.*}}(%[[AHOST_SHAPE:.*]]) {uniq_name = "_QFsub4Eahost"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
// CHECK: %[[EMBOX:.*]] = fir.embox %[[AHOST]]#0(%[[AHOST_SHAPE]]) : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<10xi32>>
// CHECK: fir.store %[[EMBOX]] to %[[TEMP_BOX]] : !fir.ref<!fir.box<!fir.array<10xi32>>>
// CHECK: %[[ADEV_BOX:.*]] = fir.convert %[[ADEV]]#0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
// CHECK: %[[AHOST_BOX:.*]] = fir.convert %[[TEMP_BOX]] : (!fir.ref<!fir.box<!fir.array<10xi32>>>) -> !fir.ref<!fir.box<none>>
// CHECK: fir.call @_FortranACUFDataTransferDescDesc(%[[ADEV_BOX]], %[[AHOST_BOX]], %c0{{.*}}, %{{.*}}, %{{.*}}) : (!fir.ref<!fir.box<none>>, !fir.ref<!fir.box<none>>, i32, !fir.ref<i8>, i32) -> none
// CHECK: %[[NBELEM:.*]] = arith.constant 10 : index
// CHECK: %[[WIDTH:.*]] = arith.constant 4 : index
// CHECK: %[[BYTES:.*]] = arith.muli %[[NBELEM]], %[[WIDTH]] : index
// CHECK: %[[AHOST_PTR:.*]] = fir.convert %[[AHOST]]#0 : (!fir.ref<!fir.array<10xi32>>) -> !fir.llvm_ptr<i8>
// CHECK: %[[ADEV_BOX:.*]] = fir.convert %[[ADEV]]#0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
// CHECK: %[[BYTES_CONV:.*]] = fir.convert %[[BYTES]] : (index) -> i64
// CHECK: fir.call @_FortranACUFDataTransferPtrDesc(%[[AHOST_PTR]], %[[ADEV_BOX]], %[[BYTES_CONV]], %c1{{.*}}, %{{.*}}, %{{.*}}) : (!fir.llvm_ptr<i8>, !fir.ref<!fir.box<none>>, i64, i32, !fir.ref<i8>, i32) -> none

func.func @_QPsub5(%arg0: !fir.ref<i32> {fir.bindc_name = "n"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %1 = cuf.alloc !fir.box<!fir.heap<!fir.array<?x?xi32>>> {bindc_name = "adev", data_attr = #cuf.cuda<device>, uniq_name = "_QFsub5Eadev"} -> !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>
  %5:2 = hlfir.declare %1 {data_attr = #cuf.cuda<device>, fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFsub5Eadev"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>)
  %6:2 = hlfir.declare %arg0 dummy_scope %0 {uniq_name = "_QFsub5En"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %7 = fir.load %6#0 : !fir.ref<i32>
  %8 = fir.convert %7 : (i32) -> i64
  %9 = fir.convert %8 : (i64) -> index
  %c0_0 = arith.constant 0 : index
  %10 = arith.cmpi sgt, %9, %c0_0 : index
  %11 = arith.select %10, %9, %c0_0 : index
  %12 = fir.load %6#0 : !fir.ref<i32>
  %13 = fir.convert %12 : (i32) -> i64
  %14 = fir.convert %13 : (i64) -> index
  %c0_1 = arith.constant 0 : index
  %15 = arith.cmpi sgt, %14, %c0_1 : index
  %16 = arith.select %15, %14, %c0_1 : index
  %17 = fir.alloca !fir.array<?x?xi32>, %11, %16 {bindc_name = "ahost", uniq_name = "_QFsub5Eahost"}
  %18 = fir.shape %11, %16 : (index, index) -> !fir.shape<2>
  %19:2 = hlfir.declare %17(%18) {uniq_name = "_QFsub5Eahost"} : (!fir.ref<!fir.array<?x?xi32>>, !fir.shape<2>) -> (!fir.box<!fir.array<?x?xi32>>, !fir.ref<!fir.array<?x?xi32>>)
  cuf.data_transfer %19#1 to %5#0, %18 : !fir.shape<2> {transfer_kind = #cuf.cuda_transfer<host_device>} : !fir.ref<!fir.array<?x?xi32>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>
  cuf.data_transfer %5#0 to %19#1, %18 : !fir.shape<2> {transfer_kind = #cuf.cuda_transfer<device_host>} : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>, !fir.ref<!fir.array<?x?xi32>>
  return
}

// CHECK-LABEL: func.func @_QPsub5
// CHECK: %[[TEMP_BOX:.*]] = fir.alloca !fir.box<!fir.array<?x?xi32>>
// CHECK: %[[ADEV:.*]]:2 = hlfir.declare %{{.*}} {data_attr = #cuf.cuda<device>, fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFsub5Eadev"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>)
// CHECK: %[[SHAPE:.*]] = fir.shape %[[I1:.*]], %[[I2:.*]] : (index, index) -> !fir.shape<2>
// CHECK: %[[AHOST:.*]]:2 = hlfir.declare %{{.*}}(%[[SHAPE]]) {uniq_name = "_QFsub5Eahost"} : (!fir.ref<!fir.array<?x?xi32>>, !fir.shape<2>) -> (!fir.box<!fir.array<?x?xi32>>, !fir.ref<!fir.array<?x?xi32>>)
// CHECK: %[[EMBOX:.*]] = fir.embox %[[AHOST]]#1(%[[SHAPE]]) : (!fir.ref<!fir.array<?x?xi32>>, !fir.shape<2>) -> !fir.box<!fir.array<?x?xi32>>
// CHECK: fir.store %[[EMBOX]] to %[[TEMP_BOX]] : !fir.ref<!fir.box<!fir.array<?x?xi32>>>
// CHECK: %[[ADEV_BOX:.*]] = fir.convert %[[ADEV]]#0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>) -> !fir.ref<!fir.box<none>>
// CHECK: %[[AHOST_BOX:.*]] = fir.convert %[[TEMP_BOX]] : (!fir.ref<!fir.box<!fir.array<?x?xi32>>>) -> !fir.ref<!fir.box<none>>
// CHECK: fir.call @_FortranACUFDataTransferDescDesc(%[[ADEV_BOX]], %[[AHOST_BOX]], %c0{{.*}}, %{{.*}}, %{{.*}}) : (!fir.ref<!fir.box<none>>, !fir.ref<!fir.box<none>>, i32, !fir.ref<i8>, i32) -> none
// CHECK: %[[NBELEM:.*]] = arith.muli %[[I1]], %[[I2]] : index
// CHECK: %[[WIDTH:.*]] = arith.constant 4 : index
// CHECK: %[[BYTES:.*]] = arith.muli %[[NBELEM]], %[[WIDTH]] : index
// CHECK: %[[AHOST_PTR:.*]] = fir.convert %[[AHOST]]#1 : (!fir.ref<!fir.array<?x?xi32>>) -> !fir.llvm_ptr<i8>
// CHECK: %[[ADEV_BOX:.*]] = fir.convert %[[ADEV]]#0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>) -> !fir.ref<!fir.box<none>>
// CHECK: %[[BYTES_CONV:.*]] = fir.convert %[[BYTES]] : (index) -> i64
// CHECK: fir.call @_FortranACUFDataTransferPtrDesc(%[[AHOST_PTR]], %[[ADEV_BOX]], %[[BYTES_CONV]], %c1{{.*}}, %{{.*}}, %{{.*}}) : (!fir.llvm_ptr<i8>, !fir.ref<!fir.box<none>>, i64, i32, !fir.ref<i8>, i32) -> none

func.func @_QPsub6() {
  %0 = cuf.alloc i32 {bindc_name = "idev", data_attr = #cuf.cuda<device>, uniq_name = "_QFsub6Eidev"} -> !fir.ref<i32>
  %1:2 = hlfir.declare %0 {data_attr = #cuf.cuda<device>, uniq_name = "_QFsub6Eidev"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %2 = fir.alloca i32 {bindc_name = "ihost", uniq_name = "_QFsub6Eihost"}
  %3:2 = hlfir.declare %2 {uniq_name = "_QFsub6Eihost"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  cuf.data_transfer %1#0 to %3#0 {transfer_kind = #cuf.cuda_transfer<device_host>} : !fir.ref<i32>, !fir.ref<i32>
  %4 = fir.load %3#0 : !fir.ref<i32>
  %5:3 = hlfir.associate %4 {uniq_name = ".cuf_host_tmp"} : (i32) -> (!fir.ref<i32>, !fir.ref<i32>, i1)
  cuf.data_transfer %5#0 to %1#0 {transfer_kind = #cuf.cuda_transfer<host_device>} : !fir.ref<i32>, !fir.ref<i32>
  hlfir.end_associate %5#1, %5#2 : !fir.ref<i32>, i1
  cuf.free %1#1 : !fir.ref<i32> {data_attr = #cuf.cuda<device>}
  return
}

// CHECK-LABEL: func.func @_QPsub6()
// CHECK: %[[IDEV:.*]]:2 = hlfir.declare %{{.*}} {data_attr = #cuf.cuda<device>, uniq_name = "_QFsub6Eidev"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK: %[[IHOST:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFsub6Eihost"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK: %[[DST:.*]] = fir.convert %[[IHOST]]#0 : (!fir.ref<i32>) -> !fir.llvm_ptr<i8>
// CHECK: %[[SRC:.*]] = fir.convert %[[IDEV]]#0 : (!fir.ref<i32>) -> !fir.llvm_ptr<i8>
// CHECK: fir.call @_FortranACUFDataTransferPtrPtr(%[[DST]], %[[SRC]], %c4{{.*}}, %c1{{.*}}, %{{.*}}, %{{.*}}) : (!fir.llvm_ptr<i8>, !fir.llvm_ptr<i8>, i64, i32, !fir.ref<i8>, i32) -> none
// CHECK: %[[LOAD:.*]] = fir.load %[[IHOST]]#0 : !fir.ref<i32>
// CHECK: %[[ASSOC:.*]]:3 = hlfir.associate %[[LOAD]] {uniq_name = ".cuf_host_tmp"} : (i32) -> (!fir.ref<i32>, !fir.ref<i32>, i1)
// CHECK: %[[DST:.*]] = fir.convert %[[IDEV]]#0 : (!fir.ref<i32>) -> !fir.llvm_ptr<i8>
// CHECK: %[[SRC:.*]] = fir.convert %[[ASSOC]]#0 : (!fir.ref<i32>) -> !fir.llvm_ptr<i8>
// CHECK: fir.call @_FortranACUFDataTransferPtrPtr(%[[DST]], %[[SRC]], %c4{{.*}}, %c0{{.*}}, %{{.*}}, %{{.*}}) : (!fir.llvm_ptr<i8>, !fir.llvm_ptr<i8>, i64, i32, !fir.ref<i8>, i32) -> none

func.func @_QPsub7() {
  %c10 = arith.constant 10 : index
  %0 = cuf.alloc !fir.array<10xi32> {bindc_name = "idev", data_attr = #cuf.cuda<device>, uniq_name = "_QFsub7Eidev"} -> !fir.ref<!fir.array<10xi32>>
  %1 = fir.shape %c10 : (index) -> !fir.shape<1>
  %2:2 = hlfir.declare %0(%1) {data_attr = #cuf.cuda<device>, uniq_name = "_QFsub7Eidev"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
  %c10_0 = arith.constant 10 : index
  %3 = fir.alloca !fir.array<10xi32> {bindc_name = "ihost", uniq_name = "_QFsub7Eihost"}
  %4 = fir.shape %c10_0 : (index) -> !fir.shape<1>
  %5:2 = hlfir.declare %3(%4) {uniq_name = "_QFsub7Eihost"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
  cuf.data_transfer %2#0 to %5#0 {transfer_kind = #cuf.cuda_transfer<device_host>} : !fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>
  cuf.data_transfer %5#0 to %2#0 {transfer_kind = #cuf.cuda_transfer<host_device>} : !fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>
  cuf.free %2#1 : !fir.ref<!fir.array<10xi32>> {data_attr = #cuf.cuda<device>}
  return
}

// CHECK-LABEL: func.func @_QPsub7()
// CHECK: %[[IDEV:.*]]:2 = hlfir.declare %{{.*}} {data_attr = #cuf.cuda<device>, uniq_name = "_QFsub7Eidev"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
// CHECK: %[[IHOST:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFsub7Eihost"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
// CHECK: %[[BYTES:.*]] = arith.muli %c10{{.*}}, %c4{{.*}} : i64
// CHECK: %[[DST:.*]] = fir.convert %[[IHOST]]#0 : (!fir.ref<!fir.array<10xi32>>) -> !fir.llvm_ptr<i8>
// CHECK: %[[SRC:.*]] = fir.convert %[[IDEV]]#0 : (!fir.ref<!fir.array<10xi32>>) -> !fir.llvm_ptr<i8>
// CHECK: fir.call @_FortranACUFDataTransferPtrPtr(%[[DST]], %[[SRC]], %[[BYTES]], %c1{{.*}}, %{{.*}}, %{{.*}}) : (!fir.llvm_ptr<i8>, !fir.llvm_ptr<i8>, i64, i32, !fir.ref<i8>, i32) -> none
// CHECK: %[[BYTES:.*]] = arith.muli %c10{{.*}}, %c4{{.*}} : i64
// CHECK: %[[DST:.*]] = fir.convert %[[IDEV]]#0 : (!fir.ref<!fir.array<10xi32>>) -> !fir.llvm_ptr<i8>
// CHECK: %[[SRC:.*]] = fir.convert %[[IHOST]]#0 : (!fir.ref<!fir.array<10xi32>>) -> !fir.llvm_ptr<i8>
// CHECK: fir.call @_FortranACUFDataTransferPtrPtr(%[[DST]], %[[SRC]], %[[BYTES]], %c0{{.*}}, %{{.*}}, %{{.*}}) : (!fir.llvm_ptr<i8>, !fir.llvm_ptr<i8>, i64, i32, !fir.ref<i8>, i32) -> none

fir.global @_QMmtestsEn(dense<[3, 4, 5, 6, 7]> : tensor<5xi32>) {data_attr = #cuf.cuda<device>} : !fir.array<5xi32>
func.func @_QPsub8() attributes {fir.bindc_name = "t"} {
  %c5 = arith.constant 5 : index
  %0 = fir.alloca !fir.array<5xi32> {bindc_name = "m", uniq_name = "_QFEm"}
  %1 = fir.shape %c5 : (index) -> !fir.shape<1>
  %2 = fir.declare %0(%1) {uniq_name = "_QFEm"} : (!fir.ref<!fir.array<5xi32>>, !fir.shape<1>) -> !fir.ref<!fir.array<5xi32>>
  %3 = fir.address_of(@_QMmtestsEn) : !fir.ref<!fir.array<5xi32>>
  %4 = fir.declare %3(%1) {data_attr = #cuf.cuda<device>, uniq_name = "_QMmtestsEn"} : (!fir.ref<!fir.array<5xi32>>, !fir.shape<1>) -> !fir.ref<!fir.array<5xi32>>
  cuf.data_transfer %4 to %2 {transfer_kind = #cuf.cuda_transfer<device_host>} : !fir.ref<!fir.array<5xi32>>, !fir.ref<!fir.array<5xi32>>
  return
}

// CHECK-LABEL: func.func @_QPsub8()
// CHECK: %[[ALLOCA:.*]] = fir.alloca !fir.array<5xi32> 
// CHECK: %[[LOCAL:.*]] = fir.declare %[[ALLOCA]]
// CHECK: %[[GBL:.*]] = fir.address_of(@_QMmtestsEn) : !fir.ref<!fir.array<5xi32>>
// CHECK: %[[DECL:.*]] = fir.declare %[[GBL]]
// CHECK: %[[HOST:.*]] = fir.convert %[[DECL]] : (!fir.ref<!fir.array<5xi32>>) -> !fir.llvm_ptr<i8>
// CHECK: %[[SRC:.*]] = fir.call @_FortranACUFGetDeviceAddress(%[[HOST]], %{{.*}}, %{{.*}}) : (!fir.llvm_ptr<i8>, !fir.ref<i8>, i32) -> !fir.llvm_ptr<i8>
// CHECK: %[[DST:.*]] = fir.convert %[[LOCAL]] : (!fir.ref<!fir.array<5xi32>>) -> !fir.llvm_ptr<i8>
// CHECK: fir.call @_FortranACUFDataTransferPtrPtr(%[[DST]], %[[SRC]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!fir.llvm_ptr<i8>, !fir.llvm_ptr<i8>, i64, i32, !fir.ref<i8>, i32) -> none


func.func @_QPsub9() {
  %c5 = arith.constant 5 : index
  %0 = fir.alloca !fir.array<5xi32> {bindc_name = "m", uniq_name = "_QFtest9Em"}
  %1 = fir.shape %c5 : (index) -> !fir.shape<1>
  %2 = fir.declare %0(%1) {uniq_name = "_QFtest9Em"} : (!fir.ref<!fir.array<5xi32>>, !fir.shape<1>) -> !fir.ref<!fir.array<5xi32>>
  %3 = fir.address_of(@_QMmtestsEn) : !fir.ref<!fir.array<5xi32>>
  %4 = fir.declare %3(%1) {data_attr = #cuf.cuda<device>, uniq_name = "_QMmtestsEn"} : (!fir.ref<!fir.array<5xi32>>, !fir.shape<1>) -> !fir.ref<!fir.array<5xi32>>
  cuf.data_transfer %2 to %4 {transfer_kind = #cuf.cuda_transfer<host_device>} : !fir.ref<!fir.array<5xi32>>, !fir.ref<!fir.array<5xi32>>
  return
}

// CHECK-LABEL: func.func @_QPsub9()
// CHECK: %[[ALLOCA:.*]] = fir.alloca !fir.array<5xi32> 
// CHECK: %[[LOCAL:.*]] = fir.declare %[[ALLOCA]]
// CHECK: %[[GBL:.*]] = fir.address_of(@_QMmtestsEn) : !fir.ref<!fir.array<5xi32>>
// CHECK: %[[DECL:.*]] = fir.declare %[[GBL]]
// CHECK: %[[HOST:.*]] = fir.convert %[[DECL]] : (!fir.ref<!fir.array<5xi32>>) -> !fir.llvm_ptr<i8>
// CHECK: %[[DST:.*]] = fir.call @_FortranACUFGetDeviceAddress(%[[HOST]], %{{.*}}, %{{.*}}) : (!fir.llvm_ptr<i8>, !fir.ref<i8>, i32) -> !fir.llvm_ptr<i8>
// CHECK: %[[SRC:.*]] = fir.convert %[[LOCAL]] : (!fir.ref<!fir.array<5xi32>>) -> !fir.llvm_ptr<i8>
// CHECK: fir.call @_FortranACUFDataTransferPtrPtr(%[[DST]], %[[SRC]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!fir.llvm_ptr<i8>, !fir.llvm_ptr<i8>, i64, i32, !fir.ref<i8>, i32) -> none

fir.global @_QMmod1Ea {data_attr = #cuf.cuda<device>} : !fir.box<!fir.heap<!fir.array<?xi32>>> {
  %c0 = arith.constant 0 : index
  %0 = fir.zero_bits !fir.heap<!fir.array<?xi32>>
  %1 = fir.shape %c0 : (index) -> !fir.shape<1>
  %2 = fir.embox %0(%1) {allocator_idx = 2 : i32} : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
  fir.has_value %2 : !fir.box<!fir.heap<!fir.array<?xi32>>>
}

func.func @_QQdesc_global() attributes {fir.bindc_name = "host_sub"} {
  %0 = fir.address_of(@_QMmod1Ea) : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  %1:2 = hlfir.declare %0 {data_attr = #cuf.cuda<device>, fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QMmod1Ea"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
  %2 = fir.address_of(@_QFEahost) : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  %3:2 = hlfir.declare %2 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFEahost"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
  cuf.data_transfer %3#0 to %1#0 {transfer_kind = #cuf.cuda_transfer<host_device>} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  return
}

// CHECK-LABEL: func.func @_QQdesc_global() attributes {fir.bindc_name = "host_sub"}
// CHECK: %[[GLOBAL_ADDRESS:.*]] = fir.address_of(@_QMmod1Ea) : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
// CHECK: %[[GLOBAL_DECL:.*]]:2 = hlfir.declare %[[GLOBAL_ADDRESS]] {data_attr = #cuf.cuda<device>, fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QMmod1Ea"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
// CHECK: %[[BOX_NONE:.*]] = fir.convert %[[GLOBAL_DECL:.*]]#0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
// CHECK: fir.call @_FortranACUFDataTransferGlobalDescDesc(%[[BOX_NONE]],{{.*}}) : (!fir.ref<!fir.box<none>>, !fir.ref<!fir.box<none>>, i32, !fir.ref<i8>, i32) -> none

fir.global @_QMmod2Eadev {data_attr = #cuf.cuda<device>} : !fir.box<!fir.heap<!fir.array<?xi32>>> {
  %c0 = arith.constant 0 : index
  %0 = fir.zero_bits !fir.heap<!fir.array<?xi32>>
  %1 = fir.shape %c0 : (index) -> !fir.shape<1>
  %2 = fir.embox %0(%1) {allocator_idx = 2 : i32} : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
  fir.has_value %2 : !fir.box<!fir.heap<!fir.array<?xi32>>>
}
func.func @_QPdesc_global_ptr() {
  %c10 = arith.constant 10 : index
  %0 = fir.address_of(@_QMmod2Eadev) : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  %1 = fir.declare %0 {data_attr = #cuf.cuda<device>, fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QMmod2Eadev"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  %2 = fir.alloca !fir.array<10xi32> {bindc_name = "ahost", uniq_name = "_QFdesc_global_ptrEahost"}
  %3 = fir.shape %c10 : (index) -> !fir.shape<1>
  %4 = fir.declare %2(%3) {uniq_name = "_QFdesc_global_ptrEahost"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.ref<!fir.array<10xi32>>
  cuf.data_transfer %4 to %1 {transfer_kind = #cuf.cuda_transfer<host_device>} : !fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  return
}

// CHECK-LABEL: func.func @_QPdesc_global_ptr()
// CHECK: %[[TEMP_BOX:.*]] = fir.alloca !fir.box<!fir.array<10xi32>>
// CHECK: %[[ADDR_ADEV:.*]] = fir.address_of(@_QMmod2Eadev) : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
// CHECK: %[[DECL_ADEV:.*]] = fir.declare %[[ADDR_ADEV]] {data_attr = #cuf.cuda<device>, fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QMmod2Eadev"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
// CHECK: %[[AHOST:.*]] = fir.alloca !fir.array<10xi32> {bindc_name = "ahost", uniq_name = "_QFdesc_global_ptrEahost"}
// CHECK: %[[SHAPE:.*]] = fir.shape %c10 : (index) -> !fir.shape<1>
// CHECK: %[[DECL_AHOST:.*]] = fir.declare %[[AHOST]](%[[SHAPE]]) {uniq_name = "_QFdesc_global_ptrEahost"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.ref<!fir.array<10xi32>>
// CHECK: %[[EMBOX:.*]] = fir.embox %[[DECL_AHOST]](%[[SHAPE]]) : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<10xi32>>
// CHECK: fir.store %[[EMBOX]] to %[[TEMP_BOX]] : !fir.ref<!fir.box<!fir.array<10xi32>>>
// CHECK: %[[ADEV_BOXNONE:.*]] = fir.convert %[[DECL_ADEV]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
// CHECK: %[[AHOST_BOXNONE:.*]] = fir.convert %[[TEMP_BOX]] : (!fir.ref<!fir.box<!fir.array<10xi32>>>) -> !fir.ref<!fir.box<none>>
// CHECK: fir.call @_FortranACUFDataTransferGlobalDescDesc(%[[ADEV_BOXNONE]], %[[AHOST_BOXNONE]], %c0{{.*}}, %{{.*}}, %{{.*}}) : (!fir.ref<!fir.box<none>>, !fir.ref<!fir.box<none>>, i32, !fir.ref<i8>, i32) -> none

} // end of module
