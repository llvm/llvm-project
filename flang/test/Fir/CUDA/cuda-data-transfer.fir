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
// CHECK: %[[AHOST_LOAD:.*]] = fir.load %[[AHOST]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
// CHECK: %[[ADEV_LOAD:.*]] = fir.load %[[ADEV]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
// CHECK: %[[AHOST_BOX:.*]] = fir.convert %[[AHOST_LOAD]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.box<none>
// CHECK: %[[ADEV_BOX:.*]] = fir.convert %[[ADEV_LOAD]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.box<none>
// CHECK: fir.call @_FortranACUFDataTransferDescDesc(%[[AHOST_BOX]], %[[ADEV_BOX]], %c1{{.*}}, %{{.*}}, %{{.*}}) : (!fir.box<none>, !fir.box<none>, i32, !fir.ref<i8>, i32) -> none

func.func @_QPsub2() {
  %0 = cuf.alloc !fir.box<!fir.heap<!fir.array<?xi32>>> {bindc_name = "adev", data_attr = #cuf.cuda<device>, uniq_name = "_QFsub2Eadev"} -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  %4:2 = hlfir.declare %0 {data_attr = #cuf.cuda<device>, fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFsub2Eadev"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
  %c2_i32 = arith.constant 2 : i32
  cuf.data_transfer %c2_i32 to %4#0 {transfer_kind = #cuf.cuda_transfer<host_device>} : i32, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  cuf.free %4#1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {data_attr = #cuf.cuda<device>}
  return
}

// CHECK-LABEL: func.func @_QPsub2()
// CHECK: %[[TEMP:.*]] = fir.alloca i32
// CHECK: %[[ADEV:.*]]:2 = hlfir.declare %{{.*}} {data_attr = #cuf.cuda<device>, fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFsub2Eadev"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
// CHECK: %[[C2:.*]] = arith.constant 2 : i32
// CHECK: fir.store %[[C2]] to %[[TEMP]] : !fir.ref<i32>
// CHECK: %[[ADEV_LOAD:.*]] = fir.load %[[ADEV]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
// CHECK: %[[ADEV_BOX:.*]] = fir.convert %[[ADEV_LOAD]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.box<none>
// CHECK: %[[TEMP_CONV:.*]] = fir.convert %[[TEMP]] : (!fir.ref<i32>) -> !fir.llvm_ptr<i8>
// CHECK: fir.call @_FortranACUFMemsetDescriptor(%[[ADEV_BOX]], %[[TEMP_CONV]], %{{.*}}, %{{.*}}) : (!fir.box<none>, !fir.llvm_ptr<i8>, !fir.ref<i8>, i32) -> none

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
// CHECK: %[[ADEV:.*]]:2 = hlfir.declare %{{.*}} {data_attr = #cuf.cuda<device>, fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFsub3Eadev"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
// CHECK: %[[V:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFsub3Ev"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK: %[[ADEV_LOAD:.*]] = fir.load %[[ADEV]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
// CHECK: %[[ADEV_BOX:.*]] = fir.convert %[[ADEV_LOAD]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.box<none>
// CHECK: %[[V_CONV:.*]] = fir.convert %[[V]]#0 : (!fir.ref<i32>) -> !fir.llvm_ptr<i8>
// CHECK: fir.call @_FortranACUFMemsetDescriptor(%[[ADEV_BOX]], %[[V_CONV]], %{{.*}}, %{{.*}}) : (!fir.box<none>, !fir.llvm_ptr<i8>, !fir.ref<i8>, i32) -> none

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
// CHECK: %[[ADEV:.*]]:2 = hlfir.declare %{{.*}} {data_attr = #cuf.cuda<device>, fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFsub4Eadev"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
// CHECK: %[[AHOST:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFsub4Eahost"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
// CHECK: %[[NBELEM:.*]] = arith.constant 10 : index
// CHECK: %[[WIDTH:.*]] = arith.constant 4 : index
// CHECK: %[[BYTES:.*]] = arith.muli %[[NBELEM]], %[[WIDTH]] : index
// CHECK: %[[ADEV_LOAD:.*]] = fir.load %[[ADEV]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
// CHECK: %[[ADEV_BOX:.*]] = fir.convert %[[ADEV_LOAD]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.box<none>
// CHECK: %[[AHOST_PTR:.*]] = fir.convert %[[AHOST]]#0 : (!fir.ref<!fir.array<10xi32>>) -> !fir.llvm_ptr<i8>
// CHECK: %[[BYTES_CONV:.*]] = fir.convert %[[BYTES]] : (index) -> i64
// CHECK: fir.call @_FortranACUFDataTransferDescPtr(%[[ADEV_BOX]], %[[AHOST_PTR]], %[[BYTES_CONV]], %c0{{.*}}, %{{.*}}, %{{.*}}) : (!fir.box<none>, !fir.llvm_ptr<i8>, i64, i32, !fir.ref<i8>, i32) -> none
// CHECK: %[[NBELEM:.*]] = arith.constant 10 : index
// CHECK: %[[WIDTH:.*]] = arith.constant 4 : index
// CHECK: %[[BYTES:.*]] = arith.muli %[[NBELEM]], %[[WIDTH]] : index
// CHECK: %[[ADEV_LOAD:.*]] = fir.load %[[ADEV]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
// CHECK: %[[AHOST_PTR:.*]] = fir.convert %[[AHOST]]#0 : (!fir.ref<!fir.array<10xi32>>) -> !fir.llvm_ptr<i8>
// CHECK: %[[ADEV_BOX:.*]] = fir.convert %[[ADEV_LOAD]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.box<none>
// CHECK: %[[BYTES_CONV:.*]] = fir.convert %[[BYTES]] : (index) -> i64
// CHECK: fir.call @_FortranACUFDataTransferPtrDesc(%[[AHOST_PTR]], %[[ADEV_BOX]], %[[BYTES_CONV]], %c1{{.*}}, %{{.*}}, %{{.*}}) : (!fir.llvm_ptr<i8>, !fir.box<none>, i64, i32, !fir.ref<i8>, i32) -> none

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
// CHECK: %[[ADEV:.*]]:2 = hlfir.declare %{{.*}} {data_attr = #cuf.cuda<device>, fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFsub5Eadev"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>)
// CHECK: %[[SHAPE:.*]] = fir.shape %[[I1:.*]], %[[I2:.*]] : (index, index) -> !fir.shape<2>
// CHECK: %[[AHOST:.*]]:2 = hlfir.declare %{{.*}}(%[[SHAPE]]) {uniq_name = "_QFsub5Eahost"} : (!fir.ref<!fir.array<?x?xi32>>, !fir.shape<2>) -> (!fir.box<!fir.array<?x?xi32>>, !fir.ref<!fir.array<?x?xi32>>)
// CHECK: %[[NBELEM:.*]] = arith.muli %[[I1]], %[[I2]] : index
// CHECK: %[[WIDTH:.*]] = arith.constant 4 : index
// CHECK: %[[BYTES:.*]] = arith.muli %[[NBELEM]], %[[WIDTH]] : index
// CHECK: %[[ADEV_LOAD:.*]] = fir.load %[[ADEV]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>
// CHECK: %[[ADEV_BOX:.*]] = fir.convert %[[ADEV_LOAD]] : (!fir.box<!fir.heap<!fir.array<?x?xi32>>>) -> !fir.box<none>
// CHECK: %[[AHOST_PTR:.*]] = fir.convert %[[AHOST]]#1 : (!fir.ref<!fir.array<?x?xi32>>) -> !fir.llvm_ptr<i8>
// CHECK: %[[BYTES_CONV:.*]] = fir.convert %[[BYTES]] : (index) -> i64
// CHECK: fir.call @_FortranACUFDataTransferDescPtr(%[[ADEV_BOX]], %[[AHOST_PTR]], %[[BYTES_CONV]], %c0{{.*}}, %{{.*}}, %{{.*}}) : (!fir.box<none>, !fir.llvm_ptr<i8>, i64, i32, !fir.ref<i8>, i32) -> none
// CHECK: %[[NBELEM:.*]] = arith.muli %[[I1]], %[[I2]] : index
// CHECK: %[[WIDTH:.*]] = arith.constant 4 : index
// CHECK: %[[BYTES:.*]] = arith.muli %[[NBELEM]], %[[WIDTH]] : index
// CHECK: %[[ADEV_LOAD:.*]] = fir.load %[[ADEV]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>
// CHECK: %[[AHOST_PTR:.*]] = fir.convert %[[AHOST]]#1 : (!fir.ref<!fir.array<?x?xi32>>) -> !fir.llvm_ptr<i8>
// CHECK: %[[ADEV_BOX:.*]] = fir.convert %[[ADEV_LOAD]] : (!fir.box<!fir.heap<!fir.array<?x?xi32>>>) -> !fir.box<none>
// CHECK: %[[BYTES_CONV:.*]] = fir.convert %[[BYTES]] : (index) -> i64
// CHECK: fir.call @_FortranACUFDataTransferPtrDesc(%[[AHOST_PTR]], %[[ADEV_BOX]], %[[BYTES_CONV]], %c1{{.*}}, %{{.*}}, %{{.*}}) : (!fir.llvm_ptr<i8>, !fir.box<none>, i64, i32, !fir.ref<i8>, i32) -> none

} // end of module
