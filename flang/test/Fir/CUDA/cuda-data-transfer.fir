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
// CHECK: fir.call @_FortranACUFDataTransferDescDesc(%[[AHOST_BOX]], %[[ADEV_BOX]], %c1{{.*}}, %{{.*}}, %{{.*}}) : (!fir.ref<!fir.box<none>>, !fir.ref<!fir.box<none>>, i32, !fir.ref<i8>, i32) -> ()

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
// CHECK: fir.call @_FortranACUFDataTransferCstDesc(%[[ADEV_BOX]], %[[TEMP_CONV]], %c0{{.*}}, %{{.*}}, %{{.*}}) : (!fir.ref<!fir.box<none>>, !fir.ref<!fir.box<none>>, i32, !fir.ref<i8>, i32) -> ()

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
// CHECK: fir.call @_FortranACUFDataTransferCstDesc(%[[ADEV_BOX]], %[[V_CONV]], %c0{{.*}}, %{{.*}}, %{{.*}}) : (!fir.ref<!fir.box<none>>, !fir.ref<!fir.box<none>>, i32, !fir.ref<i8>, i32) -> ()
  
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
// CHECK: %[[TEMP_BOX1:.*]] = fir.alloca !fir.box<!fir.array<10xi32>>
// CHECK: %[[TEMP_BOX:.*]] = fir.alloca !fir.box<!fir.array<10xi32>>
// CHECK: %[[ADEV:.*]]:2 = hlfir.declare %{{.*}} {data_attr = #cuf.cuda<device>, fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFsub4Eadev"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
// CHECK: %[[AHOST:.*]]:2 = hlfir.declare %{{.*}}(%[[AHOST_SHAPE:.*]]) {uniq_name = "_QFsub4Eahost"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
// CHECK: %[[EMBOX:.*]] = fir.embox %[[AHOST]]#0(%[[AHOST_SHAPE]]) : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<10xi32>>
// CHECK: fir.store %[[EMBOX]] to %[[TEMP_BOX]] : !fir.ref<!fir.box<!fir.array<10xi32>>>
// CHECK: %[[ADEV_BOX:.*]] = fir.convert %[[ADEV]]#0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
// CHECK: %[[AHOST_BOX:.*]] = fir.convert %[[TEMP_BOX]] : (!fir.ref<!fir.box<!fir.array<10xi32>>>) -> !fir.ref<!fir.box<none>>
// CHECK: fir.call @_FortranACUFDataTransferDescDesc(%[[ADEV_BOX]], %[[AHOST_BOX]], %c0{{.*}}, %{{.*}}, %{{.*}}) : (!fir.ref<!fir.box<none>>, !fir.ref<!fir.box<none>>, i32, !fir.ref<i8>, i32) -> ()
// CHECK: %[[EMBOX:.*]] = fir.embox %[[AHOST]]#0(%[[AHOST_SHAPE]]) : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<10xi32>>
// CHECK: fir.store %[[EMBOX]] to %[[TEMP_BOX1]] : !fir.ref<!fir.box<!fir.array<10xi32>>>
// CHECK: %[[AHOST_BOX:.*]] = fir.convert %[[TEMP_BOX1]] : (!fir.ref<!fir.box<!fir.array<10xi32>>>) -> !fir.ref<!fir.box<none>>
// CHECK: %[[ADEV_BOX:.*]] = fir.convert %[[ADEV]]#0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
// CHECK: fir.call @_FortranACUFDataTransferDescDescNoRealloc(%[[AHOST_BOX]], %[[ADEV_BOX]], %c1{{.*}}, %{{.*}}, %{{.*}}) : (!fir.ref<!fir.box<none>>, !fir.ref<!fir.box<none>>, i32, !fir.ref<i8>, i32) -> ()

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
// CHECK: %[[TEMP_BOX1:.*]] = fir.alloca !fir.box<!fir.array<?x?xi32>>
// CHECK: %[[TEMP_BOX:.*]] = fir.alloca !fir.box<!fir.array<?x?xi32>>
// CHECK: %[[ADEV:.*]]:2 = hlfir.declare %{{.*}} {data_attr = #cuf.cuda<device>, fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFsub5Eadev"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>)
// CHECK: %[[SHAPE:.*]] = fir.shape %[[I1:.*]], %[[I2:.*]] : (index, index) -> !fir.shape<2>
// CHECK: %[[AHOST:.*]]:2 = hlfir.declare %{{.*}}(%[[SHAPE]]) {uniq_name = "_QFsub5Eahost"} : (!fir.ref<!fir.array<?x?xi32>>, !fir.shape<2>) -> (!fir.box<!fir.array<?x?xi32>>, !fir.ref<!fir.array<?x?xi32>>)
// CHECK: %[[EMBOX:.*]] = fir.embox %[[AHOST]]#1(%[[SHAPE]]) : (!fir.ref<!fir.array<?x?xi32>>, !fir.shape<2>) -> !fir.box<!fir.array<?x?xi32>>
// CHECK: fir.store %[[EMBOX]] to %[[TEMP_BOX]] : !fir.ref<!fir.box<!fir.array<?x?xi32>>>
// CHECK: %[[ADEV_BOX:.*]] = fir.convert %[[ADEV]]#0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>) -> !fir.ref<!fir.box<none>>
// CHECK: %[[AHOST_BOX:.*]] = fir.convert %[[TEMP_BOX]] : (!fir.ref<!fir.box<!fir.array<?x?xi32>>>) -> !fir.ref<!fir.box<none>>
// CHECK: fir.call @_FortranACUFDataTransferDescDesc(%[[ADEV_BOX]], %[[AHOST_BOX]], %c0{{.*}}, %{{.*}}, %{{.*}}) : (!fir.ref<!fir.box<none>>, !fir.ref<!fir.box<none>>, i32, !fir.ref<i8>, i32) -> ()
// CHECK: %[[EMBOX:.*]] = fir.embox %[[AHOST]]#1(%[[SHAPE]]) : (!fir.ref<!fir.array<?x?xi32>>, !fir.shape<2>) -> !fir.box<!fir.array<?x?xi32>>
// CHECK: fir.store %[[EMBOX]] to %[[TEMP_BOX1]] : !fir.ref<!fir.box<!fir.array<?x?xi32>>>
// CHECK: %[[AHOST_BOX:.*]] = fir.convert %[[TEMP_BOX1]] : (!fir.ref<!fir.box<!fir.array<?x?xi32>>>) -> !fir.ref<!fir.box<none>>
// CHECK: %[[ADEV_BOX:.*]] = fir.convert %[[ADEV]]#0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>) -> !fir.ref<!fir.box<none>>
// CHECK: fir.call @_FortranACUFDataTransferDescDescNoRealloc(%[[AHOST_BOX]], %[[ADEV_BOX]], %c1{{.*}}, %{{.*}}, %{{.*}}) : (!fir.ref<!fir.box<none>>, !fir.ref<!fir.box<none>>, i32, !fir.ref<i8>, i32) -> ()

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
// CHECK: fir.call @_FortranACUFDataTransferPtrPtr(%[[DST]], %[[SRC]], %c4{{.*}}, %c1{{.*}}, %{{.*}}, %{{.*}}) : (!fir.llvm_ptr<i8>, !fir.llvm_ptr<i8>, i64, i32, !fir.ref<i8>, i32) -> ()
// CHECK: %[[LOAD:.*]] = fir.load %[[IHOST]]#0 : !fir.ref<i32>
// CHECK: %[[ASSOC:.*]]:3 = hlfir.associate %[[LOAD]] {uniq_name = ".cuf_host_tmp"} : (i32) -> (!fir.ref<i32>, !fir.ref<i32>, i1)
// CHECK: %[[DST:.*]] = fir.convert %[[IDEV]]#0 : (!fir.ref<i32>) -> !fir.llvm_ptr<i8>
// CHECK: %[[SRC:.*]] = fir.convert %[[ASSOC]]#0 : (!fir.ref<i32>) -> !fir.llvm_ptr<i8>
// CHECK: fir.call @_FortranACUFDataTransferPtrPtr(%[[DST]], %[[SRC]], %c4{{.*}}, %c0{{.*}}, %{{.*}}, %{{.*}}) : (!fir.llvm_ptr<i8>, !fir.llvm_ptr<i8>, i64, i32, !fir.ref<i8>, i32) -> ()

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
// CHECK: fir.call @_FortranACUFDataTransferPtrPtr(%[[DST]], %[[SRC]], %[[BYTES]], %c1{{.*}}, %{{.*}}, %{{.*}}) : (!fir.llvm_ptr<i8>, !fir.llvm_ptr<i8>, i64, i32, !fir.ref<i8>, i32) -> ()
// CHECK: %[[BYTES:.*]] = arith.muli %c10{{.*}}, %c4{{.*}} : i64
// CHECK: %[[DST:.*]] = fir.convert %[[IDEV]]#0 : (!fir.ref<!fir.array<10xi32>>) -> !fir.llvm_ptr<i8>
// CHECK: %[[SRC:.*]] = fir.convert %[[IHOST]]#0 : (!fir.ref<!fir.array<10xi32>>) -> !fir.llvm_ptr<i8>
// CHECK: fir.call @_FortranACUFDataTransferPtrPtr(%[[DST]], %[[SRC]], %[[BYTES]], %c0{{.*}}, %{{.*}}, %{{.*}}) : (!fir.llvm_ptr<i8>, !fir.llvm_ptr<i8>, i64, i32, !fir.ref<i8>, i32) -> ()

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
// CHECK: fir.address_of(@_QMmtestsEn) : !fir.ref<!fir.array<5xi32>>
// CHECK: %[[GBL:.*]] = fir.address_of(@_QMmtestsEn) : !fir.ref<!fir.array<5xi32>>
// CHECK: %[[GBL_CONV:.*]] = fir.convert %[[GBL]] : (!fir.ref<!fir.array<5xi32>>) -> !fir.llvm_ptr<i8>
// CHECK: %[[ADDR:.*]] = fir.call @_FortranACUFGetDeviceAddress(%[[GBL_CONV]], %{{.*}}, %{{.*}}) : (!fir.llvm_ptr<i8>, !fir.ref<i8>, i32) -> !fir.llvm_ptr<i8>
// CHECK: %[[ADDR_CONV:.*]] = fir.convert %[[ADDR]] : (!fir.llvm_ptr<i8>) -> !fir.ref<!fir.array<5xi32>>
// CHECK: %[[DECL:.*]] = fir.declare %[[ADDR_CONV]]
// CHECK: %[[DST:.*]] = fir.convert %[[LOCAL]] : (!fir.ref<!fir.array<5xi32>>) -> !fir.llvm_ptr<i8>
// CHECK: %[[SRC:.*]] = fir.convert %[[DECL]] : (!fir.ref<!fir.array<5xi32>>) -> !fir.llvm_ptr<i8>
// CHECK: fir.call @_FortranACUFDataTransferPtrPtr(%[[DST]], %[[SRC]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!fir.llvm_ptr<i8>, !fir.llvm_ptr<i8>, i64, i32, !fir.ref<i8>, i32) -> ()


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
// CHECK: fir.address_of(@_QMmtestsEn) : !fir.ref<!fir.array<5xi32>>
// CHECK: %[[GBL:.*]] = fir.address_of(@_QMmtestsEn) : !fir.ref<!fir.array<5xi32>>
// CHECK: %[[GBL_CONV:.*]] = fir.convert %[[GBL]] : (!fir.ref<!fir.array<5xi32>>) -> !fir.llvm_ptr<i8>
// CHECK: %[[ADDR:.*]] = fir.call @_FortranACUFGetDeviceAddress(%[[GBL_CONV]], %{{.*}}, %{{.*}}) : (!fir.llvm_ptr<i8>, !fir.ref<i8>, i32) -> !fir.llvm_ptr<i8>
// CHECK: %[[ADDR_CONV:.*]] = fir.convert %[[ADDR]] : (!fir.llvm_ptr<i8>) -> !fir.ref<!fir.array<5xi32>>
// CHECK: %[[DECL:.*]] = fir.declare %[[ADDR_CONV]]
// CHECK: %[[DST:.*]] = fir.convert %[[DECL]] : (!fir.ref<!fir.array<5xi32>>) -> !fir.llvm_ptr<i8>
// CHECK: %[[SRC:.*]] = fir.convert %[[LOCAL]] : (!fir.ref<!fir.array<5xi32>>) -> !fir.llvm_ptr<i8>
// CHECK: fir.call @_FortranACUFDataTransferPtrPtr(%[[DST]], %[[SRC]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!fir.llvm_ptr<i8>, !fir.llvm_ptr<i8>, i64, i32, !fir.ref<i8>, i32) -> ()

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
// CHECK: fir.call @_FortranACUFDataTransferGlobalDescDesc(%[[BOX_NONE]],{{.*}}) : (!fir.ref<!fir.box<none>>, !fir.ref<!fir.box<none>>, i32, !fir.ref<i8>, i32) -> ()

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
// CHECK: fir.call @_FortranACUFDataTransferGlobalDescDesc(%[[ADEV_BOXNONE]], %[[AHOST_BOXNONE]], %c0{{.*}}, %{{.*}}, %{{.*}}) : (!fir.ref<!fir.box<none>>, !fir.ref<!fir.box<none>>, i32, !fir.ref<i8>, i32) -> ()

func.func @_QPscalar_to_array() {
  %c1_i32 = arith.constant 1 : i32
  %c10 = arith.constant 10 : index
  %0 = cuf.alloc !fir.array<10xi32> {bindc_name = "a", data_attr = #cuf.cuda<device>, uniq_name = "_QFscalar_to_arrayEa"} -> !fir.ref<!fir.array<10xi32>>
  %1 = fir.shape %c10 : (index) -> !fir.shape<1>
  %2:2 = hlfir.declare %0(%1) {data_attr = #cuf.cuda<device>, uniq_name = "_QFscalar_to_arrayEa"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
  cuf.data_transfer %c1_i32 to %2#0 {transfer_kind = #cuf.cuda_transfer<host_device>} : i32, !fir.ref<!fir.array<10xi32>>
  cuf.free %2#1 : !fir.ref<!fir.array<10xi32>> {data_attr = #cuf.cuda<device>}
  return
}

// CHECK-LABEL: func.func @_QPscalar_to_array()
// CHECK: _FortranACUFDataTransferCstDesc

func.func @_QPtest_type() {
  %0 = cuf.alloc !fir.type<_QMbarTcmplx{id:i32,c:complex<f32>}> {bindc_name = "a", data_attr = #cuf.cuda<device>, uniq_name = "_QFtest_typeEa"} -> !fir.ref<!fir.type<_QMbarTcmplx{id:i32,c:complex<f32>}>>
  %1 = fir.declare %0 {data_attr = #cuf.cuda<device>, uniq_name = "_QFtest_typeEa"} : (!fir.ref<!fir.type<_QMbarTcmplx{id:i32,c:complex<f32>}>>) -> !fir.ref<!fir.type<_QMbarTcmplx{id:i32,c:complex<f32>}>>
  %2 = fir.alloca !fir.type<_QMbarTcmplx{id:i32,c:complex<f32>}> {bindc_name = "b", uniq_name = "_QFtest_typeEb"}
  %3 = fir.declare %2 {uniq_name = "_QFtest_typeEb"} : (!fir.ref<!fir.type<_QMbarTcmplx{id:i32,c:complex<f32>}>>) -> !fir.ref<!fir.type<_QMbarTcmplx{id:i32,c:complex<f32>}>>
  cuf.data_transfer %3 to %1 {transfer_kind = #cuf.cuda_transfer<host_device>} : !fir.ref<!fir.type<_QMbarTcmplx{id:i32,c:complex<f32>}>>, !fir.ref<!fir.type<_QMbarTcmplx{id:i32,c:complex<f32>}>>
  cuf.free %1 : !fir.ref<!fir.type<_QMbarTcmplx{id:i32,c:complex<f32>}>> {data_attr = #cuf.cuda<device>}
  return
}

// CHECK-LABEL: func.func @_QPtest_type()
// CHECK: fir.call @_FortranACUFDataTransferPtrPtr(%{{.*}}, %{{.*}}, %c12{{.*}}, %c0{{.*}}, %{{.*}}, %{{.*}}) : (!fir.llvm_ptr<i8>, !fir.llvm_ptr<i8>, i64, i32, !fir.ref<i8>, i32) -> ()

func.func @_QPtest_array_type() {
  %c10 = arith.constant 10 : index
  %0 = cuf.alloc !fir.array<10x!fir.type<_QMbarTcmplx{id:i32,c:complex<f32>}>> {bindc_name = "a", data_attr = #cuf.cuda<device>, uniq_name = "_QFtest_array_typeEa"} -> !fir.ref<!fir.array<10x!fir.type<_QMbarTcmplx{id:i32,c:complex<f32>}>>>
  %1 = fir.shape %c10 : (index) -> !fir.shape<1>
  %2 = fir.declare %0(%1) {data_attr = #cuf.cuda<device>, uniq_name = "_QFtest_array_typeEa"} : (!fir.ref<!fir.array<10x!fir.type<_QMbarTcmplx{id:i32,c:complex<f32>}>>>, !fir.shape<1>) -> !fir.ref<!fir.array<10x!fir.type<_QMbarTcmplx{id:i32,c:complex<f32>}>>>
  %3 = fir.alloca !fir.array<10x!fir.type<_QMbarTcmplx{id:i32,c:complex<f32>}>> {bindc_name = "b", uniq_name = "_QFtest_array_typeEb"}
  %4 = fir.declare %3(%1) {uniq_name = "_QFtest_array_typeEb"} : (!fir.ref<!fir.array<10x!fir.type<_QMbarTcmplx{id:i32,c:complex<f32>}>>>, !fir.shape<1>) -> !fir.ref<!fir.array<10x!fir.type<_QMbarTcmplx{id:i32,c:complex<f32>}>>>
  cuf.data_transfer %4 to %2 {transfer_kind = #cuf.cuda_transfer<host_device>} : !fir.ref<!fir.array<10x!fir.type<_QMbarTcmplx{id:i32,c:complex<f32>}>>>, !fir.ref<!fir.array<10x!fir.type<_QMbarTcmplx{id:i32,c:complex<f32>}>>>
  cuf.free %2 : !fir.ref<!fir.array<10x!fir.type<_QMbarTcmplx{id:i32,c:complex<f32>}>>> {data_attr = #cuf.cuda<device>}
  return
}

// CHECK-LABEL: func.func @_QPtest_array_type()
// CHECK: %[[BYTES:.*]] = arith.muli %c10{{.*}}, %c12 : index
// CHECK: %[[CONV_BYTES:.*]] = fir.convert %[[BYTES]] : (index) -> i64
// CHECK: fir.call @_FortranACUFMemAlloc(%[[CONV_BYTES]], %c0{{.*}}, %{{.*}}, %{{.*}}) {cuf.data_attr = #cuf.cuda<device>} : (i64, i32, !fir.ref<i8>, i32) -> !fir.llvm_ptr<i8>
// CHECK: %[[BYTES:.*]] = arith.muli %c10{{.*}}, %c12{{.*}} : i64
// CHECK: fir.call @_FortranACUFDataTransferPtrPtr(%{{.*}}, %{{.*}}, %[[BYTES]], %c0{{.*}}, %{{.*}}, %{{.*}}) : (!fir.llvm_ptr<i8>, !fir.llvm_ptr<i8>, i64, i32, !fir.ref<i8>, i32) -> ()

func.func @_QPshape_shift() {
  %c0_i32 = arith.constant 0 : i32
  %c11 = arith.constant 11 : index
  %c10 = arith.constant 10 : index
  %0 = cuf.alloc !fir.array<10xi32> {bindc_name = "cdev", data_attr = #cuf.cuda<device>, uniq_name = "_QFshape_shiftEcdev"} -> !fir.ref<!fir.array<10xi32>>
  %1 = fir.shape_shift %c11, %c10 : (index, index) -> !fir.shapeshift<1>
  %2:2 = hlfir.declare %0(%1) {data_attr = #cuf.cuda<device>, uniq_name = "_QFshape_shiftEcdev"} : (!fir.ref<!fir.array<10xi32>>, !fir.shapeshift<1>) -> (!fir.box<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
  cuf.data_transfer %c0_i32 to %2#1, %1 : !fir.shapeshift<1> {transfer_kind = #cuf.cuda_transfer<host_device>} : i32, !fir.ref<!fir.array<10xi32>>
  cuf.free %2#1 : !fir.ref<!fir.array<10xi32>> {data_attr = #cuf.cuda<device>}
  return
}

// CHECK-LABEL: func.func @_QPshape_shift()
// CHECK: fir.call @_FortranACUFDataTransferCstDesc

func.func @_QPshape_shift2() {
  %c11 = arith.constant 11 : index
  %c10 = arith.constant 10 : index
  %0 = fir.alloca !fir.array<10xi32> {bindc_name = "ahost", uniq_name = "_QFshape_shift2Eahost"}
  %1 = fir.shape %c10 : (index) -> !fir.shape<1>
  %2:2 = hlfir.declare %0(%1) {uniq_name = "_QFshape_shift2Eahost"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
  %3 = cuf.alloc !fir.array<10xi32> {bindc_name = "cdev", data_attr = #cuf.cuda<device>, uniq_name = "_QFshape_shift2Ecdev"} -> !fir.ref<!fir.array<10xi32>>
  %4 = fir.shape_shift %c11, %c10 : (index, index) -> !fir.shapeshift<1>
  %5:2 = hlfir.declare %3(%4) {data_attr = #cuf.cuda<device>, uniq_name = "_QFshape_shift2Ecdev"} : (!fir.ref<!fir.array<10xi32>>, !fir.shapeshift<1>) -> (!fir.box<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
  cuf.data_transfer %2#0 to %5#1, %4 : !fir.shapeshift<1> {transfer_kind = #cuf.cuda_transfer<host_device>} : !fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>
  cuf.free %5#1 : !fir.ref<!fir.array<10xi32>> {data_attr = #cuf.cuda<device>}
  return
}

// CHECK-LABEL: func.func @_QPshape_shift2()
// CHECK: %[[C10:.*]] = fir.convert %c10{{.*}} : (index) -> i64
// CHECK: %[[BYTES:.*]] = arith.muli %[[C10]], %c4{{.*}} : i64
// CHECK: fir.call @_FortranACUFDataTransferPtrPtr(%{{.*}}, %{{.*}}, %[[BYTES]], %{{.*}}, %{{.*}}, %{{.*}}) : (!fir.llvm_ptr<i8>, !fir.llvm_ptr<i8>, i64, i32, !fir.ref<i8>, i32) -> ()

fir.global @_QMmod1Ea_dev {data_attr = #cuf.cuda<device>} : !fir.array<4xf32> {
  %0 = fir.zero_bits !fir.array<4xf32>
  fir.has_value %0 : !fir.array<4xf32>
}
func.func @_QPdevice_addr_conv() {
  %cst = arith.constant 4.200000e+01 : f32
  %c4 = arith.constant 4 : index
  %0 = fir.address_of(@_QMmod1Ea_dev) : !fir.ref<!fir.array<4xf32>>
  %1 = fir.shape %c4 : (index) -> !fir.shape<1>
  %2 = fir.declare %0(%1) {data_attr = #cuf.cuda<device>, uniq_name = "_QMmod1Ea_dev"} : (!fir.ref<!fir.array<4xf32>>, !fir.shape<1>) -> !fir.ref<!fir.array<4xf32>>
  cuf.data_transfer %cst to %2 {transfer_kind = #cuf.cuda_transfer<host_device>} : f32, !fir.ref<!fir.array<4xf32>>
  return
}

// CHECK-LABEL: func.func @_QPdevice_addr_conv()
// CHECK: fir.address_of(@_QMmod1Ea_dev) : !fir.ref<!fir.array<4xf32>>
// CHECK: %[[GBL:.*]] = fir.address_of(@_QMmod1Ea_dev) : !fir.ref<!fir.array<4xf32>>
// CHECK: %[[GBL_CONV:.*]] = fir.convert %[[GBL]] : (!fir.ref<!fir.array<4xf32>>) -> !fir.llvm_ptr<i8>
// CHECK: %[[ADDR:.*]] = fir.call @_FortranACUFGetDeviceAddress(%[[GBL_CONV]], %{{.*}}, %{{.*}}) : (!fir.llvm_ptr<i8>, !fir.ref<i8>, i32) -> !fir.llvm_ptr<i8>
// CHECK: %[[ADDR_CONV:.*]] = fir.convert %[[ADDR]] : (!fir.llvm_ptr<i8>) -> !fir.ref<!fir.array<4xf32>>
// CHECK: %[[DECL:.*]] = fir.declare %[[ADDR_CONV]](%{{.*}}) {data_attr = #cuf.cuda<device>, uniq_name = "_QMmod1Ea_dev"} : (!fir.ref<!fir.array<4xf32>>, !fir.shape<1>) -> !fir.ref<!fir.array<4xf32>>
// CHECK: fir.embox %[[DECL]](%{{.*}}) : (!fir.ref<!fir.array<4xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<4xf32>>
// CHECK: fir.call @_FortranACUFDataTransferCstDesc

func.func @_QQchar_transfer() attributes {fir.bindc_name = "char_transfer"} {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %0 = cuf.alloc !fir.array<10x!fir.char<1>>(%c1 : index) {bindc_name = "a", data_attr = #cuf.cuda<device>, uniq_name = "_QFEa"} -> !fir.ref<!fir.array<10x!fir.char<1>>>
  %1 = fir.shape %c10 : (index) -> !fir.shape<1>
  %2 = fir.declare %0(%1) typeparams %c1 {data_attr = #cuf.cuda<device>, uniq_name = "_QFEa"} : (!fir.ref<!fir.array<10x!fir.char<1>>>, !fir.shape<1>, index) -> !fir.ref<!fir.array<10x!fir.char<1>>>
  %3 = fir.alloca !fir.array<10x!fir.char<1>> {bindc_name = "b", uniq_name = "_QFEb"}
  %4 = fir.declare %3(%1) typeparams %c1 {uniq_name = "_QFEb"} : (!fir.ref<!fir.array<10x!fir.char<1>>>, !fir.shape<1>, index) -> !fir.ref<!fir.array<10x!fir.char<1>>>
  cuf.data_transfer %4 to %2 {transfer_kind = #cuf.cuda_transfer<host_device>} : !fir.ref<!fir.array<10x!fir.char<1>>>, !fir.ref<!fir.array<10x!fir.char<1>>>
  cuf.free %2 : !fir.ref<!fir.array<10x!fir.char<1>>> {data_attr = #cuf.cuda<device>}
  return
}

// CHECK-LABEL:  func.func @_QQchar_transfer()
// CHECK: fir.call @_FortranACUFMemAlloc
// CHECK: %[[BYTES:.*]] = arith.muli %c10{{.*}}, %c1{{.*}} : i64
// CHECK: fir.call @_FortranACUFDataTransferPtrPtr(%{{.*}}, %{{.*}}, %[[BYTES]], %c0{{.*}}, %{{.*}}, %{{.*}}) : (!fir.llvm_ptr<i8>, !fir.llvm_ptr<i8>, i64, i32, !fir.ref<i8>, i32) -> ()

func.func @_QPdevmul(%arg0: !fir.ref<!fir.array<1x?xf32>> {fir.bindc_name = "b"}, %arg1: !fir.ref<i32> {fir.bindc_name = "wa"}, %arg2: !fir.ref<i32> {fir.bindc_name = "wb"}) {
  %c0_i64 = arith.constant 0 : i64
  %c1_i32 = arith.constant 1 : i32
  %c0_i32 = arith.constant 0 : i32
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.declare %arg2 dummy_scope %0 {uniq_name = "_QFdevmulEwb"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %2 = cuf.alloc !fir.box<!fir.heap<!fir.array<?x?xf32>>> {bindc_name = "bdev", data_attr = #cuf.cuda<device>, uniq_name = "_QFdevmulEbdev"} -> !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>
  %6 = fir.declare %2 {data_attr = #cuf.cuda<device>, fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFdevmulEbdev"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>
  %7 = fir.declare %arg1 dummy_scope %0 {uniq_name = "_QFdevmulEwa"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %8 = fir.load %1 : !fir.ref<i32>
  %9 = fir.convert %8 : (i32) -> index
  %12 = fir.shape %c1, %9 : (index, index) -> !fir.shape<2>
  %13 = fir.declare %arg0(%12) dummy_scope %0 {uniq_name = "_QFdevmulEb"} : (!fir.ref<!fir.array<1x?xf32>>, !fir.shape<2>, !fir.dscope) -> !fir.ref<!fir.array<1x?xf32>>
  %24 = fir.load %7 : !fir.ref<i32>
  %25 = fir.convert %24 : (i32) -> index
  %26 = arith.cmpi sgt, %25, %c0 : index
  %27 = arith.select %26, %25, %c0 : index
  %28 = fir.load %1 : !fir.ref<i32>
  %29 = fir.convert %28 : (i32) -> index
  %30 = arith.cmpi sgt, %29, %c0 : index
  %31 = arith.select %30, %29, %c0 : index
  %32 = fir.shape %27, %31 : (index, index) -> !fir.shape<2>
  %33 = fir.undefined index
  %34 = fir.slice %c1, %25, %c1, %c1, %29, %c1 : (index, index, index, index, index, index) -> !fir.slice<2>
  %35 = fir.embox %13(%12) [%34] : (!fir.ref<!fir.array<1x?xf32>>, !fir.shape<2>, !fir.slice<2>) -> !fir.box<!fir.array<?x?xf32>>
  cuf.data_transfer %35 to %6 {transfer_kind = #cuf.cuda_transfer<host_device>} : !fir.box<!fir.array<?x?xf32>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>
  cuf.data_transfer %6 to %35 {transfer_kind = #cuf.cuda_transfer<device_host>} : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>, !fir.box<!fir.array<?x?xf32>>
  return
}

// CHECK-LABEL: func.func @_QPdevmul(%arg0: !fir.ref<!fir.array<1x?xf32>> {fir.bindc_name = "b"}, %arg1: !fir.ref<i32> {fir.bindc_name = "wa"}, %arg2: !fir.ref<i32> {fir.bindc_name = "wb"}) {
// CHECK: %[[ALLOCA0:.*]] = fir.alloca !fir.box<!fir.array<?x?xf32>>
// CHECK: %[[ALLOCA1:.*]] = fir.alloca !fir.box<!fir.array<?x?xf32>>
// CHECK: %[[EMBOX:.*]] = fir.embox %{{.*}}(%{{.*}}) [%{{.*}}] : (!fir.ref<!fir.array<1x?xf32>>, !fir.shape<2>, !fir.slice<2>) -> !fir.box<!fir.array<?x?xf32>>
// CHECK: fir.store %[[EMBOX]] to %[[ALLOCA1]] : !fir.ref<!fir.box<!fir.array<?x?xf32>>>
// CHECK: %[[SRC:.*]] = fir.convert %[[ALLOCA1]] : (!fir.ref<!fir.box<!fir.array<?x?xf32>>>) -> !fir.ref<!fir.box<none>>
// CHECK: fir.call @_FortranACUFDataTransferDescDesc(%{{.*}}, %[[SRC]], %{{.*}}, %{{.*}}, %{{.*}}) : (!fir.ref<!fir.box<none>>, !fir.ref<!fir.box<none>>, i32, !fir.ref<i8>, i32) -> ()
// CHECK: fir.store %[[EMBOX]] to %[[ALLOCA0]] : !fir.ref<!fir.box<!fir.array<?x?xf32>>>
// CHECK: %[[DST:.*]] = fir.convert %[[ALLOCA0]] : (!fir.ref<!fir.box<!fir.array<?x?xf32>>>) -> !fir.ref<!fir.box<none>>
// CHECK: fir.call @_FortranACUFDataTransferDescDesc(%[[DST]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!fir.ref<!fir.box<none>>, !fir.ref<!fir.box<none>>, i32, !fir.ref<i8>, i32) -> ()

func.func @_QPlogical_cst() {
  %c0_i64 = arith.constant 0 : i64
  %false = arith.constant false
  %c0 = arith.constant 0 : index
  %0 = cuf.alloc !fir.box<!fir.heap<!fir.array<?x?x!fir.logical<1>>>> {bindc_name = "id2", data_attr = #cuf.cuda<device>, uniq_name = "_QFlogical_cstEid2"} -> !fir.ref<!fir.box<!fir.heap<!fir.array<?x?x!fir.logical<1>>>>>
  %4 = fir.declare %0 {data_attr = #cuf.cuda<device>, fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFlogical_cstEid2"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?x!fir.logical<1>>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?x?x!fir.logical<1>>>>>
  cuf.data_transfer %false to %4 {transfer_kind = #cuf.cuda_transfer<host_device>} : i1, !fir.ref<!fir.box<!fir.heap<!fir.array<?x?x!fir.logical<1>>>>>
  return
}

// CHECK-LABEL: func.func @_QPlogical_cst()
// CHECK: %[[DESC:.*]] = fir.alloca !fir.box<!fir.logical<1>>
// CHECK: %[[CONST:.*]] = fir.alloca !fir.logical<1>
// CHECK: %[[CONV:.*]] = fir.convert %false : (i1) -> !fir.logical<1>
// CHECK: fir.store %[[CONV]] to %[[CONST]] : !fir.ref<!fir.logical<1>>
// CHECK: %[[EMBOX:.*]] = fir.embox %[[CONST]] : (!fir.ref<!fir.logical<1>>) -> !fir.box<!fir.logical<1>>
// CHECK: fir.store %[[EMBOX]] to %[[DESC]] : !fir.ref<!fir.box<!fir.logical<1>>>
// CHECK: %[[BOX_NONE:.*]] = fir.convert %[[DESC]] : (!fir.ref<!fir.box<!fir.logical<1>>>) -> !fir.ref<!fir.box<none>>
// CHECK: fir.call @_FortranACUFDataTransferCstDesc(%{{.*}}, %[[BOX_NONE]], %{{.*}}, %{{.*}}, %{{.*}}) : (!fir.ref<!fir.box<none>>, !fir.ref<!fir.box<none>>, i32, !fir.ref<i8>, i32) -> ()

func.func @_QPcallkernel(%arg0: !fir.box<!fir.array<?x?xcomplex<f32>>> {fir.bindc_name = "a"}, %arg1: !fir.ref<f32> {fir.bindc_name = "b"}, %arg2: !fir.ref<f32> {fir.bindc_name = "c"}) {
  %c0_i64 = arith.constant 0 : i64
  %c1_i32 = arith.constant 1 : i32
  %c0_i32 = arith.constant 0 : i32
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.declare %arg0 dummy_scope %0 {uniq_name = "_QFcallkernelEa"} : (!fir.box<!fir.array<?x?xcomplex<f32>>>, !fir.dscope) -> !fir.box<!fir.array<?x?xcomplex<f32>>>
  %2 = fir.rebox %1 : (!fir.box<!fir.array<?x?xcomplex<f32>>>) -> !fir.box<!fir.array<?x?xcomplex<f32>>>
  %3 = cuf.alloc !fir.box<!fir.heap<!fir.array<?x?xcomplex<f32>>>> {bindc_name = "adev", data_attr = #cuf.cuda<device>, uniq_name = "_QFcallkernelEadev"} -> !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xcomplex<f32>>>>>
  %7 = fir.declare %3 {data_attr = #cuf.cuda<device>, fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFcallkernelEadev"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xcomplex<f32>>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xcomplex<f32>>>>>
  %8 = fir.declare %arg1 dummy_scope %0 {uniq_name = "_QFcallkernelEb"} : (!fir.ref<f32>, !fir.dscope) -> !fir.ref<f32>
  %9 = fir.declare %arg2 dummy_scope %0 {uniq_name = "_QFcallkernelEc"} : (!fir.ref<f32>, !fir.dscope) -> !fir.ref<f32>
  %10 = fir.alloca i32 {bindc_name = "m", uniq_name = "_QFcallkernelEm"}
  %11 = fir.declare %10 {uniq_name = "_QFcallkernelEm"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %12 = fir.alloca i32 {bindc_name = "n", uniq_name = "_QFcallkernelEn"}
  %13 = fir.declare %12 {uniq_name = "_QFcallkernelEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %14:3 = fir.box_dims %2, %c0 : (!fir.box<!fir.array<?x?xcomplex<f32>>>, index) -> (index, index, index)
  %15 = fir.convert %14#1 : (index) -> i32
  fir.store %15 to %13 : !fir.ref<i32>
  %16:3 = fir.box_dims %2, %c1 : (!fir.box<!fir.array<?x?xcomplex<f32>>>, index) -> (index, index, index)
  %27 = fir.load %13 : !fir.ref<i32>
  %28 = fir.convert %27 : (i32) -> index
  %29 = arith.cmpi sgt, %28, %c0 : index
  %30 = arith.select %29, %28, %c0 : index
  %31 = fir.load %11 : !fir.ref<i32>
  %32 = fir.convert %31 : (i32) -> index
  %33 = arith.cmpi sgt, %32, %c0 : index
  %34 = arith.select %33, %32, %c0 : index
  %35 = fir.shape %30, %34 : (index, index) -> !fir.shape<2>
  %36 = fir.undefined index
  %37 = fir.slice %c1, %28, %c1, %c1, %32, %c1 : (index, index, index, index, index, index) -> !fir.slice<2>
  %38 = fir.rebox %2 [%37] : (!fir.box<!fir.array<?x?xcomplex<f32>>>, !fir.slice<2>) -> !fir.box<!fir.array<?x?xcomplex<f32>>>
  cuf.data_transfer %38 to %7 {transfer_kind = #cuf.cuda_transfer<host_device>} : !fir.box<!fir.array<?x?xcomplex<f32>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xcomplex<f32>>>>>
  return
}

// CHECK-LABEL: func.func @_QPcallkernel(
// CHECK-SAME: %[[ARG0:.*]]: !fir.box<!fir.array<?x?xcomplex<f32>>> {fir.bindc_name = "a"}
// CHECK: %[[ALLOCA:.*]] = fir.alloca !fir.box<!fir.array<?x?xcomplex<f32>>>
// CHECK: %[[DECL_ARG0:.*]] = fir.declare %[[ARG0]] dummy_scope %{{.*}} {uniq_name = "_QFcallkernelEa"} : (!fir.box<!fir.array<?x?xcomplex<f32>>>, !fir.dscope) -> !fir.box<!fir.array<?x?xcomplex<f32>>>
// CHECK: %[[REBOX0:.*]] = fir.rebox %[[DECL_ARG0]] : (!fir.box<!fir.array<?x?xcomplex<f32>>>) -> !fir.box<!fir.array<?x?xcomplex<f32>>>
// CHECK: %[[REBOX1:.*]] = fir.rebox %[[REBOX0]] [%{{.*}}] : (!fir.box<!fir.array<?x?xcomplex<f32>>>, !fir.slice<2>) -> !fir.box<!fir.array<?x?xcomplex<f32>>>
// CHECK: fir.store %[[REBOX1]] to %[[ALLOCA]] : !fir.ref<!fir.box<!fir.array<?x?xcomplex<f32>>>>
// CHECK: %[[BOX_NONE:.*]] = fir.convert %[[ALLOCA]] : (!fir.ref<!fir.box<!fir.array<?x?xcomplex<f32>>>>) -> !fir.ref<!fir.box<none>>
// CHECK: fir.call @_FortranACUFDataTransferDescDesc(%{{.*}}, %[[BOX_NONE]], %{{.*}}, %{{.*}}, %{{.*}}) : (!fir.ref<!fir.box<none>>, !fir.ref<!fir.box<none>>, i32, !fir.ref<i8>, i32) -> ()

func.func @_QPsrc_cst() {
  %0 = fir.dummy_scope : !fir.dscope
  %1 = cuf.alloc !fir.box<!fir.heap<!fir.array<?xf32>>> {bindc_name = "d4", data_attr = #cuf.cuda<device>, uniq_name = "_QFsub4Ed4"} -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  %5:2 = hlfir.declare %1 {data_attr = #cuf.cuda<device>, fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFsub4Ed4"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>)
  %6 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFsub4Ei"}
  %7:2 = hlfir.declare %6 {uniq_name = "_QFsub4Ei"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %c1 = arith.constant 1 : index
  %c10_i32 = arith.constant 10 : i32
  %c0_i32 = arith.constant 0 : i32
  %9 = fir.convert %5#1 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<none>>
  %c6_i32 = arith.constant 6 : i32
  %14 = fir.convert %c6_i32 : (i32) -> index
  %c10_i32_0 = arith.constant 10 : i32
  %15 = fir.convert %c10_i32_0 : (i32) -> index
  %c1_1 = arith.constant 1 : index
  %16 = fir.convert %14 : (index) -> i32
  %17:2 = fir.do_loop %arg1 = %14 to %15 step %c1_1 iter_args(%arg2 = %16) -> (index, i32) {
    fir.store %arg2 to %7#1 : !fir.ref<i32>
    %cst = arith.constant -4.000000e+00 : f32
    %22 = fir.load %5#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
    %23 = fir.load %7#0 : !fir.ref<i32>
    %24 = fir.convert %23 : (i32) -> i64
    %25 = hlfir.designate %22 (%24)  : (!fir.box<!fir.heap<!fir.array<?xf32>>>, i64) -> !fir.ref<f32>
    cuf.data_transfer %cst to %25 {transfer_kind = #cuf.cuda_transfer<host_device>} : f32, !fir.ref<f32>
    %26 = arith.addi %arg1, %c1_1 : index
    %27 = fir.convert %c1_1 : (index) -> i32
    %28 = fir.load %7#1 : !fir.ref<i32>
    %29 = arith.addi %28, %27 : i32
    fir.result %26, %29 : index, i32
  }
  return
}

// CHECK-LABEL: func.func @_QPsrc_cst()
// CHECK: %[[ALLOCA:.*]] = fir.alloca f32
// CHECK: %[[CST:.*]] = arith.constant -4.000000e+00 : f32
// CHECK: fir.store %[[CST]] to %[[ALLOCA]] : !fir.ref<f32>
// CHECK: %[[CONV:.*]] = fir.convert %[[ALLOCA]] : (!fir.ref<f32>) -> !fir.llvm_ptr<i8>
// CHECK: fir.call @_FortranACUFDataTransferPtrPtr(%{{.*}}, %[[CONV]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!fir.llvm_ptr<i8>, !fir.llvm_ptr<i8>, i64, i32, !fir.ref<i8>, i32) -> ()

func.func @_QPchecksums(%arg0: !fir.box<!fir.array<?xf64>> {cuf.data_attr = #cuf.cuda<device>, fir.bindc_name = "a"}, %arg1: !fir.ref<i32> {fir.bindc_name = "n"}) {
  %c0 = arith.constant 0 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.declare %arg0 dummy_scope %0 {data_attr = #cuf.cuda<device>, uniq_name = "_QFchecksumsEa"} : (!fir.box<!fir.array<?xf64>>, !fir.dscope) -> !fir.box<!fir.array<?xf64>>
  %2 = fir.rebox %1 : (!fir.box<!fir.array<?xf64>>) -> !fir.box<!fir.array<?xf64>>
  %3 = fir.declare %arg1 dummy_scope %0 {uniq_name = "_QFchecksumsEn"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %4 = fir.load %3 : !fir.ref<i32>
  %5 = fir.convert %4 : (i32) -> index
  %6 = arith.cmpi sgt, %5, %c0 : index
  %7 = arith.select %6, %5, %c0 : index
  %8 = fir.alloca !fir.array<?xf64>, %7 {bindc_name = "hosttmp", uniq_name = "_QFchecksumsEhosttmp"}
  %9 = fir.shape %7 : (index) -> !fir.shape<1>
  %10 = fir.declare %8(%9) {uniq_name = "_QFchecksumsEhosttmp"} : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<?xf64>>
  %11 = fir.embox %10(%9) : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>) -> !fir.box<!fir.array<?xf64>>
  cuf.data_transfer %2 to %10, %9 : !fir.shape<1> {transfer_kind = #cuf.cuda_transfer<device_host>} : !fir.box<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>
  return
}

// CHECK-LABEL: func.func @_QPchecksums
// CHECK: %[[DST:.*]] = fir.convert %{{.*}} : (!fir.ref<!fir.box<!fir.array<?xf64>>>) -> !fir.ref<!fir.box<none>>
// CHECK: %[[SRC:.*]] = fir.convert %{{.*}} : (!fir.ref<!fir.box<!fir.array<?xf64>>>) -> !fir.ref<!fir.box<none>>
// CHECK: fir.call @_FortranACUFDataTransferDescDescNoRealloc(%[[DST]], %[[SRC]], %{{.*}}, %{{.*}}, %{{.*}}) : (!fir.ref<!fir.box<none>>, !fir.ref<!fir.box<none>>, i32, !fir.ref<i8>, i32) -> ()

func.func @_QPsub20() {
  %0 = cuf.alloc !fir.box<!fir.heap<f32>> {bindc_name = "r", data_attr = #cuf.cuda<device>, uniq_name = "_QFsub20Er"} -> !fir.ref<!fir.box<!fir.heap<f32>>>
  %1 = fir.zero_bits !fir.heap<f32>
  %2 = fir.embox %1 {allocator_idx = 2 : i32} : (!fir.heap<f32>) -> !fir.box<!fir.heap<f32>>
  fir.store %2 to %0 : !fir.ref<!fir.box<!fir.heap<f32>>>
  %3:2 = hlfir.declare %0 {data_attr = #cuf.cuda<device>, fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFsub20Er"} : (!fir.ref<!fir.box<!fir.heap<f32>>>) -> (!fir.ref<!fir.box<!fir.heap<f32>>>, !fir.ref<!fir.box<!fir.heap<f32>>>)
  %c0_i32 = arith.constant 0 : i32
  cuf.data_transfer %c0_i32 to %3#0 {transfer_kind = #cuf.cuda_transfer<host_device>} : i32, !fir.ref<!fir.box<!fir.heap<f32>>>
  return
}

// CHECK-LABEL:func.func @_QPsub20
// CHECK: %[[BOX_ALLOCA:.*]] = fir.alloca !fir.box<f32>
// CHECK: %[[TMP:.*]] = fir.alloca f32
// CHECK: %[[CONV:.*]] = fir.convert %c0{{.*}} : (i32) -> f32
// CHECK: fir.store %[[CONV]] to %[[TMP]] : !fir.ref<f32>
// CHECK: %[[BOX:.*]] = fir.embox %[[TMP]] : (!fir.ref<f32>) -> !fir.box<f32>
// CHECK: fir.store %[[BOX]] to %[[BOX_ALLOCA]] : !fir.ref<!fir.box<f32>>
// CHECK: %[[BOX_NONE:.*]] = fir.convert %[[BOX_ALLOCA]] : (!fir.ref<!fir.box<f32>>) -> !fir.ref<!fir.box<none>>
// CHECK: fir.call @_FortranACUFDataTransferCstDesc(%13, %[[BOX_NONE]], %c0{{.*}}, %{{.*}}, %{{.*}}) : (!fir.ref<!fir.box<none>>, !fir.ref<!fir.box<none>>, i32, !fir.ref<i8>, i32) -> ()

func.func @_QPsub28() {
  %0 = fir.dummy_scope : !fir.dscope
  %1 = cuf.alloc !fir.box<!fir.heap<!fir.array<?x?x!fir.logical<8>>>> {bindc_name = "id2", data_attr = #cuf.cuda<device>, uniq_name = "_QFsub28Eid2"} -> !fir.ref<!fir.box<!fir.heap<!fir.array<?x?x!fir.logical<8>>>>>
  %2 = fir.zero_bits !fir.heap<!fir.array<?x?x!fir.logical<8>>>
  %c0 = arith.constant 0 : index
  %3 = fir.shape %c0, %c0 : (index, index) -> !fir.shape<2>
  %4 = fir.embox %2(%3) {allocator_idx = 2 : i32} : (!fir.heap<!fir.array<?x?x!fir.logical<8>>>, !fir.shape<2>) -> !fir.box<!fir.heap<!fir.array<?x?x!fir.logical<8>>>>
  fir.store %4 to %1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?x!fir.logical<8>>>>>
  %5:2 = hlfir.declare %1 {data_attr = #cuf.cuda<device>, fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFsub28Eid2"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?x!fir.logical<8>>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?x!fir.logical<8>>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x?x!fir.logical<8>>>>>)
  %c1 = arith.constant 1 : index
  %c10_i32 = arith.constant 10 : i32
  %c0_i32 = arith.constant 0 : i32
  %6 = fir.convert %5#0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?x!fir.logical<8>>>>>) -> !fir.ref<!fir.box<none>>
  %7 = fir.convert %c1 : (index) -> i64
  %8 = fir.convert %c10_i32 : (i32) -> i64
  fir.call @_FortranAAllocatableSetBounds(%6, %c0_i32, %7, %8) fastmath<contract> : (!fir.ref<!fir.box<none>>, i32, i64, i64) -> ()
  %c1_0 = arith.constant 1 : index
  %c10_i32_1 = arith.constant 10 : i32
  %c1_i32 = arith.constant 1 : i32
  %9 = fir.convert %5#0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?x!fir.logical<8>>>>>) -> !fir.ref<!fir.box<none>>
  %10 = fir.convert %c1_0 : (index) -> i64
  %11 = fir.convert %c10_i32_1 : (i32) -> i64
  fir.call @_FortranAAllocatableSetBounds(%9, %c1_i32, %10, %11) fastmath<contract> : (!fir.ref<!fir.box<none>>, i32, i64, i64) -> ()
  %12 = cuf.allocate %5#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?x!fir.logical<8>>>>> {data_attr = #cuf.cuda<device>} -> i32
  %false = arith.constant false
  cuf.data_transfer %false to %5#0 {transfer_kind = #cuf.cuda_transfer<host_device>} : i1, !fir.ref<!fir.box<!fir.heap<!fir.array<?x?x!fir.logical<8>>>>>
  %13 = fir.load %5#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?x!fir.logical<8>>>>>
  %14 = fir.box_addr %13 : (!fir.box<!fir.heap<!fir.array<?x?x!fir.logical<8>>>>) -> !fir.heap<!fir.array<?x?x!fir.logical<8>>>
  %15 = fir.convert %14 : (!fir.heap<!fir.array<?x?x!fir.logical<8>>>) -> i64
  %c0_i64 = arith.constant 0 : i64
  %16 = arith.cmpi ne, %15, %c0_i64 : i64
  fir.if %16 {
    %17 = cuf.deallocate %5#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?x!fir.logical<8>>>>> {data_attr = #cuf.cuda<device>} -> i32
  }
  cuf.free %5#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?x!fir.logical<8>>>>> {data_attr = #cuf.cuda<device>}
  return
}

// CHECK-LABEL: func.func @_QPsub28()
// CHECK: %[[DESC:.*]] = fir.alloca !fir.box<!fir.logical<8>>
// CHECK: %[[L8:.*]] = fir.alloca !fir.logical<8>
// CHECK: %[[FALSE:.*]] = fir.convert %false{{.*}} : (i1) -> !fir.logical<8>
// CHECK: fir.store %[[FALSE]] to %[[L8]] : !fir.ref<!fir.logical<8>>
// CHECK: %[[EMBOX:.*]] = fir.embox %[[L8]] : (!fir.ref<!fir.logical<8>>) -> !fir.box<!fir.logical<8>>
// CHECK: fir.store %[[EMBOX]] to %[[DESC]] : !fir.ref<!fir.box<!fir.logical<8>>>
// CHECK: %[[BOX_NONE:.*]] = fir.convert %[[DESC]] : (!fir.ref<!fir.box<!fir.logical<8>>>) -> !fir.ref<!fir.box<none>>
// CHECK: fir.call @_FortranACUFDataTransferCstDesc(%{{.*}}, %[[BOX_NONE]], %{{.*}}, %{{.*}}, %{{.*}}) : (!fir.ref<!fir.box<none>>, !fir.ref<!fir.box<none>>, i32, !fir.ref<i8>, i32) -> ()

func.func @_QPtesti4(%arg0: !fir.ref<i32> {fir.bindc_name = "n1"}, %arg1: !fir.ref<i32> {fir.bindc_name = "n2"}, %arg2: !fir.ref<i32> {fir.bindc_name = "n3"}, %arg3: !fir.ref<i32> {fir.bindc_name = "n4"}) {
  %true = arith.constant true
  %c0 = arith.constant 0 : index
  %c2_i32 = arith.constant 2 : i32
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg0 dummy_scope %0 {uniq_name = "_QFtesti4En1"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %2:2 = hlfir.declare %arg1 dummy_scope %0 {uniq_name = "_QFtesti4En2"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %3:2 = hlfir.declare %arg2 dummy_scope %0 {uniq_name = "_QFtesti4En3"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %4:2 = hlfir.declare %arg3 dummy_scope %0 {uniq_name = "_QFtesti4En4"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %5 = fir.load %1#0 : !fir.ref<i32>
  %6 = arith.divsi %5, %c2_i32 : i32
  %7 = fir.convert %6 : (i32) -> index
  %8 = arith.cmpi sgt, %7, %c0 : index
  %9 = arith.select %8, %7, %c0 : index
  %10 = fir.load %2#0 : !fir.ref<i32>
  %11 = arith.divsi %10, %c2_i32 : i32
  %12 = fir.convert %11 : (i32) -> index
  %13 = arith.cmpi sgt, %12, %c0 : index
  %14 = arith.select %13, %12, %c0 : index
  %15 = fir.load %3#0 : !fir.ref<i32>
  %16 = arith.divsi %15, %c2_i32 : i32
  %17 = fir.convert %16 : (i32) -> index
  %18 = arith.cmpi sgt, %17, %c0 : index
  %19 = arith.select %18, %17, %c0 : index
  %20 = fir.load %4#0 : !fir.ref<i32>
  %21 = arith.divsi %20, %c2_i32 : i32
  %22 = fir.convert %21 : (i32) -> index
  %23 = arith.cmpi sgt, %22, %c0 : index
  %24 = arith.select %23, %22, %c0 : index
  %25 = cuf.alloc !fir.array<?x?x?x?x!fir.logical<4>>, %9, %14, %19, %24 : index, index, index, index {bindc_name = "lma", data_attr = #cuf.cuda<managed>, uniq_name = "_QFtesti4Elma"} -> !fir.ref<!fir.array<?x?x?x?x!fir.logical<4>>>
  %26 = fir.shape %9, %14, %19, %24 : (index, index, index, index) -> !fir.shape<4>
  %27:2 = hlfir.declare %25(%26) {data_attr = #cuf.cuda<managed>, uniq_name = "_QFtesti4Elma"} : (!fir.ref<!fir.array<?x?x?x?x!fir.logical<4>>>, !fir.shape<4>) -> (!fir.box<!fir.array<?x?x?x?x!fir.logical<4>>>, !fir.ref<!fir.array<?x?x?x?x!fir.logical<4>>>)
  cuf.data_transfer %true to %27#1, %26 : !fir.shape<4> {transfer_kind = #cuf.cuda_transfer<host_device>} : i1, !fir.ref<!fir.array<?x?x?x?x!fir.logical<4>>>
  cuf.free %27#1 : !fir.ref<!fir.array<?x?x?x?x!fir.logical<4>>> {data_attr = #cuf.cuda<managed>}
  return
}

// CHECK-LABEL: func.func @_QPtesti4
// CHECK: fir.call @_FortranACUFDataTransferCstDesc

// -----

func.func @_QQmain() attributes {fir.bindc_name = "T"} {
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %c80 = arith.constant 80 : index
  %c0 = arith.constant 0 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = cuf.alloc !fir.box<!fir.heap<!fir.array<?x?x?xf16>>> {bindc_name = "a", data_attr = #cuf.cuda<device>, uniq_name = "_QFEa"} -> !fir.ref<!fir.box<!fir.heap<!fir.array<?x?x?xf16>>>>
  %2 = fir.zero_bits !fir.heap<!fir.array<?x?x?xf16>>
  %3 = fir.shape %c0, %c0, %c0 : (index, index, index) -> !fir.shape<3>
  %4 = fir.embox %2(%3) {allocator_idx = 2 : i32} : (!fir.heap<!fir.array<?x?x?xf16>>, !fir.shape<3>) -> !fir.box<!fir.heap<!fir.array<?x?x?xf16>>>
  fir.store %4 to %1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?x?xf16>>>>
  %5 = fir.declare %1 {data_attr = #cuf.cuda<device>, fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFEa"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?x?xf16>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?x?x?xf16>>>>
  %6 = fir.address_of(@_QFEha) : !fir.ref<!fir.array<80x80x80xf32>>
  %7 = fir.shape %c80, %c80, %c80 : (index, index, index) -> !fir.shape<3>
  %8 = fir.declare %6(%7) {uniq_name = "_QFEha"} : (!fir.ref<!fir.array<80x80x80xf32>>, !fir.shape<3>) -> !fir.ref<!fir.array<80x80x80xf32>>
  %9 = fir.address_of(@_QFECn) : !fir.ref<i32>
  %10 = fir.declare %9 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QFECn"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %11 = fir.load %5 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?x?xf16>>>>
  %12:3 = fir.box_dims %11, %c0 : (!fir.box<!fir.heap<!fir.array<?x?x?xf16>>>, index) -> (index, index, index)
  %13:3 = fir.box_dims %11, %c1 : (!fir.box<!fir.heap<!fir.array<?x?x?xf16>>>, index) -> (index, index, index)
  %14:3 = fir.box_dims %11, %c2 : (!fir.box<!fir.heap<!fir.array<?x?x?xf16>>>, index) -> (index, index, index)
  %15 = fir.shape %12#1, %13#1, %14#1 : (index, index, index) -> !fir.shape<3>
  %16 = fir.allocmem !fir.array<?x?x?xf16>, %12#1, %13#1, %14#1 {bindc_name = ".tmp", uniq_name = ""}
  %17 = fir.declare %16(%15) {uniq_name = ".tmp"} : (!fir.heap<!fir.array<?x?x?xf16>>, !fir.shape<3>) -> !fir.heap<!fir.array<?x?x?xf16>>
  %18 = fir.embox %17(%15) : (!fir.heap<!fir.array<?x?x?xf16>>, !fir.shape<3>) -> !fir.box<!fir.array<?x?x?xf16>>
  cuf.data_transfer %11 to %18 {transfer_kind = #cuf.cuda_transfer<device_host>} : !fir.box<!fir.heap<!fir.array<?x?x?xf16>>>, !fir.box<!fir.array<?x?x?xf16>>
  return
}

// CHECK-LABEL: func.func @_QQmain() 
// CHECK: fir.call @_FortranACUFDataTransferDescDesc

} // end of module

