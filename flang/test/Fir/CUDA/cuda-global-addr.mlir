// RUN: fir-opt --split-input-file --cuf-convert %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>} {
fir.global @_QMmod1Eadev {data_attr = #cuf.cuda<device>} : !fir.array<10xi32> {
  %0 = fir.zero_bits !fir.array<10xi32>
  fir.has_value %0 : !fir.array<10xi32>
}
func.func @_QQmain() attributes {fir.bindc_name = "test"} {
  %c14_i32 = arith.constant 14 : i32
  %c6_i32 = arith.constant 6 : i32
  %c4 = arith.constant 4 : index
  %c1_i32 = arith.constant 1 : i32
  %c0_i32 = arith.constant 0 : i32
  %c10 = arith.constant 10 : index
  %1 = fir.shape %c10 : (index) -> !fir.shape<1>
  %3 = fir.address_of(@_QMmod1Eadev) : !fir.ref<!fir.array<10xi32>>
  %4 = fir.declare %3(%1) {data_attr = #cuf.cuda<device>, uniq_name = "_QMmod1Eadev"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.ref<!fir.array<10xi32>>
  %5 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFEi"}
  %6 = fir.declare %5 {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  fir.store %c0_i32 to %6 : !fir.ref<i32>
  %7 = fir.array_coor %4(%1) %c4 : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
  cuf.data_transfer %c1_i32 to %7 {transfer_kind = #cuf.cuda_transfer<host_device>} : i32, !fir.ref<i32>
  return
}

}

// CHECK-LABEL: func.func @_QQmain()
// CHECK: %[[ADDR:.*]] = fir.address_of(@_QMmod1Eadev) : !fir.ref<!fir.array<10xi32>>
// CHECK: %[[ADDRPTR:.*]] = fir.convert %[[ADDR]] : (!fir.ref<!fir.array<10xi32>>) -> !fir.llvm_ptr<i8>
// CHECK: %[[DEVICE_ADDR:.*]] = fir.call @_FortranACUFGetDeviceAddress(%[[ADDRPTR]], %{{.*}}, %{{.*}}) : (!fir.llvm_ptr<i8>, !fir.ref<i8>, i32) -> !fir.llvm_ptr<i8>
// CHECK: %[[DEVICE_ADDR_CONV:.*]] = fir.convert %[[DEVICE_ADDR]] : (!fir.llvm_ptr<i8>) -> !fir.ref<!fir.array<10xi32>>
// CHECK: %[[DECL:.*]] = fir.declare %[[DEVICE_ADDR_CONV]](%{{.*}}) {data_attr = #cuf.cuda<device>, uniq_name = "_QMmod1Eadev"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.ref<!fir.array<10xi32>>
// CHECK: %[[ARRAY_COOR:.*]] = fir.array_coor %[[DECL]](%{{.*}}) %c4{{.*}} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
// CHECK: %[[ARRAY_COOR_PTR:.*]] = fir.convert %[[ARRAY_COOR]] : (!fir.ref<i32>) -> !fir.llvm_ptr<i8>
// CHECK: fir.call @_FortranACUFDataTransferPtrPtr(%[[ARRAY_COOR_PTR]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!fir.llvm_ptr<i8>, !fir.llvm_ptr<i8>, i64, i32, !fir.ref<i8>, i32) -> none

// -----

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>} {

  fir.global @_QMdevmodEdarray {data_attr = #cuf.cuda<device>} : !fir.box<!fir.heap<!fir.array<?xf32>>> {
    %c0 = arith.constant 0 : index
    %0 = fir.zero_bits !fir.heap<!fir.array<?xf32>>
    %1 = fir.shape %c0 : (index) -> !fir.shape<1>
    %2 = fir.embox %0(%1) {allocator_idx = 2 : i32} : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xf32>>>
    fir.has_value %2 : !fir.box<!fir.heap<!fir.array<?xf32>>>
  }
  func.func @_QQmain() attributes {fir.bindc_name = "arraysize"} {
    %0 = fir.address_of(@_QMiso_c_bindingECc_int) : !fir.ref<i32>
    %1 = fir.declare %0 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_c_bindingECc_int"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %2 = fir.address_of(@_QMdevmodEdarray) : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
    %3 = fir.declare %2 {data_attr = #cuf.cuda<device>, fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QMdevmodEdarray"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
    %4 = fir.alloca i32 {bindc_name = "exp", uniq_name = "_QFEexp"}
    %5 = fir.declare %4 {uniq_name = "_QFEexp"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %6 = fir.alloca i32 {bindc_name = "hsize", uniq_name = "_QFEhsize"}
    %7 = fir.declare %6 {uniq_name = "_QFEhsize"} : (!fir.ref<i32>) -> !fir.ref<i32>
    return
  }
  fir.global @_QMiso_c_bindingECc_int constant : i32
}

// We cannot call _FortranACUFGetDeviceAddress on a constant global. 
// There is no symbol for it and the call would result into an unresolved reference.
// CHECK-NOT: fir.call {{.*}}GetDeviceAddress

