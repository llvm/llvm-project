// RUN: fir-opt --split-input-file --cuf-convert-late %s | FileCheck %s
  
fir.global @_QMmaEaa {data_attr = #cuf.cuda<device>} : !fir.array<100xi32> {
  %0 = fir.zero_bits !fir.array<100xi32>
  fir.has_value %0 : !fir.array<100xi32>
}
func.func @_QPxa(%arg0: !fir.ref<!fir.array<?xi32>> {cuf.data_attr = #cuf.cuda<device>, fir.bindc_name = "a"}, %arg1: !fir.ref<i32> {fir.bindc_name = "n"}) {
  %0 = fir.address_of(@_QMmaEaa) : !fir.ref<!fir.array<100xi32>>
  %1 = cuf.device_address @_QMmaEaa -> !fir.ref<!fir.array<100xi32>>
  return
}

// CHECK-LABEL: func.func @_QPxa
// CHECK: fir.call @_FortranACUFGetDeviceAddress

// -----

// Non-allocatable managed global with companion pointer global:
// cuf.device_address should load from the pointer global instead of
// calling CUFGetDeviceAddress.
//
// Fortran source:
//   module test
//     integer*4, managed :: manx(100)
//   end module
//   subroutine user()
//     use test
//     manx(1) = 42
//   end subroutine

fir.global @_QMtestEmanx {data_attr = #cuf.cuda<managed>} : !fir.array<100xi32> {
  %0 = fir.zero_bits !fir.array<100xi32>
  fir.has_value %0 : !fir.array<100xi32>
}

fir.global internal @_QMtestEmanx.managed.ptr {section = "__nv_managed_data__"} : !fir.llvm_ptr<i8> {
  %0 = fir.zero_bits !fir.llvm_ptr<i8>
  fir.has_value %0 : !fir.llvm_ptr<i8>
}

func.func @_QPuser() {
  %c100 = arith.constant 100 : index
  %0 = cuf.device_address @_QMtestEmanx -> !fir.ref<!fir.array<100xi32>>
  %1 = fir.shape %c100 : (index) -> !fir.shape<1>
  %2 = fir.declare %0(%1) {uniq_name = "_QMtestEmanx"} : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>) -> !fir.ref<!fir.array<100xi32>>
  return
}

// CHECK-LABEL: func.func @_QPuser
// CHECK-NOT: fir.call @_FortranACUFGetDeviceAddress
// CHECK: %[[PTR_REF:.*]] = fir.address_of(@_QMtestEmanx.managed.ptr) : !fir.ref<!fir.llvm_ptr<i8>>
// CHECK: %[[RAW_PTR:.*]] = fir.load %[[PTR_REF]] : !fir.ref<!fir.llvm_ptr<i8>>
// CHECK: %[[ADDR:.*]] = fir.convert %[[RAW_PTR]] : (!fir.llvm_ptr<i8>) -> !fir.ref<!fir.array<100xi32>>
