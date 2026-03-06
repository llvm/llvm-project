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
