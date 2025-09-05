// RUN: fir-opt --split-input-file --fir-add-alias-tags %s | FileCheck %s

// Fortran source:
// module data1
//   real :: glob1, glob2
//   equivalence (glob1, glob2)
// end module data1
// subroutine test1
//   use data1
//   glob1 = glob2
// end subroutine test1
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, i64 = dense<[32, 64]> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little">, llvm.data_layout = ""} {
  fir.global @_QMdata1Eglob1 : !fir.array<4xi8> {
    %0 = fir.zero_bits !fir.array<4xi8>
    fir.has_value %0 : !fir.array<4xi8>
  }
  func.func @_QPtest1() {
    %c0 = arith.constant 0 : index
    %0 = fir.dummy_scope : !fir.dscope
    %1 = fir.address_of(@_QMdata1Eglob1) : !fir.ref<!fir.array<4xi8>>
    %2 = fir.coordinate_of %1, %c0 : (!fir.ref<!fir.array<4xi8>>, index) -> !fir.ref<i8>
    %3 = fir.convert %2 : (!fir.ref<i8>) -> !fir.ptr<f32>
    %4 = fir.declare %3 storage(%1[0]) {uniq_name = "_QMdata1Eglob1"} : (!fir.ptr<f32>, !fir.ref<!fir.array<4xi8>>) -> !fir.ptr<f32>
    %5 = fir.declare %3 storage(%1[0]) {uniq_name = "_QMdata1Eglob2"} : (!fir.ptr<f32>, !fir.ref<!fir.array<4xi8>>) -> !fir.ptr<f32>
    %6 = fir.load %5 : !fir.ptr<f32>
    fir.store %6 to %4 : !fir.ptr<f32>
    return
  }
}
// CHECK: #[[ROOT1:.+]] = #llvm.tbaa_root<id = "Flang function root _QPtest1">
// CHECK: #[[ANYACC1:.+]] = #llvm.tbaa_type_desc<id = "any access", members = {<#[[ROOT1]], 0>}>
// CHECK: #[[ANYDATA1:.+]] = #llvm.tbaa_type_desc<id = "any data access", members = {<#[[ANYACC1]], 0>}>
// CHECK: #[[TARGETDATA1:.+]] = #llvm.tbaa_type_desc<id = "target data", members = {<#[[ANYDATA1]], 0>}>
// CHECK: #[[GLOBALDATA1:.+]] = #llvm.tbaa_type_desc<id = "global data", members = {<#[[TARGETDATA1]], 0>}>
// CHECK: #[[GLOB1COMMON:.+]] = #llvm.tbaa_type_desc<id = "global data/_QMdata1Eglob1", members = {<#[[GLOBALDATA1]], 0>}>
// CHECK: #[[GLOB1:.+]] = #llvm.tbaa_type_desc<id = "global data/_QMdata1Eglob1/bytes_0_to_3", members = {<#[[GLOB1COMMON]], 0>}>
// CHECK: #[[TAG:.+]] = #llvm.tbaa_tag<base_type = #[[GLOB1]], access_type = #[[GLOB1]], offset = 0>
// CHECK:           %[[VAL_7:.*]] = fir.load{{.*}}{tbaa = [#[[TAG]]]} : !fir.ptr<f32>
// CHECK:           fir.store{{.*}}{tbaa = [#[[TAG]]]} : !fir.ptr<f32>

// -----

// Fortran source:
// module data2
//   real :: glob1, glob2, glob3
//   equivalence (glob1, glob2)
//   common /glob1/ glob1, glob3
// end module data2
// subroutine test2
//   use data2
//   glob1 = glob2 + glob3
// end subroutine test2
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, i64 = dense<[32, 64]> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little">, llvm.data_layout = ""} {
  fir.global common @glob1_(dense<0> : vector<8xi8>) {alignment = 4 : i64} : !fir.array<8xi8>
  func.func @_QPtest2() {
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %0 = fir.dummy_scope : !fir.dscope
    %1 = fir.address_of(@glob1_) : !fir.ref<!fir.array<8xi8>>
    %2 = fir.coordinate_of %1, %c0 : (!fir.ref<!fir.array<8xi8>>, index) -> !fir.ref<i8>
    %3 = fir.convert %2 : (!fir.ref<i8>) -> !fir.ptr<f32>
    %4 = fir.declare %3 storage(%1[0]) {uniq_name = "_QMdata2Eglob1"} : (!fir.ptr<f32>, !fir.ref<!fir.array<8xi8>>) -> !fir.ptr<f32>
    %5 = fir.declare %3 storage(%1[0]) {uniq_name = "_QMdata2Eglob2"} : (!fir.ptr<f32>, !fir.ref<!fir.array<8xi8>>) -> !fir.ptr<f32>
    %6 = fir.coordinate_of %1, %c4 : (!fir.ref<!fir.array<8xi8>>, index) -> !fir.ref<i8>
    %7 = fir.convert %6 : (!fir.ref<i8>) -> !fir.ref<f32>
    %8 = fir.declare %7 storage(%1[4]) {uniq_name = "_QMdata2Eglob3"} : (!fir.ref<f32>, !fir.ref<!fir.array<8xi8>>) -> !fir.ref<f32>
    %9 = fir.load %5 : !fir.ptr<f32>
    %10 = fir.load %8 : !fir.ref<f32>
    %11 = arith.addf %9, %10 fastmath<contract> : f32
    fir.store %11 to %4 : !fir.ptr<f32>
    return
  }
}
// CHECK: #[[ROOT2:.+]] = #llvm.tbaa_root<id = "Flang function root _QPtest2">
// CHECK: #[[ANYACC2:.+]] = #llvm.tbaa_type_desc<id = "any access", members = {<#[[ROOT2]], 0>}>
// CHECK: #[[ANYDATA2:.+]] = #llvm.tbaa_type_desc<id = "any data access", members = {<#[[ANYACC2]], 0>}>
// CHECK: #[[TARGETDATA2:.+]] = #llvm.tbaa_type_desc<id = "target data", members = {<#[[ANYDATA2]], 0>}>
// CHECK: #[[GLOBALDATA2:.+]] = #llvm.tbaa_type_desc<id = "global data", members = {<#[[TARGETDATA2]], 0>}>
// CHECK: #[[GLOB1COMMON:.+]] = #llvm.tbaa_type_desc<id = "global data/glob1_", members = {<#[[GLOBALDATA2]], 0>}>
// CHECK: #[[GLOB1GLOB2:.+]] = #llvm.tbaa_type_desc<id = "global data/glob1_/bytes_0_to_3", members = {<#[[GLOB1COMMON]], 0>}>
// CHECK: #[[GLOB3:.+]] = #llvm.tbaa_type_desc<id = "global data/glob1_/bytes_4_to_7", members = {<#[[GLOB1COMMON]], 0>}>
// CHECK: #[[GLOB1GLOB2TAG:.+]] = #llvm.tbaa_tag<base_type = #[[GLOB1GLOB2]], access_type = #[[GLOB1GLOB2]], offset = 0>
// CHECK: #[[GLOB3TAG:.+]] = #llvm.tbaa_tag<base_type = #[[GLOB3]], access_type = #[[GLOB3]], offset = 0>
// CHECK:           fir.load{{.*}}{tbaa = [#[[GLOB1GLOB2TAG]]]} : !fir.ptr<f32>
// CHECK:           fir.load{{.*}}{tbaa = [#[[GLOB3TAG]]]} : !fir.ref<f32>
// CHECK:           fir.store{{.*}}{tbaa = [#[[GLOB1GLOB2TAG]]]} : !fir.ptr<f32>
