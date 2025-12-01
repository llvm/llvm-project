// RUN: fir-opt --split-input-file --fir-add-alias-tags %s | FileCheck %s

// Fortran source:
// subroutine test1
//   real :: a, b
//   common /common1/ a, b
//   a = b
// end subroutine test1
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, i64 = dense<[32, 64]> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little">, llvm.data_layout = ""} {
  fir.global common @common1_(dense<0> : vector<8xi8>) {alignment = 4 : i64} : !fir.array<8xi8>
  func.func @_QPtest1() {
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %0 = fir.dummy_scope : !fir.dscope
    %addr1 = fir.address_of(@common1_) : !fir.ref<!fir.array<8xi8>>
    %addr2 = fir.address_of(@common1_) : !fir.ref<!fir.array<8xi8>>
    %2 = fir.coordinate_of %addr1, %c0 : (!fir.ref<!fir.array<8xi8>>, index) -> !fir.ref<i8>
    %3 = fir.convert %2 : (!fir.ref<i8>) -> !fir.ref<f32>
    %4 = fir.declare %3 storage(%addr1[0]) {uniq_name = "_QFtest1Ea"} : (!fir.ref<f32>, !fir.ref<!fir.array<8xi8>>) -> !fir.ref<f32>
    %5 = fir.coordinate_of %addr2, %c4 : (!fir.ref<!fir.array<8xi8>>, index) -> !fir.ref<i8>
    %6 = fir.convert %5 : (!fir.ref<i8>) -> !fir.ref<f32>
    %7 = fir.declare %6 storage(%addr2[4]) {uniq_name = "_QFtest1Eb"} : (!fir.ref<f32>, !fir.ref<!fir.array<8xi8>>) -> !fir.ref<f32>
    %8 = fir.load %7 : !fir.ref<f32>
    fir.store %8 to %4 : !fir.ref<f32>
    return
  }
}
// CHECK: #[[$ATTR_0:.+]] = #llvm.tbaa_root<id = "Flang function root _QPtest1">
// CHECK: #[[$ATTR_1:.+]] = #llvm.tbaa_type_desc<id = "any access", members = {<#[[$ATTR_0]], 0>}>
// CHECK: #[[$ATTR_2:.+]] = #llvm.tbaa_type_desc<id = "any data access", members = {<#[[$ATTR_1]], 0>}>
// CHECK: #[[$ATTR_3:.+]] = #llvm.tbaa_type_desc<id = "target data", members = {<#[[$ATTR_2]], 0>}>
// CHECK: #[[$ATTR_4:.+]] = #llvm.tbaa_type_desc<id = "global data", members = {<#[[$ATTR_3]], 0>}>
// CHECK: #[[$ATTR_5:.+]] = #llvm.tbaa_type_desc<id = "global data/common1_", members = {<#[[$ATTR_4]], 0>}>
// CHECK: #[[$ATTR_6:.+]] = #llvm.tbaa_type_desc<id = "global data/common1_/bytes_4_to_7", members = {<#[[$ATTR_5]], 0>}>
// CHECK: #[[$ATTR_7:.+]] = #llvm.tbaa_type_desc<id = "global data/common1_/bytes_0_to_3", members = {<#[[$ATTR_5]], 0>}>
// CHECK: #[[$ATTR_8:.+]] = #llvm.tbaa_tag<base_type = #[[$ATTR_6]], access_type = #[[$ATTR_6]], offset = 0>
// CHECK: #[[$ATTR_9:.+]] = #llvm.tbaa_tag<base_type = #[[$ATTR_7]], access_type = #[[$ATTR_7]], offset = 0>
// CHECK-LABEL:   func.func @_QPtest1() {
// CHECK:           fir.load{{.*}}{tbaa = [#[[$ATTR_8]]]} : !fir.ref<f32>
// CHECK:           fir.store{{.*}}{tbaa = [#[[$ATTR_9]]]} : !fir.ref<f32>

// -----

// Fortran source:
// subroutine test2
//   real :: a, b
//   common /common2/ a
//   equivalence (a, b)
//   a = b
// end subroutine test2
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, i64 = dense<[32, 64]> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little">, llvm.data_layout = ""} {
  fir.global common @common2_(dense<0> : vector<4xi8>) {alignment = 4 : i64} : !fir.array<4xi8>
  func.func @_QPtest2() {
    %c0 = arith.constant 0 : index
    %0 = fir.dummy_scope : !fir.dscope
    %1 = fir.address_of(@common2_) : !fir.ref<!fir.array<4xi8>>
    %2 = fir.coordinate_of %1, %c0 : (!fir.ref<!fir.array<4xi8>>, index) -> !fir.ref<i8>
    %3 = fir.convert %2 : (!fir.ref<i8>) -> !fir.ptr<f32>
    %4 = fir.declare %3 storage(%1[0]) {uniq_name = "_QFtest2Ea"} : (!fir.ptr<f32>, !fir.ref<!fir.array<4xi8>>) -> !fir.ptr<f32>
    %5 = fir.declare %3 storage(%1[0]) {uniq_name = "_QFtest2Eb"} : (!fir.ptr<f32>, !fir.ref<!fir.array<4xi8>>) -> !fir.ptr<f32>
    %6 = fir.load %5 : !fir.ptr<f32>
    fir.store %6 to %4 : !fir.ptr<f32>
    return
  }
}
// CHECK: #[[$ATTR_10:.+]] = #llvm.tbaa_root<id = "Flang function root _QPtest2">
// CHECK: #[[$ATTR_11:.+]] = #llvm.tbaa_type_desc<id = "any access", members = {<#[[$ATTR_10]], 0>}>
// CHECK: #[[$ATTR_12:.+]] = #llvm.tbaa_type_desc<id = "any data access", members = {<#[[$ATTR_11]], 0>}>
// CHECK: #[[$ATTR_13:.+]] = #llvm.tbaa_type_desc<id = "target data", members = {<#[[$ATTR_12]], 0>}>
// CHECK: #[[$ATTR_14:.+]] = #llvm.tbaa_type_desc<id = "global data", members = {<#[[$ATTR_13]], 0>}>
// CHECK: #[[$ATTR_15:.+]] = #llvm.tbaa_type_desc<id = "global data/common2_", members = {<#[[$ATTR_14]], 0>}>
// CHECK: #[[$ATTR_16:.+]] = #llvm.tbaa_type_desc<id = "global data/common2_/bytes_0_to_3", members = {<#[[$ATTR_15]], 0>}>
// CHECK: #[[$ATTR_18:.+]] = #llvm.tbaa_tag<base_type = #[[$ATTR_16]], access_type = #[[$ATTR_16]], offset = 0>
// CHECK-LABEL:   func.func @_QPtest2() {
// CHECK:           fir.load{{.*}}{tbaa = [#[[$ATTR_18]]]} : !fir.ptr<f32>
// CHECK:           fir.store{{.*}}{tbaa = [#[[$ATTR_18]]]} : !fir.ptr<f32>

// -----

// Fortran source compiled with -mmlir -inline-all:
// subroutine test3
//   real :: a, b
//   common /common3/ a, b
//   a = b
//   call inner(a, b)
// contains
//   subroutine inner(c, d)
//     real :: c, d
//     c = d
//   end subroutine inner
// end subroutine test3
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, i64 = dense<[32, 64]> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little">, llvm.data_layout = ""} {
  fir.global common @common3_(dense<0> : vector<8xi8>) {alignment = 4 : i64} : !fir.array<8xi8>
  func.func @_QPtest3() {
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %0 = fir.dummy_scope : !fir.dscope
    %1 = fir.address_of(@common3_) : !fir.ref<!fir.array<8xi8>>
    %2 = fir.coordinate_of %1, %c0 : (!fir.ref<!fir.array<8xi8>>, index) -> !fir.ref<i8>
    %3 = fir.convert %2 : (!fir.ref<i8>) -> !fir.ref<f32>
    %4 = fir.declare %3 storage(%1[0]) {uniq_name = "_QFtest3Ea"} : (!fir.ref<f32>, !fir.ref<!fir.array<8xi8>>) -> !fir.ref<f32>
    %5 = fir.coordinate_of %1, %c4 : (!fir.ref<!fir.array<8xi8>>, index) -> !fir.ref<i8>
    %6 = fir.convert %5 : (!fir.ref<i8>) -> !fir.ref<f32>
    %7 = fir.declare %6 storage(%1[4]) {uniq_name = "_QFtest3Eb"} : (!fir.ref<f32>, !fir.ref<!fir.array<8xi8>>) -> !fir.ref<f32>
    %8 = fir.load %7 : !fir.ref<f32>
    fir.store %8 to %4 : !fir.ref<f32>
    %9 = fir.dummy_scope : !fir.dscope
    %10 = fir.declare %4 dummy_scope %9 {uniq_name = "_QFtest3FinnerEc"} : (!fir.ref<f32>, !fir.dscope) -> !fir.ref<f32>
    %11 = fir.declare %7 dummy_scope %9 {uniq_name = "_QFtest3FinnerEd"} : (!fir.ref<f32>, !fir.dscope) -> !fir.ref<f32>
    %12 = fir.load %11 : !fir.ref<f32>
    fir.store %12 to %10 : !fir.ref<f32>
    return
  }
}
// CHECK: #[[ROOT3:.+]] = #llvm.tbaa_root<id = "Flang function root _QPtest3">
// CHECK: #[[ROOT3INNER:.+]] = #llvm.tbaa_root<id = "Flang function root _QPtest3 - Scope 1">
// CHECK: #[[ANYACC3:.+]] = #llvm.tbaa_type_desc<id = "any access", members = {<#[[ROOT3]], 0>}>
// CHECK: #[[ANYACC3INNER:.+]] = #llvm.tbaa_type_desc<id = "any access", members = {<#[[ROOT3INNER]], 0>}>
// CHECK: #[[ANYDATA3:.+]] = #llvm.tbaa_type_desc<id = "any data access", members = {<#[[ANYACC3]], 0>}>
// CHECK: #[[ANYDATA3INNER:.+]] = #llvm.tbaa_type_desc<id = "any data access", members = {<#[[ANYACC3INNER]], 0>}>
// CHECK: #[[TARGETDATA3:.+]] = #llvm.tbaa_type_desc<id = "target data", members = {<#[[ANYDATA3]], 0>}>
// CHECK: #[[DUMMYARG3INNER:.+]] = #llvm.tbaa_type_desc<id = "dummy arg data", members = {<#[[ANYDATA3INNER]], 0>}>
// CHECK: #[[GLOBALDATA3:.+]] = #llvm.tbaa_type_desc<id = "global data", members = {<#[[TARGETDATA3]], 0>}>
// CHECK: #[[DUMMYD:.+]] = #llvm.tbaa_type_desc<id = "dummy arg data/_QFtest3FinnerEd", members = {<#[[DUMMYARG3INNER]], 0>}>
// CHECK: #[[DUMMYC:.+]] = #llvm.tbaa_type_desc<id = "dummy arg data/_QFtest3FinnerEc", members = {<#[[DUMMYARG3INNER]], 0>}>
// CHECK: #[[DUMMYDTAG:.+]] = #llvm.tbaa_tag<base_type = #[[DUMMYD]], access_type = #[[DUMMYD]], offset = 0>
// CHECK: #[[DUMMYCTAG:.+]] = #llvm.tbaa_tag<base_type = #[[DUMMYC]], access_type = #[[DUMMYC]], offset = 0>
// CHECK: #[[GLOBALDATA3COMMON3:.+]] = #llvm.tbaa_type_desc<id = "global data/common3_", members = {<#[[GLOBALDATA3]], 0>}>
// CHECK: #[[GLOBALB:.+]] = #llvm.tbaa_type_desc<id = "global data/common3_/bytes_4_to_7", members = {<#[[GLOBALDATA3COMMON3]], 0>}>
// CHECK: #[[GLOBALA:.+]] = #llvm.tbaa_type_desc<id = "global data/common3_/bytes_0_to_3", members = {<#[[GLOBALDATA3COMMON3]], 0>}>
// CHECK: #[[GLOBALBTAG:.+]] = #llvm.tbaa_tag<base_type = #[[GLOBALB]], access_type = #[[GLOBALB]], offset = 0>
// CHECK: #[[GLOBALATAG:.+]] = #llvm.tbaa_tag<base_type = #[[GLOBALA]], access_type = #[[GLOBALA]], offset = 0>
// CHECK-LABEL:   func.func @_QPtest3() {
// CHECK:           fir.load{{.*}}{tbaa = [#[[GLOBALBTAG]]]} : !fir.ref<f32>
// CHECK:           fir.store{{.*}}{tbaa = [#[[GLOBALATAG]]]} : !fir.ref<f32>
// CHECK:           fir.load{{.*}}{tbaa = [#[[DUMMYDTAG]]]} : !fir.ref<f32>
// CHECK:           fir.store{{.*}}{tbaa = [#[[DUMMYCTAG]]]} : !fir.ref<f32>

// -----

// Fortran source compiled with -mmlir -inline-all:
// subroutine test4
//   real :: a, b
//   common /common4/ a, b
//   a = b
//   call inner
// contains
//   subroutine inner
//     real :: c, d
//     common /common4/ c, d
//     c = d
//   end subroutine inner
// end subroutine test4
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, i64 = dense<[32, 64]> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little">, llvm.data_layout = ""} {
  fir.global common @common4_(dense<0> : vector<8xi8>) {alignment = 4 : i64} : !fir.array<8xi8>
  func.func @_QPtest4() {
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %0 = fir.dummy_scope : !fir.dscope
    %1 = fir.address_of(@common4_) : !fir.ref<!fir.array<8xi8>>
    %2 = fir.coordinate_of %1, %c0 : (!fir.ref<!fir.array<8xi8>>, index) -> !fir.ref<i8>
    %3 = fir.convert %2 : (!fir.ref<i8>) -> !fir.ref<f32>
    %4 = fir.declare %3 storage(%1[0]) {uniq_name = "_QFtest4Ea"} : (!fir.ref<f32>, !fir.ref<!fir.array<8xi8>>) -> !fir.ref<f32>
    %5 = fir.coordinate_of %1, %c4 : (!fir.ref<!fir.array<8xi8>>, index) -> !fir.ref<i8>
    %6 = fir.convert %5 : (!fir.ref<i8>) -> !fir.ref<f32>
    %7 = fir.declare %6 storage(%1[4]) {uniq_name = "_QFtest4Eb"} : (!fir.ref<f32>, !fir.ref<!fir.array<8xi8>>) -> !fir.ref<f32>
    %8 = fir.load %7 : !fir.ref<f32>
    fir.store %8 to %4 : !fir.ref<f32>
    %9 = fir.dummy_scope : !fir.dscope
    %10 = fir.declare %3 storage(%1[0]) {uniq_name = "_QFtest4FinnerEc"} : (!fir.ref<f32>, !fir.ref<!fir.array<8xi8>>) -> !fir.ref<f32>
    %11 = fir.declare %6 storage(%1[4]) {uniq_name = "_QFtest4FinnerEd"} : (!fir.ref<f32>, !fir.ref<!fir.array<8xi8>>) -> !fir.ref<f32>
    %12 = fir.load %11 : !fir.ref<f32>
    fir.store %12 to %10 : !fir.ref<f32>
    return
  }
}
// CHECK: #[[TEST4ROOT:.+]] = #llvm.tbaa_root<id = "Flang function root _QPtest4">
// CHECK: #[[INNER4ROOT:.+]] = #llvm.tbaa_root<id = "Flang function root _QPtest4 - Scope 1">
// CHECK: #[[TEST4ANYCC:.+]] = #llvm.tbaa_type_desc<id = "any access", members = {<#[[TEST4ROOT]], 0>}>
// CHECK: #[[INNER4ANYACC:.+]] = #llvm.tbaa_type_desc<id = "any access", members = {<#[[INNER4ROOT]], 0>}>
// CHECK: #[[TEST4ANYDATA:.+]] = #llvm.tbaa_type_desc<id = "any data access", members = {<#[[TEST4ANYCC]], 0>}>
// CHECK: #[[INNER4ANYDATA:.+]] = #llvm.tbaa_type_desc<id = "any data access", members = {<#[[INNER4ANYACC]], 0>}>
// CHECK: #[[TEST4TARGET:.+]] = #llvm.tbaa_type_desc<id = "target data", members = {<#[[TEST4ANYDATA]], 0>}>
// CHECK: #[[INNER4TARGET:.+]] = #llvm.tbaa_type_desc<id = "target data", members = {<#[[INNER4ANYDATA]], 0>}>
// CHECK: #[[TEST4GLOBAL:.+]] = #llvm.tbaa_type_desc<id = "global data", members = {<#[[TEST4TARGET]], 0>}>
// CHECK: #[[INNER4GLOBAL:.+]] = #llvm.tbaa_type_desc<id = "global data", members = {<#[[INNER4TARGET]], 0>}>
// CHECK: #[[TEST4COMMON:.+]] = #llvm.tbaa_type_desc<id = "global data/common4_", members = {<#[[TEST4GLOBAL]], 0>}>
// CHECK: #[[INNER4COMMON:.+]] = #llvm.tbaa_type_desc<id = "global data/common4_", members = {<#[[INNER4GLOBAL]], 0>}>
// CHECK: #[[TEST4B:.+]] = #llvm.tbaa_type_desc<id = "global data/common4_/bytes_4_to_7", members = {<#[[TEST4COMMON]], 0>}>
// CHECK: #[[TEST4A:.+]] = #llvm.tbaa_type_desc<id = "global data/common4_/bytes_0_to_3", members = {<#[[TEST4COMMON]], 0>}>
// CHECK: #[[INNER4D:.+]] = #llvm.tbaa_type_desc<id = "global data/common4_/bytes_4_to_7", members = {<#[[INNER4COMMON]], 0>}>
// CHECK: #[[INNER4C:.+]] = #llvm.tbaa_type_desc<id = "global data/common4_/bytes_0_to_3", members = {<#[[INNER4COMMON]], 0>}>
// CHECK: #[[TEST4BTAG:.+]] = #llvm.tbaa_tag<base_type = #[[TEST4B]], access_type = #[[TEST4B]], offset = 0>
// CHECK: #[[TEST4ATAG:.+]] = #llvm.tbaa_tag<base_type = #[[TEST4A]], access_type = #[[TEST4A]], offset = 0>
// CHECK: #[[INNER4DTAG:.+]] = #llvm.tbaa_tag<base_type = #[[INNER4D]], access_type = #[[INNER4D]], offset = 0>
// CHECK: #[[INNER4CTAG:.+]] = #llvm.tbaa_tag<base_type = #[[INNER4C]], access_type = #[[INNER4C]], offset = 0>
// CHECK-LABEL:   func.func @_QPtest4() {
// CHECK:           fir.load{{.*}}{tbaa = [#[[TEST4BTAG]]]} : !fir.ref<f32>
// CHECK:           fir.store{{.*}}{tbaa = [#[[TEST4ATAG]]]} : !fir.ref<f32>
// CHECK:           fir.load{{.*}}{tbaa = [#[[INNER4DTAG]]]} : !fir.ref<f32>
// CHECK:           fir.store{{.*}}{tbaa = [#[[INNER4CTAG]]]} : !fir.ref<f32>

// -----

// Fortran source with manually removed fir.declare for 'b':
// subroutine test5
//   real :: a, b
//   common /common5/ a, b
//   a = b
// end subroutine test5
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, i64 = dense<[32, 64]> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little">, llvm.data_layout = ""} {
  fir.global common @common5_(dense<0> : vector<8xi8>) {alignment = 4 : i64} : !fir.array<8xi8>
  func.func @_QPtest5() {
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %0 = fir.dummy_scope : !fir.dscope
    %1 = fir.address_of(@common5_) : !fir.ref<!fir.array<8xi8>>
    %2 = fir.coordinate_of %1, %c0 : (!fir.ref<!fir.array<8xi8>>, index) -> !fir.ref<i8>
    %3 = fir.convert %2 : (!fir.ref<i8>) -> !fir.ref<f32>
    %4 = fir.declare %3 storage(%1[0]) {uniq_name = "_QFtest5Ea"} : (!fir.ref<f32>, !fir.ref<!fir.array<8xi8>>) -> !fir.ref<f32>
    %5 = fir.coordinate_of %1, %c4 : (!fir.ref<!fir.array<8xi8>>, index) -> !fir.ref<i8>
    %6 = fir.convert %5 : (!fir.ref<i8>) -> !fir.ref<f32>
//    %7 = fir.declare %6 storage(%1[4]) {uniq_name = "_QFtest5Eb"} : (!fir.ref<f32>, !fir.ref<!fir.array<8xi8>>) -> !fir.ref<f32>
    %8 = fir.load %6 : !fir.ref<f32>
    fir.store %8 to %4 : !fir.ref<f32>
    return
  }
}
// CHECK: #[[TEST5ROOT:.+]] = #llvm.tbaa_root<id = "Flang function root _QPtest5">
// CHECK: #[[TEST5ANYACC:.+]] = #llvm.tbaa_type_desc<id = "any access", members = {<#[[TEST5ROOT]], 0>}>
// CHECK: #[[TEST5ANYDATA:.+]] = #llvm.tbaa_type_desc<id = "any data access", members = {<#[[TEST5ANYACC]], 0>}>
// CHECK: #[[TEST5TARGET:.+]] = #llvm.tbaa_type_desc<id = "target data", members = {<#[[TEST5ANYDATA]], 0>}>
// CHECK: #[[TEST5GLOBAL:.+]] = #llvm.tbaa_type_desc<id = "global data", members = {<#[[TEST5TARGET]], 0>}>
// CHECK: #[[TEST5COMMON5:.+]] = #llvm.tbaa_type_desc<id = "global data/common5_", members = {<#[[TEST5GLOBAL]], 0>}>
// CHECK: #[[TEST5COMMON5TAG:.+]] = #llvm.tbaa_tag<base_type = #[[TEST5COMMON5]], access_type = #[[TEST5COMMON5]], offset = 0>
// CHECK: #[[TEST5A:.+]] = #llvm.tbaa_type_desc<id = "global data/common5_/bytes_0_to_3", members = {<#[[TEST5COMMON5]], 0>}>
// CHECK: #[[TEST5ATAG:.+]] = #llvm.tbaa_tag<base_type = #[[TEST5A]], access_type = #[[TEST5A]], offset = 0>
// CHECK-LABEL:   func.func @_QPtest5() {
// CHECK:           fir.load{{.*}}{tbaa = [#[[TEST5COMMON5TAG]]]} : !fir.ref<f32>
// CHECK:           fir.store{{.*}}{tbaa = [#[[TEST5ATAG]]]} : !fir.ref<f32>

// -----

// Fortran source:
// module m1
//   common /block/ a(10), b(10)
// end
// module m2
//   common /block/ c(20)
// end
// subroutine test6
//   use m1
//   use m2
//   a(1) = c(1) + b(1)
// end subroutine test6
//
// Test that all accesses are using the same TBAA tag.
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, i64 = dense<[32, 64]> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little">, llvm.data_layout = ""} {
  fir.global common @block_(dense<0> : vector<80xi8>) {alignment = 4 : i64} : !fir.array<80xi8>
  func.func @_QPtest6() {
    %c1 = arith.constant 1 : index
    %c20 = arith.constant 20 : index
    %c40 = arith.constant 40 : index
    %c10 = arith.constant 10 : index
    %c0 = arith.constant 0 : index
    %0 = fir.dummy_scope : !fir.dscope
    %1 = fir.address_of(@block_) : !fir.ref<!fir.array<80xi8>>
    %2 = fir.coordinate_of %1, %c0 : (!fir.ref<!fir.array<80xi8>>, index) -> !fir.ref<i8>
    %3 = fir.convert %2 : (!fir.ref<i8>) -> !fir.ref<!fir.array<10xf32>>
    %4 = fir.shape %c10 : (index) -> !fir.shape<1>
    %5 = fir.declare %3(%4) storage(%1[0]) {uniq_name = "_QMm1Ea"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>, !fir.ref<!fir.array<80xi8>>) -> !fir.ref<!fir.array<10xf32>>
    %6 = fir.coordinate_of %1, %c40 : (!fir.ref<!fir.array<80xi8>>, index) -> !fir.ref<i8>
    %7 = fir.convert %6 : (!fir.ref<i8>) -> !fir.ref<!fir.array<10xf32>>
    %8 = fir.declare %7(%4) storage(%1[40]) {uniq_name = "_QMm1Eb"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>, !fir.ref<!fir.array<80xi8>>) -> !fir.ref<!fir.array<10xf32>>
    %9 = fir.convert %2 : (!fir.ref<i8>) -> !fir.ref<!fir.array<20xf32>>
    %10 = fir.shape %c20 : (index) -> !fir.shape<1>
    %11 = fir.declare %9(%10) storage(%1[0]) {uniq_name = "_QMm2Ec"} : (!fir.ref<!fir.array<20xf32>>, !fir.shape<1>, !fir.ref<!fir.array<80xi8>>) -> !fir.ref<!fir.array<20xf32>>
    %12 = fir.array_coor %11(%10) %c1 : (!fir.ref<!fir.array<20xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
    %13 = fir.load %12 : !fir.ref<f32>
    %14 = fir.array_coor %8(%4) %c1 : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
    %15 = fir.load %14 : !fir.ref<f32>
    %16 = arith.addf %13, %15 fastmath<contract> : f32
    %17 = fir.array_coor %5(%4) %c1 : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
    fir.store %16 to %17 : !fir.ref<f32>
    return
  }
}
// CHECK: #[[$ATTR_0:.+]] = #llvm.tbaa_root<id = "Flang function root _QPtest6">
// CHECK: #[[$ATTR_1:.+]] = #llvm.tbaa_type_desc<id = "any access", members = {<#[[$ATTR_0]], 0>}>
// CHECK: #[[$ATTR_2:.+]] = #llvm.tbaa_type_desc<id = "any data access", members = {<#[[$ATTR_1]], 0>}>
// CHECK: #[[$ATTR_3:.+]] = #llvm.tbaa_type_desc<id = "target data", members = {<#[[$ATTR_2]], 0>}>
// CHECK: #[[$ATTR_4:.+]] = #llvm.tbaa_type_desc<id = "global data", members = {<#[[$ATTR_3]], 0>}>
// CHECK: #[[$ATTR_5:.+]] = #llvm.tbaa_type_desc<id = "global data/block_", members = {<#[[$ATTR_4]], 0>}>
// CHECK: #[[$ATTR_6:.+]] = #llvm.tbaa_type_desc<id = "global data/block_/bytes_0_to_79", members = {<#[[$ATTR_5]], 0>}>
// CHECK: #[[$ATTR_7:.+]] = #llvm.tbaa_tag<base_type = #[[$ATTR_6]], access_type = #[[$ATTR_6]], offset = 0>
// CHECK-LABEL:   func.func @_QPtest6() {
// CHECK:           fir.load{{.*}}{tbaa = [#[[$ATTR_7]]]} : !fir.ref<f32>
// CHECK:           fir.load{{.*}}{tbaa = [#[[$ATTR_7]]]} : !fir.ref<f32>
// CHECK:           fir.store{{.*}}{tbaa = [#[[$ATTR_7]]]} : !fir.ref<f32>

// -----

// Fortran source:
// module m1
//   integer :: b
//   common /block/ a(10), b
// end
// module m2
//   real, pointer :: p
//   common /block/ p
// end
// subroutine test7
//   use m1
//   use m2
//   a(1) = p + b
// end subroutine test7
//
// Test that:
//   * access to 'p' is tagged with "target data",
//   * access to 'b' is tagged with global data/block_/bytes_40_to_43
//   * access to 'a' is not tagged, because it overlaps with
//     a descriptor.
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, i64 = dense<[32, 64]> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little">, llvm.data_layout = ""} {  fir.global common @block_(dense<0> : vector<44xi8>) {alignment = 4 : i64} : !fir.array<44xi8>
  func.func @_QPtest7() {
    %c1 = arith.constant 1 : index
    %c40 = arith.constant 40 : index
    %c10 = arith.constant 10 : index
    %c0 = arith.constant 0 : index
    %0 = fir.dummy_scope : !fir.dscope
    %1 = fir.address_of(@block_) : !fir.ref<!fir.array<44xi8>>
    %2 = fir.coordinate_of %1, %c0 : (!fir.ref<!fir.array<44xi8>>, index) -> !fir.ref<i8>
    %3 = fir.convert %2 : (!fir.ref<i8>) -> !fir.ref<!fir.array<10xf32>>
    %4 = fir.shape %c10 : (index) -> !fir.shape<1>
    %5 = fir.declare %3(%4) storage(%1[0]) {uniq_name = "_QMm1Ea"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>, !fir.ref<!fir.array<44xi8>>) -> !fir.ref<!fir.array<10xf32>>
    %6 = fir.coordinate_of %1, %c40 : (!fir.ref<!fir.array<44xi8>>, index) -> !fir.ref<i8>
    %7 = fir.convert %6 : (!fir.ref<i8>) -> !fir.ref<i32>
    %8 = fir.declare %7 storage(%1[40]) {uniq_name = "_QMm1Eb"} : (!fir.ref<i32>, !fir.ref<!fir.array<44xi8>>) -> !fir.ref<i32>
    %9 = fir.convert %1 : (!fir.ref<!fir.array<44xi8>>) -> !fir.ref<!fir.array<24xi8>>
    %10 = fir.coordinate_of %9, %c0 : (!fir.ref<!fir.array<24xi8>>, index) -> !fir.ref<i8>
    %11 = fir.convert %10 : (!fir.ref<i8>) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
    %12 = fir.declare %11 storage(%9[0]) {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QMm2Ep"} : (!fir.ref<!fir.box<!fir.ptr<f32>>>, !fir.ref<!fir.array<24xi8>>) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
    %13 = fir.load %12 : !fir.ref<!fir.box<!fir.ptr<f32>>>
    %14 = fir.box_addr %13 : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
    %15 = fir.load %14 : !fir.ptr<f32>
    %16 = fir.load %8 : !fir.ref<i32>
    %17 = fir.convert %16 : (i32) -> f32
    %18 = arith.addf %15, %17 fastmath<contract> : f32
    %19 = fir.array_coor %5(%4) %c1 : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
    fir.store %18 to %19 : !fir.ref<f32>
    return
  }
}
// CHECK: #[[$ATTR_73:.+]] = #llvm.tbaa_root<id = "Flang function root _QPtest7">
// CHECK: #[[$ATTR_74:.+]] = #llvm.tbaa_type_desc<id = "any access", members = {<#[[$ATTR_73]], 0>}>
// CHECK: #[[$ATTR_75:.+]] = #llvm.tbaa_type_desc<id = "any data access", members = {<#[[$ATTR_74]], 0>}>
// CHECK: #[[$ATTR_76:.+]] = #llvm.tbaa_type_desc<id = "target data", members = {<#[[$ATTR_75]], 0>}>
// CHECK: #[[$ATTR_77:.+]] = #llvm.tbaa_tag<base_type = #[[$ATTR_76]], access_type = #[[$ATTR_76]], offset = 0>
// CHECK: #[[$ATTR_78:.+]] = #llvm.tbaa_type_desc<id = "global data", members = {<#[[$ATTR_76]], 0>}>
// CHECK: #[[$ATTR_79:.+]] = #llvm.tbaa_type_desc<id = "global data/block_", members = {<#[[$ATTR_78]], 0>}>
// CHECK: #[[$ATTR_80:.+]] = #llvm.tbaa_type_desc<id = "global data/block_/bytes_40_to_43", members = {<#[[$ATTR_79]], 0>}>
// CHECK: #[[$ATTR_81:.+]] = #llvm.tbaa_tag<base_type = #[[$ATTR_80]], access_type = #[[$ATTR_80]], offset = 0>
// CHECK-LABEL:   func.func @_QPtest7() {
// CHECK:           fir.load %{{[0-9]+}} : !fir.ref<!fir.box<!fir.ptr<f32>>>
// CHECK:           fir.load{{.*}} {tbaa = [#[[$ATTR_77]]]} : !fir.ptr<f32>
// CHECK:           fir.load{{.*}}{tbaa = [#[[$ATTR_81]]]} : !fir.ref<i32>
// CHECK:           fir.store %{{[0-9]+}} to %{{[0-9]+}} : !fir.ref<f32>

// -----

// Fortran source:
// module m1
//   integer :: b
//   common /block/ a(10), b
// end
// module m2
//   real, pointer :: p
//   common /block/ p
// end
// subroutine test8
//   use m1
//   use m2
//   a(1) = p + b
// end subroutine test8
//
// The storage for 'b' is manually made ambiguous.
// Test that none of the global data accesses is tagged,
// because there is member containing descriptor in 'block',
// and we cannot reliably trace the storage for 'b'.
// In most cases, the storage should be traceable.
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, i64 = dense<[32, 64]> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little">, llvm.data_layout = ""} {  fir.global common @block_(dense<0> : vector<44xi8>) {alignment = 4 : i64} : !fir.array<44xi8>
  func.func @_QPtest8() {
    %c1 = arith.constant 1 : index
    %c40 = arith.constant 40 : index
    %c10 = arith.constant 10 : index
    %c0 = arith.constant 0 : index
    %0 = fir.dummy_scope : !fir.dscope
    %1 = fir.address_of(@block_) : !fir.ref<!fir.array<44xi8>>
    %ptrcast = fir.convert %1 : (!fir.ref<!fir.array<44xi8>>) -> !fir.llvm_ptr<!fir.array<44xi8>>
    %tmpmem = fir.alloca !fir.llvm_ptr<!fir.array<44xi8>>
    fir.store %ptrcast to %tmpmem : !fir.ref<!fir.llvm_ptr<!fir.array<44xi8>>>
    %2 = fir.coordinate_of %1, %c0 : (!fir.ref<!fir.array<44xi8>>, index) -> !fir.ref<i8>
    %3 = fir.convert %2 : (!fir.ref<i8>) -> !fir.ref<!fir.array<10xf32>>
    %4 = fir.shape %c10 : (index) -> !fir.shape<1>
    %5 = fir.declare %3(%4) storage(%1[0]) {uniq_name = "_QMm1Ea"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>, !fir.ref<!fir.array<44xi8>>) -> !fir.ref<!fir.array<10xf32>>
    %6 = fir.coordinate_of %1, %c40 : (!fir.ref<!fir.array<44xi8>>, index) -> !fir.ref<i8>
    %7 = fir.convert %6 : (!fir.ref<i8>) -> !fir.ref<i32>
    %addr = fir.load %tmpmem : !fir.ref<!fir.llvm_ptr<!fir.array<44xi8>>>
    %addrcast = fir.convert %addr : (!fir.llvm_ptr<!fir.array<44xi8>>) -> !fir.ref<!fir.array<44xi8>>
    %8 = fir.declare %7 storage(%addrcast[40]) {uniq_name = "_QMm1Eb"} : (!fir.ref<i32>, !fir.ref<!fir.array<44xi8>>) -> !fir.ref<i32>
    %9 = fir.convert %1 : (!fir.ref<!fir.array<44xi8>>) -> !fir.ref<!fir.array<24xi8>>
    %10 = fir.coordinate_of %9, %c0 : (!fir.ref<!fir.array<24xi8>>, index) -> !fir.ref<i8>
    %11 = fir.convert %10 : (!fir.ref<i8>) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
    %12 = fir.declare %11 storage(%9[0]) {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QMm2Ep"} : (!fir.ref<!fir.box<!fir.ptr<f32>>>, !fir.ref<!fir.array<24xi8>>) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
    %13 = fir.load %12 : !fir.ref<!fir.box<!fir.ptr<f32>>>
    %14 = fir.box_addr %13 : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
    %15 = fir.load %14 : !fir.ptr<f32>
    %16 = fir.load %8 : !fir.ref<i32>
    %17 = fir.convert %16 : (i32) -> f32
    %18 = arith.addf %15, %17 fastmath<contract> : f32
    %19 = fir.array_coor %5(%4) %c1 : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
    fir.store %18 to %19 : !fir.ref<f32>
    return
  }
}
// CHECK: #[[$ATTR_82:.+]] = #llvm.tbaa_root<id = "Flang function root _QPtest8">
// CHECK: #[[$ATTR_83:.+]] = #llvm.tbaa_type_desc<id = "any access", members = {<#[[$ATTR_82]], 0>}>
// CHECK: #[[$ATTR_84:.+]] = #llvm.tbaa_type_desc<id = "any data access", members = {<#[[$ATTR_83]], 0>}>
// CHECK: #[[$ATTR_85:.+]] = #llvm.tbaa_type_desc<id = "target data", members = {<#[[$ATTR_84]], 0>}>
// CHECK: #[[$ATTR_86:.+]] = #llvm.tbaa_tag<base_type = #[[$ATTR_85]], access_type = #[[$ATTR_85]], offset = 0>
// CHECK: #[[$ATTR_87:.+]] = #llvm.tbaa_type_desc<id = "allocated data", members = {<#[[$ATTR_85]], 0>}>
// CHECK: #[[$ATTR_88:.+]] = #llvm.tbaa_tag<base_type = #[[$ATTR_87]], access_type = #[[$ATTR_87]], offset = 0>
// CHECK-LABEL:   func.func @_QPtest8() {
// CHECK:           fir.load %{{[0-9]+}} : !fir.ref<!fir.box<!fir.ptr<f32>>>
// CHECK:           fir.load %{{[0-9]+}} {tbaa = [#[[$ATTR_86]]]} : !fir.ptr<f32>
// CHECK:           fir.load %{{[0-9]+}} : !fir.ref<i32>
// CHECK:           fir.store %{{[0-9]+}} to %{{[0-9]+}} : !fir.ref<f32>
