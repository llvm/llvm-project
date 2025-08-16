// RUN: fir-opt --split-input-file --fir-add-alias-tags %s | FileCheck %s

// Fortran source:
// subroutine test1
//   real :: a, b
//   common /common1/ a, b
//   a = b
// end subroutine test1
fir.global common @common1_(dense<0> : vector<8xi8>) {alignment = 4 : i64} : !fir.array<8xi8>
func.func @_QPtest1() {
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.address_of(@common1_) : !fir.ref<!fir.array<8xi8>>
  %2 = fir.convert %1 : (!fir.ref<!fir.array<8xi8>>) -> !fir.ref<!fir.array<?xi8>>
  %3 = fir.coordinate_of %2, %c0 : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
  %4 = fir.convert %3 : (!fir.ref<i8>) -> !fir.ref<f32>
  %5 = fir.declare %4 {uniq_name = "_QFtest1Ea"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %6 = fir.coordinate_of %2, %c4 : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
  %7 = fir.convert %6 : (!fir.ref<i8>) -> !fir.ref<f32>
  %8 = fir.declare %7 {uniq_name = "_QFtest1Eb"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %9 = fir.load %8 : !fir.ref<f32>
  fir.store %9 to %5 : !fir.ref<f32>
  return
}
// CHECK: #[[$ATTR_0:.+]] = #llvm.tbaa_root<id = "Flang function root _QPtest1">
// CHECK: #[[$ATTR_1:.+]] = #llvm.tbaa_type_desc<id = "any access", members = {<#[[$ATTR_0]], 0>}>
// CHECK: #[[$ATTR_2:.+]] = #llvm.tbaa_type_desc<id = "any data access", members = {<#[[$ATTR_1]], 0>}>
// CHECK: #[[$ATTR_3:.+]] = #llvm.tbaa_type_desc<id = "target data", members = {<#[[$ATTR_2]], 0>}>
// CHECK: #[[$ATTR_4:.+]] = #llvm.tbaa_type_desc<id = "global data", members = {<#[[$ATTR_3]], 0>}>
// CHECK: #[[$ATTR_5:.+]] = #llvm.tbaa_type_desc<id = "global data/common1_", members = {<#[[$ATTR_4]], 0>}>
// CHECK: #[[$ATTR_6:.+]] = #llvm.tbaa_type_desc<id = "global data/common1_/_QFtest1Eb", members = {<#[[$ATTR_5]], 0>}>
// CHECK: #[[$ATTR_7:.+]] = #llvm.tbaa_type_desc<id = "global data/common1_/_QFtest1Ea", members = {<#[[$ATTR_5]], 0>}>
// CHECK: #[[$ATTR_8:.+]] = #llvm.tbaa_tag<base_type = #[[$ATTR_6]], access_type = #[[$ATTR_6]], offset = 0>
// CHECK: #[[$ATTR_9:.+]] = #llvm.tbaa_tag<base_type = #[[$ATTR_7]], access_type = #[[$ATTR_7]], offset = 0>
// CHECK-LABEL:   func.func @_QPtest1() {
// CHECK:           fir.load{{.*}}{tbaa = [#[[$ATTR_8]]]} : !fir.ref<f32>
// CHECK:           fir.store{{.*}}{tbaa = [#[[$ATTR_9]]]} : !fir.ref<f32>

// -----

// Fortran source:
// subroutine test2
//   real :: a, b
//   common /common2/ a, b
//   a = b
// end subroutine test2
fir.global common @common2_(dense<0> : vector<8xi8>) {alignment = 4 : i64} : !fir.array<8xi8>
func.func @_QPtest2() {
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.address_of(@common2_) : !fir.ref<!fir.array<8xi8>>
  %2 = fir.convert %1 : (!fir.ref<!fir.array<8xi8>>) -> !fir.ref<!fir.array<?xi8>>
  %3 = fir.coordinate_of %2, %c0 : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
  %4 = fir.convert %3 : (!fir.ref<i8>) -> !fir.ref<f32>
  %5 = fir.declare %4 {uniq_name = "_QFtest2Ea"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %6 = fir.coordinate_of %2, %c4 : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
  %7 = fir.convert %6 : (!fir.ref<i8>) -> !fir.ref<f32>
  %8 = fir.declare %7 {uniq_name = "_QFtest2Eb"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %9 = fir.load %8 : !fir.ref<f32>
  fir.store %9 to %5 : !fir.ref<f32>
  return
}
// CHECK: #[[$ATTR_10:.+]] = #llvm.tbaa_root<id = "Flang function root _QPtest2">
// CHECK: #[[$ATTR_11:.+]] = #llvm.tbaa_type_desc<id = "any access", members = {<#[[$ATTR_10]], 0>}>
// CHECK: #[[$ATTR_12:.+]] = #llvm.tbaa_type_desc<id = "any data access", members = {<#[[$ATTR_11]], 0>}>
// CHECK: #[[$ATTR_13:.+]] = #llvm.tbaa_type_desc<id = "target data", members = {<#[[$ATTR_12]], 0>}>
// CHECK: #[[$ATTR_14:.+]] = #llvm.tbaa_type_desc<id = "global data", members = {<#[[$ATTR_13]], 0>}>
// CHECK: #[[$ATTR_15:.+]] = #llvm.tbaa_type_desc<id = "global data/common2_", members = {<#[[$ATTR_14]], 0>}>
// CHECK: #[[$ATTR_16:.+]] = #llvm.tbaa_type_desc<id = "global data/common2_/_QFtest2Eb", members = {<#[[$ATTR_15]], 0>}>
// CHECK: #[[$ATTR_17:.+]] = #llvm.tbaa_type_desc<id = "global data/common2_/_QFtest2Ea", members = {<#[[$ATTR_15]], 0>}>
// CHECK: #[[$ATTR_18:.+]] = #llvm.tbaa_tag<base_type = #[[$ATTR_16]], access_type = #[[$ATTR_16]], offset = 0>
// CHECK: #[[$ATTR_19:.+]] = #llvm.tbaa_tag<base_type = #[[$ATTR_17]], access_type = #[[$ATTR_17]], offset = 0>
// CHECK-LABEL:   func.func @_QPtest2() {
// CHECK:           fir.load{{.*}}{tbaa = [#[[$ATTR_18]]]} : !fir.ref<f32>
// CHECK:           fir.store{{.*}}{tbaa = [#[[$ATTR_19]]]} : !fir.ref<f32>

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
fir.global common @common3_(dense<0> : vector<8xi8>) {alignment = 4 : i64} : !fir.array<8xi8>
func.func @_QPtest3() {
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.address_of(@common3_) : !fir.ref<!fir.array<8xi8>>
  %2 = fir.convert %1 : (!fir.ref<!fir.array<8xi8>>) -> !fir.ref<!fir.array<?xi8>>
  %3 = fir.coordinate_of %2, %c0 : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
  %4 = fir.convert %3 : (!fir.ref<i8>) -> !fir.ref<f32>
  %5 = fir.declare %4 {uniq_name = "_QFtest3Ea"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %6 = fir.coordinate_of %2, %c4 : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
  %7 = fir.convert %6 : (!fir.ref<i8>) -> !fir.ref<f32>
  %8 = fir.declare %7 {uniq_name = "_QFtest3Eb"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %9 = fir.load %8 : !fir.ref<f32>
  fir.store %9 to %5 : !fir.ref<f32>
  %10 = fir.dummy_scope : !fir.dscope
  %11 = fir.declare %5 dummy_scope %10 {uniq_name = "_QFtest3FinnerEc"} : (!fir.ref<f32>, !fir.dscope) -> !fir.ref<f32>
  %12 = fir.declare %8 dummy_scope %10 {uniq_name = "_QFtest3FinnerEd"} : (!fir.ref<f32>, !fir.dscope) -> !fir.ref<f32>
  %13 = fir.load %12 : !fir.ref<f32>
  fir.store %13 to %11 : !fir.ref<f32>
  return
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
// CHECK: #[[GLOBALB:.+]] = #llvm.tbaa_type_desc<id = "global data/common3_/_QFtest3Eb", members = {<#[[GLOBALDATA3COMMON3]], 0>}>
// CHECK: #[[GLOBALA:.+]] = #llvm.tbaa_type_desc<id = "global data/common3_/_QFtest3Ea", members = {<#[[GLOBALDATA3COMMON3]], 0>}>
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
fir.global common @common4_(dense<0> : vector<8xi8>) {alignment = 4 : i64} : !fir.array<8xi8>
func.func @_QPtest4() {
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.address_of(@common4_) : !fir.ref<!fir.array<8xi8>>
  %2 = fir.convert %1 : (!fir.ref<!fir.array<8xi8>>) -> !fir.ref<!fir.array<?xi8>>
  %3 = fir.coordinate_of %2, %c0 : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
  %4 = fir.convert %3 : (!fir.ref<i8>) -> !fir.ref<f32>
  %5 = fir.declare %4 {uniq_name = "_QFtest4Ea"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %6 = fir.coordinate_of %2, %c4 : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
  %7 = fir.convert %6 : (!fir.ref<i8>) -> !fir.ref<f32>
  %8 = fir.declare %7 {uniq_name = "_QFtest4Eb"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %9 = fir.load %8 : !fir.ref<f32>
  fir.store %9 to %5 : !fir.ref<f32>
  %10 = fir.dummy_scope : !fir.dscope
  %11 = fir.declare %4 {uniq_name = "_QFtest4FinnerEc"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %12 = fir.declare %7 {uniq_name = "_QFtest4FinnerEd"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %13 = fir.load %12 : !fir.ref<f32>
  fir.store %13 to %11 : !fir.ref<f32>
  return
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
// CHECK: #[[TEST4B:.+]] = #llvm.tbaa_type_desc<id = "global data/common4_/_QFtest4Eb", members = {<#[[TEST4COMMON]], 0>}>
// CHECK: #[[TEST4A:.+]] = #llvm.tbaa_type_desc<id = "global data/common4_/_QFtest4Ea", members = {<#[[TEST4COMMON]], 0>}>
// CHECK: #[[INNER4D:.+]] = #llvm.tbaa_type_desc<id = "global data/common4_/_QFtest4FinnerEd", members = {<#[[INNER4COMMON]], 0>}>
// CHECK: #[[INNER4C:.+]] = #llvm.tbaa_type_desc<id = "global data/common4_/_QFtest4FinnerEc", members = {<#[[INNER4COMMON]], 0>}>
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
fir.global common @common5_(dense<0> : vector<8xi8>) {alignment = 4 : i64} : !fir.array<8xi8>
func.func @_QPtest5() {
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.address_of(@common5_) : !fir.ref<!fir.array<8xi8>>
  %2 = fir.convert %1 : (!fir.ref<!fir.array<8xi8>>) -> !fir.ref<!fir.array<?xi8>>
  %3 = fir.coordinate_of %2, %c0 : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
  %4 = fir.convert %3 : (!fir.ref<i8>) -> !fir.ref<f32>
  %5 = fir.declare %4 {uniq_name = "_QFtest5Ea"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %6 = fir.coordinate_of %2, %c4 : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
  %7 = fir.convert %6 : (!fir.ref<i8>) -> !fir.ref<f32>
  %9 = fir.load %7 : !fir.ref<f32>
  fir.store %9 to %5 : !fir.ref<f32>
  return
}
// CHECK: #[[TEST5ROOT:.+]] = #llvm.tbaa_root<id = "Flang function root _QPtest5">
// CHECK: #[[TEST5ANYACC:.+]] = #llvm.tbaa_type_desc<id = "any access", members = {<#[[TEST5ROOT]], 0>}>
// CHECK: #[[TEST5ANYDATA:.+]] = #llvm.tbaa_type_desc<id = "any data access", members = {<#[[TEST5ANYACC]], 0>}>
// CHECK: #[[TEST5TARGET:.+]] = #llvm.tbaa_type_desc<id = "target data", members = {<#[[TEST5ANYDATA]], 0>}>
// CHECK: #[[TEST5GLOBAL:.+]] = #llvm.tbaa_type_desc<id = "global data", members = {<#[[TEST5TARGET]], 0>}>
// CHECK: #[[TEST5COMMON5:.+]] = #llvm.tbaa_type_desc<id = "global data/common5_", members = {<#[[TEST5GLOBAL]], 0>}>
// CHECK: #[[TEST5COMMON5TAG:.+]] = #llvm.tbaa_tag<base_type = #[[TEST5COMMON5]], access_type = #[[TEST5COMMON5]], offset = 0>
// CHECK: #[[TEST5A:.+]] = #llvm.tbaa_type_desc<id = "global data/common5_/_QFtest5Ea", members = {<#[[TEST5COMMON5]], 0>}>
// CHECK: #[[TEST5ATAG:.+]] = #llvm.tbaa_tag<base_type = #[[TEST5A]], access_type = #[[TEST5A]], offset = 0>
// CHECK-LABEL:   func.func @_QPtest5() {
// CHECK:           fir.load{{.*}}{tbaa = [#[[TEST5COMMON5TAG]]]} : !fir.ref<f32>
// CHECK:           fir.store{{.*}}{tbaa = [#[[TEST5ATAG]]]} : !fir.ref<f32>
