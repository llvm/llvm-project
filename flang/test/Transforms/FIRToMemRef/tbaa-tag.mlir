// RUN: fir-opt %s --fir-to-memref --allow-unregistered-dialect | FileCheck %s
// Derived from:
// program main
//   real :: array(10)
//   array = 1327
//   array = array + scalar
// end program
// CHECK: #tbaa_root = #llvm.tbaa_root<id = "Flang function root _QQmain">
// CHECK-NEXT: #tbaa_type_desc = #llvm.tbaa_type_desc<id = "any access", members = {<#tbaa_root, 0>}>
// CHECK-NEXT: #tbaa_type_desc1 = #llvm.tbaa_type_desc<id = "any data access", members = {<#tbaa_type_desc, 0>}>
// CHECK-NEXT: #tbaa_type_desc2 = #llvm.tbaa_type_desc<id = "global data", members = {<#tbaa_type_desc1, 0>}>
// CHECK-NEXT: #tbaa_type_desc3 = #llvm.tbaa_type_desc<id = "global data/_QFEarray", members = {<#tbaa_type_desc2, 0>}>
// CHECK-NEXT: #tbaa_tag = #llvm.tbaa_tag<base_type = #tbaa_type_desc3, access_type = #tbaa_type_desc3, offset = 0>
// CHECK-LABEL: func.func @tbaa
// CHECK: memref.store %cst, %{{[0-9]+}}[%{{[0-9]+}}] {tbaa = [#tbaa_tag]}
// CHECK: memref.load %{{[0-9]+}}[%{{[0-9]+}}] {tbaa = [#tbaa_tag]}
// CHECK: memref.store %{{[0-9]+}}, %{{[0-9]+}}[%{{[0-9]+}}] {tbaa = [#tbaa_tag]}
func.func @tbaa() {
  %c1 = arith.constant 1 : index
  %cst = arith.constant 1.327000e+03 : f32
  %c10 = arith.constant 10 : index
  %0 = fir.address_of(@_QFEarray) : !fir.ref<!fir.array<10xf32>>
  %shape = fir.shape %c10 : (index) -> !fir.shape<1>
  %1 = fir.declare %0(%shape) {uniq_name = "_QFEarray"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.ref<!fir.array<10xf32>>
  %2 = fir.alloca f32 {bindc_name = "scalar", uniq_name = "_QFEscalar"}
  %3 = fir.declare %2 {uniq_name = "_QFEscalar"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %c1_0 = arith.constant 1 : index
  %4 = arith.addi %c10, %c1_0 : index
  scf.for %arg0 = %c1 to %4 step %c1 {
    %7 = fir.array_coor %1(%shape) %arg0 : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
    fir.store %cst to %7 {tbaa = [#llvm.tbaa_tag<base_type = <id = "global data/_QFEarray", members = {<#llvm.tbaa_type_desc<id = "global data", members = {<#llvm.tbaa_type_desc<id = "any data access", members = {<#llvm.tbaa_type_desc<id = "any access", members = {<#llvm.tbaa_root<id = "Flang function root _QQmain">, 0>}>, 0>}>, 0>}>, 0>}>, access_type = <id = "global data/_QFEarray", members = {<#llvm.tbaa_type_desc<id = "global data", members = {<#llvm.tbaa_type_desc<id = "any data access", members = {<#llvm.tbaa_type_desc<id = "any access", members = {<#llvm.tbaa_root<id = "Flang function root _QQmain">, 0>}>, 0>}>, 0>}>, 0>}>, offset = 0>]} : !fir.ref<f32>
  }
  %5 = fir.load %3 : !fir.ref<f32>
  %c1_1 = arith.constant 1 : index
  %6 = arith.addi %c10, %c1_1 : index
  scf.for %arg0 = %c1 to %6 step %c1 {
    %7 = fir.array_coor %1(%shape) %arg0 : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
    %8 = fir.load %7 {tbaa = [#llvm.tbaa_tag<base_type = <id = "global data/_QFEarray", members = {<#llvm.tbaa_type_desc<id = "global data", members = {<#llvm.tbaa_type_desc<id = "any data access", members = {<#llvm.tbaa_type_desc<id = "any access", members = {<#llvm.tbaa_root<id = "Flang function root _QQmain">, 0>}>, 0>}>, 0>}>, 0>}>, access_type = <id = "global data/_QFEarray", members = {<#llvm.tbaa_type_desc<id = "global data", members = {<#llvm.tbaa_type_desc<id = "any data access", members = {<#llvm.tbaa_type_desc<id = "any access", members = {<#llvm.tbaa_root<id = "Flang function root _QQmain">, 0>}>, 0>}>, 0>}>, 0>}>, offset = 0>]} : !fir.ref<f32>
    %9 = arith.addf %8, %5 fastmath<contract> : f32
    fir.store %9 to %7 {tbaa = [#llvm.tbaa_tag<base_type = <id = "global data/_QFEarray", members = {<#llvm.tbaa_type_desc<id = "global data", members = {<#llvm.tbaa_type_desc<id = "any data access", members = {<#llvm.tbaa_type_desc<id = "any access", members = {<#llvm.tbaa_root<id = "Flang function root _QQmain">, 0>}>, 0>}>, 0>}>, 0>}>, access_type = <id = "global data/_QFEarray", members = {<#llvm.tbaa_type_desc<id = "global data", members = {<#llvm.tbaa_type_desc<id = "any data access", members = {<#llvm.tbaa_type_desc<id = "any access", members = {<#llvm.tbaa_root<id = "Flang function root _QQmain">, 0>}>, 0>}>, 0>}>, 0>}>, offset = 0>]} : !fir.ref<f32>
  }
  return
}

