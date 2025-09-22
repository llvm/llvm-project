// RUN: fir-opt --fir-add-alias-tags %s | FileCheck --check-prefixes=ALL,DEFAULT %s
// RUN: fir-opt --fir-add-alias-tags --local-alloc-tbaa %s | FileCheck --check-prefixes=ALL,LOCAL %s

// Test AddAliasTagsPass creating sub-tree for TARGET/POINTER variables.

// module data
//   real :: glob(10)
//   real, target :: globt(10)
//   real, allocatable :: globa(:)
//   real, allocatable, target :: globat(:)
//   real, pointer :: globp(:)
// end module data
// subroutine test1(dummyf, dummyft, dummyas, dummyast, dummya, dummyat, dummyp)
//   use data
//   real :: dummyf(10)
//   real, target :: dummyft(10)
//   real :: dummyas(:)
//   real, target :: dummyast(:)
//   real, allocatable :: dummya(:)
//   real, allocatable, target :: dummyat(:)
//   real, pointer :: dummyp(:)
//   real :: local(10)
//   real, target :: localt(10)
//   real, allocatable :: locala(:)
//   real, allocatable, target :: localat(:)
//   real, pointer :: localp(:)
//   glob(1) = 1.0
//   globt(1) = 1.0
//   globa(1) = 1.0
//   globat(1) = 1.0
//   globp(1) = 1.0
//   dummyf(1) = 1.0
//   dummyft(1) = 1.0
//   dummyas(1) = 1.0
//   dummyast(1) = 1.0
//   dummya(1) = 1.0
//   dummyat(1) = 1.0
//   dummyp(1) = 1.0
//   local(1) = 1.0
//   localt(1) = 1.0
//   locala(1) = 1.0
//   localat(1) = 1.0
//   localp(1) = 1.0
// end subroutine test1

// "Flang function root _QPtest1"
// |
// "any access"
// |
// |- "descriptor member"
// |- "any data access"
//    |
//    |- "dummy arg data"
//    |  |
//    |  |- "dummy arg data/_QFtest1Edummyf"
//    |  |- "dummy arg data/_QFtest1Edummyas"
//    |  |- "dummy arg data/_QFtest1Edummya"
//    |
//    |- "target data" <- all pointers and taget dummys
//       |
//       |- "global data"
//       |  |
//       |  |- "global data/_QMdataEglob"
//       |  |- "global data/_QMdataEglobt"
//       |
//       |- "direct data"
//       |  |
//       |  |- "direct data/_QMdataEgloba"
//       |  |- "direct data/_QMdataEglobat"
//       |
//       |- "allocated data"
//          |
//          |- "allocated data/_QFtest1Elocal"
//          |- "allocated data/_QFtest1Elocalt"
//          |- "allocated data/_QFtest1Elocala"
//          |- "allocated data/_QFtest1Elocalat"

// ALL: #[[FUNCROOT:.+]] = #llvm.tbaa_root<id = "Flang function root _QPtest1">
// ALL: #[[ANYACCESS:.+]] = #llvm.tbaa_type_desc<id = "any access", members = {<#[[FUNCROOT]], 0>}>
// ALL: #[[ANYDATA:.+]] = #llvm.tbaa_type_desc<id = "any data access", members = {<#[[ANYACCESS]], 0>}>
// ALL: #[[TARGETDATA:.+]] = #llvm.tbaa_type_desc<id = "target data", members = {<#[[ANYDATA]], 0>}>
// ALL: #[[DUMMYDATA:.+]] = #llvm.tbaa_type_desc<id = "dummy arg data", members = {<#[[ANYDATA]], 0>}>
// ALL: #[[TARGETTAG:.+]] = #llvm.tbaa_tag<base_type = #[[TARGETDATA]], access_type = #[[TARGETDATA]], offset = 0>
// ALL: #[[GLOBALDATA:.+]] = #llvm.tbaa_type_desc<id = "global data", members = {<#[[TARGETDATA]], 0>}>
// ALL: #[[DIRECTDATA:.+]] = #llvm.tbaa_type_desc<id = "direct data", members = {<#[[TARGETDATA]], 0>}>
// ALL: #[[DUMMYFVAR:.+]] = #llvm.tbaa_type_desc<id = "dummy arg data/_QFtest1Edummyf", members = {<#[[DUMMYDATA]], 0>}>
// ALL: #[[DUMMYASVAR:.+]] = #llvm.tbaa_type_desc<id = "dummy arg data/_QFtest1Edummyas", members = {<#[[DUMMYDATA]], 0>}>
// ALL: #[[DUMMYAVAR:.+]] = #llvm.tbaa_type_desc<id = "dummy arg data/_QFtest1Edummya", members = {<#[[DUMMYDATA]], 0>}>
// LOCAL: #[[LOCALDATA:.+]] = #llvm.tbaa_type_desc<id = "allocated data", members = {<#[[TARGETDATA]], 0>}>
// ALL: #[[DUMMYFTAG:.+]] = #llvm.tbaa_tag<base_type = #[[DUMMYFVAR]], access_type = #[[DUMMYFVAR]], offset = 0>
// ALL: #[[DUMMYASTAG:.+]] = #llvm.tbaa_tag<base_type = #[[DUMMYASVAR]], access_type = #[[DUMMYASVAR]], offset = 0>
// ALL: #[[DUMMYATAG:.+]] = #llvm.tbaa_tag<base_type = #[[DUMMYAVAR]], access_type = #[[DUMMYAVAR]], offset = 0>
// ALL: #[[GLOBVAR:.+]] = #llvm.tbaa_type_desc<id = "global data/_QMdataEglob", members = {<#[[GLOBALDATA]], 0>}>
// ALL: #[[GLOBTVAR:.+]] = #llvm.tbaa_type_desc<id = "global data/_QMdataEglobt", members = {<#[[GLOBALDATA]], 0>}>
// ALL: #[[GLOBAVAR:.+]] = #llvm.tbaa_type_desc<id = "direct data/_QMdataEgloba", members = {<#[[DIRECTDATA]], 0>}>
// ALL: #[[GLOBATVAR:.+]] = #llvm.tbaa_type_desc<id = "direct data/_QMdataEglobat", members = {<#[[DIRECTDATA]], 0>}>
// LOCAL: #[[LOCALVAR:.+]] = #llvm.tbaa_type_desc<id = "allocated data/_QFtest1Elocal", members = {<#[[LOCALDATA]], 0>}>
// LOCAL: #[[LOCALTVAR:.+]] = #llvm.tbaa_type_desc<id = "allocated data/_QFtest1Elocalt", members = {<#[[LOCALDATA]], 0>}>
// LOCAL: #[[LOCALAVAR:.+]] = #llvm.tbaa_type_desc<id = "allocated data/_QFtest1Elocala", members = {<#[[LOCALDATA]], 0>}>
// LOCAL: #[[LOCALATVAR:.+]] = #llvm.tbaa_type_desc<id = "allocated data/_QFtest1Elocalat", members = {<#[[LOCALDATA]], 0>}>
// ALL: #[[GLOBTAG:.+]] = #llvm.tbaa_tag<base_type = #[[GLOBVAR]], access_type = #[[GLOBVAR]], offset = 0>
// ALL: #[[GLOBTTAG:.+]] = #llvm.tbaa_tag<base_type = #[[GLOBTVAR]], access_type = #[[GLOBTVAR]], offset = 0>
// ALL: #[[GLOBATAG:.+]] = #llvm.tbaa_tag<base_type = #[[GLOBAVAR]], access_type = #[[GLOBAVAR]], offset = 0>
// ALL: #[[GLOBATTAG:.+]] = #llvm.tbaa_tag<base_type = #[[GLOBATVAR]], access_type = #[[GLOBATVAR]], offset = 0>
// LOCAL: #[[LOCALTAG:.+]] = #llvm.tbaa_tag<base_type = #[[LOCALVAR]], access_type = #[[LOCALVAR]], offset = 0>
// LOCAL: #[[LOCALTTAG:.+]] = #llvm.tbaa_tag<base_type = #[[LOCALTVAR]], access_type = #[[LOCALTVAR]], offset = 0>
// LOCAL: #[[LOCALATAG:.+]] = #llvm.tbaa_tag<base_type = #[[LOCALAVAR]], access_type = #[[LOCALAVAR]], offset = 0>
// LOCAL: #[[LOCALATTAG:.+]] = #llvm.tbaa_tag<base_type = #[[LOCALATVAR]], access_type = #[[LOCALATVAR]], offset = 0>

module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, i64 = dense<[32, 64]> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little">, llvm.data_layout = ""} {
  fir.global @_QMdataEglob : !fir.array<10xf32> {
    %0 = fir.zero_bits !fir.array<10xf32>
    fir.has_value %0 : !fir.array<10xf32>
  }
  fir.global @_QMdataEgloba : !fir.box<!fir.heap<!fir.array<?xf32>>> {
    %c0 = arith.constant 0 : index
    %0 = fir.zero_bits !fir.heap<!fir.array<?xf32>>
    %1 = fir.shape %c0 : (index) -> !fir.shape<1>
    %2 = fir.embox %0(%1) : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xf32>>>
    fir.has_value %2 : !fir.box<!fir.heap<!fir.array<?xf32>>>
  }
  fir.global @_QMdataEglobat target : !fir.box<!fir.heap<!fir.array<?xf32>>> {
    %c0 = arith.constant 0 : index
    %0 = fir.zero_bits !fir.heap<!fir.array<?xf32>>
    %1 = fir.shape %c0 : (index) -> !fir.shape<1>
    %2 = fir.embox %0(%1) : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xf32>>>
    fir.has_value %2 : !fir.box<!fir.heap<!fir.array<?xf32>>>
  }
  fir.global @_QMdataEglobp : !fir.box<!fir.ptr<!fir.array<?xf32>>> {
    %c0 = arith.constant 0 : index
    %0 = fir.zero_bits !fir.ptr<!fir.array<?xf32>>
    %1 = fir.shape %c0 : (index) -> !fir.shape<1>
    %2 = fir.embox %0(%1) : (!fir.ptr<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
    fir.has_value %2 : !fir.box<!fir.ptr<!fir.array<?xf32>>>
  }
  fir.global @_QMdataEglobt target : !fir.array<10xf32> {
    %0 = fir.zero_bits !fir.array<10xf32>
    fir.has_value %0 : !fir.array<10xf32>
  }

// ALL-LABEL:   func.func @_QPtest1(
  func.func @_QPtest1(%arg0: !fir.ref<!fir.array<10xf32>> {fir.bindc_name = "dummyf"}, %arg1: !fir.ref<!fir.array<10xf32>> {fir.bindc_name = "dummyft", fir.target}, %arg2: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "dummyas"}, %arg3: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "dummyast", fir.target}, %arg4: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>> {fir.bindc_name = "dummya"}, %arg5: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>> {fir.bindc_name = "dummyat", fir.target}, %arg6: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>> {fir.bindc_name = "dummyp"}) {
    %c0_i64 = arith.constant 0 : i64
    %c1 = arith.constant 1 : index
    %cst = arith.constant 1.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c10 = arith.constant 10 : index
    %0 = fir.dummy_scope : !fir.dscope
    %1 = fir.declare %arg4 dummy_scope %0 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest1Edummya"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.dscope) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
    %2 = fir.declare %arg2 dummy_scope %0 {uniq_name = "_QFtest1Edummyas"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> !fir.box<!fir.array<?xf32>>
    %3 = fir.declare %arg3 dummy_scope %0 {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFtest1Edummyast"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> !fir.box<!fir.array<?xf32>>
    %4 = fir.declare %arg5 dummy_scope %0 {fortran_attrs = #fir.var_attrs<allocatable, target>, uniq_name = "_QFtest1Edummyat"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.dscope) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
    %5 = fir.shape %c10 : (index) -> !fir.shape<1>
    %6 = fir.declare %arg0(%5) dummy_scope %0 {uniq_name = "_QFtest1Edummyf"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<10xf32>>
    %7 = fir.declare %arg1(%5) dummy_scope %0 {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFtest1Edummyft"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<10xf32>>
    %8 = fir.declare %arg6 dummy_scope %0 {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtest1Edummyp"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>, !fir.dscope) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
    %9 = fir.address_of(@_QMdataEglob) : !fir.ref<!fir.array<10xf32>>
    %10 = fir.declare %9(%5) {uniq_name = "_QMdataEglob"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.ref<!fir.array<10xf32>>
    %11 = fir.address_of(@_QMdataEgloba) : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
    %12 = fir.declare %11 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QMdataEgloba"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
    %13 = fir.address_of(@_QMdataEglobat) : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
    %14 = fir.declare %13 {fortran_attrs = #fir.var_attrs<allocatable, target>, uniq_name = "_QMdataEglobat"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
    %15 = fir.address_of(@_QMdataEglobp) : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
    %16 = fir.declare %15 {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QMdataEglobp"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
    %17 = fir.address_of(@_QMdataEglobt) : !fir.ref<!fir.array<10xf32>>
    %18 = fir.declare %17(%5) {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QMdataEglobt"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.ref<!fir.array<10xf32>>
    %19 = fir.alloca !fir.array<10xf32> {bindc_name = "local", uniq_name = "_QFtest1Elocal"}
    %20 = fir.declare %19(%5) {uniq_name = "_QFtest1Elocal"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.ref<!fir.array<10xf32>>
    %21 = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>> {bindc_name = "locala", uniq_name = "_QFtest1Elocala"}
    %22 = fir.zero_bits !fir.heap<!fir.array<?xf32>>
    %23 = fir.shape %c0 : (index) -> !fir.shape<1>
    %24 = fir.embox %22(%23) : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xf32>>>
// ALL-NOT: fir.store{{.*}}tbaa
    fir.store %24 to %21 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
    %25 = fir.declare %21 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest1Elocala"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
    %26 = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>> {bindc_name = "localat", fir.target, uniq_name = "_QFtest1Elocalat"}
// ALL-NOT: fir.store{{.*}}tbaa
    fir.store %24 to %26 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
    %27 = fir.declare %26 {fortran_attrs = #fir.var_attrs<allocatable, target>, uniq_name = "_QFtest1Elocalat"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
    %28 = fir.alloca !fir.box<!fir.ptr<!fir.array<?xf32>>> {bindc_name = "localp", uniq_name = "_QFtest1Elocalp"}
    %29 = fir.zero_bits !fir.ptr<!fir.array<?xf32>>
    %30 = fir.embox %29(%23) : (!fir.ptr<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
// ALL-NOT: fir.store{{.*}}tbaa
    fir.store %30 to %28 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
    %31 = fir.declare %28 {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtest1Elocalp"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
    %32 = fir.alloca !fir.array<10xf32> {bindc_name = "localt", fir.target, uniq_name = "_QFtest1Elocalt"}
    %33 = fir.declare %32(%5) {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFtest1Elocalt"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.ref<!fir.array<10xf32>>
    %34 = fir.array_coor %10(%5) %c1 : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
// module data :: real :: glob(10)
// ALL: fir.store{{.*}}{tbaa = [#[[GLOBTAG]]]} : !fir.ref<f32>
    fir.store %cst to %34 : !fir.ref<f32>
    %35 = fir.array_coor %18(%5) %c1 : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
// module data :: real, target :: globt(10)
// ALL: fir.store{{.*}}{tbaa = [#[[GLOBTTAG]]]} : !fir.ref<f32>
    fir.store %cst to %35 : !fir.ref<f32>
// ALL-NOT: fir.load{{.*}}tbaa
    %36 = fir.load %12 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
    %37 = fir.box_addr %36 : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
    %38:3 = fir.box_dims %36, %c0 : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> (index, index, index)
    %39 = fir.shape_shift %38#0, %38#1 : (index, index) -> !fir.shapeshift<1>
    %40 = fir.array_coor %37(%39) %c1 : (!fir.heap<!fir.array<?xf32>>, !fir.shapeshift<1>, index) -> !fir.ref<f32>
// module data :: real, allocatable :: globa(:)
// ALL: fir.store{{.*}}{tbaa = [#[[GLOBATAG]]]} : !fir.ref<f32>
    fir.store %cst to %40 : !fir.ref<f32>
// ALL-NOT: fir.load{{.*}}tbaa
    %41 = fir.load %14 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
    %42 = fir.box_addr %41 : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
    %43:3 = fir.box_dims %41, %c0 : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> (index, index, index)
    %44 = fir.shape_shift %43#0, %43#1 : (index, index) -> !fir.shapeshift<1>
    %45 = fir.array_coor %42(%44) %c1 : (!fir.heap<!fir.array<?xf32>>, !fir.shapeshift<1>, index) -> !fir.ref<f32>
// module data :: real, allocatable, target :: globat(:)
// ALL: fir.store{{.*}}{tbaa = [#[[GLOBATTAG]]]} : !fir.ref<f32>
    fir.store %cst to %45 : !fir.ref<f32>
// ALL-NOT: fir.load{{.*}}tbaa
    %46 = fir.load %16 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
    %47:3 = fir.box_dims %46, %c0 : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, index) -> (index, index, index)
    %48 = fir.shift %47#0 : (index) -> !fir.shift<1>
    %49 = fir.array_coor %46(%48) %c1 : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, !fir.shift<1>, index) -> !fir.ref<f32>
// module data :: real, pointer :: globp(:)
// ALL: fir.store{{.*}}{tbaa = [#[[TARGETTAG]]]} : !fir.ref<f32>
    fir.store %cst to %49 : !fir.ref<f32>
    %50 = fir.array_coor %6(%5) %c1 : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
// real :: dummyf(10)
// ALL: fir.store{{.*}}{tbaa = [#[[DUMMYFTAG]]]} : !fir.ref<f32>
    fir.store %cst to %50 : !fir.ref<f32>
    %51 = fir.array_coor %7(%5) %c1 : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
// real, target :: dummyft(10)
// ALL: fir.store{{.*}}{tbaa = [#[[TARGETTAG]]]} : !fir.ref<f32>
    fir.store %cst to %51 : !fir.ref<f32>
    %52 = fir.array_coor %2 %c1 : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
// real :: dummyas(:)
// ALL: fir.store{{.*}}{tbaa = [#[[DUMMYASTAG]]]} : !fir.ref<f32>
    fir.store %cst to %52 : !fir.ref<f32>
    %53 = fir.array_coor %3 %c1 : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
// real, target :: dummyast(:)
// ALL: fir.store{{.*}}{tbaa = [#[[TARGETTAG]]]} : !fir.ref<f32>
    fir.store %cst to %53 : !fir.ref<f32>
// ALL-NOT: fir.load{{.*}}tbaa
    %54 = fir.load %1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
    %55 = fir.box_addr %54 : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
    %56:3 = fir.box_dims %54, %c0 : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> (index, index, index)
    %57 = fir.shape_shift %56#0, %56#1 : (index, index) -> !fir.shapeshift<1>
    %58 = fir.array_coor %55(%57) %c1 : (!fir.heap<!fir.array<?xf32>>, !fir.shapeshift<1>, index) -> !fir.ref<f32>
// real, allocatable :: dummya(:)
// ALL: fir.store{{.*}}{tbaa = [#[[DUMMYATAG]]]} : !fir.ref<f32>
    fir.store %cst to %58 : !fir.ref<f32>
// ALL-NOT: fir.load{{.*}}tbaa
    %59 = fir.load %4 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
    %60 = fir.box_addr %59 : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
    %61:3 = fir.box_dims %59, %c0 : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> (index, index, index)
    %62 = fir.shape_shift %61#0, %61#1 : (index, index) -> !fir.shapeshift<1>
    %63 = fir.array_coor %60(%62) %c1 : (!fir.heap<!fir.array<?xf32>>, !fir.shapeshift<1>, index) -> !fir.ref<f32>
// real, allocatable, target :: dummyat(:)
// ALL: fir.store{{.*}}{tbaa = [#[[TARGETTAG]]]} : !fir.ref<f32>
    fir.store %cst to %63 : !fir.ref<f32>
// ALL-NOT: fir.load{{.*}}tbaa
    %64 = fir.load %8 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
    %65:3 = fir.box_dims %64, %c0 : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, index) -> (index, index, index)
    %66 = fir.shift %65#0 : (index) -> !fir.shift<1>
    %67 = fir.array_coor %64(%66) %c1 : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, !fir.shift<1>, index) -> !fir.ref<f32>
// real, pointer :: dummyp(:)
// ALL: fir.store{{.*}}{tbaa = [#[[TARGETTAG]]]} : !fir.ref<f32>
    fir.store %cst to %67 : !fir.ref<f32>
    %68 = fir.array_coor %20(%5) %c1 : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
// real :: local(10)
// DEFAULT: fir.store{{.*}}tbaa
// LOCAL: fir.store{{.*}}{tbaa = [#[[LOCALTAG]]]} : !fir.ref<f32>
    fir.store %cst to %68 : !fir.ref<f32>
    %69 = fir.array_coor %33(%5) %c1 : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
// real, target :: localt(10)
// DEFAULT: fir.store{{.*}}tbaa
// LOCAL: fir.store{{.*}}{tbaa = [#[[LOCALTTAG]]]} : !fir.ref<f32>
    fir.store %cst to %69 : !fir.ref<f32>
// ALL-NOT: fir.load{{.*}}tbaa
    %70 = fir.load %25 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
    %71 = fir.box_addr %70 : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
    %72:3 = fir.box_dims %70, %c0 : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> (index, index, index)
    %73 = fir.shape_shift %72#0, %72#1 : (index, index) -> !fir.shapeshift<1>
    %74 = fir.array_coor %71(%73) %c1 : (!fir.heap<!fir.array<?xf32>>, !fir.shapeshift<1>, index) -> !fir.ref<f32>
// real, allocatable :: locala(:)
// DEFAULT: fir.store{{.*}}tbaa
// LOCAL: fir.store{{.*}}{tbaa = [#[[LOCALATAG]]]} : !fir.ref<f32>
    fir.store %cst to %74 : !fir.ref<f32>
// ALL-NOT: fir.load{{.*}}tbaa
    %75 = fir.load %27 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
    %76 = fir.box_addr %75 : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
    %77:3 = fir.box_dims %75, %c0 : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> (index, index, index)
    %78 = fir.shape_shift %77#0, %77#1 : (index, index) -> !fir.shapeshift<1>
    %79 = fir.array_coor %76(%78) %c1 : (!fir.heap<!fir.array<?xf32>>, !fir.shapeshift<1>, index) -> !fir.ref<f32>
// real, allocatable, target :: localat(:)
// DEFAULT: fir.store{{.*}}tbaa
// LOCAL: fir.store{{.*}}{tbaa = [#[[LOCALATTAG]]]} : !fir.ref<f32>
    fir.store %cst to %79 : !fir.ref<f32>
// ALL-NOT: fir.load{{.*}}tbaa
    %80 = fir.load %31 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
    %81:3 = fir.box_dims %80, %c0 : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, index) -> (index, index, index)
    %82 = fir.shift %81#0 : (index) -> !fir.shift<1>
    %83 = fir.array_coor %80(%82) %c1 : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, !fir.shift<1>, index) -> !fir.ref<f32>
// real, pointer :: localp(:)
// DEFAULT: fir.store{{.*}}tbaa
// LOCAL: fir.store{{.*}}{tbaa = [#[[TARGETTAG]]]} : !fir.ref<f32>
    fir.store %cst to %83 : !fir.ref<f32>
// ALL-NOT: fir.load{{.*}}tbaa
    %84 = fir.load %27 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
    %85 = fir.box_addr %84 : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
    %86 = fir.convert %85 : (!fir.heap<!fir.array<?xf32>>) -> i64
    %87 = arith.cmpi ne, %86, %c0_i64 : i64
    fir.if %87 {
// ALL-NOT: fir.load{{.*}}tbaa
      %92 = fir.load %27 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
      %93 = fir.box_addr %92 : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
      fir.freemem %93 : !fir.heap<!fir.array<?xf32>>
// ALL-NOT: fir.store{{.*}}tbaa
      fir.store %24 to %27 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
    }
// ALL-NOT: fir.load{{.*}}tbaa
    %88 = fir.load %25 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
    %89 = fir.box_addr %88 : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
    %90 = fir.convert %89 : (!fir.heap<!fir.array<?xf32>>) -> i64
    %91 = arith.cmpi ne, %90, %c0_i64 : i64
    fir.if %91 {
// ALL-NOT: fir.load{{.*}}tbaa
      %92 = fir.load %25 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
      %93 = fir.box_addr %92 : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
      fir.freemem %93 : !fir.heap<!fir.array<?xf32>>
// ALL-NOT: fir.store{{.*}}tbaa
      fir.store %24 to %25 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
    }
    return
  }
}
