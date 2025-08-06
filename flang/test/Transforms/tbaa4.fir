// Test TBAA tags for common and equivalence.
// RUN: fir-opt --fir-add-alias-tags --split-input-file %s | FileCheck --check-prefixes=ALL,DEFAULT %s
// RUN: fir-opt --fir-add-alias-tags --local-alloc-tbaa --split-input-file %s | FileCheck --check-prefixes=ALL,LOCAL %s

// ALL: #[[ROOT:.+]] = #llvm.tbaa_root<id = "Flang function root _QPtest_common">
// ALL: #[[ANY:.+]] = #llvm.tbaa_type_desc<id = "any access", members = {<#[[ROOT]], 0>}>
// ALL: #[[ANYDATA:.+]] = #llvm.tbaa_type_desc<id = "any data access", members = {<#[[ANY]], 0>}>
// ALL: #[[TARGETDATA:.+]] = #llvm.tbaa_type_desc<id = "target data", members = {<#[[ANYDATA]], 0>}>
// ALL: #[[GLOBALDATA:.+]] = #llvm.tbaa_type_desc<id = "global data", members = {<#[[TARGETDATA]], 0>}>
// ALL: #[[BLK:.+]] = #llvm.tbaa_type_desc<id = "global data/blk_", members = {<#[[GLOBALDATA]], 0>}>
// ALL: #[[TAG:.+]] = #llvm.tbaa_tag<base_type = #[[BLK]], access_type = #[[BLK]], offset = 0>

module {
// ALL-LABEL:   fir.global common @blk_(dense<0> : vector<48xi8>) {alignment = 4 : i64} : !fir.array<48xi8>
  fir.global common @blk_(dense<0> : vector<48xi8>) {alignment = 4 : i64} : !fir.array<48xi8>

// ALL-LABEL:   func.func @_QPtest_common() {
// ALL:           fir.store{{.*}}{tbaa = [#[[TAG]]]} : !fir.ref<f32>
// ALL:           fir.store{{.*}}{tbaa = [#[[TAG]]]} : !fir.ref<i32>
// ALL:           fir.store{{.*}}{tbaa = [#[[TAG]]]} : !fir.ref<f32>
  func.func @_QPtest_common() {
    %c1 = arith.constant 1 : index
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 1.000000e+00 : f32
    %c10 = arith.constant 10 : index
    %c8 = arith.constant 8 : index
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %0 = fir.dummy_scope : !fir.dscope
    %1 = fir.address_of(@blk_) : !fir.ref<!fir.array<48xi8>>
    %2 = fir.convert %1 : (!fir.ref<!fir.array<48xi8>>) -> !fir.ref<!fir.array<?xi8>>
    %3 = fir.coordinate_of %2, %c0 : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
    %4 = fir.convert %3 : (!fir.ref<i8>) -> !fir.ref<f32>
    %5 = fir.declare %4 {uniq_name = "_QFtest_commonEa"} : (!fir.ref<f32>) -> !fir.ref<f32>
    %6 = fir.coordinate_of %2, %c4 : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
    %7 = fir.convert %6 : (!fir.ref<i8>) -> !fir.ref<f32>
    %8 = fir.declare %7 {uniq_name = "_QFtest_commonEb"} : (!fir.ref<f32>) -> !fir.ref<f32>
    %9 = fir.coordinate_of %2, %c8 : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
    %10 = fir.convert %9 : (!fir.ref<i8>) -> !fir.ref<!fir.array<10xi32>>
    %11 = fir.shape %c10 : (index) -> !fir.shape<1>
    %12 = fir.declare %10(%11) {uniq_name = "_QFtest_commonEc"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.ref<!fir.array<10xi32>>
    fir.store %cst to %5 : !fir.ref<f32>
    %13 = fir.array_coor %12(%11) %c1 : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
    fir.store %c1_i32 to %13 : !fir.ref<i32>
    fir.store %cst to %8 : !fir.ref<f32>
    return
  }
}

// -----

// LOCAL: #[[ROOT:.+]] = #llvm.tbaa_root<id = "Flang function root _QPtest_local_equiv">
// LOCAL: #[[ANY:.+]] = #llvm.tbaa_type_desc<id = "any access", members = {<#[[ROOT]], 0>}>
// LOCAL: #[[ANYDATA:.+]] = #llvm.tbaa_type_desc<id = "any data access", members = {<#[[ANY]], 0>}>
// LOCAL: #[[TARGETDATA:.+]] = #llvm.tbaa_type_desc<id = "target data", members = {<#[[ANYDATA]], 0>}>
// LOCAL: #[[ALLOCATEDDATA:.+]] = #llvm.tbaa_type_desc<id = "allocated data", members = {<#[[TARGETDATA]], 0>}>
// LOCAL: #[[EQUIV:.+]] = #llvm.tbaa_type_desc<id = "allocated data/_QFtest_local_equivEa", members = {<#[[ALLOCATEDDATA]], 0>}>
// LOCAL: #[[$ATTR_13:.+]] = #llvm.tbaa_tag<base_type = #[[EQUIV]], access_type = #[[EQUIV]], offset = 0>

// ALL-LABEL:   func.func @_QPtest_local_equiv() {
// LOCAL:         fir.store{{.*}}{tbaa = [#[[TAG]]]} : !fir.ptr<f32>
// LOCAL:         fir.store{{.*}}{tbaa = [#[[TAG]]]} : !fir.ref<i32>
// LOCAL:         fir.store{{.*}}{tbaa = [#[[TAG]]]} : !fir.ptr<f32>
// DEFAULT-NOT:   fir.store{{.}}tbaa
func.func @_QPtest_local_equiv() {
  %c1 = arith.constant 1 : index
  %c1_i32 = arith.constant 1 : i32
  %cst = arith.constant 1.000000e+00 : f32
  %c10 = arith.constant 10 : index
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.alloca !fir.array<40xi8> {uniq_name = "_QFtest_local_equivEa"}
  %2 = fir.coordinate_of %1, %c8 : (!fir.ref<!fir.array<40xi8>>, index) -> !fir.ref<i8>
  %3 = fir.convert %2 : (!fir.ref<i8>) -> !fir.ptr<f32>
  %4 = fir.declare %3 {uniq_name = "_QFtest_local_equivEa"} : (!fir.ptr<f32>) -> !fir.ptr<f32>
  %5 = fir.declare %3 {uniq_name = "_QFtest_local_equivEb"} : (!fir.ptr<f32>) -> !fir.ptr<f32>
  %6 = fir.coordinate_of %1, %c0 : (!fir.ref<!fir.array<40xi8>>, index) -> !fir.ref<i8>
  %7 = fir.convert %6 : (!fir.ref<i8>) -> !fir.ptr<!fir.array<10xi32>>
  %8 = fir.shape %c10 : (index) -> !fir.shape<1>
  %9 = fir.declare %7(%8) {uniq_name = "_QFtest_local_equivEc"} : (!fir.ptr<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.ptr<!fir.array<10xi32>>
  fir.store %cst to %4 : !fir.ptr<f32>
  %10 = fir.array_coor %9(%8) %c1 : (!fir.ptr<!fir.array<10xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
  fir.store %c1_i32 to %10 : !fir.ref<i32>
  fir.store %cst to %5 : !fir.ptr<f32>
  return
}

// -----

// ALL: #[[ROOT:.+]] = #llvm.tbaa_root<id = "Flang function root _QPtest_save_equiv">
// ALL: #[[ANY:.+]] = #llvm.tbaa_type_desc<id = "any access", members = {<#[[ROOT]], 0>}>
// ALL: #[[ANYDATA:.+]] = #llvm.tbaa_type_desc<id = "any data access", members = {<#[[ANY]], 0>}>
// ALL: #[[TARGETDATA:.+]] = #llvm.tbaa_type_desc<id = "target data", members = {<#[[ANYDATA]], 0>}>
// ALL: #[[GLOBALDATA:.+]] = #llvm.tbaa_type_desc<id = "global data", members = {<#[[TARGETDATA]], 0>}>
// ALL: #[[EQUIV:.+]] = #llvm.tbaa_type_desc<id = "global data/_QFtest_save_equivEa", members = {<#[[GLOBALDATA]], 0>}>
// ALL: #[[TAG:.+]] = #llvm.tbaa_tag<base_type = #[[EQUIV]], access_type = #[[EQUIV]], offset = 0>
module {
// ALL-LABEL:   fir.global internal @_QFtest_save_equivEa : !fir.array<40xi8> {
  fir.global internal @_QFtest_save_equivEa : !fir.array<40xi8> {
    %0 = fir.zero_bits !fir.array<40xi8>
    fir.has_value %0 : !fir.array<40xi8>
  }

// ALL-LABEL:   func.func @_QPtest_save_equiv() {
// ALL:           fir.store{{.*}}{tbaa = [#[[TAG]]]} : !fir.ptr<f32>
// ALL:           fir.store{{.*}}{tbaa = [#[[TAG]]]} : !fir.ref<i32>
// ALL:           fir.store{{.*}}{tbaa = [#[[TAG]]]} : !fir.ptr<f32>
  func.func @_QPtest_save_equiv() {
    %c1 = arith.constant 1 : index
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 1.000000e+00 : f32
    %c10 = arith.constant 10 : index
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %0 = fir.dummy_scope : !fir.dscope
    %1 = fir.address_of(@_QFtest_save_equivEa) : !fir.ref<!fir.array<40xi8>>
    %2 = fir.coordinate_of %1, %c8 : (!fir.ref<!fir.array<40xi8>>, index) -> !fir.ref<i8>
    %3 = fir.convert %2 : (!fir.ref<i8>) -> !fir.ptr<f32>
    %4 = fir.declare %3 {uniq_name = "_QFtest_save_equivEa"} : (!fir.ptr<f32>) -> !fir.ptr<f32>
    %5 = fir.declare %3 {uniq_name = "_QFtest_save_equivEb"} : (!fir.ptr<f32>) -> !fir.ptr<f32>
    %6 = fir.coordinate_of %1, %c0 : (!fir.ref<!fir.array<40xi8>>, index) -> !fir.ref<i8>
    %7 = fir.convert %6 : (!fir.ref<i8>) -> !fir.ptr<!fir.array<10xi32>>
    %8 = fir.shape %c10 : (index) -> !fir.shape<1>
    %9 = fir.declare %7(%8) {uniq_name = "_QFtest_save_equivEc"} : (!fir.ptr<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.ptr<!fir.array<10xi32>>
    fir.store %cst to %4 : !fir.ptr<f32>
    %10 = fir.array_coor %9(%8) %c1 : (!fir.ptr<!fir.array<10xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
    fir.store %c1_i32 to %10 : !fir.ref<i32>
    fir.store %cst to %5 : !fir.ptr<f32>
    return
  }
}

// -----

// ALL: #[[ROOT:.+]] = #llvm.tbaa_root<id = "Flang function root _QPtest_global_equiv">
// ALL: #[[ANY:.+]] = #llvm.tbaa_type_desc<id = "any access", members = {<#[[ROOT]], 0>}>
// ALL: #[[ANYDATA:.+]] = #llvm.tbaa_type_desc<id = "any data access", members = {<#[[ANY]], 0>}>
// ALL: #[[TARGETDATA:.+]] = #llvm.tbaa_type_desc<id = "target data", members = {<#[[ANYDATA]], 0>}>
// ALL: #[[GLOBALDATA:.+]] = #llvm.tbaa_type_desc<id = "global data", members = {<#[[TARGETDATA]], 0>}>
// ALL: #[[EQUIV:.+]] = #llvm.tbaa_type_desc<id = "global data/_QMdataEa", members = {<#[[GLOBALDATA]], 0>}>
// ALL: #[[TAG:.+]] = #llvm.tbaa_tag<base_type = #[[EQUIV]], access_type = #[[EQUIV]], offset = 0>

module {
// ALL-LABEL:   fir.global @_QMdataEa : !fir.array<40xi8> {
  fir.global @_QMdataEa : !fir.array<40xi8> {
    %0 = fir.zero_bits !fir.array<40xi8>
    fir.has_value %0 : !fir.array<40xi8>
  }
// ALL-LABEL:   func.func @_QPtest_global_equiv() {
// ALL:           fir.store{{.*}}{tbaa = [#[[TAG]]]} : !fir.ptr<f32>
// ALL:           fir.store{{.*}}{tbaa = [#[[TAG]]]} : !fir.ref<i32>
// ALL:           fir.store{{.*}}{tbaa = [#[[TAG]]]} : !fir.ptr<f32>
  func.func @_QPtest_global_equiv() {
    %c1 = arith.constant 1 : index
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 1.000000e+00 : f32
    %c10 = arith.constant 10 : index
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %0 = fir.dummy_scope : !fir.dscope
    %1 = fir.address_of(@_QMdataEa) : !fir.ref<!fir.array<40xi8>>
    %2 = fir.coordinate_of %1, %c8 : (!fir.ref<!fir.array<40xi8>>, index) -> !fir.ref<i8>
    %3 = fir.convert %2 : (!fir.ref<i8>) -> !fir.ptr<f32>
    %4 = fir.declare %3 {uniq_name = "_QMdataEa"} : (!fir.ptr<f32>) -> !fir.ptr<f32>
    %5 = fir.declare %3 {uniq_name = "_QMdataEb"} : (!fir.ptr<f32>) -> !fir.ptr<f32>
    %6 = fir.coordinate_of %1, %c0 : (!fir.ref<!fir.array<40xi8>>, index) -> !fir.ref<i8>
    %7 = fir.convert %6 : (!fir.ref<i8>) -> !fir.ptr<!fir.array<10xi32>>
    %8 = fir.shape %c10 : (index) -> !fir.shape<1>
    %9 = fir.declare %7(%8) {uniq_name = "_QMdataEc"} : (!fir.ptr<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.ptr<!fir.array<10xi32>>
    fir.store %cst to %4 : !fir.ptr<f32>
    %10 = fir.array_coor %9(%8) %c1 : (!fir.ptr<!fir.array<10xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
    fir.store %c1_i32 to %10 : !fir.ref<i32>
    fir.store %cst to %5 : !fir.ptr<f32>
    return
  }
}
