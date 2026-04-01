// TBAA tags for derived-type *descriptor* components (fir.box in records) after
// multi-level fir.coordinate_of chains; pointer pointees use the "target data"
// subtree. Nested allocatable components still attach TBAA to the heap payload
// access under the enclosing allocation/
//
// RUN: fir-opt --split-input-file --fir-add-alias-tags %s | FileCheck %s

// type(outer) with type(inner) and real, pointer :: p — local composite,
// two record levels before the pointer component box.

// CHECK: #[[R0:.+]] = #llvm.tbaa_root<id = "Flang function root _QPtest_ptr_component">
// CHECK: #[[ANY0:.+]] = #llvm.tbaa_type_desc<id = "any access", members = {<#[[R0]], 0>}>
// CHECK: #[[ANYDATA0:.+]] = #llvm.tbaa_type_desc<id = "any data access", members = {<#[[ANY0]], 0>}>
// CHECK: #[[TGT0:.+]] = #llvm.tbaa_type_desc<id = "target data", members = {<#[[ANYDATA0]], 0>}>
// CHECK: #[[PTRTAG0:.+]] = #llvm.tbaa_tag<base_type = #[[TGT0]], access_type = #[[TGT0]], offset = 0>
// CHECK-LABEL: func.func @_QPtest_ptr_component
// CHECK: fir.load %{{[0-9]+}} : !fir.ref<!fir.box<!fir.ptr<f32>>>
// CHECK: fir.load %{{[0-9]+}} {tbaa = [#[[PTRTAG0]]]} : !fir.ptr<f32>

module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, i64 = dense<[32, 64]> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little">, llvm.data_layout = ""} {
  func.func @_QPtest_ptr_component() {
    %0 = fir.alloca !fir.type<_QMtbaaTouterP{in:!fir.type<_QMtbaaTinnerP{p:!fir.box<!fir.ptr<f32>>}>}> {uniq_name = "_QMtbaaFtest_ptr_componentEz", bindc_name = "z"}
    %z = fir.declare %0 {uniq_name = "_QMtbaaFtest_ptr_componentEz"} : (!fir.ref<!fir.type<_QMtbaaTouterP{in:!fir.type<_QMtbaaTinnerP{p:!fir.box<!fir.ptr<f32>>}>}>>) -> !fir.ref<!fir.type<_QMtbaaTouterP{in:!fir.type<_QMtbaaTinnerP{p:!fir.box<!fir.ptr<f32>>}>}>>
    %in = fir.coordinate_of %z, in : (!fir.ref<!fir.type<_QMtbaaTouterP{in:!fir.type<_QMtbaaTinnerP{p:!fir.box<!fir.ptr<f32>>}>}>>) -> !fir.ref<!fir.type<_QMtbaaTinnerP{p:!fir.box<!fir.ptr<f32>>}>>
    %pref = fir.coordinate_of %in, p : (!fir.ref<!fir.type<_QMtbaaTinnerP{p:!fir.box<!fir.ptr<f32>>}>>) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
    %bl = fir.load %pref : !fir.ref<!fir.box<!fir.ptr<f32>>>
    %pt = fir.box_addr %bl : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
    %v = fir.load %pt : !fir.ptr<f32>
    return
  }
}

// -----

// Three record nesting levels (a, b) before pointer component p.

// CHECK: #[[R1:.+]] = #llvm.tbaa_root<id = "Flang function root _QPtest_deep_ptr_component">
// CHECK: #[[ANY1:.+]] = #llvm.tbaa_type_desc<id = "any access", members = {<#[[R1]], 0>}>
// CHECK: #[[ANYDATA1:.+]] = #llvm.tbaa_type_desc<id = "any data access", members = {<#[[ANY1]], 0>}>
// CHECK: #[[TGT1:.+]] = #llvm.tbaa_type_desc<id = "target data", members = {<#[[ANYDATA1]], 0>}>
// CHECK: #[[PTRTAG1:.+]] = #llvm.tbaa_tag<base_type = #[[TGT1]], access_type = #[[TGT1]], offset = 0>
// CHECK-LABEL: func.func @_QPtest_deep_ptr_component
// CHECK: fir.load %{{[0-9]+}} : !fir.ref<!fir.box<!fir.ptr<f32>>>
// CHECK: fir.load %{{[0-9]+}} {tbaa = [#[[PTRTAG1]]]} : !fir.ptr<f32>

module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, i64 = dense<[32, 64]> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little">, llvm.data_layout = ""} {
  func.func @_QPtest_deep_ptr_component() {
    %0 = fir.alloca !fir.type<_QMtbaaTl0{a:!fir.type<_QMtbaaTl1{b:!fir.type<_QMtbaaTl2{p:!fir.box<!fir.ptr<f32>>}>}>}> {uniq_name = "_QMtbaaFtest_deep_ptr_componentEz"}
    %z = fir.declare %0 {uniq_name = "_QMtbaaFtest_deep_ptr_componentEz"} : (!fir.ref<!fir.type<_QMtbaaTl0{a:!fir.type<_QMtbaaTl1{b:!fir.type<_QMtbaaTl2{p:!fir.box<!fir.ptr<f32>>}>}>}>>) -> !fir.ref<!fir.type<_QMtbaaTl0{a:!fir.type<_QMtbaaTl1{b:!fir.type<_QMtbaaTl2{p:!fir.box<!fir.ptr<f32>>}>}>}>>
    %l1 = fir.coordinate_of %z, a : (!fir.ref<!fir.type<_QMtbaaTl0{a:!fir.type<_QMtbaaTl1{b:!fir.type<_QMtbaaTl2{p:!fir.box<!fir.ptr<f32>>}>}>}>>) -> !fir.ref<!fir.type<_QMtbaaTl1{b:!fir.type<_QMtbaaTl2{p:!fir.box<!fir.ptr<f32>>}>}>>
    %l2 = fir.coordinate_of %l1, b : (!fir.ref<!fir.type<_QMtbaaTl1{b:!fir.type<_QMtbaaTl2{p:!fir.box<!fir.ptr<f32>>}>}>>) -> !fir.ref<!fir.type<_QMtbaaTl2{p:!fir.box<!fir.ptr<f32>>}>>
    %pref = fir.coordinate_of %l2, p : (!fir.ref<!fir.type<_QMtbaaTl2{p:!fir.box<!fir.ptr<f32>>}>>) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
    %bl = fir.load %pref : !fir.ref<!fir.box<!fir.ptr<f32>>>
    %pt = fir.box_addr %bl : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
    %v = fir.load %pt : !fir.ptr<f32>
    return
  }
}

// -----

// Global composite: two coordinate_of steps (a, then p) to the pointer box.

// CHECK: #[[R2:.+]] = #llvm.tbaa_root<id = "Flang function root _QPtest_glob_nested_ptr">
// CHECK: #[[ANY2:.+]] = #llvm.tbaa_type_desc<id = "any access", members = {<#[[R2]], 0>}>
// CHECK: #[[ANYDATA2:.+]] = #llvm.tbaa_type_desc<id = "any data access", members = {<#[[ANY2]], 0>}>
// CHECK: #[[TGT2:.+]] = #llvm.tbaa_type_desc<id = "target data", members = {<#[[ANYDATA2]], 0>}>
// CHECK: #[[PTRTAG2:.+]] = #llvm.tbaa_tag<base_type = #[[TGT2]], access_type = #[[TGT2]], offset = 0>
// CHECK-LABEL: func.func @_QPtest_glob_nested_ptr
// CHECK: fir.load %{{[0-9]+}} : !fir.ref<!fir.box<!fir.ptr<f32>>>
// CHECK: fir.load %{{[0-9]+}} {tbaa = [#[[PTRTAG2]]]} : !fir.ptr<f32>

module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, i64 = dense<[32, 64]> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little">, llvm.data_layout = ""} {
  fir.global @_QMtbaaEgdeep : !fir.type<_QMtbaaTg0{a:!fir.type<_QMtbaaTg1{p:!fir.box<!fir.ptr<f32>>}>}> {
    %0 = fir.undefined !fir.type<_QMtbaaTg0{a:!fir.type<_QMtbaaTg1{p:!fir.box<!fir.ptr<f32>>}>}>
    fir.has_value %0 : !fir.type<_QMtbaaTg0{a:!fir.type<_QMtbaaTg1{p:!fir.box<!fir.ptr<f32>>}>}>
  }
  func.func @_QPtest_glob_nested_ptr() {
    %0 = fir.address_of(@_QMtbaaEgdeep) : !fir.ref<!fir.type<_QMtbaaTg0{a:!fir.type<_QMtbaaTg1{p:!fir.box<!fir.ptr<f32>>}>}>>
    %g = fir.declare %0 {uniq_name = "_QMtbaaEgdeep"} : (!fir.ref<!fir.type<_QMtbaaTg0{a:!fir.type<_QMtbaaTg1{p:!fir.box<!fir.ptr<f32>>}>}>>) -> !fir.ref<!fir.type<_QMtbaaTg0{a:!fir.type<_QMtbaaTg1{p:!fir.box<!fir.ptr<f32>>}>}>>
    %in = fir.coordinate_of %g, a : (!fir.ref<!fir.type<_QMtbaaTg0{a:!fir.type<_QMtbaaTg1{p:!fir.box<!fir.ptr<f32>>}>}>>) -> !fir.ref<!fir.type<_QMtbaaTg1{p:!fir.box<!fir.ptr<f32>>}>>
    %pref = fir.coordinate_of %in, p : (!fir.ref<!fir.type<_QMtbaaTg1{p:!fir.box<!fir.ptr<f32>>}>>) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
    %bl = fir.load %pref : !fir.ref<!fir.box<!fir.ptr<f32>>>
    %pt = fir.box_addr %bl : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
    %v = fir.load %pt : !fir.ptr<f32>
    return
  }
}

// -----

// Two separate local objects, each with nested p%q pointer boxes (two box
// loads without TBAA; pointer load/store use target data).

// CHECK: #[[R3:.+]] = #llvm.tbaa_root<id = "Flang function root _QPtest_two_nested_ptr_paths">
// CHECK: #[[ANY3:.+]] = #llvm.tbaa_type_desc<id = "any access", members = {<#[[R3]], 0>}>
// CHECK: #[[ANYDATA3:.+]] = #llvm.tbaa_type_desc<id = "any data access", members = {<#[[ANY3]], 0>}>
// CHECK: #[[TGT3:.+]] = #llvm.tbaa_type_desc<id = "target data", members = {<#[[ANYDATA3]], 0>}>
// CHECK: #[[PTRTAG3:.+]] = #llvm.tbaa_tag<base_type = #[[TGT3]], access_type = #[[TGT3]], offset = 0>
// CHECK-LABEL: func.func @_QPtest_two_nested_ptr_paths
// CHECK: fir.load %{{[0-9]+}} : !fir.ref<!fir.box<!fir.ptr<f32>>>
// CHECK: fir.load %{{[0-9]+}} : !fir.ref<!fir.box<!fir.ptr<f32>>>
// CHECK: fir.load %{{[0-9]+}} {tbaa = [#[[PTRTAG3]]]} : !fir.ptr<f32>
// CHECK: fir.store %{{[0-9]+}} to %{{[0-9]+}} {tbaa = [#[[PTRTAG3]]]} : !fir.ptr<f32>

module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, i64 = dense<[32, 64]> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little">, llvm.data_layout = ""} {
  func.func @_QPtest_two_nested_ptr_paths() {
    %0 = fir.alloca !fir.type<_QMtbaaTx0{p:!fir.type<_QMtbaaTx1{q:!fir.box<!fir.ptr<f32>>}>}> {uniq_name = "_QMtbaaFtest_two_nested_ptr_pathsEx"}
    %1 = fir.alloca !fir.type<_QMtbaaTy0{p:!fir.type<_QMtbaaTy1{q:!fir.box<!fir.ptr<f32>>}>}> {uniq_name = "_QMtbaaFtest_two_nested_ptr_pathsEy"}
    %x = fir.declare %0 {uniq_name = "_QMtbaaFtest_two_nested_ptr_pathsEx"} : (!fir.ref<!fir.type<_QMtbaaTx0{p:!fir.type<_QMtbaaTx1{q:!fir.box<!fir.ptr<f32>>}>}>>) -> !fir.ref<!fir.type<_QMtbaaTx0{p:!fir.type<_QMtbaaTx1{q:!fir.box<!fir.ptr<f32>>}>}>>
    %y = fir.declare %1 {uniq_name = "_QMtbaaFtest_two_nested_ptr_pathsEy"} : (!fir.ref<!fir.type<_QMtbaaTy0{p:!fir.type<_QMtbaaTy1{q:!fir.box<!fir.ptr<f32>>}>}>>) -> !fir.ref<!fir.type<_QMtbaaTy0{p:!fir.type<_QMtbaaTy1{q:!fir.box<!fir.ptr<f32>>}>}>>
    %x1 = fir.coordinate_of %x, p : (!fir.ref<!fir.type<_QMtbaaTx0{p:!fir.type<_QMtbaaTx1{q:!fir.box<!fir.ptr<f32>>}>}>>) -> !fir.ref<!fir.type<_QMtbaaTx1{q:!fir.box<!fir.ptr<f32>>}>>
    %xq = fir.coordinate_of %x1, q : (!fir.ref<!fir.type<_QMtbaaTx1{q:!fir.box<!fir.ptr<f32>>}>>) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
    %y1 = fir.coordinate_of %y, p : (!fir.ref<!fir.type<_QMtbaaTy0{p:!fir.type<_QMtbaaTy1{q:!fir.box<!fir.ptr<f32>>}>}>>) -> !fir.ref<!fir.type<_QMtbaaTy1{q:!fir.box<!fir.ptr<f32>>}>>
    %yq = fir.coordinate_of %y1, q : (!fir.ref<!fir.type<_QMtbaaTy1{q:!fir.box<!fir.ptr<f32>>}>>) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
    %bx = fir.load %xq : !fir.ref<!fir.box<!fir.ptr<f32>>>
    %by = fir.load %yq : !fir.ref<!fir.box<!fir.ptr<f32>>>
    %px = fir.box_addr %bx : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
    %py = fir.box_addr %by : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
    %vx = fir.load %px : !fir.ptr<f32>
    fir.store %vx to %py : !fir.ptr<f32>
    return
  }
}

// -----

// real, allocatable :: f after three record nesting levels (v%w%mid%f) — box
// load is untagged; heap payload is tagged under the enclosing allocation.

// CHECK: #[[R4:.+]] = #llvm.tbaa_root<id = "Flang function root _QPtest_nested_alloc_descriptor">
// CHECK: #[[ANY4:.+]] = #llvm.tbaa_type_desc<id = "any access", members = {<#[[R4]], 0>}>
// CHECK: #[[ANYDATA4:.+]] = #llvm.tbaa_type_desc<id = "any data access", members = {<#[[ANY4]], 0>}>
// CHECK: #[[ALLOC4:.+]] = #llvm.tbaa_type_desc<id = "allocated data", members = {<#[[ANYDATA4]], 0>}>
// CHECK: #[[UNODE:.+]] = #llvm.tbaa_type_desc<id = "allocated data/_QMtbaaFtest_nested_alloc_descriptorEv", members = {<#[[ALLOC4]], 0>}>
// CHECK: #[[HEAPTAG:.+]] = #llvm.tbaa_tag<base_type = #[[UNODE]], access_type = #[[UNODE]], offset = 0>
// CHECK-LABEL: func.func @_QPtest_nested_alloc_descriptor
// CHECK: fir.load %{{[0-9]+}} : !fir.ref<!fir.box<!fir.heap<f32>>>
// CHECK: fir.load %{{[0-9]+}} {tbaa = [#[[HEAPTAG]]]} : !fir.ref<f32>

module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, i64 = dense<[32, 64]> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little">, llvm.data_layout = ""} {
  func.func @_QPtest_nested_alloc_descriptor() {
    %0 = fir.alloca !fir.type<_QMtbaaTd0{w:!fir.type<_QMtbaaTd1{mid:!fir.type<_QMtbaaTd2{f:!fir.box<!fir.heap<f32>>}>}>}> {uniq_name = "_QMtbaaFtest_nested_alloc_descriptorEv"}
    %v = fir.declare %0 {uniq_name = "_QMtbaaFtest_nested_alloc_descriptorEv"} : (!fir.ref<!fir.type<_QMtbaaTd0{w:!fir.type<_QMtbaaTd1{mid:!fir.type<_QMtbaaTd2{f:!fir.box<!fir.heap<f32>>}>}>}>>) -> !fir.ref<!fir.type<_QMtbaaTd0{w:!fir.type<_QMtbaaTd1{mid:!fir.type<_QMtbaaTd2{f:!fir.box<!fir.heap<f32>>}>}>}>>
    %w = fir.coordinate_of %v, w : (!fir.ref<!fir.type<_QMtbaaTd0{w:!fir.type<_QMtbaaTd1{mid:!fir.type<_QMtbaaTd2{f:!fir.box<!fir.heap<f32>>}>}>}>>) -> !fir.ref<!fir.type<_QMtbaaTd1{mid:!fir.type<_QMtbaaTd2{f:!fir.box<!fir.heap<f32>>}>}>>
    %mid = fir.coordinate_of %w, mid : (!fir.ref<!fir.type<_QMtbaaTd1{mid:!fir.type<_QMtbaaTd2{f:!fir.box<!fir.heap<f32>>}>}>>) -> !fir.ref<!fir.type<_QMtbaaTd2{f:!fir.box<!fir.heap<f32>>}>>
    %fref = fir.coordinate_of %mid, f : (!fir.ref<!fir.type<_QMtbaaTd2{f:!fir.box<!fir.heap<f32>>}>>) -> !fir.ref<!fir.box<!fir.heap<f32>>>
    %bl = fir.load %fref : !fir.ref<!fir.box<!fir.heap<f32>>>
    %hp = fir.box_addr %bl : (!fir.box<!fir.heap<f32>>) -> !fir.heap<f32>
    %r = fir.convert %hp : (!fir.heap<f32>) -> !fir.ref<f32>
    %x = fir.load %r : !fir.ref<f32>
    return
  }
}
