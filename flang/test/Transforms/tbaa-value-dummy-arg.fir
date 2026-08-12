// Test that TBAA tags for a VALUE dummy argument local copy alloca are placed
// under "dummy arg data" (not "allocated data"), so that DSEPass cannot
// incorrectly treat the copy-stores as dead relative to dummy-arg reads.
//
// The VALUE dummy arg is lowered as a local fir.alloca with
// fortran_attrs = #fir.var_attrs<value>. Both the copy-store into the alloca
// and the subsequent loads from it must be tagged under "dummy arg data",
// not "allocated data".  Before the fix, the alloca was classified as
// SourceKind::Allocate and received an "allocated data" tag, which is a
// sibling of "dummy arg data" under "any data access", causing DSE to
// incorrectly eliminate the copy stores.

// RUN: fir-opt --fir-add-alias-tags %s | FileCheck %s

// CHECK: #[[ROOT:.+]] = #llvm.tbaa_root<id = "Flang function root fnt4">
// CHECK: #[[ANYACCESS:.+]] = #llvm.tbaa_type_desc<id = "any access", members = {<#[[ROOT]], 0>}>
// CHECK: #[[ANYDATA:.+]] = #llvm.tbaa_type_desc<id = "any data access", members = {<#[[ANYACCESS]], 0>}>
// CHECK: #[[DUMMYARG:.+]] = #llvm.tbaa_type_desc<id = "dummy arg data", members = {<#[[ANYDATA]], 0>}>
// CHECK-NOT: "allocated data"

// The copy-store and the load from the VALUE alloca must both use
// a "dummy arg data" tag, not an "allocated data" tag.
// CHECK-LABEL: func.func @fnt4(
// CHECK:         fir.store {{.*}} {tbaa = [#{{.+}}]}
// CHECK:         fir.load {{.*}} {tbaa = [#{{.+}}]}

module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, i64 = dense<[32, 64]> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "big">, llvm.data_layout = ""} {

// Represents:
//   subroutine fnt4(dt) bind(c)
//     type(dtd1), value :: dt   ! large struct passed on stack (byval)
//     ...
//   end subroutine
func.func @fnt4(%arg0: !fir.ref<!fir.type<_QMmxisob27bTdtd1{a:!fir.array<50xi8>,d0:!fir.type<_QMmxisob27bTdtd0{a:!fir.array<50xi8>}>}>> {fir.bindc_name = "dt"}) {
  %0 = fir.dummy_scope : !fir.dscope
  // VALUE dummy: local alloca copy, declared with fortran_attrs = value
  %1 = fir.alloca !fir.type<_QMmxisob27bTdtd1{a:!fir.array<50xi8>,d0:!fir.type<_QMmxisob27bTdtd0{a:!fir.array<50xi8>}>}> {bindc_name = "dt", uniq_name = "_QFfnt4Edt"}
  %2 = fir.declare %1 {fortran_attrs = #fir.var_attrs<value>, uniq_name = "_QFfnt4Edt"} : (!fir.ref<!fir.type<_QMmxisob27bTdtd1{a:!fir.array<50xi8>,d0:!fir.type<_QMmxisob27bTdtd0{a:!fir.array<50xi8>}>}>>) -> !fir.ref<!fir.type<_QMmxisob27bTdtd1{a:!fir.array<50xi8>,d0:!fir.type<_QMmxisob27bTdtd0{a:!fir.array<50xi8>}>}>>
  // Copy incoming byval arg into the local alloca — this store must be
  // tagged under "dummy arg data" so DSE cannot treat it as dead.
  %3 = fir.load %arg0 : !fir.ref<!fir.type<_QMmxisob27bTdtd1{a:!fir.array<50xi8>,d0:!fir.type<_QMmxisob27bTdtd0{a:!fir.array<50xi8>}>}>>
  fir.store %3 to %2 : !fir.ref<!fir.type<_QMmxisob27bTdtd1{a:!fir.array<50xi8>,d0:!fir.type<_QMmxisob27bTdtd0{a:!fir.array<50xi8>}>}>>
  // Load dt%a from the VALUE copy — must share the same TBAA subtree.
  %4 = fir.coordinate_of %2, a : (!fir.ref<!fir.type<_QMmxisob27bTdtd1{a:!fir.array<50xi8>,d0:!fir.type<_QMmxisob27bTdtd0{a:!fir.array<50xi8>}>}>>) -> !fir.ref<!fir.array<50xi8>>
  %c0 = arith.constant 0 : index
  %5 = fir.coordinate_of %4, %c0 : (!fir.ref<!fir.array<50xi8>>, index) -> !fir.ref<i8>
  %6 = fir.load %5 : !fir.ref<i8>
  return
}
}
