// test that tbaa attributes can be added to fir.load and fir.store
// and that these attributes are propagated to LLVMIR

// RUN: tco %s | FileCheck %s

// subroutine simple(a)
//   integer, intent(inout) :: a(:)
//   a(1) = a(2)
// end subroutine
#tbaa_root = #llvm.tbaa_root<id = "Flang function root _QPsimple">
#tbaa_type_desc = #llvm.tbaa_type_desc<id = "any access", members = {<#tbaa_root, 0>}>
#tbaa_type_desc1 = #llvm.tbaa_type_desc<id = "any data access", members = {<#tbaa_type_desc, 0>}>
#tbaa_type_desc2 = #llvm.tbaa_type_desc<id = "dummy arg data", members = {<#tbaa_type_desc1, 0>}>
#tbaa_type_desc3 = #llvm.tbaa_type_desc<id = "dummy arg data/_QFfuncEa", members = {<#tbaa_type_desc2, 0>}>
#tbaa_tag = #llvm.tbaa_tag<base_type = #tbaa_type_desc3, access_type = #tbaa_type_desc3, offset = 0>
module attributes {fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @_QPsimple(%arg0: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "a"}) {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %0 = fir.declare %arg0 {fortran_attrs = #fir.var_attrs<intent_inout>, uniq_name = "_QFfuncEa"} : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>>
    %1 = fir.rebox %0 : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>>
    %2 = fir.array_coor %1 %c2 : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
    %3 = fir.load %2 {tbaa = [#tbaa_tag]} : !fir.ref<i32>
    %4 = fir.array_coor %1 %c1 : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
    fir.store %3 to %4 {tbaa = [#tbaa_tag]} : !fir.ref<i32>
    return
  }
}

// CHECK-LABEL: define void @_QPsimple(
// CHECK-SAME:      ptr %[[ARG0:.*]]) {
// [...]
// load  a(2):
// CHECK:  %[[VAL20:.*]] = getelementptr i8, ptr %{{.*}}, i64 %{{.*}}
// CHECK:  %[[A2:.*]] = load i32, ptr %[[VAL20]], align 4, !tbaa ![[A_ACCESS_TAG:.*]]
// [...]
// store a(2) to a(1):
// CHECK:  %[[A1:.*]] = getelementptr i8, ptr %{{.*}}, i64 %{{.*}}
// CHECK:  store i32 %[[A2]], ptr %[[A1]], align 4, !tbaa ![[A_ACCESS_TAG]]
// CHECK:  ret void
// CHECK: }
// CHECK: ![[ANY_ACCESS_TYPE:.*]] = !{!"any access", ![[ROOT:.*]], i64 0}
// CHECK: ![[ROOT]] = !{!"Flang function root _QPsimple"}
// CHECK: ![[A_ACCESS_TAG]] = !{![[A_ACCESS_TYPE:.*]], ![[A_ACCESS_TYPE]], i64 0}
// CHECK: ![[A_ACCESS_TYPE]] = !{!"dummy arg data/_QFfuncEa", ![[DUMMY_ARG_TYPE:.*]], i64 0}
// CHECK: ![[DUMMY_ARG_TYPE]] = !{!"dummy arg data", ![[DATA_ACCESS_TYPE:.*]], i64 0}
// CHECK: ![[DATA_ACCESS_TYPE]] = !{!"any data access", ![[ANY_ACCESS_TYPE]], i64 0}
