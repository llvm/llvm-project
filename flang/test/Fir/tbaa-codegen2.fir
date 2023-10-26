// test that tbaa attributes can be added to fir.load and fir.store
// and that these attributes are propagated to LLVMIR and that these
// interoperrate with tbaa tags added during codegen

// RUN: tco %s | FileCheck %s

// subroutine func(a)
//   integer, intent(inout) :: a(:)
//   a = a+1
//   a(1) = a(2)
#tbaa_root = #llvm.tbaa_root<id = "Flang function root _QPfunc">
#tbaa_type_desc = #llvm.tbaa_type_desc<id = "any access", members = {<#tbaa_root, 0>}>
#tbaa_type_desc1 = #llvm.tbaa_type_desc<id = "any data access", members = {<#tbaa_type_desc, 0>}>
#tbaa_type_desc2 = #llvm.tbaa_type_desc<id = "dummy arg data", members = {<#tbaa_type_desc1, 0>}>
#tbaa_type_desc3 = #llvm.tbaa_type_desc<id = "dummy arg data/_QFfuncEa", members = {<#tbaa_type_desc2, 0>}>
#tbaa_tag = #llvm.tbaa_tag<base_type = #tbaa_type_desc3, access_type = #tbaa_type_desc3, offset = 0>
module attributes {fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @_QPfunc(%arg0: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "a"}) {
    %c3_i32 = arith.constant 3 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %0 = fir.alloca !fir.box<!fir.array<?xi32>>
    %1 = fir.declare %arg0 {fortran_attrs = #fir.var_attrs<intent_inout>, uniq_name = "_QFfuncEa"} : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>>
    %2 = fir.rebox %1 : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>>
    %3:3 = fir.box_dims %2, %c0 : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
    %4 = fir.shape %3#1 : (index) -> !fir.shape<1>
    %5 = fir.allocmem !fir.array<?xi32>, %3#1 {bindc_name = ".tmp.array", uniq_name = ""}
    %6 = fir.declare %5(%4) {uniq_name = ".tmp.array"} : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.heap<!fir.array<?xi32>>
    %7 = fir.embox %6(%4) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
    fir.do_loop %arg1 = %c1 to %3#1 step %c1 unordered {
      %16 = fir.array_coor %2 %arg1 : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
      // load with tbaa
      %17 = fir.load %16 {tbaa = [#tbaa_tag]} : !fir.ref<i32>
      %18 = arith.addi %17, %c1_i32 : i32
      %19 = fir.array_coor %6(%4) %arg1 : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
      // store without tbaa
      fir.store %18 to %19 : !fir.ref<i32>
    }
    fir.store %2 to %0 : !fir.ref<!fir.box<!fir.array<?xi32>>>
    %8 = fir.address_of(@_QQcl.2F746D702F73696D706C652E66393000) : !fir.ref<!fir.char<1,16>>
    %9 = fir.convert %0 : (!fir.ref<!fir.box<!fir.array<?xi32>>>) -> !fir.ref<!fir.box<none>>
    %10 = fir.convert %7 : (!fir.box<!fir.array<?xi32>>) -> !fir.box<none>
    %11 = fir.convert %8 : (!fir.ref<!fir.char<1,16>>) -> !fir.ref<i8>
    %12 = fir.call @_FortranAAssign(%9, %10, %11, %c3_i32) : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.ref<i8>, i32) -> none
    fir.freemem %6 : !fir.heap<!fir.array<?xi32>>
    %13 = fir.array_coor %2 %c2 : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
    // load modified not to have tbaa
    %14 = fir.load %13 : !fir.ref<i32>
    %15 = fir.array_coor %2 %c1 : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
    // store with tbaa
    fir.store %14 to %15 {tbaa = [#tbaa_tag]} : !fir.ref<i32>
    return
  }
  func.func private @_FortranAAssign(!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.ref<i8>, i32) -> none attributes {fir.runtime}
  fir.global linkonce @_QQcl.2F746D702F73696D706C652E66393000 constant : !fir.char<1,16> {
    %0 = fir.string_lit "/tmp/simple.f90\00"(16) : !fir.char<1,16>
    fir.has_value %0 : !fir.char<1,16>
  }
}
// CHECK-LABEL: define void @_QPfunc(
// CHECK-SAME:      ptr %[[ARG0:.*]]) {
// [...]
// CHECK:  %[[VAL5:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, ptr %[[ARG0]], i32 0, i32 7, i32 0, i32 0
// box access:
// CHECK:  %[[VAL6:.*]] = load i64, ptr %[[VAL5]], align 4, !tbaa ![[BOX_ACCESS_TAG:.*]]
// CHECK:  %[[VAL7:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, ptr %0, i32 0, i32 7, i32 0, i32 1
// box access:
// CHECK:  %[[VAL8:.*]] = load i64, ptr %[[VAL7]], align 4, !tbaa ![[BOX_ACCESS_TAG]]
// CHECK:  %[[VAL9:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, ptr %[[ARG0]], i32 0, i32 7, i32 0, i32 2
// box access:
// CHECK:  %[[VAL10:.*]] = load i64, ptr %[[VAL9]], align 4, !tbaa ![[BOX_ACCESS_TAG]]
// CHECK:  %[[VAL11:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, ptr %[[ARG0]], i32 0, i32 0
// box access:
// CHECK:  %[[VAL12:.*]] = load ptr, ptr %[[VAL11]], align 8, !tbaa ![[BOX_ACCESS_TAG]]
// CHECK:  %[[VAL15:.*]] = insertvalue { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] } %14, ptr %[[VAL12]], 0
// CHECK:  store { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] } %[[VAL15]], ptr %{{.*}}, align 8, !tbaa ![[BOX_ACCESS_TAG]]
// CHECK:  %[[VAL16:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, ptr %{{.*}}, i32 0, i32 7, i64 0, i32 0
// box access:
// CHECK:  %[[VAL17:.*]] = load i64, ptr %[[VAL16]], align 4, !tbaa ![[BOX_ACCESS_TAG]]
// CHECK:  %[[VAL18:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, ptr %{{.*}}, i32 0, i32 7, i64 0, i32 1
// box access:
// CHECK:  %[[VAL19:.*]] = load i64, ptr %[[VAL18]], align 4, !tbaa ![[BOX_ACCESS_TAG]]
// CHECK:  %[[VAL20:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, ptr %{{.*}}, i32 0, i32 7, i64 0, i32 2
// box access:
// CHECK:  %[[VAL21:.*]] = load i64, ptr %[[VAL20]], align 4, !tbaa ![[BOX_ACCESS_TAG]]
// [...]
// box access:
// CHECK:  store { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] } %{{.*}}, ptr %{{.*}}, align 8, !tbaa ![[BOX_ACCESS_TAG]]
// [...]

// [...]
// CHECK:  %[[VAL40:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, ptr %{{.*}}, i32 0, i32 0
// box access:
// CHECK:  %[[VAL41:.*]] = load ptr, ptr %[[VAL40]], align 8, !tbaa ![[BOX_ACCESS_TAG]]
// CHECK:  %[[VAL42:.*]] = getelementptr i8, ptr %[[VAL41]], i64 %{{.*}}
// access to 'a':
// CHECK:  %[[VAL43:.*]] = load i32, ptr %[[VAL42]], align 4, !tbaa ![[A_ACCESS_TAG:.*]]
// [...]
// CHECK:  %[[VAL50:.*]] = getelementptr i32, ptr %{{.*}}, i64 %{{.*}}
// store to the temporary:
// CHECK:  store i32 %{{.*}}, ptr %[[VAL50]], align 4, !tbaa ![[DATA_ACCESS_TAG:.*]]
// [...]

// CHECK: [[BOX_ACCESS_TAG]] = !{![[BOX_ACCESS_TYPE:.*]], ![[BOX_ACCESS_TYPE]], i64 0}
// CHECK: ![[BOX_ACCESS_TYPE]] = !{!"descriptor member", ![[ANY_ACCESS_TYPE:.*]], i64 0}
// CHECK: ![[ANY_ACCESS_TYPE]] = !{!"any access", ![[ROOT_TYPE:.*]], i64 0}
// CHECK: ![[ROOT_TYPE]] = !{!"Flang function root _QPfunc"}
// CHECK: ![[A_ACCESS_TAG]] = !{![[A_ACCESS_TYPE:.*]], ![[A_ACCESS_TYPE]], i64 0}
// CHECK: ![[A_ACCESS_TYPE]] = !{!"dummy arg data/_QFfuncEa", ![[ARG_ACCESS_TYPE:.*]], i64 0}
// CHECK: ![[ARG_ACCESS_TYPE]] = !{!"dummy arg data", ![[DATA_ACCESS_TYPE:.*]], i64 0}
// CHECK: ![[DATA_ACCESS_TYPE]] = !{!"any data access", ![[ANY_ACCESS_TYPE]], i64 0}
// CHECK: ![[DATA_ACCESS_TAG]] = !{![[DATA_ACCESS_TYPE]], ![[DATA_ACCESS_TYPE]], i64 0}
