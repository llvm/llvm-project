// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fexperimental-relative-c++-abi-vtables -std=c++20 -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-lowering-prepare %s -o %t.cir 2>&1 | FileCheck %s -check-prefix=BEFORE
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fexperimental-relative-c++-abi-vtables -std=c++20 -fclangir -emit-cir -mmlir --mlir-print-ir-after=cir-lowering-prepare %s -o %t.cir 2>&1 | FileCheck %s -check-prefix=AFTER

struct Base {
  virtual ~Base();
};

// BEFORE: !rec_Base = !cir.record<struct "Base"

void *ptr_cast_to_complete(Base *ptr) {
  return dynamic_cast<void *>(ptr);
}

// BEFORE: cir.func {{.*}} @_Z20ptr_cast_to_completeP4Base
// BEFORE:   %{{.+}} = cir.dyn_cast ptr relative_layout %{{.+}} : !cir.ptr<!rec_Base> -> !cir.ptr<!void>
// BEFORE: }

//      AFTER: cir.func {{.*}} @_Z20ptr_cast_to_completeP4Base
//      AFTER:   %[[#SRC:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!cir.ptr<!rec_Base>>, !cir.ptr<!rec_Base>
// AFTER-NEXT:   %[[#SRC_IS_NOT_NULL:]] = cir.cast ptr_to_bool %[[#SRC]] : !cir.ptr<!rec_Base> -> !cir.bool
// AFTER-NEXT:   %{{.+}} = cir.ternary(%[[#SRC_IS_NOT_NULL]], true {
// AFTER-NEXT:     %[[#VPTR_PTR:]] = cir.vtable.get_vptr %[[#SRC:]] : !cir.ptr<!rec_Base> -> !cir.ptr<!cir.vptr>
// AFTER-NEXT:     %[[#VPTR:]] = cir.load %[[#VPTR_PTR]] : !cir.ptr<!cir.vptr>, !cir.vptr
// AFTER-NEXT:     %[[#ELEM_PTR:]] = cir.cast bitcast %[[#VPTR:]] : !cir.vptr -> !cir.ptr<!s32i>
// AFTER-NEXT:     %[[#MINUS_TWO:]] = cir.const #cir.int<-2> : !s64i
// AFTER-NEXT:     %[[#OFFSET_TO_TOP_PTR:]] = cir.ptr_stride %[[#ELEM_PTR]], %[[#MINUS_TWO:]] : (!cir.ptr<!s32i>, !s64i) -> !cir.ptr<!s32i>
// AFTER-NEXT:     %[[#OFFSET_TO_TOP:]] = cir.load align(4) %[[#OFFSET_TO_TOP_PTR]] : !cir.ptr<!s32i>, !s32i
// AFTER-NEXT:     %[[#SRC_BYTES_PTR:]] = cir.cast bitcast %[[#SRC]] : !cir.ptr<!rec_Base> -> !cir.ptr<!u8i>
// AFTER-NEXT:     %[[#DST_BYTES_PTR:]] = cir.ptr_stride %[[#SRC_BYTES_PTR]], %[[#OFFSET_TO_TOP]] : (!cir.ptr<!u8i>, !s32i) -> !cir.ptr<!u8i>
// AFTER-NEXT:     %[[#DST:]] = cir.cast bitcast %[[#DST_BYTES_PTR]] : !cir.ptr<!u8i> -> !cir.ptr<!void>
// AFTER-NEXT:     cir.yield %[[#DST]] : !cir.ptr<!void>
// AFTER-NEXT:   }, false {
// AFTER-NEXT:     %[[#NULL:]] = cir.const #cir.ptr<null> : !cir.ptr<!void>
// AFTER-NEXT:     cir.yield %[[#NULL]] : !cir.ptr<!void>
// AFTER-NEXT:   }) : (!cir.bool) -> !cir.ptr<!void>
//      AFTER: }
