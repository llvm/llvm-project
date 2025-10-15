// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-lowering-prepare %s -o %t.cir 2> %t.before.log
// RUN: FileCheck %s --input-file=%t.before.log -check-prefix=CIR-BEFORE
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -emit-cir -mmlir --mlir-print-ir-after=cir-lowering-prepare %s -o %t.cir 2> %t.after.log
// RUN: FileCheck %s --input-file=%t.after.log -check-prefix=CIR-AFTER
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck %s --input-file=%t-cir.ll -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -emit-llvm %s -o %t.ll
// RUN: FileCheck %s --input-file=%t.ll -check-prefix=OGCG

struct Base {
  virtual ~Base();
};

struct Derived : Base {};

// CIR-BEFORE-DAG: !rec_Base = !cir.record
// CIR-BEFORE-DAG: !rec_Derived = !cir.record
// CIR-BEFORE-DAG: #dyn_cast_info__ZTI4Base__ZTI7Derived = #cir.dyn_cast_info<src_rtti = #cir.global_view<@_ZTI4Base> : !cir.ptr<!u8i>, dest_rtti = #cir.global_view<@_ZTI7Derived> : !cir.ptr<!u8i>, runtime_func = @__dynamic_cast, bad_cast_func = @__cxa_bad_cast, offset_hint = #cir.int<0> : !s64i>

Derived *ptr_cast(Base *b) {
  return dynamic_cast<Derived *>(b);
}

// CIR-BEFORE: cir.func dso_local @_Z8ptr_castP4Base
// CIR-BEFORE:   %{{.+}} = cir.dyn_cast ptr %{{.+}} : !cir.ptr<!rec_Base> -> !cir.ptr<!rec_Derived> #dyn_cast_info__ZTI4Base__ZTI7Derived
// CIR-BEFORE: }

//      CIR-AFTER: cir.func dso_local @_Z8ptr_castP4Base
//      CIR-AFTER:   %[[SRC:.*]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!cir.ptr<!rec_Base>>, !cir.ptr<!rec_Base>
// CIR-AFTER-NEXT:   %[[SRC_IS_NOT_NULL:.*]] = cir.cast ptr_to_bool %[[SRC]] : !cir.ptr<!rec_Base> -> !cir.bool
// CIR-AFTER-NEXT:   %{{.+}} = cir.ternary(%[[SRC_IS_NOT_NULL]], true {
// CIR-AFTER-NEXT:     %[[SRC_VOID_PTR:.*]] = cir.cast bitcast %[[SRC]] : !cir.ptr<!rec_Base> -> !cir.ptr<!void>
// CIR-AFTER-NEXT:     %[[BASE_RTTI:.*]] = cir.const #cir.global_view<@_ZTI4Base> : !cir.ptr<!u8i>
// CIR-AFTER-NEXT:     %[[DERIVED_RTTI:.*]] = cir.const #cir.global_view<@_ZTI7Derived> : !cir.ptr<!u8i>
// CIR-AFTER-NEXT:     %[[HINT:.*]] = cir.const #cir.int<0> : !s64i
// CIR-AFTER-NEXT:     %[[RT_CALL_RET:.*]] = cir.call @__dynamic_cast(%[[SRC_VOID_PTR]], %[[BASE_RTTI]], %[[DERIVED_RTTI]], %[[HINT]]) : (!cir.ptr<!void>, !cir.ptr<!u8i>, !cir.ptr<!u8i>, !s64i) -> !cir.ptr<!void>
// CIR-AFTER-NEXT:     %[[CASTED:.*]] = cir.cast bitcast %[[RT_CALL_RET]] : !cir.ptr<!void> -> !cir.ptr<!rec_Derived>
// CIR-AFTER-NEXT:     cir.yield %[[CASTED]] : !cir.ptr<!rec_Derived>
// CIR-AFTER-NEXT:   }, false {
// CIR-AFTER-NEXT:     %[[NULL_PTR:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!rec_Derived>
// CIR-AFTER-NEXT:     cir.yield %[[NULL_PTR]] : !cir.ptr<!rec_Derived>
// CIR-AFTER-NEXT:   }) : (!cir.bool) -> !cir.ptr<!rec_Derived>
//      CIR-AFTER: }

// LLVM: define {{.*}} @_Z8ptr_castP4Base
// LLVM:   %[[IS_NOT_NULL:.*]] = icmp ne ptr %[[PTR:.*]], null
// LLVM:   br i1 %[[IS_NOT_NULL]], label %[[NOT_NULL:.*]], label %[[NULL:.*]]
// LLVM: [[NOT_NULL]]:
// LLVM:   %[[RESULT:.*]] = call ptr @__dynamic_cast(ptr %[[PTR]], ptr @_ZTI4Base, ptr @_ZTI7Derived, i64 0)
// LLVM:   br label %[[DONE:.*]]
// LLVM: [[NULL]]:
// LLVM:   br label %[[DONE]]
// LLVM: [[DONE]]:
// LLVM:   %[[RET:.*]] = phi ptr [ null, %[[NULL]] ], [ %[[RESULT]], %[[NOT_NULL]] ]

// OGCG: define {{.*}} @_Z8ptr_castP4Base
// OGCG:   %[[IS_NULL:.*]] = icmp eq ptr %[[PTR:.*]], null
// OGCG:   br i1 %[[IS_NULL]], label %[[NULL:.*]], label %[[NOT_NULL:.*]]
// OGCG: [[NOT_NULL]]:
// OGCG:   %[[RESULT:.*]] = call ptr @__dynamic_cast(ptr %[[PTR]], ptr @_ZTI4Base, ptr @_ZTI7Derived, i64 0)
// OGCG:   br label %[[DONE:.*]]
// OGCG: [[NULL]]:
// OGCG:   br label %[[DONE]]
// OGCG: [[DONE]]:
// OGCG:   %[[RET:.*]] = phi ptr [ %[[RESULT]], %[[NOT_NULL]] ], [ null, %[[NULL]] ]


Derived &ref_cast(Base &b) {
  return dynamic_cast<Derived &>(b);
}

// CIR-BEFORE: cir.func dso_local @_Z8ref_castR4Base
// CIR-BEFORE:   %{{.+}} = cir.dyn_cast ref %{{.+}} : !cir.ptr<!rec_Base> -> !cir.ptr<!rec_Derived> #dyn_cast_info__ZTI4Base__ZTI7Derived
// CIR-BEFORE: }

//      CIR-AFTER: cir.func dso_local @_Z8ref_castR4Base
//      CIR-AFTER:   %[[SRC_VOID_PTR:.*]] = cir.cast bitcast %{{.+}} : !cir.ptr<!rec_Base> -> !cir.ptr<!void>
// CIR-AFTER-NEXT:   %[[SRC_RTTI:.*]] = cir.const #cir.global_view<@_ZTI4Base> : !cir.ptr<!u8i>
// CIR-AFTER-NEXT:   %[[DEST_RTTI:.*]] = cir.const #cir.global_view<@_ZTI7Derived> : !cir.ptr<!u8i>
// CIR-AFTER-NEXT:   %[[OFFSET_HINT:.*]] = cir.const #cir.int<0> : !s64i
// CIR-AFTER-NEXT:   %[[CASTED_PTR:.*]] = cir.call @__dynamic_cast(%[[SRC_VOID_PTR]], %[[SRC_RTTI]], %[[DEST_RTTI]], %[[OFFSET_HINT]]) : (!cir.ptr<!void>, !cir.ptr<!u8i>, !cir.ptr<!u8i>, !s64i) -> !cir.ptr<!void>
// CIR-AFTER-NEXT:   %[[NULL_PTR:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!void>
// CIR-AFTER-NEXT:   %[[CASTED_PTR_IS_NULL:.*]] = cir.cmp(eq, %[[CASTED_PTR]], %[[NULL_PTR]]) : !cir.ptr<!void>, !cir.bool
// CIR-AFTER-NEXT:   cir.if %[[CASTED_PTR_IS_NULL]] {
// CIR-AFTER-NEXT:     cir.call @__cxa_bad_cast() : () -> ()
// CIR-AFTER-NEXT:     cir.unreachable
// CIR-AFTER-NEXT:   }
// CIR-AFTER-NEXT:   %{{.+}} = cir.cast bitcast %[[CASTED_PTR]] : !cir.ptr<!void> -> !cir.ptr<!rec_Derived>
//      CIR-AFTER: }

// LLVM: define {{.*}} ptr @_Z8ref_castR4Base
// LLVM:   %[[RESULT:.*]] = call ptr @__dynamic_cast(ptr %{{.*}}, ptr @_ZTI4Base, ptr @_ZTI7Derived, i64 0)
// LLVM:   %[[IS_NULL:.*]] = icmp eq ptr %[[RESULT]], null
// LLVM:   br i1 %[[IS_NULL]], label %[[BAD_CAST:.*]], label %[[DONE:.*]]
// LLVM: [[BAD_CAST]]:
// LLVM:   call void @__cxa_bad_cast()

// OGCG: define {{.*}} ptr @_Z8ref_castR4Base
// OGCG:   %[[RESULT:.*]] = call ptr @__dynamic_cast(ptr %0, ptr @_ZTI4Base, ptr @_ZTI7Derived, i64 0)
// OGCG:   %[[IS_NULL:.*]] = icmp eq ptr %[[RESULT]], null
// OGCG:   br i1 %[[IS_NULL]], label %[[BAD_CAST:.*]], label %[[DONE:.*]]
// OGCG: [[BAD_CAST]]:
// OGCG:   call void @__cxa_bad_cast()
