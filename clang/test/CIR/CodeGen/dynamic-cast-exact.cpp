// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -O1 -fclangir -clangir-disable-passes -emit-cir -o %t.cir %s
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -O1 -fclangir -emit-llvm -o %t-cir.ll %s
// RUN: FileCheck --input-file=%t-cir.ll --check-prefix=LLVM %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -O1 -emit-llvm -o %t.ll %s
// RUN: FileCheck --input-file=%t.ll --check-prefix=OGCG %s

struct Base1 {
  virtual ~Base1();
};

struct Base2 {
  virtual ~Base2();
};

struct Derived final : Base1 {};

Derived *ptr_cast(Base1 *ptr) {
  return dynamic_cast<Derived *>(ptr);
}

//      CIR: cir.func {{.*}} @_Z8ptr_castP5Base1
//      CIR:   %[[SRC:.*]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!cir.ptr<!rec_Base1>>, !cir.ptr<!rec_Base1>
// CIR-NEXT:   %[[NULL_PTR:.*]] = cir.const #cir.ptr<null>
// CIR-NEXT:   %[[SRC_IS_NULL:.*]] = cir.cmp(eq, %[[SRC]], %[[NULL_PTR]])
// CIR-NEXT:   %[[RESULT:.*]] = cir.ternary(%[[SRC_IS_NULL]], true {
// CIR-NEXT:     %[[NULL_PTR_DEST:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!rec_Derived>
// CIR-NEXT:     cir.yield %[[NULL_PTR_DEST]] : !cir.ptr<!rec_Derived>
// CIR-NEXT:   }, false {
// CIR-NEXT:     %[[EXPECTED_VPTR:.*]] = cir.vtable.address_point(@_ZTV7Derived, address_point = <index = 0, offset = 2>) : !cir.vptr
// CIR-NEXT:     %[[SRC_VPTR_PTR:.*]] = cir.cast bitcast %[[SRC]] : !cir.ptr<!rec_Base1> -> !cir.ptr<!cir.vptr>
// CIR-NEXT:     %[[SRC_VPTR:.*]] = cir.load{{.*}} %[[SRC_VPTR_PTR]] : !cir.ptr<!cir.vptr>, !cir.vptr
// CIR-NEXT:     %[[SUCCESS:.*]] = cir.cmp(eq, %[[SRC_VPTR]], %[[EXPECTED_VPTR]]) : !cir.vptr, !cir.bool
// CIR-NEXT:     %[[EXACT_RESULT:.*]] = cir.ternary(%[[SUCCESS]], true {
// CIR-NEXT:       %[[RES:.*]] = cir.cast bitcast %[[SRC]] : !cir.ptr<!rec_Base1> -> !cir.ptr<!rec_Derived>
// CIR-NEXT:       cir.yield %[[RES]] : !cir.ptr<!rec_Derived>
// CIR-NEXT:     }, false {
// CIR-NEXT:       %[[NULL:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!rec_Derived>
// CIR-NEXT:       cir.yield %[[NULL]] : !cir.ptr<!rec_Derived>
// CIR-NEXT:     }) : (!cir.bool) -> !cir.ptr<!rec_Derived>
// CIR-NEXT:     cir.yield %[[EXACT_RESULT]] : !cir.ptr<!rec_Derived>
// CIR-NEXT:   }) : (!cir.bool) -> !cir.ptr<!rec_Derived>

// Note: The LLVM output omits the label for the entry block (which is
//       implicitly %1), so we use %{{.*}} to match the implicit label in the
//       phi check.

//      LLVM: define dso_local ptr @_Z8ptr_castP5Base1(ptr{{.*}} %[[SRC:.*]])
// LLVM-NEXT:   %[[SRC_IS_NULL:.*]] = icmp eq ptr %0, null
// LLVM-NEXT:   br i1 %[[SRC_IS_NULL]], label %[[LABEL_END:.*]], label %[[LABEL_NOTNULL:.*]]
//      LLVM: [[LABEL_NOTNULL]]:
// LLVM-NEXT:   %[[VPTR:.*]] = load ptr, ptr %[[SRC]], align 8
// LLVM-NEXT:   %[[SUCCESS:.*]] = icmp eq ptr %[[VPTR]], getelementptr inbounds nuw (i8, ptr @_ZTV7Derived, i64 16)
// LLVM-NEXT:   %[[EXACT_RESULT:.*]] = select i1 %[[SUCCESS]], ptr %[[SRC]], ptr null
// LLVM-NEXT:   br label %[[LABEL_END]]
//      LLVM: [[LABEL_END]]:
// LLVM-NEXT:   %[[RESULT:.*]] = phi ptr [ %[[EXACT_RESULT]], %[[LABEL_NOTNULL]] ], [ null, %{{.*}} ]
// LLVM-NEXT:   ret ptr %[[RESULT]]
// LLVM-NEXT: }

//      OGCG: define{{.*}} ptr @_Z8ptr_castP5Base1(ptr {{.*}} %[[SRC:.*]])
// OGCG-NEXT: entry:
// OGCG-NEXT:   %[[NULL_CHECK:.*]] = icmp eq ptr %[[SRC]], null
// OGCG-NEXT:   br i1 %[[NULL_CHECK]], label %[[LABEL_NULL:.*]], label %[[LABEL_NOTNULL:.*]]
//      OGCG: [[LABEL_NOTNULL]]:
// OGCG-NEXT:   %[[VTABLE:.*]] = load ptr, ptr %[[SRC]], align 8
// OGCG-NEXT:   %[[VTABLE_CHECK:.*]] = icmp eq ptr %[[VTABLE]], getelementptr inbounds {{.*}} (i8, ptr @_ZTV7Derived, i64 16)
// OGCG-NEXT:   br i1 %[[VTABLE_CHECK]], label %[[LABEL_END:.*]], label %[[LABEL_NULL]]
//      OGCG: [[LABEL_NULL]]:
// OGCG-NEXT:   br label %[[LABEL_END]]
//      OGCG: [[LABEL_END]]:
// OGCG-NEXT:   %[[RESULT:.*]] = phi ptr [ %[[SRC]], %[[LABEL_NOTNULL]] ], [ null, %[[LABEL_NULL]] ]
// OGCG-NEXT:   ret ptr %[[RESULT]]
// OGCG-NEXT: }

Derived &ref_cast(Base1 &ref) {
  return dynamic_cast<Derived &>(ref);
}

//      CIR: cir.func {{.*}} @_Z8ref_castR5Base1
//      CIR:   %[[SRC:.*]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!cir.ptr<!rec_Base1>>, !cir.ptr<!rec_Base1>
// CIR-NEXT:   %[[EXPECTED_VPTR:.*]] = cir.vtable.address_point(@_ZTV7Derived, address_point = <index = 0, offset = 2>) : !cir.vptr
// CIR-NEXT:   %[[SRC_VPTR_PTR:.*]] = cir.cast bitcast %[[SRC]] : !cir.ptr<!rec_Base1> -> !cir.ptr<!cir.vptr>
// CIR-NEXT:   %[[SRC_VPTR:.*]] = cir.load{{.*}} %[[SRC_VPTR_PTR]] : !cir.ptr<!cir.vptr>, !cir.vptr
// CIR-NEXT:   %[[SUCCESS:.*]] = cir.cmp(eq, %[[SRC_VPTR]], %[[EXPECTED_VPTR]]) : !cir.vptr, !cir.bool
// CIR-NEXT:   %[[FAILED:.*]] = cir.unary(not, %[[SUCCESS]]) : !cir.bool, !cir.bool
// CIR-NEXT:   cir.if %[[FAILED]] {
// CIR-NEXT:     cir.call @__cxa_bad_cast() : () -> ()
// CIR-NEXT:     cir.unreachable
// CIR-NEXT:   }
// CIR-NEXT:   %{{.+}} = cir.cast bitcast %[[SRC]] : !cir.ptr<!rec_Base1> -> !cir.ptr<!rec_Derived>

//      LLVM: define{{.*}} ptr @_Z8ref_castR5Base1(ptr{{.*}} %[[SRC:.*]])
// LLVM-NEXT:   %[[VPTR:.*]] = load ptr, ptr %[[SRC]], align 8
// LLVM-NEXT:   %[[OK:.*]] = icmp eq ptr %[[VPTR]], getelementptr inbounds nuw (i8, ptr @_ZTV7Derived, i64 16)
// LLVM-NEXT:   br i1 %[[OK]], label %[[LABEL_OK:.*]], label %[[LABEL_FAIL:.*]]
//      LLVM: [[LABEL_FAIL]]:
// LLVM-NEXT:   tail call void @__cxa_bad_cast()
// LLVM-NEXT:   unreachable
//      LLVM: [[LABEL_OK]]:
// LLVM-NEXT:   ret ptr %[[SRC]]
// LLVM-NEXT: }

//      OGCG: define{{.*}} ptr @_Z8ref_castR5Base1(ptr {{.*}} %[[REF:.*]])
// OGCG-NEXT: entry:
// OGCG-NEXT:   %[[VTABLE:.*]] = load ptr, ptr %[[REF]], align 8
// OGCG-NEXT:   %[[VTABLE_CHECK:.*]] = icmp eq ptr %[[VTABLE]], getelementptr inbounds {{.*}} (i8, ptr @_ZTV7Derived, i64 16)
// OGCG-NEXT:   br i1 %[[VTABLE_CHECK]], label %[[LABEL_END:.*]], label %[[LABEL_NULL:.*]]
//      OGCG: [[LABEL_NULL]]:
// OGCG-NEXT:   {{.*}}call void @__cxa_bad_cast()
// OGCG-NEXT:   unreachable
//      OGCG: [[LABEL_END]]:
// OGCG-NEXT:   ret ptr %[[REF]]
// OGCG-NEXT: }

struct Offset { virtual ~Offset(); };
struct A { virtual ~A(); };
struct B final : Offset, A { };

B *offset_cast(A *a) {
  return dynamic_cast<B*>(a);
}

//      CIR: cir.func {{.*}} @_Z11offset_castP1A
//      CIR:   %[[SRC:.*]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!cir.ptr<!rec_A>>, !cir.ptr<!rec_A>
// CIR-NEXT:   %[[NULL_PTR:.*]] = cir.const #cir.ptr<null>
// CIR-NEXT:   %[[SRC_IS_NULL:.*]] = cir.cmp(eq, %[[SRC]], %[[NULL_PTR]])
// CIR-NEXT:   %[[RESULT:.*]] = cir.ternary(%[[SRC_IS_NULL]], true {
// CIR-NEXT:     %[[NULL_PTR_DEST:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!rec_B>
// CIR-NEXT:     cir.yield %[[NULL_PTR_DEST]] : !cir.ptr<!rec_B>
// CIR-NEXT:   }, false {
// CIR-NEXT:     %[[EXPECTED_VPTR:.*]] = cir.vtable.address_point(@_ZTV1B, address_point = <index = 1, offset = 2>) : !cir.vptr
// CIR-NEXT:     %[[SRC_VPTR_PTR:.*]] = cir.cast bitcast %[[SRC]] : !cir.ptr<!rec_A> -> !cir.ptr<!cir.vptr>
// CIR-NEXT:     %[[SRC_VPTR:.*]] = cir.load{{.*}} %[[SRC_VPTR_PTR]] : !cir.ptr<!cir.vptr>, !cir.vptr
// CIR-NEXT:     %[[SUCCESS:.*]] = cir.cmp(eq, %[[SRC_VPTR]], %[[EXPECTED_VPTR]]) : !cir.vptr, !cir.bool
// CIR-NEXT:     %[[EXACT_RESULT:.*]] = cir.ternary(%[[SUCCESS]], true {
// CIR-NEXT:       %[[MINUS_EIGHT:.*]] = cir.const #cir.int<18446744073709551608> : !u64i
// CIR-NEXT:       %[[SRC_VOID:.*]] = cir.cast bitcast %[[SRC]] : !cir.ptr<!rec_A> -> !cir.ptr<!u8i>
// CIR-NEXT:       %[[SRC_OFFSET:.*]] = cir.ptr_stride %[[SRC_VOID]], %[[MINUS_EIGHT]]
// CIR-NEXT:       %[[RES:.*]] = cir.cast bitcast %[[SRC_OFFSET]] : !cir.ptr<!u8i> -> !cir.ptr<!rec_B>
// CIR-NEXT:       cir.yield %[[RES]] : !cir.ptr<!rec_B>
// CIR-NEXT:     }, false {
// CIR-NEXT:       %[[NULL:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!rec_B>
// CIR-NEXT:       cir.yield %[[NULL]] : !cir.ptr<!rec_B>
// CIR-NEXT:     }) : (!cir.bool) -> !cir.ptr<!rec_B>
// CIR-NEXT:     cir.yield %[[EXACT_RESULT]] : !cir.ptr<!rec_B>
// CIR-NEXT:   }) : (!cir.bool) -> !cir.ptr<!rec_B>

//      LLVM: define dso_local ptr @_Z11offset_castP1A(ptr{{.*}} %[[SRC:.*]])
// LLVM-NEXT:   %[[SRC_IS_NULL:.*]] = icmp eq ptr %0, null
// LLVM-NEXT:   br i1 %[[SRC_IS_NULL]], label %[[LABEL_END:.*]], label %[[LABEL_NOTNULL:.*]]
//      LLVM: [[LABEL_NOTNULL]]:
// LLVM-NEXT:   %[[VTABLE:.*]] = load ptr, ptr %[[SRC]]
// LLVM-NEXT:   %[[VTABLE_CHECK:.*]] = icmp eq ptr %[[VTABLE]], getelementptr inbounds nuw (i8, ptr @_ZTV1B, i64 48)
// LLVM-NEXT:   %[[SRC_OFFSET:.*]] = getelementptr i8, ptr %[[SRC]], i64 -8
// LLVM-NEXT:   %[[EXACT_RESULT:.*]] = select i1 %[[VTABLE_CHECK]], ptr %[[SRC_OFFSET]], ptr null
// LLVM-NEXT:   br label %[[LABEL_END]]
//      LLVM: [[LABEL_END]]:
// LLVM-NEXT:   %[[RESULT:.*]] = phi ptr [ %[[EXACT_RESULT]], %[[LABEL_NOTNULL]] ], [ null, %{{.*}} ]
// LLVM-NEXT:   ret ptr %[[RESULT]]
// LLVM-NEXT: }

//      OGCG: define{{.*}} ptr @_Z11offset_castP1A(ptr{{.*}} %[[SRC:.*]])
//      OGCG:   %[[SRV_NULL:.*]] = icmp eq ptr %[[SRC]], null
// OGCG-NEXT:   br i1 %[[SRV_NULL]], label %[[LABEL_NULL:.*]], label %[[LABEL_NOTNULL:.*]]
//      OGCG: [[LABEL_NOTNULL]]:
// OGCG-NEXT:   %[[VTABLE:.*]] = load ptr, ptr %[[SRC]]
// OGCG-NEXT:   %[[VTABLE_CHECK:.*]] = icmp eq ptr %[[VTABLE]], getelementptr inbounds nuw inrange(-16, 16) (i8, ptr @_ZTV1B, i64 48)
// OGCG-NEXT:   %[[RESULT:.*]] = getelementptr inbounds i8, ptr %[[SRC]], i64 -8
// OGCG-NEXT:   br i1 %[[VTABLE_CHECK]], label %[[LABEL_END:.*]], label %[[LABEL_NULL]]
//      OGCG: [[LABEL_NULL]]:
// OGCG-NEXT:   br label %[[LABEL_END]]
//      OGCG: [[LABEL_END]]:
// OGCG-NEXT:   phi ptr [ %[[RESULT]], %[[LABEL_NOTNULL]] ], [ null, %[[LABEL_NULL]] ]

Derived *ptr_cast_always_fail(Base2 *ptr) {
    return dynamic_cast<Derived *>(ptr);
  }

//      CIR: cir.func {{.*}} @_Z20ptr_cast_always_failP5Base2
//      CIR:   %{{.+}} = cir.load{{.*}} %{{.+}} : !cir.ptr<!cir.ptr<!rec_Base2>>, !cir.ptr<!rec_Base2>
// CIR-NEXT:   %[[RESULT:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!rec_Derived>
// CIR-NEXT:   cir.store %[[RESULT]], %{{.*}} : !cir.ptr<!rec_Derived>, !cir.ptr<!cir.ptr<!rec_Derived>>

//      LLVM: define {{.*}} ptr @_Z20ptr_cast_always_failP5Base2
// LLVM-NEXT:   ret ptr null

//      OGCG: define {{.*}} ptr @_Z20ptr_cast_always_failP5Base2
// OGCG-NEXT: entry:
// OGCG-NEXT:   ret ptr null

Derived &ref_cast_always_fail(Base2 &ref) {
  return dynamic_cast<Derived &>(ref);
}

//      CIR: cir.func {{.*}} @_Z20ref_cast_always_failR5Base2
//      CIR:   %{{.+}} = cir.load{{.*}} %{{.+}} : !cir.ptr<!cir.ptr<!rec_Base2>>, !cir.ptr<!rec_Base2>
// CIR-NEXT:   cir.call @__cxa_bad_cast() : () -> ()
// CIR-NEXT:   cir.unreachable

//      LLVM: define {{.*}} ptr @_Z20ref_cast_always_failR5Base2
// LLVM-NEXT:   tail call void @__cxa_bad_cast()
// LLVM-NEXT:   unreachable

//      OGCG: define {{.*}} ptr @_Z20ref_cast_always_failR5Base2
// OGCG-NEXT: entry:
// OGCG-NEXT:   tail call void @__cxa_bad_cast()
// OGCG-NEXT:   unreachable
