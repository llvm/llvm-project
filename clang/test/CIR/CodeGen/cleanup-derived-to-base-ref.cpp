// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fcxx-exceptions -fexceptions -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fcxx-exceptions -fexceptions -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM

struct Base {};
struct Derived : Base {
  ~Derived();
};

Derived make_derived();

void bind_base_ref_to_derived_temp() {
  const Base &r = make_derived();
}

// CIR-LABEL: cir.func {{.*}}@_Z29bind_base_ref_to_derived_tempv
// CIR:   %[[REF_TMP:.*]] = cir.alloca !rec_Derived, !cir.ptr<!rec_Derived>, ["ref.tmp0"]
// CIR:   %[[R:.*]] = cir.alloca !cir.ptr<!rec_Base>, !cir.ptr<!cir.ptr<!rec_Base>>, ["r", init, const]
// CIR:   %[[SPILL:.*]] = cir.alloca !cir.ptr<!rec_Base>, !cir.ptr<!cir.ptr<!rec_Base>>, ["tmp.exprcleanup"]
// CIR:   cir.call @_Z12make_derivedv()
// CIR:   cir.cleanup.scope {
// CIR:     %[[BASE:.*]] = cir.base_class_addr %[[REF_TMP]] : !cir.ptr<!rec_Derived> nonnull [0] -> !cir.ptr<!rec_Base>
// CIR:     cir.store {{.*}} %[[BASE]], %[[SPILL]]
// CIR:     cir.yield
// CIR:   } cleanup eh {
// CIR:     cir.call @_ZN7DerivedD1Ev(%[[REF_TMP]])
// CIR:     cir.yield
// CIR:   }
// CIR:   cir.cleanup.scope {
// CIR:     %[[RELOAD:.*]] = cir.load {{.*}} %[[SPILL]] : !cir.ptr<!cir.ptr<!rec_Base>>, !cir.ptr<!rec_Base>
// CIR:     cir.store {{.*}} %[[RELOAD]], %[[R]]
// CIR:     cir.yield
// CIR:   } cleanup all {
// CIR:     cir.call @_ZN7DerivedD1Ev(%[[REF_TMP]])
// CIR:     cir.yield
// CIR:   }
// CIR:   cir.return

// LLVM-LABEL: define {{.*}}@_Z29bind_base_ref_to_derived_tempv()
// LLVM:   %[[REF_TMP:.*]] = alloca %struct.Derived
// LLVM:   %[[R:.*]] = alloca ptr
// LLVM:   %[[SPILL:.*]] = alloca ptr
// LLVM:   store ptr %[[REF_TMP]], ptr %[[SPILL]]
// LLVM:   %[[RELOAD:.*]] = load ptr, ptr %[[SPILL]]
// LLVM:   store ptr %[[RELOAD]], ptr %[[R]]
// LLVM:   call void @_ZN7DerivedD1Ev(ptr {{.*}} %[[REF_TMP]])
// LLVM:   ret void
