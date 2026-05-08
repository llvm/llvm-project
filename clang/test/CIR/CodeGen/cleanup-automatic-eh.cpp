// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fexceptions -fcxx-exceptions -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fexceptions -fcxx-exceptions -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fexceptions -fcxx-exceptions -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG

struct Struk {
  Struk();
  ~Struk();
};

void mayThrow();

void test_cleanup_with_automatic_storage_duration() {
  const Struk &ref = Struk{};
  mayThrow();
}

// CIR: cir.func{{.*}} @_Z44test_cleanup_with_automatic_storage_durationv()
// CIR:   %[[REF_TMP:.*]] = cir.alloca !rec_Struk, !cir.ptr<!rec_Struk>, ["ref.tmp0"]
// CIR:   %[[REF:.*]] = cir.alloca !cir.ptr<!rec_Struk>, !cir.ptr<!cir.ptr<!rec_Struk>>, ["ref", init, const]
// CIR:   cir.call @_ZN5StrukC1Ev(%[[REF_TMP]])
// CIR:   cir.cleanup.scope {
// CIR:     cir.store{{.*}} %[[REF_TMP]], %[[REF]]
// CIR:     cir.call @_Z8mayThrowv()
// CIR:     cir.yield
// CIR:   } cleanup all {
// CIR:     cir.call @_ZN5StrukD1Ev(%[[REF_TMP]]) nothrow
// CIR:     cir.yield
// CIR:   }
// CIR:   cir.return

// LLVM: define {{.*}} void @_Z44test_cleanup_with_automatic_storage_durationv()
// LLVM:   call void @_ZN5StrukC1Ev(
// LLVM:   invoke void @_Z8mayThrowv()
// LLVM:     to label %[[CONT:.*]] unwind label %[[LPAD:.*]]
// LLVM: [[CONT]]:
// LLVM:   call void @_ZN5StrukD1Ev(
// LLVM:   br label %[[EXIT_NORMAL_CLEANUP:.*]]
// LLVM: [[EXIT_NORMAL_CLEANUP]]:
// LLVM:   br label %[[EXIT:.*]]
// LLVM: [[LPAD]]:
// LLVM:   landingpad { ptr, i32 }
// LLVM:     cleanup
// LLVM:   call void @_ZN5StrukD1Ev(
// LLVM:   resume
// LLVM: [[EXIT]]:
// LLVM:   ret void

// OGCG: define {{.*}} void @_Z44test_cleanup_with_automatic_storage_durationv()
// OGCG:   call void @_ZN5StrukC1Ev(
// OGCG:   invoke void @_Z8mayThrowv()
// OGCG:     to label %[[CONT:.*]] unwind label %[[LPAD:.*]]
// OGCG: [[CONT]]:
// OGCG:   call void @_ZN5StrukD1Ev(
// OGCG:   ret void
// OGCG: [[LPAD]]:
// OGCG:   landingpad { ptr, i32 }
// OGCG:     cleanup
// OGCG:   call void @_ZN5StrukD1Ev(
// OGCG:   resume
