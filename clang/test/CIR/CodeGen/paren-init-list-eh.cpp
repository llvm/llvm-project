// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fexceptions -fcxx-exceptions -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fexceptions -fcxx-exceptions -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fexceptions -fcxx-exceptions -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

struct Struk {
  int val;
  Struk(int);
  ~Struk();
};

struct Outer {
  Struk s1;
  Struk s2;
  int x;
};

void test_init_list_with_dtor() {
  Outer o = {Struk{1}, Struk{2}, 3};
}

// CIR: cir.func {{.*}} @_Z24test_init_list_with_dtorv
// CIR:   %[[O:.*]] = cir.alloca !rec_Outer, !cir.ptr<!rec_Outer>, ["o", init]
// CIR:   cir.scope {
// CIR:     %[[S1:.*]] = cir.get_member %[[O]][0] {name = "s1"} : !cir.ptr<!rec_Outer> -> !cir.ptr<!rec_Struk>
// CIR:     %[[ONE:.*]] = cir.const #cir.int<1>
// CIR:     cir.call @_ZN5StrukC1Ei(%[[S1]], %[[ONE]])
// CIR:     cir.cleanup.scope {
// CIR:       %[[S2:.*]] = cir.get_member %[[O]][1] {name = "s2"} : !cir.ptr<!rec_Outer> -> !cir.ptr<!rec_Struk>
// CIR:       %[[TWO:.*]] = cir.const #cir.int<2>
// CIR:       cir.call @_ZN5StrukC1Ei(%[[S2]], %[[TWO]])
// CIR:       cir.cleanup.scope {
// CIR:         %[[X:.*]] = cir.get_member %[[O]][2] {name = "x"}
// CIR:         %[[THREE:.*]] = cir.const #cir.int<3> : !s32i
// CIR:         cir.store align(4) %[[THREE]], %[[X]]
// CIR:         cir.yield
// CIR:       } cleanup eh {
// CIR:         cir.call @_ZN5StrukD1Ev(%[[S2]])
// CIR:         cir.yield
// CIR:       }
// CIR:       cir.yield
// CIR:     } cleanup eh {
// CIR:       cir.call @_ZN5StrukD1Ev(%[[S1]])
// CIR:       cir.yield
// CIR:     }
// CIR:   }
// CIR:   cir.cleanup.scope {
// CIR:     cir.yield
// CIR:   } cleanup all {
// CIR:     cir.call @_ZN5OuterD1Ev(%[[O]])
// CIR:     cir.yield
// CIR:   }
// CIR:   cir.return
// CIR: }

// LLVM: define {{.*}} void @_Z24test_init_list_with_dtorv
// LLVM:   %[[O:.*]] = alloca %struct.Outer
// LLVM:   %[[S1_ADDR:.*]] = getelementptr %struct.Outer, ptr %[[O]], i32 0, i32 0
// LLVM:   call void @_ZN5StrukC1Ei(ptr {{.*}} %[[S1_ADDR]], i32 {{.*}} 1)
// LLVM:   %[[S2_ADDR:.*]] = getelementptr %struct.Outer, ptr %[[O]], i32 0, i32 1
// LLVM:   invoke void @_ZN5StrukC1Ei(ptr {{.*}} %[[S2_ADDR]], i32 {{.*}} 2)
// LLVM:           to label %[[CONT:.*]] unwind label %[[LPAD:.*]]
// LLVM: [[CONT]]:
// LLVM:   %[[X_ADDR:.*]] = getelementptr %struct.Outer, ptr %[[O]], i32 0, i32 2
// LLVM:   store i32 3, ptr %[[X_ADDR]]
// LLVM:   br label %[[EXIT_CLEANUP_SCOPE:.*]]
// LLVM: [[EXIT_CLEANUP_SCOPE]]:
// LLVM:   br label %[[DONE:.*]]
// LLVM: [[LPAD]]:
// LLVM:   %[[EXN:.*]] = landingpad { ptr, i32 }
// LLVM:                    cleanup
// LLVM:   %[[EXN_ADDR:.*]] = extractvalue { ptr, i32 } %[[EXN]], 0
// LLVM:   %[[EXN_SELECTOR:.*]] = extractvalue { ptr, i32 } %[[EXN]], 1
// LLVM:   br label %[[EH_CLEANUP:.*]]
// LLVM: [[EH_CLEANUP]]:
// LLVM:   %[[EXN_ADDR_PHI:.*]] = phi ptr [ %[[EXN_ADDR]], %[[LPAD]] ]
// LLVM:   %[[EXN_SELECTOR_PHI:.*]] = phi i32 [ %[[EXN_SELECTOR]], %[[LPAD]] ]
// LLVM:   call void @_ZN5StrukD1Ev(ptr {{.*}} %[[S1_ADDR]])
// LLVM:   %[[EXN_INS:.*]] = insertvalue { ptr, i32 } poison, ptr %[[EXN_ADDR_PHI]], 0
// LLVM:   %[[EXN_INS2:.*]] = insertvalue { ptr, i32 } %[[EXN_INS]], i32 %[[EXN_SELECTOR_PHI]], 1
// LLVM:   resume { ptr, i32 } %[[EXN_INS2]]
// LLVM: [[DONE]]:
// LLVM:   ret void

// OGCG: define {{.*}} void @_Z24test_init_list_with_dtorv
// OGCG:   %[[O:.*]] = alloca %struct.Outer
// OGCG:   %[[EXN_SLOT:.*]] = alloca ptr
// OGCG:   %[[EHSELECTOR_SLOT:.*]] = alloca i32
// OGCG:   %[[S1_ADDR:.*]] = getelementptr inbounds nuw %struct.Outer, ptr %[[O]], i32 0, i32 0
// OGCG:   call void @_ZN5StrukC1Ei(ptr {{.*}} %[[S1_ADDR]], i32 {{.*}} 1)
// OGCG:   %[[S2_ADDR:.*]] = getelementptr inbounds nuw %struct.Outer, ptr %[[O]], i32 0, i32 1
// OGCG:   invoke void @_ZN5StrukC1Ei(ptr {{.*}} %[[S2_ADDR]], i32 {{.*}} 2)
// OGCG:           to label %[[CONT:.*]] unwind label %[[LPAD:.*]]
// OGCG: [[CONT]]:
// OGCG:   %[[X_ADDR:.*]] = getelementptr inbounds nuw %struct.Outer, ptr %[[O]], i32 0, i32 2
// OGCG:   store i32 3, ptr %[[X_ADDR]]
// OGCG:   call void @_ZN5OuterD1Ev(ptr{{.*}} %[[O]])
// OGCG:   ret void
// OGCG: [[LPAD]]:
// OGCG:   %[[EXN:.*]] = landingpad { ptr, i32 }
// OGCG:                    cleanup
// OGCG:   %[[EXN_ADDR:.*]] = extractvalue { ptr, i32 } %[[EXN]], 0
// OGCG:   store ptr %[[EXN_ADDR]], ptr %[[EXN_SLOT]]
// OGCG:   %[[EXN_SELECTOR:.*]] = extractvalue { ptr, i32 } %[[EXN]], 1
// OGCG:   store i32 %[[EXN_SELECTOR]], ptr %[[EHSELECTOR_SLOT]]
// OGCG:   call void @_ZN5StrukD1Ev(ptr {{.*}} %[[S1_ADDR]])
// OGCG:   br label %[[EH_RESUME:.*]]
// OGCG: [[EH_RESUME]]:
// OGCG:   %[[EXN_SLOT_LOAD:.*]] = load ptr, ptr %[[EXN_SLOT]]
// OGCG:   %[[EHSELECTOR_SLOT_LOAD:.*]] = load i32, ptr %[[EHSELECTOR_SLOT]]
// OGCG:   %[[EXN_INSERT:.*]] = insertvalue { ptr, i32 } poison, ptr %[[EXN_SLOT_LOAD]], 0
// OGCG:   %[[EXN_INSERT_2:.*]] = insertvalue { ptr, i32 } %[[EXN_INSERT]], i32 %[[EHSELECTOR_SLOT_LOAD]], 1
// OGCG:   resume { ptr, i32 } %[[EXN_INSERT_2]]
