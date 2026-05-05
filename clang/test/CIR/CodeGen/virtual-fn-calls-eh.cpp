// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -mconstructor-aliases -fcxx-exceptions -fexceptions -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: cir-opt --cir-flatten-cfg %t.cir -o %t-flat.cir
// RUN: FileCheck --input-file=%t-flat.cir %s --check-prefix=CIR-FLAT
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -mconstructor-aliases -fcxx-exceptions -fexceptions -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -mconstructor-aliases -fcxx-exceptions -fexceptions -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

struct B {
  ~B();
  virtual void f(char);
};

void call_virtual_fn_in_cleanup_scope() {
  B b;
  b.f('c');
}

// CIR: cir.func {{.*}} @_Z32call_virtual_fn_in_cleanup_scopev()
// CIR:   %[[B:.*]] = cir.alloca !rec_B, !cir.ptr<!rec_B>, ["b", init]
// CIR:   cir.call @_ZN1BC2Ev(%[[B]])
// CIR:   cir.cleanup.scope {
// CIR:     %[[C_LITERAL:.*]] = cir.const #cir.int<99> : !s8i
// CIR:     cir.call @_ZN1B1fEc(%[[B]], %[[C_LITERAL]]) : (!cir.ptr<!rec_B> {{.*}}, !s8i {{.*}}) -> ()
// CIR:     cir.yield
// CIR:   } cleanup  all {
// CIR:     cir.call @_ZN1BD1Ev(%[[B]]) nothrow : (!cir.ptr<!rec_B> {{.*}}) -> ()
// CIR:     cir.yield
// CIR:   }

// CIR-FLAT: cir.func {{.*}} @_Z32call_virtual_fn_in_cleanup_scopev()
// CIR-FLAT:   %[[B:.*]] = cir.alloca !rec_B, !cir.ptr<!rec_B>, ["b", init]
// CIR-FLAT:   cir.call @_ZN1BC2Ev(%[[B]]) nothrow : (!cir.ptr<!rec_B> {{.*}}) -> ()
// CIR-FLAT:   cir.br ^[[CLEANUP_SCOPE:bb[0-9]+]]
// CIR-FLAT: ^[[CLEANUP_SCOPE]]:
// CIR-FLAT:   %[[C_LITERAL:.*]] = cir.const #cir.int<99> : !s8i
// CIR-FLAT:   cir.try_call @_ZN1B1fEc(%[[B]], %[[C_LITERAL]]) ^[[NORMAL:bb[0-9]+]], ^[[UNWIND:bb[0-9]+]] : (!cir.ptr<!rec_B> {{.*}}, !s8i {{.*}}) -> ()
// CIR-FLAT: ^[[NORMAL]]:  // pred: ^bb1
// CIR-FLAT:   cir.br ^[[NORMAL_CLEANUP:bb[0-9]+]]
// CIR-FLAT: ^[[NORMAL_CLEANUP]]:
// CIR-FLAT:   cir.call @_ZN1BD1Ev(%[[B]]) nothrow : (!cir.ptr<!rec_B> {{.*}}) -> ()
// CIR-FLAT:   cir.br ^[[NORMAL_CONTINUE:bb[0-9]+]]
// CIR-FLAT: ^[[NORMAL_CONTINUE]]:
// CIR-FLAT:   cir.br ^[[TRY_CONTINUE:bb[0-9]+]]
// CIR-FLAT: ^[[UNWIND]]:  // pred: ^bb1
// CIR-FLAT:   %[[EH_TOKEN:.*]] = cir.eh.initiate cleanup : !cir.eh_token
// CIR-FLAT:   cir.br ^[[EH_CLEANUP:bb[0-9]+]](%[[EH_TOKEN]] : !cir.eh_token)
// CIR-FLAT: ^[[EH_CLEANUP]](%[[EH_TOKEN:.*]]: !cir.eh_token):
// CIR-FLAT:   %[[CT:.*]] = cir.begin_cleanup %[[EH_TOKEN]] : !cir.eh_token -> !cir.cleanup_token
// CIR-FLAT:   cir.call @_ZN1BD1Ev(%[[B]]) nothrow : (!cir.ptr<!rec_B> {{.*}}) -> ()
// CIR-FLAT:   cir.end_cleanup %[[CT]] : !cir.cleanup_token
// CIR-FLAT:   cir.resume %[[EH_TOKEN]] : !cir.eh_token
// CIR-FLAT: ^[[TRY_CONTINUE]]:
// CIR-FLAT:   cir.return

// LLVM: define {{.*}} void @_Z32call_virtual_fn_in_cleanup_scopev()
// LLVM:   %[[B:.*]] = alloca %struct.B
// LLVM:   call void @_ZN1BC2Ev(ptr {{.*}} %[[B]])
// LLVM:   br label %[[CLEANUP_SCOPE:.*]]
// LLVM: [[CLEANUP_SCOPE]]:
// LLVM:   invoke void @_ZN1B1fEc(ptr {{.*}} %[[B]], i8 noundef 99)
// LLVM:           to label %[[NORMAL_CONTINUE:.*]] unwind label %[[UNWIND:.*]]
// LLVM: [[NORMAL_CONTINUE]]
// LLVM:   br label %[[NORMAL_CLEANUP:.*]]
// LLVM: [[NORMAL_CLEANUP]]:
// LLVM:   call void @_ZN1BD1Ev(ptr {{.*}} %[[B]])
// LLVM:   br label %[[EXIT_CLEANUP_SCOPE:.*]]
// LLVM: [[EXIT_CLEANUP_SCOPE]]:
// LLVM:   br label %[[DONE:.*]]
// LLVM: [[UNWIND]]:
// LLVM:   %[[EXN:.*]] = landingpad { ptr, i32 }
// LLVM:                   cleanup
// LLVM:   %[[EXN_PTR:.*]] = extractvalue { ptr, i32 } %[[EXN]], 0
// LLVM:   %[[TYPEID:.*]] = extractvalue { ptr, i32 } %[[EXN]], 1
// LLVM:   br label %[[EH_CLEANUP:.*]]
// LLVM: [[EH_CLEANUP]]:
// LLVM:   %[[EXN_PTR_PHI:.*]] = phi ptr [ %[[EXN_PTR]], %[[UNWIND]] ]
// LLVM:   %[[TYPEID_PHI:.*]] = phi i32 [ %[[TYPEID]], %[[UNWIND]] ]
// LLVM:   call void @_ZN1BD1Ev(ptr {{.*}} %[[B]])
// LLVM:   %[[EXN_INSERT:.*]] = insertvalue { ptr, i32 } poison, ptr %[[EXN_PTR_PHI]], 0
// LLVM:   %[[EXN_INSERT_2:.*]] = insertvalue { ptr, i32 } %[[EXN_INSERT]], i32 %[[TYPEID_PHI]], 1
// LLVM:   resume { ptr, i32 } %[[EXN_INSERT_2]]
// LLVM: [[DONE]]:
// LLVM:   ret void

// OGCG: define {{.*}} void @_Z32call_virtual_fn_in_cleanup_scopev()
// OGCG:   %[[B:.*]] = alloca %struct.B, align 8
// OGCG:   %[[EXN_SLOT:.*]] = alloca ptr
// OGCG:   %[[EHSELECTOR_SLOT:.*]] = alloca i32
// OGCG:   call void @_ZN1BC2Ev(ptr {{.*}} %[[B]])
// OGCG:   invoke void @_ZN1B1fEc(ptr {{.*}} %[[B]], i8 {{.*}} 99)
// OGCG:           to label %[[INVOKE_CONT:.*]] unwind label %[[UNWIND:.*]]
// OGCG: [[INVOKE_CONT]]:
// OGCG:   call void @_ZN1BD1Ev(ptr {{.*}} %[[B]])
// OGCG:   ret void
// OGCG: [[UNWIND]]:
// OGCG:   %[[EXN:.*]] = landingpad { ptr, i32 }
// OGCG:           cleanup
// OGCG:   %[[EXN_PTR:.*]] = extractvalue { ptr, i32 } %[[EXN]], 0
// OGCG:   store ptr %[[EXN_PTR]], ptr %[[EXN_SLOT]]
// OGCG:   %[[TYPEID:.*]] = extractvalue { ptr, i32 } %[[EXN]], 1
// OGCG:   store i32 %[[TYPEID]], ptr %[[EHSELECTOR_SLOT]]
// OGCG:   call void @_ZN1BD1Ev(ptr {{.*}} %[[B]])
// OGCG:   br label %[[EH_RESUME:.*]]
// OGCG: [[EH_RESUME]]:
// OGCG:   %[[EXN:.*]] = load ptr, ptr %[[EXN_SLOT]]
// OGCG:   %[[SEL:.*]] = load i32, ptr %[[EHSELECTOR_SLOT]]
// OGCG:   %[[LPAD_VAL:.*]] = insertvalue { ptr, i32 } poison, ptr %[[EXN]], 0
// OGCG:   %[[LPAD_VAL_1:.*]] = insertvalue { ptr, i32 } %[[LPAD_VAL]], i32 %[[SEL]], 1
// OGCG:   resume { ptr, i32 } %[[LPAD_VAL_1]]
  