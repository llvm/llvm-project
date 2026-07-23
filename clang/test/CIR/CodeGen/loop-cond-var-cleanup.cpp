// RUN: %clang_cc1 -no-enable-noundef-analysis %s -triple=x86_64-linux-gnu -fclangir -emit-cir -std=c++17 -fcxx-exceptions -fexceptions -o %t.cir
// RUN: FileCheck -check-prefixes=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -no-enable-noundef-analysis %s -triple=x86_64-linux-gnu -fclangir -emit-llvm -std=c++17 -fcxx-exceptions -fexceptions -o %t-cir.ll
// RUN: FileCheck -check-prefixes=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -no-enable-noundef-analysis %s -triple=x86_64-linux-gnu -emit-llvm -std=c++17 -fcxx-exceptions -fexceptions -o %t.ll
// RUN: FileCheck -check-prefixes=OGCG --input-file=%t.ll %s

struct S {
  S();
  ~S();
  explicit operator bool();
};

void while_cond_var(int n) {
  while (S s{})
    --n;
}

// CIR-LABEL: cir.func {{.*}} @_Z14while_cond_vari
// CIR:     %[[FLAG:.*]] = cir.alloca {{.*}}"cond.cleanup.isactive"{{.*}} : !cir.ptr<!cir.bool>
// CIR:   cir.scope {
// CIR:     %[[S:.*]] = cir.alloca {{.*}}"s"{{.*}} : !cir.ptr<!rec_S>
// CIR:     cir.while {
// The cleanup active flag is cleared before the initializer and set once the
// condition variable is constructed, so we don't call the destructor if the
// variable wasn't successfully constructed.
// CIR:       %[[FALSE:.*]] = cir.const #false
// CIR:       cir.store %[[FALSE]], %[[FLAG]]
// CIR:       cir.call @_ZN1SC1Ev(%[[S]])
// CIR:       %[[TRUE:.*]] = cir.const #true
// CIR:       cir.store %[[TRUE]], %[[FLAG]]
// CIR:       %[[COND:.*]] = cir.call @_ZN1ScvbEv(%[[S]])
// CIR:       cir.condition(%[[COND]])
// CIR:     } do {
// CIR:     } cleanup all {
// CIR:       %[[ACTIVE:.*]] = cir.load {{.*}}%[[FLAG]]
// CIR:       cir.if %[[ACTIVE]] {
// CIR:         cir.call @_ZN1SD1Ev(%[[S]]) nothrow
// CIR:       }
// CIR:       cir.yield
// CIR:     }

// LLVM-LABEL: define dso_local void @_Z14while_cond_vari(i32 %0) {{.*}} personality ptr @__gxx_personality_v0 {
// LLVM:         %[[S:.*]] = alloca %struct.S
// LLVM:         %[[FLAG:.*]] = alloca i8
// The loop header branches into the condition region or to the loop exit.
// LLVM:         br label %{{.*}}
// LLVM:         br label %[[COND:.*]]
// LLVM:       [[COND]]:
// LLVM:         br i1 true, label %{{.*}}, label %[[EXIT:.*]]
// LLVM:         store i8 0, ptr %[[FLAG]]
// Unlike OGCG, the LLVM path emits an invoke here because our cleanup scope is
// the entire condition region. The active flag is used to skip the cleanup if
// we throw from here.
// LLVM:         invoke void @_ZN1SC1Ev(ptr {{.*}} %[[S]])
// LLVM-NEXT:            to label %[[CTOR_CONT:.*]] unwind label %[[LPAD:.*]]
// LLVM:       [[CTOR_CONT]]:
// LLVM:         store i8 1, ptr %[[FLAG]]
// LLVM:         %[[CALL:.*]] = invoke i1 @_ZN1ScvbEv(ptr {{.*}} %[[S]])
// LLVM-NEXT:            to label %[[CVB_CONT:.*]] unwind label %[[LPAD]]
// LLVM:       [[CVB_CONT]]:
// LLVM:         br i1 %[[CALL]], label %[[BODY:.*]], label %[[COND_FALSE:.*]]
// The body sets the cleanup destination selector to the "continue" value and
// the false (loop-exit) edge sets it to the "break" value; both then fall
// through to the shared cleanup.
// LLVM:       [[BODY]]:
// LLVM:         store i32 0, ptr %[[DEST:.*]],
// LLVM:         br label %[[CLEANUP:.*]]
// LLVM:       [[COND_FALSE]]:
// LLVM:         store i32 1, ptr %[[DEST]]
// LLVM:         br label %[[CLEANUP]]
// The cleanup tests the active flag, destroys the condition variable when set,
// then dispatches on the destination selector to the loop back-edge (continue)
// or the loop exit (break).
// LLVM:       [[CLEANUP]]:
// LLVM:         %[[ACT:.*]] = load i8, ptr %[[FLAG]]
// LLVM:         %[[ACTB:.*]] = trunc i8 %[[ACT]] to i1
// LLVM:         br i1 %[[ACTB]], label %[[DTOR:.*]], label %{{.*}}
// LLVM:       [[DTOR]]:
// LLVM:         call void @_ZN1SD1Ev(ptr {{.*}} %[[S]])
// LLVM:         %[[SEL:.*]] = load i32, ptr %[[DEST]]
// LLVM:         switch i32 %[[SEL]], label %{{.*}}
// LLVM:           i32 0, label %[[CONTINUE:.*]]
// LLVM:           i32 1, label %[[BREAK:.*]]
// LLVM:       [[CONTINUE]]:
// LLVM:         br label %[[BACKEDGE:.*]]
// LLVM:       [[BREAK]]:
// LLVM:         br label %[[EXIT]]
// LLVM:       [[LPAD]]:
// LLVM:         landingpad { ptr, i32 }
// LLVM-NEXT:            cleanup
// LLVM:         %[[EACT:.*]] = load i8, ptr %[[FLAG]]
// LLVM:         %[[EACTB:.*]] = trunc i8 %[[EACT]] to i1
// LLVM:         br i1 %[[EACTB]], label %[[EHDTOR:.*]], label %{{.*}}
// LLVM:       [[EHDTOR]]:
// LLVM:         call void @_ZN1SD1Ev(ptr {{.*}} %[[S]])
// LLVM:         resume { ptr, i32 }
// LLVM:       [[BACKEDGE]]:
// LLVM:         br label %[[COND]]

// OGCG-LABEL: define dso_local void @_Z14while_cond_vari(i32 %n) {{.*}} personality ptr @__gxx_personality_v0 {
// OGCG:         %[[S:.*]] = alloca %struct.S
// OGCG:         br label %[[COND:.*]]
// OGCG:       [[COND]]:
// OGCG:         call void @_ZN1SC1Ev(ptr {{.*}} %[[S]])
// OGCG:         %[[CALL:.*]] = invoke {{.*}} i1 @_ZN1ScvbEv(ptr {{.*}} %[[S]])
// OGCG-NEXT:            to label %[[CONT:.*]] unwind label %[[LPAD:.*]]
// OGCG:       [[CONT]]:
// OGCG:         br i1 %[[CALL]], label %[[BODY:.*]], label %[[EXIT:.*]]
// OGCG:       [[EXIT]]:
// OGCG:         store i32 3, ptr %[[DEST:.*]],
// OGCG:         br label %[[CLEANUP:.*]]
// OGCG:       [[LPAD]]:
// OGCG:         landingpad { ptr, i32 }
// OGCG-NEXT:            cleanup
// OGCG:         call void @_ZN1SD1Ev(ptr {{.*}} %[[S]])
// OGCG:         br label %[[EHRESUME:.*]]
// OGCG:       [[BODY]]:
// OGCG:         store i32 0, ptr %[[DEST]]
// OGCG:         br label %[[CLEANUP]]
// OGCG:       [[CLEANUP]]:
// OGCG:         call void @_ZN1SD1Ev(ptr {{.*}} %[[S]])
// OGCG:         %[[SEL:.*]] = load i32, ptr %[[DEST]]
// OGCG:         switch i32 %[[SEL]], label %{{.*}}
// OGCG:           i32 0, label %[[CONTINUE:.*]]
// OGCG:           i32 3, label %[[END:.*]]
// OGCG:       [[CONTINUE]]:
// OGCG:         br label %[[COND]]
// OGCG:       [[END]]:
// OGCG:         ret void
// OGCG:       [[EHRESUME]]:
// OGCG:         resume { ptr, i32 }

void while_cond_var_break() {
  while (S s{}) {
    break;
  }
}

// CIR-LABEL: cir.func {{.*}} @_Z20while_cond_var_breakv
// CIR:     %[[FLAG:.*]] = cir.alloca {{.*}}"cond.cleanup.isactive"{{.*}} : !cir.ptr<!cir.bool>
// CIR:   cir.scope {
// CIR:     %[[S:.*]] = cir.alloca {{.*}}"s"{{.*}} : !cir.ptr<!rec_S>
// CIR:     cir.while {
// CIR:       %[[FALSE:.*]] = cir.const #false
// CIR:       cir.store %[[FALSE]], %[[FLAG]]
// CIR:       cir.call @_ZN1SC1Ev(%[[S]])
// CIR:       %[[TRUE:.*]] = cir.const #true
// CIR:       cir.store %[[TRUE]], %[[FLAG]]
// CIR:       %[[COND:.*]] = cir.call @_ZN1ScvbEv(%[[S]])
// CIR-NOT:   cir.cleanup.scope
// CIR:       cir.condition(%[[COND]])
// CIR:     } do {
// CIR:       cir.break
// CIR:     } cleanup all {
// CIR:       %[[ACTIVE:.*]] = cir.load {{.*}}%[[FLAG]]
// CIR:       cir.if %[[ACTIVE]] {
// CIR:         cir.call @_ZN1SD1Ev(%[[S]]) nothrow
// CIR:       }
// CIR:       cir.yield
// CIR:     }

// LLVM-LABEL: define dso_local void @_Z20while_cond_var_breakv() {{.*}} personality ptr @__gxx_personality_v0 {
// LLVM:         %[[S:.*]] = alloca %struct.S
// LLVM:         %[[FLAG:.*]] = alloca i8
// LLVM:         br label %{{.*}}
// LLVM:         br label %[[COND:.*]]
// LLVM:       [[COND]]:
// LLVM:         br i1 true, label %{{.*}}, label %[[EXIT:.*]]
// LLVM:         store i8 0, ptr %[[FLAG]]
// LLVM:         invoke void @_ZN1SC1Ev(ptr {{.*}} %[[S]])
// LLVM-NEXT:            to label %[[CTOR_CONT:.*]] unwind label %[[LPAD:.*]]
// LLVM:       [[CTOR_CONT]]:
// LLVM:         store i8 1, ptr %[[FLAG]]
// LLVM:         %[[CALL:.*]] = invoke i1 @_ZN1ScvbEv(ptr {{.*}} %[[S]])
// LLVM-NEXT:            to label %[[CVB_CONT:.*]] unwind label %[[LPAD]]
// LLVM:       [[CVB_CONT]]:
// LLVM:         br i1 %[[CALL]], label %[[BODY:.*]], label %[[COND_FALSE:.*]]
// LLVM:       [[BODY]]:
// LLVM:         store i32 1, ptr %[[DEST:.*]], align 4
// LLVM:         br label %[[CLEANUP:.*]]
// LLVM:       [[COND_FALSE]]:
// LLVM:         store i32 2, ptr %[[DEST]], align 4
// LLVM:         br label %[[CLEANUP]]
// LLVM:       [[CLEANUP]]:
// LLVM:         %[[ACT:.*]] = load i8, ptr %[[FLAG]]
// LLVM:         %[[ACTB:.*]] = trunc i8 %[[ACT]] to i1
// LLVM:         br i1 %[[ACTB]], label %[[DTOR:.*]], label %{{.*}}
// LLVM:       [[DTOR]]:
// LLVM:         call void @_ZN1SD1Ev(ptr {{.*}} %[[S]])
// LLVM:         %[[SEL:.*]] = load i32, ptr %[[DEST]], align 4
// LLVM:         switch i32 %[[SEL]], label %{{.*}}
// LLVM:           i32 0, label %[[CONTINUE:.*]]
// LLVM:           i32 1, label %[[BREAK:.*]]
// LLVM:       [[BREAK]]:
// LLVM:         br label %[[EXIT]]
// LLVM:       [[LPAD]]:
// LLVM:         landingpad { ptr, i32 }
// LLVM-NEXT:            cleanup
// LLVM:         %[[EACT:.*]] = load i8, ptr %[[FLAG]]
// LLVM:         %[[EACTB:.*]] = trunc i8 %[[EACT]] to i1
// LLVM:         br i1 %[[EACTB]], label %[[EHDTOR:.*]], label %{{.*}}
// LLVM:       [[EHDTOR]]:
// LLVM:         call void @_ZN1SD1Ev(ptr {{.*}} %[[S]])
// LLVM:         resume { ptr, i32 }

// OGCG-LABEL: define dso_local void @_Z20while_cond_var_breakv() {{.*}} personality ptr @__gxx_personality_v0 {
// OGCG:         %[[S:.*]] = alloca %struct.S
// OGCG:         br label %[[COND:.*]]
// OGCG:       [[COND]]:
// OGCG:         call void @_ZN1SC1Ev(ptr {{.*}} %[[S]])
// OGCG:         %[[CALL:.*]] = invoke {{.*}} i1 @_ZN1ScvbEv(ptr {{.*}} %[[S]])
// OGCG-NEXT:            to label %[[CONT:.*]] unwind label %[[LPAD:.*]]
// OGCG:       [[CONT]]:
// OGCG:         br i1 %[[CALL]], label %[[BODY:.*]], label %[[COND_FALSE:.*]]
// OGCG:       [[COND_FALSE]]:
// OGCG:         store i32 3, ptr %[[DEST:.*]], align 4
// OGCG:         br label %[[CLEANUP:.*]]
// OGCG:       [[LPAD]]:
// OGCG:         landingpad { ptr, i32 }
// OGCG-NEXT:            cleanup
// OGCG:         call void @_ZN1SD1Ev(ptr {{.*}} %[[S]])
// OGCG:         br label %[[EHRESUME:.*]]
// OGCG:       [[BODY]]:
// OGCG:         store i32 3, ptr %[[DEST]], align 4
// OGCG:         br label %[[CLEANUP]]
// OGCG:       [[CLEANUP]]:
// OGCG:         call void @_ZN1SD1Ev(ptr {{.*}} %[[S]])
// OGCG:         br label %[[END:.*]]
// OGCG:       [[END]]:
// OGCG:         ret void
// OGCG:       [[EHRESUME]]:
// OGCG:         resume { ptr, i32 }

void while_cond_var_continue() {
  while (S s{}) {
    continue;
  }
}

// CIR-LABEL: cir.func {{.*}} @_Z23while_cond_var_continuev
// CIR:     %[[FLAG:.*]] = cir.alloca {{.*}}"cond.cleanup.isactive"{{.*}} : !cir.ptr<!cir.bool>
// CIR:   cir.scope {
// CIR:     %[[S:.*]] = cir.alloca {{.*}}"s"{{.*}} : !cir.ptr<!rec_S>
// CIR:     cir.while {
// CIR:       %[[FALSE:.*]] = cir.const #false
// CIR:       cir.store %[[FALSE]], %[[FLAG]]
// CIR:       cir.call @_ZN1SC1Ev(%[[S]])
// CIR:       %[[TRUE:.*]] = cir.const #true
// CIR:       cir.store %[[TRUE]], %[[FLAG]]
// CIR:       %[[COND:.*]] = cir.call @_ZN1ScvbEv(%[[S]])
// CIR-NOT:   cir.cleanup.scope
// CIR:       cir.condition(%[[COND]])
// CIR:     } do {
// CIR:       cir.continue
// CIR:     } cleanup all {
// CIR:       %[[ACTIVE:.*]] = cir.load {{.*}}%[[FLAG]]
// CIR:       cir.if %[[ACTIVE]] {
// CIR:         cir.call @_ZN1SD1Ev(%[[S]]) nothrow
// CIR:       }
// CIR:       cir.yield
// CIR:     }

// LLVM-LABEL: define dso_local void @_Z23while_cond_var_continuev() {{.*}} personality ptr @__gxx_personality_v0 {
// LLVM:         %[[S:.*]] = alloca %struct.S
// LLVM:         %[[FLAG:.*]] = alloca i8
// LLVM:         br label %{{.*}}
// LLVM:         br label %[[COND:.*]]
// LLVM:       [[COND]]:
// LLVM:         br i1 true, label %{{.*}}, label %[[EXIT:.*]]
// LLVM:         store i8 0, ptr %[[FLAG]]
// LLVM:         invoke void @_ZN1SC1Ev(ptr {{.*}} %[[S]])
// LLVM-NEXT:            to label %[[CTOR_CONT:.*]] unwind label %[[LPAD:.*]]
// LLVM:       [[CTOR_CONT]]:
// LLVM:         store i8 1, ptr %[[FLAG]]
// LLVM:         %[[CALL:.*]] = invoke i1 @_ZN1ScvbEv(ptr {{.*}} %[[S]])
// LLVM-NEXT:            to label %[[CVB_CONT:.*]] unwind label %[[LPAD]]
// LLVM:       [[CVB_CONT]]:
// LLVM:         br i1 %[[CALL]], label %[[BODY:.*]], label %[[COND_FALSE:.*]]
// LLVM:       [[BODY]]:
// LLVM:         store i32 1, ptr %[[DEST:.*]], align 4
// LLVM:         br label %[[CLEANUP:.*]]
// LLVM:       [[COND_FALSE]]:
// LLVM:         store i32 2, ptr %[[DEST]], align 4
// LLVM:         br label %[[CLEANUP]]
// LLVM:       [[CLEANUP]]:
// LLVM:         %[[ACT:.*]] = load i8, ptr %[[FLAG]]
// LLVM:         %[[ACTB:.*]] = trunc i8 %[[ACT]] to i1
// LLVM:         br i1 %[[ACTB]], label %[[DTOR:.*]], label %{{.*}}
// LLVM:       [[DTOR]]:
// LLVM:         call void @_ZN1SD1Ev(ptr {{.*}} %[[S]])
// LLVM:         %[[SEL:.*]] = load i32, ptr %[[DEST]], align 4
// LLVM:         switch i32 %[[SEL]], label %{{.*}}
// LLVM:           i32 1, label %[[CONTINUE:.*]]
// LLVM:           i32 2, label %[[BREAK:.*]]
// LLVM:       [[CONTINUE]]:
// LLVM:         br label %[[COND]]
// LLVM:       [[BREAK]]:
// LLVM:         br label %[[EXIT]]
// LLVM:       [[LPAD]]:
// LLVM:         landingpad { ptr, i32 }
// LLVM-NEXT:            cleanup
// LLVM:         %[[EACT:.*]] = load i8, ptr %[[FLAG]]
// LLVM:         %[[EACTB:.*]] = trunc i8 %[[EACT]] to i1
// LLVM:         br i1 %[[EACTB]], label %[[EHDTOR:.*]], label %{{.*}}
// LLVM:       [[EHDTOR]]:
// LLVM:         call void @_ZN1SD1Ev(ptr {{.*}} %[[S]])
// LLVM:         resume { ptr, i32 }

// OGCG-LABEL: define dso_local void @_Z23while_cond_var_continuev() {{.*}} personality ptr @__gxx_personality_v0 {
// OGCG:         %[[S:.*]] = alloca %struct.S
// OGCG:         br label %[[COND:.*]]
// OGCG:       [[COND]]:
// OGCG:         call void @_ZN1SC1Ev(ptr {{.*}} %[[S]])
// OGCG:         %[[CALL:.*]] = invoke {{.*}} i1 @_ZN1ScvbEv(ptr {{.*}} %[[S]])
// OGCG-NEXT:            to label %[[CONT:.*]] unwind label %[[LPAD:.*]]
// OGCG:       [[CONT]]:
// OGCG:         br i1 %[[CALL]], label %[[BODY:.*]], label %[[COND_FALSE:.*]]
// OGCG:       [[COND_FALSE]]:
// OGCG:         store i32 3, ptr %[[DEST:.*]], align 4
// OGCG:         br label %[[CLEANUP:.*]]
// OGCG:       [[LPAD]]:
// OGCG:         landingpad { ptr, i32 }
// OGCG-NEXT:            cleanup
// OGCG:         call void @_ZN1SD1Ev(ptr {{.*}} %[[S]])
// OGCG:         br label %[[EHRESUME:.*]]
// OGCG:       [[BODY]]:
// OGCG:         store i32 2, ptr %[[DEST]], align 4
// OGCG:         br label %[[CLEANUP]]
// OGCG:       [[CLEANUP]]:
// OGCG:         call void @_ZN1SD1Ev(ptr {{.*}} %[[S]])
// OGCG:         %[[SEL:.*]] = load i32, ptr %[[DEST]], align 4
// OGCG:         switch i32 %[[SEL]], label %{{.*}}
// OGCG:           i32 3, label %[[END:.*]]
// OGCG:           i32 2, label %[[COND]]
// OGCG:       [[END]]:
// OGCG:         ret void
// OGCG:       [[EHRESUME]]:
// OGCG:         resume { ptr, i32 }

void for_cond_var_break() {
  for (; S s{};) {
    break;
  }
}

// CIR-LABEL: cir.func {{.*}} @_Z18for_cond_var_breakv
// CIR:     %[[FLAG:.*]] = cir.alloca {{.*}}"cond.cleanup.isactive"{{.*}} : !cir.ptr<!cir.bool>
// CIR:   cir.scope {
// CIR:     %[[S:.*]] = cir.alloca {{.*}}"s"{{.*}} : !cir.ptr<!rec_S>
// CIR:     cir.for : cond {
// CIR:       %[[FALSE:.*]] = cir.const #false
// CIR:       cir.store %[[FALSE]], %[[FLAG]]
// CIR:       cir.call @_ZN1SC1Ev(%[[S]])
// CIR:       %[[TRUE:.*]] = cir.const #true
// CIR:       cir.store %[[TRUE]], %[[FLAG]]
// CIR:       %[[COND:.*]] = cir.call @_ZN1ScvbEv(%[[S]])
// CIR-NOT:   cir.cleanup.scope
// CIR:       cir.condition(%[[COND]])
// CIR:     } body {
// CIR:       cir.break
// CIR:     } step {
// CIR:     } cleanup all {
// CIR:       %[[ACTIVE:.*]] = cir.load {{.*}}%[[FLAG]]
// CIR:       cir.if %[[ACTIVE]] {
// CIR:         cir.call @_ZN1SD1Ev(%[[S]]) nothrow
// CIR:       }
// CIR:       cir.yield
// CIR:     }

// LLVM-LABEL: define dso_local void @_Z18for_cond_var_breakv() {{.*}} personality ptr @__gxx_personality_v0 {
// LLVM:         %[[S:.*]] = alloca %struct.S
// LLVM:         %[[FLAG:.*]] = alloca i8
// LLVM:         br label %{{.*}}
// LLVM:         br label %[[COND:.*]]
// LLVM:       [[COND]]:
// LLVM:         br i1 true, label %{{.*}}, label %[[EXIT:.*]]
// LLVM:         store i8 0, ptr %[[FLAG]]
// LLVM:         invoke void @_ZN1SC1Ev(ptr {{.*}} %[[S]])
// LLVM-NEXT:            to label %[[CTOR_CONT:.*]] unwind label %[[LPAD:.*]]
// LLVM:       [[CTOR_CONT]]:
// LLVM:         store i8 1, ptr %[[FLAG]]
// LLVM:         %[[CALL:.*]] = invoke i1 @_ZN1ScvbEv(ptr {{.*}} %[[S]])
// LLVM-NEXT:            to label %[[CVB_CONT:.*]] unwind label %[[LPAD]]
// LLVM:       [[CVB_CONT]]:
// LLVM:         br i1 %[[CALL]], label %[[BODY:.*]], label %[[COND_FALSE:.*]]
// LLVM:       [[BODY]]:
// LLVM:         store i32 1, ptr %[[DEST:.*]], align 4
// LLVM:         br label %[[CLEANUP:.*]]
// LLVM:       [[COND_FALSE]]:
// LLVM:         store i32 2, ptr %[[DEST]], align 4
// LLVM:         br label %[[CLEANUP]]
// LLVM:       [[CLEANUP]]:
// LLVM:         %[[ACT:.*]] = load i8, ptr %[[FLAG]]
// LLVM:         %[[ACTB:.*]] = trunc i8 %[[ACT]] to i1
// LLVM:         br i1 %[[ACTB]], label %[[DTOR:.*]], label %{{.*}}
// LLVM:       [[DTOR]]:
// LLVM:         call void @_ZN1SD1Ev(ptr {{.*}} %[[S]])
// LLVM:         %[[SEL:.*]] = load i32, ptr %[[DEST]], align 4
// LLVM:         switch i32 %[[SEL]], label %{{.*}}
// LLVM:           i32 0, label %[[CONTINUE:.*]]
// LLVM:           i32 1, label %[[BREAK:.*]]
// LLVM:       [[BREAK]]:
// LLVM:         br label %[[EXIT]]
// LLVM:       [[LPAD]]:
// LLVM:         landingpad { ptr, i32 }
// LLVM-NEXT:            cleanup
// LLVM:         %[[EACT:.*]] = load i8, ptr %[[FLAG]]
// LLVM:         %[[EACTB:.*]] = trunc i8 %[[EACT]] to i1
// LLVM:         br i1 %[[EACTB]], label %[[EHDTOR:.*]], label %{{.*}}
// LLVM:       [[EHDTOR]]:
// LLVM:         call void @_ZN1SD1Ev(ptr {{.*}} %[[S]])
// LLVM:         resume { ptr, i32 }

// OGCG-LABEL: define dso_local void @_Z18for_cond_var_breakv() {{.*}} personality ptr @__gxx_personality_v0 {
// OGCG:         %[[S:.*]] = alloca %struct.S
// OGCG:         br label %[[COND:.*]]
// OGCG:       [[COND]]:
// OGCG:         call void @_ZN1SC1Ev(ptr {{.*}} %[[S]])
// OGCG:         %[[CALL:.*]] = invoke {{.*}} i1 @_ZN1ScvbEv(ptr {{.*}} %[[S]])
// OGCG-NEXT:            to label %[[CONT:.*]] unwind label %[[LPAD:.*]]
// OGCG:       [[CONT]]:
// OGCG:         br i1 %[[CALL]], label %[[BODY:.*]], label %[[COND_FALSE:.*]]
// OGCG:       [[COND_FALSE]]:
// OGCG:         store i32 2, ptr %[[DEST:.*]], align 4
// OGCG:         br label %[[CLEANUP:.*]]
// OGCG:       [[LPAD]]:
// OGCG:         landingpad { ptr, i32 }
// OGCG-NEXT:            cleanup
// OGCG:         call void @_ZN1SD1Ev(ptr {{.*}} %[[S]])
// OGCG:         br label %[[EHRESUME:.*]]
// OGCG:       [[BODY]]:
// OGCG:         store i32 2, ptr %[[DEST]], align 4
// OGCG:         br label %[[CLEANUP]]
// OGCG:       [[CLEANUP]]:
// OGCG:         call void @_ZN1SD1Ev(ptr {{.*}} %[[S]])
// OGCG:         br label %[[END:.*]]
// OGCG:       [[END]]:
// OGCG:         ret void
// OGCG:       [[EHRESUME]]:
// OGCG:         resume { ptr, i32 }

void for_cond_var_continue() {
  for (; S s{};) {
    continue;
  }
}

// CIR-LABEL: cir.func {{.*}} @_Z21for_cond_var_continuev
// CIR:     %[[FLAG:.*]] = cir.alloca {{.*}}"cond.cleanup.isactive"{{.*}} : !cir.ptr<!cir.bool>
// CIR:   cir.scope {
// CIR:     %[[S:.*]] = cir.alloca {{.*}}"s"{{.*}} : !cir.ptr<!rec_S>
// CIR:     cir.for : cond {
// CIR:       %[[FALSE:.*]] = cir.const #false
// CIR:       cir.store %[[FALSE]], %[[FLAG]]
// CIR:       cir.call @_ZN1SC1Ev(%[[S]])
// CIR:       %[[TRUE:.*]] = cir.const #true
// CIR:       cir.store %[[TRUE]], %[[FLAG]]
// CIR:       %[[COND:.*]] = cir.call @_ZN1ScvbEv(%[[S]])
// CIR-NOT:   cir.cleanup.scope
// CIR:       cir.condition(%[[COND]])
// CIR:     } body {
// CIR:       cir.continue
// CIR:     } step {
// CIR:     } cleanup all {
// CIR:       %[[ACTIVE:.*]] = cir.load {{.*}}%[[FLAG]]
// CIR:       cir.if %[[ACTIVE]] {
// CIR:         cir.call @_ZN1SD1Ev(%[[S]]) nothrow
// CIR:       }
// CIR:       cir.yield
// CIR:     }

// LLVM-LABEL: define dso_local void @_Z21for_cond_var_continuev() {{.*}} personality ptr @__gxx_personality_v0 {
// LLVM:         %[[S:.*]] = alloca %struct.S
// LLVM:         %[[FLAG:.*]] = alloca i8
// LLVM:         br label %{{.*}}
// LLVM:         br label %[[COND:.*]]
// LLVM:       [[COND]]:
// LLVM:         br i1 true, label %{{.*}}, label %[[EXIT:.*]]
// LLVM:         store i8 0, ptr %[[FLAG]]
// LLVM:         invoke void @_ZN1SC1Ev(ptr {{.*}} %[[S]])
// LLVM-NEXT:            to label %[[CTOR_CONT:.*]] unwind label %[[LPAD:.*]]
// LLVM:       [[CTOR_CONT]]:
// LLVM:         store i8 1, ptr %[[FLAG]]
// LLVM:         %[[CALL:.*]] = invoke i1 @_ZN1ScvbEv(ptr {{.*}} %[[S]])
// LLVM-NEXT:            to label %[[CVB_CONT:.*]] unwind label %[[LPAD]]
// LLVM:       [[CVB_CONT]]:
// LLVM:         br i1 %[[CALL]], label %[[BODY:.*]], label %[[COND_FALSE:.*]]
// LLVM:       [[BODY]]:
// LLVM:         br label %[[STEP:.*]]
// LLVM:       [[STEP]]:
// LLVM:         store i32 0, ptr %[[DEST:.*]], align 4
// LLVM:         br label %[[CLEANUP:.*]]
// LLVM:       [[COND_FALSE]]:
// LLVM:         store i32 1, ptr %[[DEST]], align 4
// LLVM:         br label %[[CLEANUP]]
// LLVM:       [[CLEANUP]]:
// LLVM:         %[[ACT:.*]] = load i8, ptr %[[FLAG]]
// LLVM:         %[[ACTB:.*]] = trunc i8 %[[ACT]] to i1
// LLVM:         br i1 %[[ACTB]], label %[[DTOR:.*]], label %{{.*}}
// LLVM:       [[DTOR]]:
// LLVM:         call void @_ZN1SD1Ev(ptr {{.*}} %[[S]])
// LLVM:         %[[SEL:.*]] = load i32, ptr %[[DEST]], align 4
// LLVM:         switch i32 %[[SEL]], label %{{.*}}
// LLVM:           i32 0, label %[[CONTINUE:.*]]
// LLVM:           i32 1, label %[[BREAK:.*]]
// LLVM:       [[CONTINUE]]:
// LLVM:         br label %[[BACKEDGE0:.*]]
// LLVM:       [[BREAK]]:
// LLVM:         br label %[[EXIT]]
// LLVM:       [[LPAD]]:
// LLVM:         landingpad { ptr, i32 }
// LLVM-NEXT:            cleanup
// LLVM:         %[[EACT:.*]] = load i8, ptr %[[FLAG]]
// LLVM:         %[[EACTB:.*]] = trunc i8 %[[EACT]] to i1
// LLVM:         br i1 %[[EACTB]], label %[[EHDTOR:.*]], label %{{.*}}
// LLVM:       [[EHDTOR]]:
// LLVM:         call void @_ZN1SD1Ev(ptr {{.*}} %[[S]])
// LLVM:         resume { ptr, i32 }
// LLVM:       [[BACKEDGE0]]:
// LLVM:         br label %[[BACKEDGE1:.*]]
// LLVM:       [[BACKEDGE1]]:
// LLVM:         br label %[[COND]]

// OGCG-LABEL: define dso_local void @_Z21for_cond_var_continuev() {{.*}} personality ptr @__gxx_personality_v0 {
// OGCG:         %[[S:.*]] = alloca %struct.S
// OGCG:         br label %[[COND:.*]]
// OGCG:       [[COND]]:
// OGCG:         call void @_ZN1SC1Ev(ptr {{.*}} %[[S]])
// OGCG:         %[[CALL:.*]] = invoke {{.*}} i1 @_ZN1ScvbEv(ptr {{.*}} %[[S]])
// OGCG-NEXT:            to label %[[CONT:.*]] unwind label %[[LPAD:.*]]
// OGCG:       [[CONT]]:
// OGCG:         br i1 %[[CALL]], label %[[BODY:.*]], label %[[COND_FALSE:.*]]
// OGCG:       [[COND_FALSE]]:
// OGCG:         store i32 2, ptr %[[DEST:.*]], align 4
// OGCG:         br label %[[CLEANUP:.*]]
// OGCG:       [[LPAD]]:
// OGCG:         landingpad { ptr, i32 }
// OGCG-NEXT:            cleanup
// OGCG:         call void @_ZN1SD1Ev(ptr {{.*}} %[[S]])
// OGCG:         br label %[[EHRESUME:.*]]
// OGCG:       [[BODY]]:
// OGCG:         store i32 3, ptr %[[DEST]], align 4
// OGCG:         br label %[[CLEANUP]]
// OGCG:       [[CLEANUP]]:
// OGCG:         call void @_ZN1SD1Ev(ptr {{.*}} %[[S]])
// OGCG:         %[[SEL:.*]] = load i32, ptr %[[DEST]], align 4
// OGCG:         switch i32 %[[SEL]], label %{{.*}}
// OGCG:           i32 2, label %[[END:.*]]
// OGCG:           i32 3, label %[[COND]]
// OGCG:       [[END]]:
// OGCG:         ret void
// OGCG:       [[EHRESUME]]:
// OGCG:         resume { ptr, i32 }

void for_cond_var(int n) {
  for (int i = 0; S s{}; ++i)
    --n;
}

// CIR-LABEL: cir.func {{.*}} @_Z12for_cond_vari
// CIR:     %[[FLAG:.*]] = cir.alloca {{.*}}"cond.cleanup.isactive"{{.*}} : !cir.ptr<!cir.bool>
// CIR:   cir.scope {
// CIR:     %[[S:.*]] = cir.alloca {{.*}}"s"{{.*}} : !cir.ptr<!rec_S>
// CIR:     cir.for : cond {
// The active flag is cleared before the initializer and set once the condition
// variable is constructed.
// CIR:       %[[FALSE:.*]] = cir.const #false
// CIR:       cir.store %[[FALSE]], %[[FLAG]]
// CIR:       cir.call @_ZN1SC1Ev(%[[S]])
// CIR:       %[[TRUE:.*]] = cir.const #true
// CIR:       cir.store %[[TRUE]], %[[FLAG]]
// CIR:       %[[COND:.*]] = cir.call @_ZN1ScvbEv(%[[S]])
// CIR-NOT:   cir.cleanup.scope
// CIR:       cir.condition(%[[COND]])
// CIR:     } body {
// CIR:     } step {
// The per-iteration destructor is guarded by the active flag.
// CIR:     } cleanup all {
// CIR:       %[[ACTIVE:.*]] = cir.load {{.*}}%[[FLAG]]
// CIR:       cir.if %[[ACTIVE]] {
// CIR:         cir.call @_ZN1SD1Ev(%[[S]]) nothrow
// CIR:       }
// CIR:       cir.yield
// CIR:     }

// LLVM-LABEL: define dso_local void @_Z12for_cond_vari(i32 %0) {{.*}} personality ptr @__gxx_personality_v0 {
// LLVM:         %[[S:.*]] = alloca %struct.S
// LLVM:         %[[FLAG:.*]] = alloca i8
// LLVM:         br label %{{.*}}
// LLVM:         br label %[[COND:.*]]
// LLVM:       [[COND]]:
// LLVM:         br i1 true, label %{{.*}}, label %[[EXIT:.*]]
// LLVM:         store i8 0, ptr %[[FLAG]]
// LLVM:         invoke void @_ZN1SC1Ev(ptr {{.*}} %[[S]])
// LLVM-NEXT:            to label %[[CTOR_CONT:.*]] unwind label %[[LPAD:.*]]
// LLVM:       [[CTOR_CONT]]:
// LLVM:         store i8 1, ptr %[[FLAG]]
// LLVM:         %[[CALL:.*]] = invoke i1 @_ZN1ScvbEv(ptr {{.*}} %[[S]])
// LLVM-NEXT:            to label %[[CVB_CONT:.*]] unwind label %[[LPAD]]
// LLVM:       [[CVB_CONT]]:
// LLVM:         br i1 %[[CALL]], label %[[BODY:.*]], label %[[COND_FALSE:.*]]
// LLVM:       [[BODY]]:
// LLVM:         br label %[[STEP:.*]]
// LLVM:       [[STEP]]:
// LLVM:         store i32 0, ptr %[[DEST:.*]], align 4
// LLVM:         br label %[[CLEANUP:.*]]
// LLVM:       [[COND_FALSE]]:
// LLVM:         store i32 1, ptr %[[DEST]], align 4
// LLVM:         br label %[[CLEANUP]]
// LLVM:       [[CLEANUP]]:
// LLVM:         %[[ACT:.*]] = load i8, ptr %[[FLAG]]
// LLVM:         %[[ACTB:.*]] = trunc i8 %[[ACT]] to i1
// LLVM:         br i1 %[[ACTB]], label %[[DTOR:.*]], label %{{.*}}
// LLVM:       [[DTOR]]:
// LLVM:         call void @_ZN1SD1Ev(ptr {{.*}} %[[S]])
// LLVM:         %[[SEL:.*]] = load i32, ptr %[[DEST]], align 4
// LLVM:         switch i32 %[[SEL]], label %{{.*}}
// LLVM:           i32 0, label %[[CONTINUE:.*]]
// LLVM:           i32 1, label %[[BREAK:.*]]
// LLVM:       [[CONTINUE]]:
// LLVM:         br label %[[BACKEDGE0:.*]]
// LLVM:       [[BREAK]]:
// LLVM:         br label %[[EXIT]]
// LLVM:       [[LPAD]]:
// LLVM:         landingpad { ptr, i32 }
// LLVM-NEXT:            cleanup
// LLVM:         %[[EACT:.*]] = load i8, ptr %[[FLAG]]
// LLVM:         %[[EACTB:.*]] = trunc i8 %[[EACT]] to i1
// LLVM:         br i1 %[[EACTB]], label %[[EHDTOR:.*]], label %{{.*}}
// LLVM:       [[EHDTOR]]:
// LLVM:         call void @_ZN1SD1Ev(ptr {{.*}} %[[S]])
// LLVM:         resume { ptr, i32 }
// LLVM:       [[BACKEDGE0]]:
// LLVM:         br label %[[BACKEDGE1:.*]]
// LLVM:       [[BACKEDGE1]]:
// LLVM:         br label %[[COND]]

// OGCG-LABEL: define dso_local void @_Z12for_cond_vari(i32 %n) {{.*}} personality ptr @__gxx_personality_v0 {
// OGCG:         %[[S:.*]] = alloca %struct.S
// OGCG:         br label %[[COND:.*]]
// OGCG:       [[COND]]:
// OGCG:         call void @_ZN1SC1Ev(ptr {{.*}} %[[S]])
// OGCG:         %[[CALL:.*]] = invoke {{.*}} i1 @_ZN1ScvbEv(ptr {{.*}} %[[S]])
// OGCG-NEXT:            to label %[[CONT:.*]] unwind label %[[LPAD:.*]]
// OGCG:       [[CONT]]:
// OGCG:         br i1 %[[CALL]], label %[[BODY:.*]], label %[[COND_CLEANUP:.*]]
// The loop-exit edge sets the cleanup destination selector to the "break"
// value.
// OGCG:       [[COND_CLEANUP]]:
// OGCG:         store i32 2, ptr %[[DEST:.*]],
// OGCG:         br label %[[CLEANUP:.*]]
// OGCG:       [[LPAD]]:
// OGCG:         landingpad { ptr, i32 }
// OGCG-NEXT:            cleanup
// OGCG:         call void @_ZN1SD1Ev(ptr {{.*}} %[[S]])
// OGCG:         br label %[[EHRESUME:.*]]
// OGCG:       [[BODY]]:
// OGCG:         br label %[[STEP:.*]]
// OGCG:       [[STEP]]:
// OGCG:         store i32 0, ptr %[[DEST]]
// OGCG:         br label %[[CLEANUP]]
// OGCG:       [[CLEANUP]]:
// OGCG:         call void @_ZN1SD1Ev(ptr {{.*}} %[[S]])
// OGCG:         %[[SEL:.*]] = load i32, ptr %[[DEST]]
// OGCG:         switch i32 %[[SEL]], label %{{.*}}
// OGCG:           i32 0, label %[[CONTINUE:.*]]
// OGCG:           i32 2, label %[[END:.*]]
// OGCG:       [[CONTINUE]]:
// OGCG:         br label %[[COND]]
// OGCG:       [[END]]:
// OGCG:         ret void
// OGCG:       [[EHRESUME]]:
// OGCG:         resume { ptr, i32 }

S chainS(const S &);
S makeS();

void while_cond_var_temp(int n) {
  while (S s = chainS(makeS()))
    --n;
}

// CIR-LABEL: cir.func {{.*}} @_Z19while_cond_var_tempi
// CIR:     %[[FLAG:.*]] = cir.alloca {{.*}}"cond.cleanup.isactive"{{.*}} : !cir.ptr<!cir.bool>
// CIR:     %[[S:.*]] = cir.alloca {{.*}}"s"{{.*}} : !cir.ptr<!rec_S>
// CIR:     %[[TMP:.*]] = cir.alloca {{.*}}"ref.tmp0"{{.*}} : !cir.ptr<!rec_S>
// CIR:     cir.while {
// CIR:       %[[FALSE:.*]] = cir.const #false
// CIR:       cir.store %[[FALSE]], %[[FLAG]]
// CIR:       cir.call @_Z5makeSv() : () -> !rec_S
// CIR:       cir.cleanup.scope {
// CIR:         cir.call @_Z6chainSRK1S(%[[TMP]])
// CIR:       } cleanup all {
// CIR:         cir.call @_ZN1SD1Ev(%[[TMP]]) nothrow
// CIR:         cir.yield
// CIR:       }
// CIR:       %[[TRUE:.*]] = cir.const #true
// CIR:       cir.store %[[TRUE]], %[[FLAG]]
// CIR-NOT:   cir.cleanup.scope
// CIR:       %[[COND:.*]] = cir.call @_ZN1ScvbEv(%[[S]])
// CIR:       cir.condition(%[[COND]])
// CIR:     } do {
// CIR:     } cleanup all {
// CIR:       %[[ACTIVE:.*]] = cir.load {{.*}}%[[FLAG]]
// CIR:       cir.if %[[ACTIVE]] {
// CIR:         cir.call @_ZN1SD1Ev(%[[S]]) nothrow
// CIR:       }
// CIR:       cir.yield
// CIR:     }

// LLVM-LABEL: define dso_local void @_Z19while_cond_var_tempi(i32 %0) {{.*}} personality ptr @__gxx_personality_v0 {
// LLVM:         %[[S:.*]] = alloca %struct.S
// LLVM:         %[[TMP:.*]] = alloca %struct.S
// LLVM:         %[[FLAG:.*]] = alloca i8
// LLVM:         br label %{{.*}}
// LLVM:         br label %[[COND:.*]]
// LLVM:       [[COND]]:
// LLVM:         br i1 true, label %{{.*}}, label %[[EXIT:.*]]
// LLVM:         store i8 0, ptr %[[FLAG]], align 1
// LLVM:         %[[MK:.*]] = invoke %struct.S @_Z5makeSv()
// LLVM-NEXT:            to label %[[MK_CONT:.*]] unwind label %[[LPAD_OUTER:.*]]
// LLVM:       [[MK_CONT]]:
// LLVM:         store %struct.S %[[MK]], ptr %[[TMP]], align 1
// LLVM:         %[[CH:.*]] = invoke %struct.S @_Z6chainSRK1S(ptr {{.*}} %[[TMP]])
// LLVM-NEXT:            to label %[[CH_CONT:.*]] unwind label %[[LPAD_TMP:.*]]
// LLVM:       [[CH_CONT]]:
// LLVM:         store %struct.S %[[CH]], ptr %[[S]], align 1
// LLVM:         call void @_ZN1SD1Ev(ptr {{.*}} %[[TMP]])
// LLVM:       [[LPAD_TMP]]:
// LLVM:         landingpad { ptr, i32 }
// LLVM-NEXT:            cleanup
// LLVM:         call void @_ZN1SD1Ev(ptr {{.*}} %[[TMP]])
// LLVM:         br label %[[EHCOMMON:.*]]
// LLVM:         store i8 1, ptr %[[FLAG]], align 1
// LLVM:         %[[CALL:.*]] = invoke i1 @_ZN1ScvbEv(ptr {{.*}} %[[S]])
// LLVM-NEXT:            to label %[[CVB_CONT:.*]] unwind label %[[LPAD_OUTER]]
// LLVM:       [[CVB_CONT]]:
// LLVM:         br i1 %[[CALL]], label %[[BODY:.*]], label %[[COND_FALSE:.*]]
// LLVM:       [[BODY]]:
// LLVM:         store i32 0, ptr %[[DEST:.*]], align 4
// LLVM:         br label %[[CLEANUP:.*]]
// LLVM:       [[COND_FALSE]]:
// LLVM:         store i32 1, ptr %[[DEST]], align 4
// LLVM:         br label %[[CLEANUP]]
// LLVM:       [[CLEANUP]]:
// LLVM:         %[[ACT:.*]] = load i8, ptr %[[FLAG]]
// LLVM:         %[[ACTB:.*]] = trunc i8 %[[ACT]] to i1
// LLVM:         br i1 %[[ACTB]], label %[[DTOR:.*]], label %{{.*}}
// LLVM:       [[DTOR]]:
// LLVM:         call void @_ZN1SD1Ev(ptr {{.*}} %[[S]])
// LLVM:         %[[SEL:.*]] = load i32, ptr %[[DEST]], align 4
// LLVM:         switch i32 %[[SEL]], label %{{.*}}
// LLVM:           i32 0, label %[[CONTINUE:.*]]
// LLVM:           i32 1, label %[[BREAK:.*]]
// LLVM:       [[CONTINUE]]:
// LLVM:         br label %[[BACKEDGE:.*]]
// LLVM:       [[BREAK]]:
// LLVM:         br label %[[EXIT]]
// LLVM:       [[LPAD_OUTER]]:
// LLVM:         landingpad { ptr, i32 }
// LLVM-NEXT:            cleanup
// LLVM:         br label %[[EHCOMMON]]
// LLVM:       [[EHCOMMON]]:
// LLVM:         %[[EACT:.*]] = load i8, ptr %[[FLAG]]
// LLVM:         %[[EACTB:.*]] = trunc i8 %[[EACT]] to i1
// LLVM:         br i1 %[[EACTB]], label %[[EHDTOR:.*]], label %{{.*}}
// LLVM:       [[EHDTOR]]:
// LLVM:         call void @_ZN1SD1Ev(ptr {{.*}} %[[S]])
// LLVM:         resume { ptr, i32 }
// LLVM:       [[BACKEDGE]]:
// LLVM:         br label %[[COND]]

// OGCG-LABEL: define dso_local void @_Z19while_cond_var_tempi(i32 %n) {{.*}} personality ptr @__gxx_personality_v0 {
// OGCG:         %[[S:.*]] = alloca %struct.S
// OGCG:         %[[TMP:.*]] = alloca %struct.S
// OGCG:         br label %[[COND:.*]]
// OGCG:       [[COND]]:
// OGCG:         call void @_Z5makeSv(ptr {{.*}} sret(%struct.S) {{.*}} %[[TMP]])
// OGCG:         invoke void @_Z6chainSRK1S(ptr {{.*}} sret(%struct.S) {{.*}} %[[S]], ptr {{.*}} %[[TMP]])
// OGCG-NEXT:            to label %[[CONT:.*]] unwind label %[[LPAD_TMP:.*]]
// OGCG:       [[CONT]]:
// OGCG:         call void @_ZN1SD1Ev(ptr {{.*}} %[[TMP]])
// OGCG:         %[[CALL:.*]] = invoke {{.*}} i1 @_ZN1ScvbEv(ptr {{.*}} %[[S]])
// OGCG-NEXT:            to label %[[CONT2:.*]] unwind label %[[LPAD_S:.*]]
// OGCG:       [[CONT2]]:
// OGCG:         br i1 %[[CALL]], label %[[BODY:.*]], label %[[EXIT:.*]]
// OGCG:       [[EXIT]]:
// OGCG:         store i32 3, ptr %[[DEST:.*]], align 4
// OGCG:         br label %[[CLEANUP:.*]]
// OGCG:       [[LPAD_TMP]]:
// OGCG:         landingpad { ptr, i32 }
// OGCG-NEXT:            cleanup
// OGCG:         call void @_ZN1SD1Ev(ptr {{.*}} %[[TMP]])
// OGCG:         br label %[[EHRESUME:.*]]
// OGCG:       [[LPAD_S]]:
// OGCG:         landingpad { ptr, i32 }
// OGCG-NEXT:            cleanup
// OGCG:         call void @_ZN1SD1Ev(ptr {{.*}} %[[S]])
// OGCG:         br label %[[EHRESUME]]
// OGCG:       [[BODY]]:
// OGCG:         store i32 0, ptr %[[DEST]], align 4
// OGCG:         br label %[[CLEANUP]]
// OGCG:       [[CLEANUP]]:
// OGCG:         call void @_ZN1SD1Ev(ptr {{.*}} %[[S]])
// OGCG:         %[[SEL:.*]] = load i32, ptr %[[DEST]], align 4
// OGCG:         switch i32 %[[SEL]], label %{{.*}}
// OGCG:           i32 0, label %[[CONTINUE:.*]]
// OGCG:           i32 3, label %[[END:.*]]
// OGCG:       [[CONTINUE]]:
// OGCG:         br label %[[COND]]
// OGCG:       [[END]]:
// OGCG:         ret void
// OGCG:       [[EHRESUME]]:
// OGCG:         resume { ptr, i32 }

void while_cond_var_nested(int n) {
  while (S s = ({
           while (S t{}) {
             S bodyVar;
             (void)bodyVar;
           }
           S{};
         }))
    --n;
}

// CIR-LABEL: cir.func {{.*}} @_Z21while_cond_var_nestedi
// Each loop's condition variable gets its own active flag.
// CIR:     %[[FLAG_OUTER:.*]] = cir.alloca {{.*}}"cond.cleanup.isactive"{{.*}} : !cir.ptr<!cir.bool>
// CIR:     %[[FLAG_INNER:.*]] = cir.alloca {{.*}}"cond.cleanup.isactive"{{.*}} : !cir.ptr<!cir.bool>
// CIR:     %[[S:.*]] = cir.alloca {{.*}}"s"{{.*}} : !cir.ptr<!rec_S>
// CIR:     cir.while {
// CIR:       cir.store %{{.*}}, %[[FLAG_OUTER]]
// CIR:       %[[T:.*]] = cir.alloca {{.*}}"t"{{.*}} : !cir.ptr<!rec_S>
// CIR:       cir.while {
// CIR:         cir.store %{{.*}}, %[[FLAG_INNER]]
// CIR:         cir.call @_ZN1SC1Ev(%[[T]])
// CIR:         cir.store %{{.*}}, %[[FLAG_INNER]]
// CIR:         %[[TCOND:.*]] = cir.call @_ZN1ScvbEv(%[[T]])
// CIR:         cir.condition(%[[TCOND]])
// CIR:       } do {
// CIR:         %[[BV:.*]] = cir.alloca {{.*}}"bodyVar"{{.*}} : !cir.ptr<!rec_S>
// CIR:         cir.call @_ZN1SC1Ev(%[[BV]])
// CIR:         cir.cleanup.scope {
// CIR:         } cleanup all {
// CIR:           cir.call @_ZN1SD1Ev(%[[BV]]) nothrow
// CIR:           cir.yield
// CIR:         }
// CIR:       } cleanup all {
// CIR:         %[[IACT:.*]] = cir.load {{.*}}%[[FLAG_INNER]]
// CIR:         cir.if %[[IACT]] {
// CIR:           cir.call @_ZN1SD1Ev(%[[T]]) nothrow
// CIR:         }
// CIR:         cir.yield
// CIR:       }
// CIR:       cir.call @_ZN1SC1Ev(%[[S]])
// CIR:       cir.store %{{.*}}, %[[FLAG_OUTER]]
// CIR:       %[[SCOND:.*]] = cir.call @_ZN1ScvbEv(%[[S]])
// CIR:       cir.condition(%[[SCOND]])
// CIR:     } do {
// CIR:     } cleanup all {
// CIR:       %[[OACT:.*]] = cir.load {{.*}}%[[FLAG_OUTER]]
// CIR:       cir.if %[[OACT]] {
// CIR:         cir.call @_ZN1SD1Ev(%[[S]]) nothrow
// CIR:       }
// CIR:       cir.yield
// CIR:     }

// LLVM-LABEL: define dso_local void @_Z21while_cond_var_nestedi(i32 %0) {{.*}} personality ptr @__gxx_personality_v0 {
// LLVM:         %[[S:.*]] = alloca %struct.S
// LLVM:         %[[FLAG_OUTER:.*]] = alloca i8
// LLVM:         %[[FLAG_INNER:.*]] = alloca i8
// LLVM:         br label %{{.*}}
// LLVM:         br label %[[OUTER_COND:.*]]
// LLVM:       [[OUTER_COND]]:
// LLVM:         br i1 true, label %{{.*}}, label %[[OUTER_EXIT:.*]]
// LLVM:         store i8 0, ptr %[[FLAG_OUTER]], align 1
// LLVM:         br label %{{.*}}
// LLVM:         br label %{{.*}}
// LLVM:         br label %[[INNER_COND:.*]]
// LLVM:       [[INNER_COND]]:
// LLVM:         br i1 true, label %{{.*}}, label %[[INNER_EXIT:.*]]
// LLVM:         store i8 0, ptr %[[FLAG_INNER]], align 1
// LLVM:         invoke void @_ZN1SC1Ev(ptr {{.*}} %[[T:.*]])
// LLVM-NEXT:            to label %[[ICTOR_CONT:.*]] unwind label %[[INNER_LPAD:.*]]
// LLVM:       [[ICTOR_CONT]]:
// LLVM:         store i8 1, ptr %[[FLAG_INNER]], align 1
// LLVM:         %[[TCALL:.*]] = invoke i1 @_ZN1ScvbEv(ptr {{.*}} %[[T]])
// LLVM-NEXT:            to label %[[TCVB_CONT:.*]] unwind label %[[INNER_LPAD]]
// LLVM:       [[TCVB_CONT]]:
// LLVM:         br i1 %[[TCALL]], label %[[IBODY:.*]], label %[[ICOND_FALSE:.*]]
// LLVM:       [[IBODY]]:
// LLVM:         invoke void @_ZN1SC1Ev(ptr {{.*}} %[[BV:.*]])
// LLVM-NEXT:            to label %[[BV_CONT:.*]] unwind label %[[INNER_LPAD]]
// LLVM:       [[BV_CONT]]:
// LLVM:         call void @_ZN1SD1Ev(ptr {{.*}} %[[BV]])
// LLVM:         store i32 0, ptr %[[DEST:.*]], align 4
// LLVM:         br label %[[ICLEANUP:.*]]
// LLVM:       [[ICOND_FALSE]]:
// LLVM:         store i32 1, ptr %[[DEST]], align 4
// LLVM:         br label %[[ICLEANUP]]
// LLVM:       [[ICLEANUP]]:
// LLVM:         %[[IACT:.*]] = load i8, ptr %[[FLAG_INNER]]
// LLVM:         %[[IACTB:.*]] = trunc i8 %[[IACT]] to i1
// LLVM:         br i1 %[[IACTB]], label %[[IDTOR:.*]], label %{{.*}}
// LLVM:       [[IDTOR]]:
// LLVM:         call void @_ZN1SD1Ev(ptr {{.*}} %[[T]])
// LLVM:         %[[ISEL:.*]] = load i32, ptr %[[DEST]], align 4
// LLVM:         switch i32 %[[ISEL]], label %{{.*}}
// LLVM:           i32 0, label %[[ICONTINUE:.*]]
// LLVM:           i32 1, label %[[IBREAK:.*]]
// LLVM:       [[ICONTINUE]]:
// LLVM:         br label %[[IBACKEDGE:.*]]
// LLVM:       [[IBREAK]]:
// LLVM:         br label %[[INNER_EXIT]]
// LLVM:       [[INNER_LPAD]]:
// LLVM:         landingpad { ptr, i32 }
// LLVM-NEXT:            cleanup
// LLVM:         %[[EIACT:.*]] = load i8, ptr %[[FLAG_INNER]]
// LLVM:         %[[EIACTB:.*]] = trunc i8 %[[EIACT]] to i1
// LLVM:         br i1 %[[EIACTB]], label %[[EIDTOR:.*]], label %{{.*}}
// LLVM:       [[EIDTOR]]:
// LLVM:         call void @_ZN1SD1Ev(ptr {{.*}} %[[T]])
// LLVM:         br label %{{.*}}
// LLVM:         br label %[[OUTER_EH:.*]]
// LLVM:       [[IBACKEDGE]]:
// LLVM:         br label %[[INNER_COND]]
// LLVM:       [[INNER_EXIT]]:
// LLVM:         invoke void @_ZN1SC1Ev(ptr {{.*}} %[[S]])
// LLVM-NEXT:            to label %[[OCTOR_CONT:.*]] unwind label %[[OUTER_LPAD:.*]]
// LLVM:       [[OCTOR_CONT]]:
// LLVM:         store i8 1, ptr %[[FLAG_OUTER]], align 1
// LLVM:         %[[SCALL:.*]] = invoke i1 @_ZN1ScvbEv(ptr {{.*}} %[[S]])
// LLVM-NEXT:            to label %[[SCVB_CONT:.*]] unwind label %[[OUTER_LPAD]]
// LLVM:       [[SCVB_CONT]]:
// LLVM:         br i1 %[[SCALL]], label %[[OBODY:.*]], label %[[OCOND_FALSE:.*]]
// LLVM:       [[OBODY]]:
// LLVM:         store i32 0, ptr %[[DEST]], align 4
// LLVM:         br label %[[OCLEANUP:.*]]
// LLVM:       [[OCOND_FALSE]]:
// LLVM:         store i32 1, ptr %[[DEST]], align 4
// LLVM:         br label %[[OCLEANUP]]
// LLVM:       [[OCLEANUP]]:
// LLVM:         %[[OACT:.*]] = load i8, ptr %[[FLAG_OUTER]]
// LLVM:         %[[OACTB:.*]] = trunc i8 %[[OACT]] to i1
// LLVM:         br i1 %[[OACTB]], label %[[ODTOR:.*]], label %{{.*}}
// LLVM:       [[ODTOR]]:
// LLVM:         call void @_ZN1SD1Ev(ptr {{.*}} %[[S]])
// LLVM:         %[[OSEL:.*]] = load i32, ptr %[[DEST]], align 4
// LLVM:         switch i32 %[[OSEL]], label %{{.*}}
// LLVM:           i32 0, label %[[OCONTINUE:.*]]
// LLVM:           i32 1, label %[[OBREAK:.*]]
// LLVM:       [[OCONTINUE]]:
// LLVM:         br label %[[OBACKEDGE:.*]]
// LLVM:       [[OBREAK]]:
// LLVM:         br label %[[OUTER_EXIT]]
// LLVM:       [[OUTER_LPAD]]:
// LLVM:         landingpad { ptr, i32 }
// LLVM-NEXT:            cleanup
// LLVM:         br label %[[OUTER_EH]]
// LLVM:       [[OUTER_EH]]:
// LLVM:         %[[EOACT:.*]] = load i8, ptr %[[FLAG_OUTER]]
// LLVM:         %[[EOACTB:.*]] = trunc i8 %[[EOACT]] to i1
// LLVM:         br i1 %[[EOACTB]], label %[[EODTOR:.*]], label %{{.*}}
// LLVM:       [[EODTOR]]:
// LLVM:         call void @_ZN1SD1Ev(ptr {{.*}} %[[S]])
// LLVM:         resume { ptr, i32 }
// LLVM:       [[OBACKEDGE]]:
// LLVM:         br label %[[OUTER_COND]]

// OGCG-LABEL: define dso_local void @_Z21while_cond_var_nestedi(i32 %n) {{.*}} personality ptr @__gxx_personality_v0 {
// OGCG:         %[[S:.*]] = alloca %struct.S
// OGCG:         %[[T:.*]] = alloca %struct.S
// OGCG:         %[[BV:.*]] = alloca %struct.S
// OGCG:         br label %[[OUTER_COND:.*]]
// OGCG:       [[OUTER_COND]]:
// OGCG:         br label %[[INNER_COND:.*]]
// OGCG:       [[INNER_COND]]:
// OGCG:         call void @_ZN1SC1Ev(ptr {{.*}} %[[T]])
// OGCG:         %[[TCALL:.*]] = invoke {{.*}} i1 @_ZN1ScvbEv(ptr {{.*}} %[[T]])
// OGCG-NEXT:            to label %[[ICONT:.*]] unwind label %[[INNER_LPAD:.*]]
// OGCG:       [[ICONT]]:
// OGCG:         br i1 %[[TCALL]], label %[[IBODY:.*]], label %[[INNER_EXIT:.*]]
// OGCG:       [[INNER_EXIT]]:
// OGCG:         store i32 5, ptr %[[DEST:.*]], align 4
// OGCG:         br label %[[ICLEANUP:.*]]
// OGCG:       [[INNER_LPAD]]:
// OGCG:         landingpad { ptr, i32 }
// OGCG-NEXT:            cleanup
// OGCG:         call void @_ZN1SD1Ev(ptr {{.*}} %[[T]])
// OGCG:         br label %[[EHRESUME:.*]]
// OGCG:       [[IBODY]]:
// OGCG:         invoke void @_ZN1SC1Ev(ptr {{.*}} %[[BV]])
// OGCG-NEXT:            to label %[[IBODY_CONT:.*]] unwind label %[[INNER_LPAD]]
// OGCG:       [[IBODY_CONT]]:
// OGCG:         call void @_ZN1SD1Ev(ptr {{.*}} %[[BV]])
// OGCG:         store i32 0, ptr %[[DEST]], align 4
// OGCG:         br label %[[ICLEANUP]]
// OGCG:       [[ICLEANUP]]:
// OGCG:         call void @_ZN1SD1Ev(ptr {{.*}} %[[T]])
// OGCG:         %[[ISEL:.*]] = load i32, ptr %[[DEST]], align 4
// OGCG:         switch i32 %[[ISEL]], label %{{.*}}
// OGCG:           i32 0, label %[[ICONTINUE:.*]]
// OGCG:           i32 5, label %[[INNER_END:.*]]
// OGCG:       [[ICONTINUE]]:
// OGCG:         br label %[[INNER_COND]]
// OGCG:       [[INNER_END]]:
// OGCG:         call void @_ZN1SC1Ev(ptr {{.*}} %[[S]])
// OGCG:         %[[SCALL:.*]] = invoke {{.*}} i1 @_ZN1ScvbEv(ptr {{.*}} %[[S]])
// OGCG-NEXT:            to label %[[OCONT:.*]] unwind label %[[OUTER_LPAD:.*]]
// OGCG:       [[OCONT]]:
// OGCG:         br i1 %[[SCALL]], label %[[OBODY:.*]], label %[[OUTER_EXIT:.*]]
// OGCG:       [[OUTER_EXIT]]:
// OGCG:         store i32 3, ptr %[[DEST]], align 4
// OGCG:         br label %[[OCLEANUP:.*]]
// OGCG:       [[OUTER_LPAD]]:
// OGCG:         landingpad { ptr, i32 }
// OGCG-NEXT:            cleanup
// OGCG:         call void @_ZN1SD1Ev(ptr {{.*}} %[[S]])
// OGCG:         br label %[[EHRESUME]]
// OGCG:       [[OBODY]]:
// OGCG:         store i32 0, ptr %[[DEST]], align 4
// OGCG:         br label %[[OCLEANUP]]
// OGCG:       [[OCLEANUP]]:
// OGCG:         call void @_ZN1SD1Ev(ptr {{.*}} %[[S]])
// OGCG:         %[[OSEL:.*]] = load i32, ptr %[[DEST]], align 4
// OGCG:         switch i32 %[[OSEL]], label %{{.*}}
// OGCG:           i32 0, label %[[OCONTINUE:.*]]
// OGCG:           i32 3, label %[[OUTER_END:.*]]
// OGCG:       [[OCONTINUE]]:
// OGCG:         br label %[[OUTER_COND]]
// OGCG:       [[OUTER_END]]:
// OGCG:         ret void
// OGCG:       [[EHRESUME]]:
// OGCG:         resume { ptr, i32 }
