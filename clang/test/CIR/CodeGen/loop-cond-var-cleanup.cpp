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
// LLVM:         %[[FLAG:.*]] = alloca i8
// LLVM:         %[[S:.*]] = alloca %struct.S
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
// LLVM:         %[[CALL:.*]] = invoke zeroext i1 @_ZN1ScvbEv(ptr {{.*}} %[[S]])
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
// LLVM:         %[[FLAG:.*]] = alloca i8
// LLVM:         %[[S:.*]] = alloca %struct.S
// LLVM:         br label %{{.*}}
// LLVM:         br label %[[COND:.*]]
// LLVM:       [[COND]]:
// LLVM:         br i1 true, label %{{.*}}, label %[[EXIT:.*]]
// LLVM:         store i8 0, ptr %[[FLAG]]
// LLVM:         invoke void @_ZN1SC1Ev(ptr {{.*}} %[[S]])
// LLVM-NEXT:            to label %[[CTOR_CONT:.*]] unwind label %[[LPAD:.*]]
// LLVM:       [[CTOR_CONT]]:
// LLVM:         store i8 1, ptr %[[FLAG]]
// LLVM:         %[[CALL:.*]] = invoke zeroext i1 @_ZN1ScvbEv(ptr {{.*}} %[[S]])
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
// LLVM:         %[[FLAG:.*]] = alloca i8
// LLVM:         %[[S:.*]] = alloca %struct.S
// LLVM:         br label %{{.*}}
// LLVM:         br label %[[COND:.*]]
// LLVM:       [[COND]]:
// LLVM:         br i1 true, label %{{.*}}, label %[[EXIT:.*]]
// LLVM:         store i8 0, ptr %[[FLAG]]
// LLVM:         invoke void @_ZN1SC1Ev(ptr {{.*}} %[[S]])
// LLVM-NEXT:            to label %[[CTOR_CONT:.*]] unwind label %[[LPAD:.*]]
// LLVM:       [[CTOR_CONT]]:
// LLVM:         store i8 1, ptr %[[FLAG]]
// LLVM:         %[[CALL:.*]] = invoke zeroext i1 @_ZN1ScvbEv(ptr {{.*}} %[[S]])
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
// LLVM:         %[[FLAG:.*]] = alloca i8
// LLVM:         %[[S:.*]] = alloca %struct.S
// LLVM:         br label %{{.*}}
// LLVM:         br label %[[COND:.*]]
// LLVM:       [[COND]]:
// LLVM:         br i1 true, label %{{.*}}, label %[[EXIT:.*]]
// LLVM:         store i8 0, ptr %[[FLAG]]
// LLVM:         invoke void @_ZN1SC1Ev(ptr {{.*}} %[[S]])
// LLVM-NEXT:            to label %[[CTOR_CONT:.*]] unwind label %[[LPAD:.*]]
// LLVM:       [[CTOR_CONT]]:
// LLVM:         store i8 1, ptr %[[FLAG]]
// LLVM:         %[[CALL:.*]] = invoke zeroext i1 @_ZN1ScvbEv(ptr {{.*}} %[[S]])
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
// LLVM:         %[[FLAG:.*]] = alloca i8
// LLVM:         %[[S:.*]] = alloca %struct.S
// LLVM:         br label %{{.*}}
// LLVM:         br label %[[COND:.*]]
// LLVM:       [[COND]]:
// LLVM:         br i1 true, label %{{.*}}, label %[[EXIT:.*]]
// LLVM:         store i8 0, ptr %[[FLAG]]
// LLVM:         invoke void @_ZN1SC1Ev(ptr {{.*}} %[[S]])
// LLVM-NEXT:            to label %[[CTOR_CONT:.*]] unwind label %[[LPAD:.*]]
// LLVM:       [[CTOR_CONT]]:
// LLVM:         store i8 1, ptr %[[FLAG]]
// LLVM:         %[[CALL:.*]] = invoke zeroext i1 @_ZN1ScvbEv(ptr {{.*}} %[[S]])
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
// LLVM:         %[[FLAG:.*]] = alloca i8
// LLVM:         %[[S:.*]] = alloca %struct.S
// LLVM:         br label %{{.*}}
// LLVM:         br label %[[COND:.*]]
// LLVM:       [[COND]]:
// LLVM:         br i1 true, label %{{.*}}, label %[[EXIT:.*]]
// LLVM:         store i8 0, ptr %[[FLAG]]
// LLVM:         invoke void @_ZN1SC1Ev(ptr {{.*}} %[[S]])
// LLVM-NEXT:            to label %[[CTOR_CONT:.*]] unwind label %[[LPAD:.*]]
// LLVM:       [[CTOR_CONT]]:
// LLVM:         store i8 1, ptr %[[FLAG]]
// LLVM:         %[[CALL:.*]] = invoke zeroext i1 @_ZN1ScvbEv(ptr {{.*}} %[[S]])
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
// CIR:       cir.call @_Z5makeSv(%[[TMP]]) : (!cir.ptr<!rec_S> {{.*}}llvm.sret = !rec_S{{.*}}) -> ()
// CIR:       cir.cleanup.scope {
// CIR:         cir.call @_Z6chainSRK1S(%[[S]], %[[TMP]]) : (!cir.ptr<!rec_S> {{.*}}llvm.sret = !rec_S{{.*}}, !cir.ptr<!rec_S> {{.*}}) -> ()
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
// LLVM:         %[[FLAG:.*]] = alloca i8
// LLVM:         %[[S:.*]] = alloca %struct.S
// LLVM:         %[[TMP:.*]] = alloca %struct.S
// LLVM:         br label %{{.*}}
// LLVM:         br label %[[COND:.*]]
// LLVM:       [[COND]]:
// LLVM:         br i1 true, label %{{.*}}, label %[[EXIT:.*]]
// LLVM:         store i8 0, ptr %[[FLAG]], align 1
// LLVM:         invoke void @_Z5makeSv(ptr {{.*}} sret(%struct.S) {{.*}} %[[TMP]])
// LLVM-NEXT:            to label %[[MK_CONT:.*]] unwind label %[[LPAD_OUTER:.*]]
// LLVM:       [[MK_CONT]]:
// LLVM:         invoke void @_Z6chainSRK1S(ptr {{.*}} sret(%struct.S) {{.*}} %[[S]], ptr {{.*}} %[[TMP]])
// LLVM-NEXT:            to label %[[CH_CONT:.*]] unwind label %[[LPAD_TMP:.*]]
// LLVM:       [[CH_CONT]]:
// LLVM:         call void @_ZN1SD1Ev(ptr {{.*}} %[[TMP]])
// LLVM:       [[LPAD_TMP]]:
// LLVM:         landingpad { ptr, i32 }
// LLVM-NEXT:            cleanup
// LLVM:         call void @_ZN1SD1Ev(ptr {{.*}} %[[TMP]])
// LLVM:         br label %[[EHCOMMON:.*]]
// LLVM:         store i8 1, ptr %[[FLAG]], align 1
// LLVM:         %[[CALL:.*]] = invoke zeroext i1 @_ZN1ScvbEv(ptr {{.*}} %[[S]])
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
// LLVM:         %[[FLAG_OUTER:.*]] = alloca i8
// LLVM:         %[[FLAG_INNER:.*]] = alloca i8
// LLVM:         %[[S:.*]] = alloca %struct.S
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
// LLVM:         %[[TCALL:.*]] = invoke zeroext i1 @_ZN1ScvbEv(ptr {{.*}} %[[T]])
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
// LLVM:         %[[SCALL:.*]] = invoke zeroext i1 @_ZN1ScvbEv(ptr {{.*}} %[[S]])
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

void use(S &);

struct Holder {
  S &&a;
  S &&b;
  ~Holder();
  explicit operator bool();
};

// Destroy the variable first, then its extended temporaries in reverse order.
// All three cleanups share the active flag.
void for_cond_var_extended_member() {
  for (; Holder h{makeS(), makeS()};)
    use(h.a);
}

// CIR-LABEL: cir.func {{.*}} @_Z28for_cond_var_extended_memberv
// CIR:         %[[FLAG:.*]] = cir.alloca {{.*}}"cond.cleanup.isactive"{{.*}} : !cir.ptr<!cir.bool>
// CIR:         %[[A:.*]] = cir.alloca {{.*}}"ref.tmp0"{{.*}} : !cir.ptr<!rec_S>
// CIR:         %[[B:.*]] = cir.alloca {{.*}}"ref.tmp1"{{.*}} : !cir.ptr<!rec_S>
// CIR:         %[[H:.*]] = cir.alloca {{.*}}"h"{{.*}} : !cir.ptr<!rec_Holder>
// CIR:         cir.for : cond {
// CIR:           %[[FALSE:.*]] = cir.const #false
// CIR-NEXT:      cir.store %[[FALSE]], %[[FLAG]]
// CIR:           cir.call @_Z5makeSv(%[[A]])
// The first temporary is protected while the second is initialized.
// CIR:           cir.cleanup.scope {
// CIR:             cir.call @_Z5makeSv(%[[B]])
// CIR:           } cleanup eh {
// CIR:             cir.call @_ZN1SD1Ev(%[[B]]) nothrow
// CIR:           } cleanup eh {
// CIR-NEXT:        cir.call @_ZN1SD1Ev(%[[A]]) nothrow
// CIR:           %[[TRUE:.*]] = cir.const #true
// CIR-NEXT:      cir.store %[[TRUE]], %[[FLAG]]
// CIR:           cir.condition
// CIR:         } cleanup all {
// CIR-NEXT:      %[[ACTIVE:.*]] = cir.load {{.*}}%[[FLAG]]
// CIR-NEXT:      cir.if %[[ACTIVE]] {
// CIR-NEXT:        cir.call @_ZN6HolderD1Ev(%[[H]]) nothrow{{.*}}
// CIR-NEXT:      }
// CIR-NEXT:      %[[ACTIVE_B:.*]] = cir.load {{.*}}%[[FLAG]]
// CIR-NEXT:      cir.if %[[ACTIVE_B]] {
// CIR-NEXT:        cir.call @_ZN1SD1Ev(%[[B]]) nothrow{{.*}}
// CIR-NEXT:      }
// CIR-NEXT:      %[[ACTIVE_A:.*]] = cir.load {{.*}}%[[FLAG]]
// CIR-NEXT:      cir.if %[[ACTIVE_A]] {
// CIR-NEXT:        cir.call @_ZN1SD1Ev(%[[A]]) nothrow{{.*}}
// CIR-NEXT:      }
// CIR-NEXT:      cir.yield
// CIR-NEXT:    }

// LLVM-LABEL: define dso_local void @_Z28for_cond_var_extended_memberv() {{.*}} personality ptr @__gxx_personality_v0 {
// LLVM:         %[[FLAG:.*]] = alloca i8
// LLVM:         %[[A:.*]] = alloca %struct.S
// LLVM:         %[[B:.*]] = alloca %struct.S
// LLVM:         %[[H:.*]] = alloca %struct.Holder
// LLVM:         store i8 0, ptr %[[FLAG]]
// LLVM:         invoke void @_Z5makeSv(ptr {{.*}} %[[A]])
// LLVM-NEXT:            to label %{{.*}} unwind label %[[LPAD:.*]]
// LLVM:         invoke void @_Z5makeSv(ptr {{.*}} %[[B]])
// LLVM:         store i8 1, ptr %[[FLAG]]
// LLVM:         call void @_ZN6HolderD1Ev(ptr {{.*}} %[[H]])
// LLVM:         call void @_ZN1SD1Ev(ptr {{.*}} %[[B]])
// LLVM:         call void @_ZN1SD1Ev(ptr {{.*}} %[[A]])
// LLVM:       [[LPAD]]:
// LLVM:         landingpad { ptr, i32 }
// LLVM-NEXT:            cleanup
// LLVM:         load i8, ptr %[[FLAG]]
// LLVM:         call void @_ZN6HolderD1Ev(ptr {{.*}} %[[H]])
// LLVM:         load i8, ptr %[[FLAG]]
// LLVM:         call void @_ZN1SD1Ev(ptr {{.*}} %[[B]])
// LLVM:         load i8, ptr %[[FLAG]]
// LLVM:         call void @_ZN1SD1Ev(ptr {{.*}} %[[A]])
// LLVM:         resume { ptr, i32 }

// OGCG-LABEL: define dso_local void @_Z28for_cond_var_extended_memberv() {{.*}} personality ptr @__gxx_personality_v0 {
// OGCG:         call void @_Z5makeSv(ptr {{.*}} %[[A:[^)]*]])
// OGCG:         invoke void @_Z5makeSv(ptr {{.*}} %[[B:[^)]*]])
// OGCG:       {{^}}cleanup:
// OGCG:         call void @_ZN6HolderD1Ev(
// OGCG-NEXT:    call void @_ZN1SD1Ev(ptr {{.*}} %[[B]])
// OGCG-NEXT:    call void @_ZN1SD1Ev(ptr {{.*}} %[[A]])

S pick(S &);

// A temporary that a variable in a statement expression in the initializer
// extends is destroyed at the end of the statement expression.
void while_cond_var_extended_in_stmt_expr() {
  while (S &&s = ({
           S &&t = makeS();
           pick(t);
         }))
    use(s);
}

// CIR-LABEL: cir.func {{.*}} @_Z36while_cond_var_extended_in_stmt_exprv
// CIR:         %[[FLAG:.*]] = cir.alloca {{.*}}"cond.cleanup.isactive"{{.*}} : !cir.ptr<!cir.bool>
// CIR:         %[[OUTER_TMP:.*]] = cir.alloca {{.*}}"ref.tmp0"{{.*}} : !cir.ptr<!rec_S>
// CIR:         %[[S:.*]] = cir.alloca {{.*}}"s"{{.*}} : !cir.ptr<!cir.ptr<!rec_S>>
// CIR:         cir.while {
// CIR:           %[[FALSE:.*]] = cir.const #false
// CIR-NEXT:      cir.store %[[FALSE]], %[[FLAG]]
// CIR-NEXT:      cir.scope {
// CIR:             %[[INNER_TMP:.*]] = cir.alloca {{.*}}"ref.tmp1"{{.*}} : !cir.ptr<!rec_S>
// CIR:             cir.call @_Z5makeSv(%[[INNER_TMP]])
// CIR:             cir.cleanup.scope {
// CIR:               cir.call @_Z4pickR1S(%[[OUTER_TMP]],
// CIR:             } cleanup all {
// CIR-NEXT:          cir.call @_ZN1SD1Ev(%[[INNER_TMP]]) nothrow{{.*}}
// CIR-NEXT:          cir.yield
// CIR-NEXT:        }
// CIR-NEXT:      }
// CIR-NEXT:      cir.store {{.*}}%[[OUTER_TMP]], %[[S]]
// CIR-NEXT:      %[[TRUE:.*]] = cir.const #true
// CIR-NEXT:      cir.store %[[TRUE]], %[[FLAG]]
// CIR:           cir.condition
// CIR:         } cleanup all {
// CIR-NEXT:      %[[ACTIVE:.*]] = cir.load {{.*}}%[[FLAG]]
// CIR-NEXT:      cir.if %[[ACTIVE]] {
// CIR-NEXT:        cir.call @_ZN1SD1Ev(%[[OUTER_TMP]]) nothrow{{.*}}
// CIR-NEXT:      }
// CIR-NEXT:      cir.yield
// CIR-NEXT:    }

// OGCG-LABEL: define dso_local void @_Z36while_cond_var_extended_in_stmt_exprv() {{.*}} personality ptr @__gxx_personality_v0 {
// OGCG:         call void @_Z5makeSv(ptr {{.*}} %[[INNER_TMP:.*]])
// OGCG:         invoke void @_Z4pickR1S(ptr {{.*}} %[[OUTER_TMP:[^,]*]],
// OGCG:         call void @_ZN1SD1Ev(ptr {{.*}} %[[INNER_TMP]])
// OGCG:         invoke {{.*}} @_ZN1ScvbEv(
// OGCG:       {{^}}cleanup:
// OGCG:         call void @_ZN1SD1Ev(ptr {{.*}} %[[OUTER_TMP]])

struct DefaultInit {
  S &&s = makeS();
  explicit operator bool();
};

// A temporary in a default member initializer is extended as well.
void while_cond_var_extended_default_init() {
  while (DefaultInit d{})
    use(d.s);
}

// CIR-LABEL: cir.func {{.*}} @_Z36while_cond_var_extended_default_initv
// CIR:         %[[FLAG:.*]] = cir.alloca {{.*}}"cond.cleanup.isactive"{{.*}} : !cir.ptr<!cir.bool>
// CIR:         %[[TMP:.*]] = cir.alloca {{.*}}"ref.tmp0"{{.*}} : !cir.ptr<!rec_S>
// CIR:         cir.while {
// CIR:           %[[FALSE:.*]] = cir.const #false
// CIR-NEXT:      cir.store %[[FALSE]], %[[FLAG]]
// CIR:           cir.call @_Z5makeSv(%[[TMP]])
// CIR:           cir.cleanup.scope {
// CIR:           } cleanup eh {
// CIR-NEXT:        cir.call @_ZN1SD1Ev(%[[TMP]]) nothrow
// CIR:           %[[TRUE:.*]] = cir.const #true
// CIR-NEXT:      cir.store %[[TRUE]], %[[FLAG]]
// CIR-NOT:       @_ZN1SD1Ev
// CIR:         } cleanup all {
// CIR-NEXT:      %[[ACTIVE:.*]] = cir.load {{.*}}%[[FLAG]]
// CIR-NEXT:      cir.if %[[ACTIVE]] {
// CIR-NEXT:        cir.call @_ZN1SD1Ev(%[[TMP]]) nothrow{{.*}}
// CIR-NEXT:      }
// CIR-NEXT:      cir.yield
// CIR-NEXT:    }

// OGCG-LABEL: define dso_local void @_Z36while_cond_var_extended_default_initv() {{.*}} personality ptr @__gxx_personality_v0 {
// OGCG:         call void @_Z5makeSv(ptr {{.*}} %[[TMP:.*]])
// OGCG:       {{^}}cleanup:
// OGCG:         call void @_ZN1SD1Ev(ptr {{.*}} %[[TMP]])

namespace std {
template <class E> class initializer_list {
  const E *b;
  decltype(sizeof 0) n;

public:
  constexpr initializer_list() : b(nullptr), n(0) {}
};
} // namespace std

struct List {
  std::initializer_list<S> l;
  explicit operator bool();
};

void use(List &);

// So does a std::initializer_list member for its backing array.
void while_cond_var_extended_init_list() {
  while (List l{{S{}, S{}}})
    use(l);
}

// CIR-LABEL: cir.func {{.*}} @_Z33while_cond_var_extended_init_listv
// CIR:         %[[FLAG:.*]] = cir.alloca {{.*}}"cond.cleanup.isactive"{{.*}} : !cir.ptr<!cir.bool>
// CIR:         %[[ARRAY:.*]] = cir.alloca {{.*}}"ref.tmp0"{{.*}} : !cir.ptr<!cir.array<!rec_S x 2>>
// CIR:         cir.while {
// CIR:           %[[FALSE:.*]] = cir.const #false
// CIR-NEXT:      cir.store %[[FALSE]], %[[FLAG]]
// The elements constructed so far are destroyed if a constructor throws.
// CIR:           cir.cleanup.scope {
// CIR:             cir.call @_ZN1SC1Ev(
// CIR:             cir.call @_ZN1SC1Ev(
// CIR-NEXT:          cir.yield
// CIR-NEXT:        } cleanup eh {
// The whole array is destroyed if the rest of the initialization throws.
// CIR:           cir.cleanup.scope {
// CIR:             cir.get_member {{.*}} {name = "n"}
// CIR:           } cleanup eh {
// CIR:           %[[TRUE:.*]] = cir.const #true
// CIR-NEXT:      cir.store %[[TRUE]], %[[FLAG]]
// CIR:         } cleanup all {
// CIR-NEXT:      %[[ACTIVE:.*]] = cir.load {{.*}}%[[FLAG]]
// CIR-NEXT:      cir.if %[[ACTIVE]] {
// CIR-NEXT:        %[[SIZE:.*]] = cir.const #cir.int<2> : !u64i
// CIR-NEXT:        %[[BEGIN:.*]] = cir.cast array_to_ptrdecay %[[ARRAY]]
// CIR-NEXT:        cir.ptr_stride %[[BEGIN]], %[[SIZE]]
// CIR:             cir.do {
// CIR:               cir.call @_ZN1SD1Ev({{.*}}) nothrow{{.*}}
// CIR:             } while {

// LLVM-LABEL: define dso_local void @_Z33while_cond_var_extended_init_listv() {{.*}} personality ptr @__gxx_personality_v0 {
// LLVM:         %[[FLAG:.*]] = alloca i8
// LLVM:         store i8 0, ptr %[[FLAG]]
// LLVM:         invoke void @_ZN1SC1Ev(
// LLVM-NEXT:            to label %{{.*}} unwind label %[[PARTIAL_LPAD:.*]]
// LLVM:         invoke void @_ZN1SC1Ev(
// LLVM-NEXT:            to label %{{.*}} unwind label %[[PARTIAL_LPAD]]
// LLVM:       [[PARTIAL_LPAD]]:
// LLVM:         landingpad { ptr, i32 }
// LLVM:         call void @_ZN1SD1Ev(
// LLVM:         store i8 1, ptr %[[FLAG]]
// LLVM:         landingpad { ptr, i32 }
// The partial-array cleanup and the loop's EH cleanup share this block, so the
// whole array is destroyed only once the flag is set.
// LLVM:         %[[EH_ACTIVE:.*]] = load i8, ptr %[[FLAG]]
// LLVM:         %[[EH_ACTIVE_BOOL:.*]] = trunc i8 %[[EH_ACTIVE]] to i1
// LLVM:         br i1 %[[EH_ACTIVE_BOOL]], label %{{.*}}, label %{{.*}}
// LLVM:         call void @_ZN1SD1Ev(
// LLVM:         resume { ptr, i32 }

// OGCG-LABEL: define dso_local void @_Z33while_cond_var_extended_init_listv() {{.*}} personality ptr @__gxx_personality_v0 {
// OGCG:       {{^}}cleanup:
// OGCG:       {{^}}arraydestroy.body{{[0-9]*}}:
// OGCG:         call void @_ZN1SD1Ev(

// The extended temporary is destroyed on iteration exits, early returns, and
// unwinding. Its cleanup is inactive until initialization completes.
int while_cond_var_extended_return(int n) {
  while (S &&s = makeS()) {
    if (n)
      return n;
    use(s);
  }
  return 0;
}

// CIR-LABEL: cir.func {{.*}} @_Z30while_cond_var_extended_returni
// CIR:         %[[FLAG:.*]] = cir.alloca {{.*}}"cond.cleanup.isactive"{{.*}} : !cir.ptr<!cir.bool>
// CIR:         %[[TMP:.*]] = cir.alloca {{.*}}"ref.tmp0"{{.*}} : !cir.ptr<!rec_S>
// CIR:         %[[S:.*]] = cir.alloca {{.*}}"s"{{.*}} : !cir.ptr<!cir.ptr<!rec_S>>
// CIR:         cir.while {
// CIR:           %[[FALSE:.*]] = cir.const #false
// CIR-NEXT:      cir.store %[[FALSE]], %[[FLAG]]
// CIR-NEXT:      cir.call @_Z5makeSv(%[[TMP]])
// CIR-NEXT:      cir.store {{.*}}%[[TMP]], %[[S]]
// CIR-NEXT:      %[[TRUE:.*]] = cir.const #true
// CIR-NEXT:      cir.store %[[TRUE]], %[[FLAG]]
// CIR-NOT:       @_ZN1SD1Ev
// CIR:           cir.condition
// CIR-NOT:       @_ZN1SD1Ev
// CIR:         } do {
// CIR-NOT:       @_ZN1SD1Ev
// CIR:             cir.return
// CIR-NOT:       @_ZN1SD1Ev
// CIR:           cir.call @_Z3useR1S
// CIR-NOT:       @_ZN1SD1Ev
// CIR:         } cleanup all {
// CIR-NEXT:      %[[ACTIVE:.*]] = cir.load {{.*}}%[[FLAG]]
// CIR-NEXT:      cir.if %[[ACTIVE]] {
// CIR-NEXT:        cir.call @_ZN1SD1Ev(%[[TMP]]) nothrow{{.*}}
// CIR-NEXT:      }
// CIR-NEXT:      cir.yield
// CIR-NEXT:    }

// LLVM-LABEL: define dso_local {{.*}}i32 @_Z30while_cond_var_extended_returni(
// LLVM:         %[[FLAG:.*]] = alloca i8
// LLVM:         %[[TMP:.*]] = alloca %struct.S
// LLVM:         store i8 0, ptr %[[FLAG]]
// LLVM-NEXT:    invoke void @_Z5makeSv(ptr {{.*}} %[[TMP]])
// LLVM-NEXT:            to label %[[MAKE_CONT:.*]] unwind label %[[LPAD:.*]]
// LLVM:       [[MAKE_CONT]]:
// LLVM:         store i8 1, ptr %[[FLAG]]
// LLVM:         store i32 %{{.*}}, ptr %[[RETVAL:.*]], align 4
// LLVM-NEXT:    store i32 [[RET_DEST:[0-9]+]], ptr %[[DEST:.*]], align 4
// LLVM-NEXT:    br label %[[CLEANUP:.*]]
// LLVM:         invoke void @_Z3useR1S(
// LLVM:       [[CLEANUP]]:
// LLVM:         %[[ACTIVE:.*]] = load i8, ptr %[[FLAG]]
// LLVM:         %[[ACTIVE_BOOL:.*]] = trunc i8 %[[ACTIVE]] to i1
// LLVM:         br i1 %[[ACTIVE_BOOL]], label %[[DTOR:.*]], label %{{.*}}
// LLVM:       [[DTOR]]:
// LLVM-NEXT:    call void @_ZN1SD1Ev(ptr {{.*}} %[[TMP]])
// LLVM:         %[[SEL:.*]] = load i32, ptr %[[DEST]]
// LLVM:         switch i32 %[[SEL]], label %{{.*}} [
// LLVM:           i32 [[RET_DEST]], label %[[RETURN:.*]]
// LLVM:       [[RETURN]]:
// LLVM-NEXT:    %[[RV:.*]] = load i32, ptr %[[RETVAL]]
// LLVM-NEXT:    ret i32 %[[RV]]
// LLVM:       [[LPAD]]:
// LLVM:         landingpad { ptr, i32 }
// LLVM-NEXT:            cleanup
// LLVM:         %[[EH_ACTIVE:.*]] = load i8, ptr %[[FLAG]]
// LLVM:         %[[EH_ACTIVE_BOOL:.*]] = trunc i8 %[[EH_ACTIVE]] to i1
// LLVM:         br i1 %[[EH_ACTIVE_BOOL]], label %[[EH_DTOR:.*]], label %{{.*}}
// LLVM:       [[EH_DTOR]]:
// LLVM:         call void @_ZN1SD1Ev(ptr {{.*}} %[[TMP]])
// LLVM:         resume { ptr, i32 }

// OGCG-LABEL: define dso_local {{.*}}i32 @_Z30while_cond_var_extended_returni(
// OGCG:         call void @_Z5makeSv(ptr {{.*}} %[[TMP:.*]])
// OGCG:       {{^}}lpad:
// OGCG:         call void @_ZN1SD1Ev(ptr {{.*}} %[[TMP]])
// OGCG:         store i32 %{{.*}}, ptr %[[RETVAL:.*]], align 4
// OGCG-NEXT:    store i32 [[RET_DEST:[0-9]+]], ptr %[[DEST:.*]], align 4
// OGCG-NEXT:    br label %[[CLEANUP:.*]]
// OGCG:       [[CLEANUP]]:
// OGCG-NEXT:    call void @_ZN1SD1Ev(ptr {{.*}} %[[TMP]])
// OGCG:         switch i32 %{{.*}}, label %{{.*}} [
// OGCG:           i32 [[RET_DEST]], label %[[RETURN:.*]]
// OGCG:       [[RETURN]]:
// OGCG:         ret i32

// A for loop gets a cleanup region for the extended temporary even though the
// reference itself needs no cleanup.
void for_cond_var_extended() {
  for (; S &&s = makeS();)
    use(s);
}

// CIR-LABEL: cir.func {{.*}} @_Z21for_cond_var_extendedv
// CIR:         %[[FLAG:.*]] = cir.alloca {{.*}}"cond.cleanup.isactive"{{.*}} : !cir.ptr<!cir.bool>
// CIR:         %[[TMP:.*]] = cir.alloca {{.*}}"ref.tmp0"{{.*}} : !cir.ptr<!rec_S>
// CIR:         cir.for : cond {
// CIR:           %[[FALSE:.*]] = cir.const #false
// CIR-NEXT:      cir.store %[[FALSE]], %[[FLAG]]
// CIR-NEXT:      cir.call @_Z5makeSv(%[[TMP]])
// CIR-NEXT:      cir.store {{.*}}%[[TMP]], %{{.*}}
// CIR-NEXT:      %[[TRUE:.*]] = cir.const #true
// CIR-NEXT:      cir.store %[[TRUE]], %[[FLAG]]
// CIR:           cir.condition
// CIR:         } cleanup all {
// CIR-NEXT:      %[[ACTIVE:.*]] = cir.load {{.*}}%[[FLAG]]
// CIR-NEXT:      cir.if %[[ACTIVE]] {
// CIR-NEXT:        cir.call @_ZN1SD1Ev(%[[TMP]]) nothrow{{.*}}
// CIR-NEXT:      }
// CIR-NEXT:      cir.yield
// CIR-NEXT:    }

// The temporary in one arm of the conditional operator is destroyed at the end
// of the full-expression, in that full-expression's own cleanup scope. Only the
// result, which the variable extends, is destroyed with the variable.
void while_cond_var_extended_conditional(int n) {
  while (S &&s = n ? chainS(makeS()) : makeS())
    use(s);
}

// CIR-LABEL: cir.func {{.*}} @_Z35while_cond_var_extended_conditionali
// CIR:         %[[FLAG:.*]] = cir.alloca {{.*}}"cond.cleanup.isactive"{{.*}} : !cir.ptr<!cir.bool>
// CIR:         %[[ARG_TMP:.*]] = cir.alloca {{.*}}"ref.tmp1"{{.*}} : !cir.ptr<!rec_S>
// CIR:         %[[TMP:.*]] = cir.alloca {{.*}}"ref.tmp0"{{.*}} : !cir.ptr<!rec_S>
// CIR:         cir.while {
// CIR:           %[[FALSE:.*]] = cir.const #false
// CIR-NEXT:      cir.store %[[FALSE]], %[[FLAG]]
// CIR:           cir.cleanup.scope {
// CIR:             cir.if {{.*}} {
// CIR:               cir.call @_Z6chainSRK1S(%[[TMP]], %[[ARG_TMP]])
// CIR:             } else {
// CIR:               cir.call @_Z5makeSv(%[[TMP]])
// CIR:             }
// CIR:           } cleanup all {
// CIR:             cir.call @_ZN1SD1Ev(%[[ARG_TMP]]) nothrow
// CIR:           }
// CIR-NOT:       @_ZN1SD1Ev
// CIR:           %[[TRUE:.*]] = cir.const #true
// CIR-NEXT:      cir.store %[[TRUE]], %[[FLAG]]
// CIR-NOT:       @_ZN1SD1Ev
// CIR:           cir.condition
// CIR:         } do {
// CIR:         } cleanup all {
// CIR-NEXT:      %[[ACTIVE:.*]] = cir.load {{.*}}%[[FLAG]]
// CIR-NEXT:      cir.if %[[ACTIVE]] {
// CIR-NEXT:        cir.call @_ZN1SD1Ev(%[[TMP]]) nothrow{{.*}}
// CIR-NEXT:      }
// CIR-NEXT:      cir.yield
// CIR-NEXT:    }

struct P {
  int x;
  S s;
};
P makeP();
void use(const int &);

// A reference to a member of a temporary extends the whole temporary, so its
// destructor runs at the end of each iteration even though the variable's type
// is trivially destructible.
void while_cond_var_extended_subobject() {
  while (const int &i = makeP().x)
    use(i);
}

// CIR-LABEL: cir.func {{.*}} @_Z33while_cond_var_extended_subobjectv
// CIR:         %[[FLAG:.*]] = cir.alloca {{.*}}"cond.cleanup.isactive"{{.*}} : !cir.ptr<!cir.bool>
// CIR:         %[[TMP:.*]] = cir.alloca {{.*}}"ref.tmp0"{{.*}} : !cir.ptr<!rec_P>
// CIR:         cir.while {
// CIR:           %[[FALSE:.*]] = cir.const #false
// CIR-NEXT:      cir.store %[[FALSE]], %[[FLAG]]
// CIR-NEXT:      cir.call @_Z5makePv(%[[TMP]])
// CIR-NEXT:      cir.cleanup.scope {
// CIR-NEXT:        cir.get_member %[[TMP]][0] {name = "x"}
// CIR:           } cleanup eh {
// CIR-NEXT:        cir.call @_ZN1PD1Ev(%[[TMP]]) nothrow
// CIR:           %[[TRUE:.*]] = cir.const #true
// CIR-NEXT:      cir.store %[[TRUE]], %[[FLAG]]
// CIR:           cir.condition
// CIR:         } cleanup all {
// CIR-NEXT:      %[[ACTIVE:.*]] = cir.load {{.*}}%[[FLAG]]
// CIR-NEXT:      cir.if %[[ACTIVE]] {
// CIR-NEXT:        cir.call @_ZN1PD1Ev(%[[TMP]]) nothrow{{.*}}
// CIR-NEXT:      }
// CIR-NEXT:      cir.yield
// CIR-NEXT:    }

struct ThrowingDtor {
  ~ThrowingDtor() noexcept(false);
  explicit operator bool() const;
};
ThrowingDtor makeThrowingDtor();

struct ThrowingDtorHolder {
  const ThrowingDtor &t;
  ~ThrowingDtorHolder();
  explicit operator bool() const;
};
void use(const ThrowingDtorHolder &);

// The destructor of the temporary may throw, but it runs last, so an exception
// from it skips no other destructor. On the EH path, it terminates.
void while_cond_var_extended_throwing_last() {
  while (ThrowingDtorHolder h{makeThrowingDtor()})
    use(h);
}

// CIR-LABEL: cir.func {{.*}} @_Z37while_cond_var_extended_throwing_lastv
// CIR:         %[[FLAG:.*]] = cir.alloca {{.*}}"cond.cleanup.isactive"{{.*}} : !cir.ptr<!cir.bool>
// CIR:         %[[TMP:.*]] = cir.alloca {{.*}}"ref.tmp0"{{.*}} : !cir.ptr<!rec_ThrowingDtor>
// CIR:         %[[H:.*]] = cir.alloca {{.*}}"h"{{.*}} : !cir.ptr<!rec_ThrowingDtorHolder>
// CIR:         cir.while {
// CIR:         } cleanup all {
// CIR-NEXT:      %[[ACTIVE:.*]] = cir.load {{.*}}%[[FLAG]]
// CIR-NEXT:      cir.if %[[ACTIVE]] {
// CIR-NEXT:        cir.call @_ZN18ThrowingDtorHolderD1Ev(%[[H]]) nothrow{{.*}}
// CIR-NEXT:      }
// CIR-NEXT:      %[[ACTIVE2:.*]] = cir.load {{.*}}%[[FLAG]]
// CIR-NEXT:      cir.if %[[ACTIVE2]] {
// CIR-NEXT:        cir.call @_ZN12ThrowingDtorD1Ev(%[[TMP]]) : {{.*}}
// CIR-NEXT:      }
// CIR-NEXT:      cir.yield
// CIR-NEXT:    }

// LLVM-LABEL: define dso_local void @_Z37while_cond_var_extended_throwing_lastv() {{.*}} personality ptr @__gxx_personality_v0 {
// LLVM:         %[[TMP:.*]] = alloca %struct.ThrowingDtor
// LLVM:         %[[H:.*]] = alloca %struct.ThrowingDtorHolder
// LLVM:         call void @_ZN18ThrowingDtorHolderD1Ev(ptr {{.*}} %[[H]])
// LLVM:         call void @_ZN12ThrowingDtorD1Ev(ptr {{.*}} %[[TMP]])
// LLVM:         landingpad { ptr, i32 }
// LLVM-NEXT:            cleanup
// LLVM:         call void @_ZN18ThrowingDtorHolderD1Ev(ptr {{.*}} %[[H]])
// LLVM:         invoke void @_ZN12ThrowingDtorD1Ev(ptr {{.*}} %[[TMP]])
// LLVM-NEXT:            to label %{{.*}} unwind label %[[TERMINATE:.*]]
// LLVM:         resume { ptr, i32 }
// LLVM:       [[TERMINATE]]:
// LLVM-NEXT:    landingpad { ptr, i32 }
// LLVM-NEXT:            catch ptr null
// LLVM:         call void @__clang_call_terminate(

// OGCG-LABEL: define dso_local void @_Z37while_cond_var_extended_throwing_lastv() {{.*}} personality ptr @__gxx_personality_v0 {
// OGCG:       lpad:
// OGCG:         call void @_ZN18ThrowingDtorHolderD1Ev(ptr {{.*}} %[[H:[^)]*]])
// OGCG-NEXT:    invoke void @_ZN12ThrowingDtorD1Ev(ptr {{.*}} %[[TMP:[^)]*]])
// OGCG-NEXT:            to label %{{.*}} unwind label %terminate.lpad
// OGCG:       {{^}}cleanup:
// OGCG-NEXT:    call void @_ZN18ThrowingDtorHolderD1Ev(ptr {{.*}} %[[H]])
// OGCG-NEXT:    call void @_ZN12ThrowingDtorD1Ev(ptr {{.*}} %[[TMP]])
