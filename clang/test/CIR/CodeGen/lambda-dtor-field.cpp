// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fexceptions -fcxx-exceptions -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fexceptions -fcxx-exceptions -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fexceptions -fcxx-exceptions -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

struct S {
  S();
  S(int);
  S(const S &);
  ~S();
  int x;
};

void capture_one(S s) {
  auto lam = [s]() {};
}

// CIR-LABEL: @_Z11capture_one1S
// CIR:         %[[LAM:.*]] = cir.alloca !rec_anon{{.*}}, {{.*}} ["lam", init]
// CIR:         %[[FIELD:.*]] = cir.get_member %[[LAM]][0] {name = "s"}
// CIR:         cir.call @_ZN1SC1ERKS_(%[[FIELD]],
// CIR:         cir.cleanup.scope {
// CIR:           cir.yield
// CIR:         } cleanup all {
// CIR:           cir.call @_ZZ11capture_one1SEN3$_0D1Ev(%[[LAM]]){{.*}}
// CIR:           cir.yield
// CIR:         }

// LLVM-LABEL: define internal void @"_ZZ11capture_one1SEN3$_0D2Ev"(
// LLVM:   %[[THIS1:.*]] = load ptr, ptr
// LLVM:   %[[FIELD1:.*]] = getelementptr %[[LAM_TY_1:.*]], ptr %[[THIS1]], i32 0, i32 0
// LLVM:   call void @_ZN1SD1Ev(ptr {{.*}} %[[FIELD1]])
// LLVM:   ret void

// LLVM-LABEL: define dso_local void @_Z11capture_one1S(
// LLVM:   %[[S_ALLOCA:.*]] = alloca %struct.S
// LLVM:   %[[LAM1:.*]] = alloca %[[LAM_TY_1]]
// LLVM:   %[[F1:.*]] = getelementptr %[[LAM_TY_1]], ptr %[[LAM1]], i32 0, i32 0
// LLVM:   call void @_ZN1SC1ERKS_(ptr {{.*}} %[[F1]], ptr {{.*}} %[[S_ALLOCA]])
// LLVM:   call void @"_ZZ11capture_one1SEN3$_0D1Ev"(ptr {{.*}} %[[LAM1]])
// LLVM:   ret void

// OGCG-LABEL: define dso_local void @_Z11capture_one1S(
// OGCG:   %[[LAM1:.*]] = alloca %[[LAM_TY_1:.*]], align 4
// OGCG:   %[[FIELD1:.*]] = getelementptr inbounds nuw %[[LAM_TY_1]], ptr %[[LAM1]], i32 0, i32 0
// OGCG:   call void @_ZN1SC1ERKS_(ptr {{.*}} %[[FIELD1]], ptr {{.*}} %s)
// OGCG:   call void @"_ZZ11capture_one1SEN3$_0D1Ev"(ptr {{.*}} %[[LAM1]])
// OGCG:   ret void

void capture_two(S a, S b) {
  auto lam = [a, b]() {};
}

// CIR-LABEL: @_Z11capture_two1SS_
// CIR:         %[[LAM2:.*]] = cir.alloca !rec_anon{{.*}}, {{.*}} ["lam", init]
// CIR:         %[[FA:.*]] = cir.get_member %[[LAM2]][0] {name = "a"}
// CIR:         cir.call @_ZN1SC1ERKS_(%[[FA]],
// CIR:         cir.cleanup.scope {
// CIR:           %[[FB:.*]] = cir.get_member %[[LAM2]][1] {name = "b"}
// CIR:           cir.call @_ZN1SC1ERKS_(%[[FB]],
// CIR:           cir.yield
// CIR:         } cleanup eh {
// CIR:           cir.call @_ZN1SD1Ev(%[[FA]]){{.*}}
// CIR:           cir.yield
// CIR:         }
// CIR:         cir.cleanup.scope {
// CIR:           cir.yield
// CIR:         } cleanup all {
// CIR:           cir.call @_ZZ11capture_two1SS_EN3$_0D1Ev(%[[LAM2]]){{.*}}
// CIR:           cir.yield
// CIR:         }

// LLVM-LABEL: define internal void @"_ZZ11capture_two1SS_EN3$_0D2Ev"(
// LLVM:   %[[THIS2:.*]] = load ptr, ptr
// LLVM:   %[[FB_D:.*]] = getelementptr %[[LAM_TY_2:.*]], ptr %[[THIS2]], i32 0, i32 1
// LLVM:   call void @_ZN1SD1Ev(ptr {{.*}} %[[FB_D]])
// LLVM:   %[[FA_D:.*]] = getelementptr %[[LAM_TY_2]], ptr %[[THIS2]], i32 0, i32 0
// LLVM:   call void @_ZN1SD1Ev(ptr {{.*}} %[[FA_D]])
// LLVM:   ret void

// LLVM-LABEL: define dso_local void @_Z11capture_two1SS_(%struct.S {{.*}}, %struct.S {{.*}}) #{{.*}} personality ptr @__gxx_personality_v0 {
// LLVM:   %[[A_ALLOCA:.*]] = alloca %struct.S
// LLVM:   %[[B_ALLOCA:.*]] = alloca %struct.S
// LLVM:   %[[LAM2:.*]] = alloca %[[LAM_TY_2]]
// LLVM:   %[[FA:.*]] = getelementptr %[[LAM_TY_2]], ptr %[[LAM2]], i32 0, i32 0
// LLVM:   call void @_ZN1SC1ERKS_(ptr {{.*}} %[[FA]], ptr {{.*}} %[[A_ALLOCA]])
// LLVM:   %[[FB:.*]] = getelementptr %[[LAM_TY_2]], ptr %[[LAM2]], i32 0, i32 1
// LLVM:   invoke void @_ZN1SC1ERKS_(ptr {{.*}} %[[FB]], ptr {{.*}} %[[B_ALLOCA]])
// LLVM:           to label %{{.*}} unwind label %{{.*}}
// LLVM:   call void @"_ZZ11capture_two1SS_EN3$_0D1Ev"(ptr {{.*}} %[[LAM2]])
// LLVM:   ret void

// OGCG-LABEL: define dso_local void @_Z11capture_two1SS_(ptr noundef %a, ptr noundef %b){{.*}}personality ptr @__gxx_personality_v0
// OGCG:   %[[LAM2:.*]] = alloca %[[LAM_TY_2:.*]], align 4
// OGCG:   %[[FA:.*]] = getelementptr inbounds nuw %[[LAM_TY_2]], ptr %[[LAM2]], i32 0, i32 0
// OGCG:   call void @_ZN1SC1ERKS_(ptr {{.*}} %[[FA]], ptr {{.*}} %a)
// OGCG:   %[[FB:.*]] = getelementptr inbounds nuw %[[LAM_TY_2]], ptr %[[LAM2]], i32 0, i32 1
// OGCG:   invoke void @_ZN1SC1ERKS_(ptr {{.*}} %[[FB]], ptr {{.*}} %b)
// OGCG:           to label %{{.*}} unwind label %{{.*}}
// OGCG:   call void @"_ZZ11capture_two1SS_EN3$_0D1Ev"(ptr {{.*}} %[[LAM2]])
// OGCG:   ret void

void capture_mixed(int n, S s) {
  auto lam = [n, s]() {};
}

// CIR-LABEL: @_Z13capture_mixedi1S
// CIR:         %[[LAM3:.*]] = cir.alloca !rec_anon{{.*}}, {{.*}} ["lam", init]
// CIR:         %[[FN:.*]] = cir.get_member %[[LAM3]][0] {name = "n"}
// CIR:         cir.load
// CIR:         cir.store
// CIR:         %[[FS:.*]] = cir.get_member %[[LAM3]][1] {name = "s"}
// CIR:         cir.call @_ZN1SC1ERKS_(%[[FS]],
// CIR:         cir.cleanup.scope {
// CIR:           cir.yield
// CIR:         } cleanup all {
// CIR:           cir.call @_ZZ13capture_mixedi1SEN3$_0D1Ev(%[[LAM3]]){{.*}}
// CIR:           cir.yield
// CIR:         }

// LLVM-LABEL: define internal void @"_ZZ13capture_mixedi1SEN3$_0D2Ev"(
// LLVM:   %[[THIS3:.*]] = load ptr, ptr
// LLVM:   %[[FS_D:.*]] = getelementptr %[[LAM_TY_3:.*]], ptr %[[THIS3]], i32 0, i32 1
// LLVM:   call void @_ZN1SD1Ev(ptr {{.*}} %[[FS_D]])
// LLVM:   ret void

// LLVM-LABEL: define dso_local void @_Z13capture_mixedi1S(
// LLVM:   %[[N_ALLOCA:.*]] = alloca i32
// LLVM:   %[[S_ALLOCA2:.*]] = alloca %struct.S
// LLVM:   %[[LAM3:.*]] = alloca %[[LAM_TY_3]]
// LLVM:   %[[FN:.*]] = getelementptr %[[LAM_TY_3]], ptr %[[LAM3]], i32 0, i32 0
// LLVM:   %[[NVAL:.*]] = load i32, ptr %[[N_ALLOCA]]
// LLVM:   store i32 %[[NVAL]], ptr %[[FN]]
// LLVM:   %[[FS:.*]] = getelementptr %[[LAM_TY_3]], ptr %[[LAM3]], i32 0, i32 1
// LLVM:   call void @_ZN1SC1ERKS_(ptr {{.*}} %[[FS]], ptr {{.*}} %[[S_ALLOCA2]])
// LLVM:   call void @"_ZZ13capture_mixedi1SEN3$_0D1Ev"(ptr {{.*}} %[[LAM3]])
// LLVM:   ret void

// OGCG-LABEL: define dso_local void @_Z13capture_mixedi1S(
// OGCG:   %[[LAM3:.*]] = alloca %[[LAM_TY_3:.*]], align 4
// OGCG:   %[[FN:.*]] = getelementptr inbounds nuw %[[LAM_TY_3]], ptr %[[LAM3]], i32 0, i32 0
// OGCG:   %[[NVAL:.*]] = load i32, ptr %n.addr
// OGCG:   store i32 %[[NVAL]], ptr %[[FN]]
// OGCG:   %[[FS:.*]] = getelementptr inbounds nuw %[[LAM_TY_3]], ptr %[[LAM3]], i32 0, i32 1
// OGCG:   call void @_ZN1SC1ERKS_(ptr {{.*}} %[[FS]], ptr {{.*}} %s)
// OGCG:   call void @"_ZZ13capture_mixedi1SEN3$_0D1Ev"(ptr {{.*}} %[[LAM3]])
// OGCG:   ret void

void capture_local() {
  S s;
  auto lam = [s]() {};
}

// CIR-LABEL: @_Z13capture_localv
// CIR:         %[[S4:.*]] = cir.alloca !rec_S, {{.*}} ["s", init]
// CIR:         %[[LAM4:.*]] = cir.alloca !rec_anon{{.*}}, {{.*}} ["lam", init]
// CIR:         cir.call @_ZN1SC1Ev(%[[S4]])
// CIR:         cir.cleanup.scope {
// CIR:           %[[FL:.*]] = cir.get_member %[[LAM4]][0] {name = "s"}
// CIR:           cir.call @_ZN1SC1ERKS_(%[[FL]],
// CIR:           cir.cleanup.scope {
// CIR:             cir.yield
// CIR:           } cleanup all {
// CIR:             cir.call @_ZZ13capture_localvEN3$_0D1Ev(%[[LAM4]]){{.*}}
// CIR:             cir.yield
// CIR:           }
// CIR:           cir.yield
// CIR:         } cleanup all {
// CIR:           cir.call @_ZN1SD1Ev(%[[S4]]){{.*}}
// CIR:           cir.yield
// CIR:         }

// LLVM-LABEL: define internal void @"_ZZ13capture_localvEN3$_0D2Ev"(
// LLVM:   %[[THIS4:.*]] = load ptr, ptr
// LLVM:   %[[FL_D:.*]] = getelementptr %[[LAM_TY_4:.*]], ptr %[[THIS4]], i32 0, i32 0
// LLVM:   call void @_ZN1SD1Ev(ptr {{.*}} %[[FL_D]])
// LLVM:   ret void

// LLVM-LABEL: define dso_local void @_Z13capture_localv(){{.*}} personality ptr @__gxx_personality_v0
// LLVM:   %[[S_LOCAL:.*]] = alloca %struct.S
// LLVM:   %[[LAM4:.*]] = alloca %[[LAM_TY_4]]
// LLVM:   call void @_ZN1SC1Ev(ptr {{.*}} %[[S_LOCAL]])
// LLVM:   %[[FL:.*]] = getelementptr %[[LAM_TY_4]], ptr %[[LAM4]], i32 0, i32 0
// LLVM:   invoke void @_ZN1SC1ERKS_(ptr {{.*}} %[[FL]], ptr {{.*}} %[[S_LOCAL]])
// LLVM:           to label %{{.*}} unwind label %{{.*}}
// LLVM:   call void @"_ZZ13capture_localvEN3$_0D1Ev"(ptr {{.*}} %[[LAM4]])
// LLVM:   call void @_ZN1SD1Ev(ptr {{.*}} %[[S_LOCAL]])
// LLVM:   ret void

// OGCG-LABEL: define dso_local void @_Z13capture_localv(){{.*}} personality ptr @__gxx_personality_v0
// OGCG:   %[[S_LOCAL:.*]] = alloca %struct.S
// OGCG:   %[[LAM4:.*]] = alloca %[[LAM_TY_4:.*]], align 4
// OGCG:   call void @_ZN1SC1Ev(ptr {{.*}} %[[S_LOCAL]])
// OGCG:   %[[FL:.*]] = getelementptr inbounds nuw %[[LAM_TY_4]], ptr %[[LAM4]], i32 0, i32 0
// OGCG:   invoke void @_ZN1SC1ERKS_(ptr {{.*}} %[[FL]], ptr {{.*}} %[[S_LOCAL]])
// OGCG:           to label %{{.*}} unwind label %{{.*}}
// OGCG:   call void @"_ZZ13capture_localvEN3$_0D1Ev"(ptr {{.*}} %[[LAM4]])
// OGCG:   call void @_ZN1SD1Ev(ptr {{.*}} %[[S_LOCAL]])
// OGCG:   ret void

// This test uses a GNU statement expression inside a lambda init-capture to
// exercise the case where a cleanup is deactivated while its body region
// contains a non-yield exit (cir.return). The cleanup for capture 'a' must
// fire on the early return path but be skipped on normal fallthrough. CIR
// handles this by guarding the cleanup with an active flag that is set to true
// when the cleanup is pushed and false at the deactivation point.
void stmt_expr_return(bool cond) {
  auto lam = [a = S(0), b = S(({
      if (cond) return;
      42;
  }))]() {};
}

// CIR-LABEL: @_Z16stmt_expr_returnb
// CIR:         %[[LAM5:.*]] = cir.alloca !rec_anon{{.*}}, {{.*}} ["lam", init]
// CIR:         %[[ACTIVE:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["cleanup.isactive"]
// CIR:         %[[FA5:.*]] = cir.get_member %[[LAM5]][0] {name = "a"}
// CIR:         cir.call @_ZN1SC1Ei(%[[FA5]],
// CIR:         %[[TRUE:.*]] = cir.const #true
// CIR:         cir.store %[[TRUE]], %[[ACTIVE]]
// CIR:         cir.cleanup.scope {
// CIR:           %[[FB5:.*]] = cir.get_member %[[LAM5]][1] {name = "b"}
// CIR:           cir.if
// CIR:             cir.return
// CIR:           cir.call @_ZN1SC1Ei(%[[FB5]],
// CIR:           %[[FALSE:.*]] = cir.const #false
// CIR:           cir.store %[[FALSE]], %[[ACTIVE]]
// CIR:           cir.yield
// CIR:         } cleanup all {
// CIR:           %[[FLAG:.*]] = cir.load{{.*}} %[[ACTIVE]]
// CIR:           cir.if %[[FLAG]] {
// CIR:             cir.call @_ZN1SD1Ev(%[[FA5]]){{.*}}
// CIR:           }
// CIR:           cir.yield
// CIR:         }
// CIR:         cir.cleanup.scope {
// CIR:           cir.yield
// CIR:         } cleanup all {
// CIR:           cir.call @_ZZ16stmt_expr_returnbEN3$_0D1Ev(%[[LAM5]]){{.*}}
// CIR:           cir.yield
// CIR:         }

// LLVM-LABEL: define internal void @"_ZZ16stmt_expr_returnbEN3$_0D2Ev"(
// LLVM:   %[[THIS5:.*]] = load ptr, ptr
// LLVM:   %[[FB5_D:.*]] = getelementptr %[[LAM_TY_5:.*]], ptr %[[THIS5]], i32 0, i32 1
// LLVM:   call void @_ZN1SD1Ev(ptr {{.*}} %[[FB5_D]])
// LLVM:   %[[FA5_D:.*]] = getelementptr %[[LAM_TY_5]], ptr %[[THIS5]], i32 0, i32 0
// LLVM:   call void @_ZN1SD1Ev(ptr {{.*}} %[[FA5_D]])
// LLVM:   ret void

// LLVM-LABEL: define dso_local void @_Z16stmt_expr_returnb({{.*}}) {{.*}} personality ptr @__gxx_personality_v0 {
// LLVM:   %[[LAM5:.*]] = alloca %[[LAM_TY_5]]
// LLVM:   %[[ACTIVE_ALLOCA:.*]] = alloca i8
// LLVM:   %[[FA5:.*]] = getelementptr %[[LAM_TY_5]], ptr %[[LAM5]], i32 0, i32 0
// LLVM:   call void @_ZN1SC1Ei(ptr {{.*}} %[[FA5]], i32 {{.*}} 0)
// LLVM:   store i8 1, ptr %[[ACTIVE_ALLOCA]]
// LLVM:   %[[FB5:.*]] = getelementptr %[[LAM_TY_5]], ptr %[[LAM5]], i32 0, i32 1
// LLVM:   br i1 %{{.*}},
// The early return path — the active flag is still true, so the cleanup fires.
// LLVM:   invoke void @_ZN1SC1Ei(ptr {{.*}} %[[FB5]],
// LLVM:           to label %{{.*}} unwind label %{{.*}}
// LLVM:   store i8 0, ptr %[[ACTIVE_ALLOCA]]
// The active flag is checked and the cleanup conditionally fires.
// LLVM:   call void @_ZN1SD1Ev(ptr {{.*}} %[[FA5]])
// LLVM:   ret void
// Normal fallthrough path completes with the lambda destructor.
// LLVM:   call void @"_ZZ16stmt_expr_returnbEN3$_0D1Ev"(ptr {{.*}} %[[LAM5]])
// LLVM:   ret void

// OGCG-LABEL: define dso_local void @_Z16stmt_expr_returnb(i1 noundef zeroext %cond){{.*}} personality ptr @__gxx_personality_v0
// OGCG:   %[[LAM5:.*]] = alloca %[[LAM_TY_5:.*]], align 4
// OGCG:   %[[FA5:.*]] = getelementptr inbounds nuw %[[LAM_TY_5]], ptr %[[LAM5]], i32 0, i32 0
// OGCG:   call void @_ZN1SC1Ei(ptr {{.*}} %[[FA5]], i32 {{.*}} 0)
// OGCG:   %[[FB5:.*]] = getelementptr inbounds nuw %[[LAM_TY_5]], ptr %[[LAM5]], i32 0, i32 1
// OGCG:   br i1 %{{.*}},
// The early return path correctly destroys capture 'a' in classic codegen.
// OGCG:   call void @_ZN1SD1Ev(ptr {{.*}} %[[FA5]])
// OGCG:   invoke void @_ZN1SC1Ei(ptr {{.*}} %[[FB5]],
// OGCG:           to label %{{.*}} unwind label %{{.*}}
// OGCG:   call void @"_ZZ16stmt_expr_returnbEN3$_0D1Ev"(ptr {{.*}} %[[LAM5]])
// OGCG:   ret void

// The D2 destructors are emitted after all other functions in OGCG.

// OGCG-LABEL: define internal void @"_ZZ11capture_one1SEN3$_0D2Ev"(
// OGCG:   %[[THIS1:.*]] = load ptr, ptr %this.addr
// OGCG:   %[[FIELD1_D:.*]] = getelementptr inbounds nuw %[[LAM_TY_1]], ptr %[[THIS1]], i32 0, i32 0
// OGCG:   call void @_ZN1SD1Ev(ptr {{.*}} %[[FIELD1_D]])
// OGCG:   ret void

// OGCG-LABEL: define internal void @"_ZZ11capture_two1SS_EN3$_0D2Ev"(
// OGCG:   %[[THIS2:.*]] = load ptr, ptr %this.addr
// OGCG:   %[[FB_D:.*]] = getelementptr inbounds nuw %[[LAM_TY_2]], ptr %[[THIS2]], i32 0, i32 1
// OGCG:   call void @_ZN1SD1Ev(ptr {{.*}} %[[FB_D]])
// OGCG:   %[[FA_D:.*]] = getelementptr inbounds nuw %[[LAM_TY_2]], ptr %[[THIS2]], i32 0, i32 0
// OGCG:   call void @_ZN1SD1Ev(ptr {{.*}} %[[FA_D]])
// OGCG:   ret void

// OGCG-LABEL: define internal void @"_ZZ13capture_mixedi1SEN3$_0D2Ev"(
// OGCG:   %[[THIS3:.*]] = load ptr, ptr %this.addr
// OGCG:   %[[FS_D:.*]] = getelementptr inbounds nuw %[[LAM_TY_3]], ptr %[[THIS3]], i32 0, i32 1
// OGCG:   call void @_ZN1SD1Ev(ptr {{.*}} %[[FS_D]])
// OGCG:   ret void

// OGCG-LABEL: define internal void @"_ZZ13capture_localvEN3$_0D2Ev"(
// OGCG:   %[[THIS4:.*]] = load ptr, ptr %this.addr
// OGCG:   %[[FL_D:.*]] = getelementptr inbounds nuw %[[LAM_TY_4]], ptr %[[THIS4]], i32 0, i32 0
// OGCG:   call void @_ZN1SD1Ev(ptr {{.*}} %[[FL_D]])
// OGCG:   ret void

// OGCG-LABEL: define internal void @"_ZZ16stmt_expr_returnbEN3$_0D2Ev"(
// OGCG:   %[[THIS5:.*]] = load ptr, ptr %this.addr
// OGCG:   %[[FB5_D:.*]] = getelementptr inbounds nuw %[[LAM_TY_5]], ptr %[[THIS5]], i32 0, i32 1
// OGCG:   call void @_ZN1SD1Ev(ptr {{.*}} %[[FB5_D]])
// OGCG:   %[[FA5_D:.*]] = getelementptr inbounds nuw %[[LAM_TY_5]], ptr %[[THIS5]], i32 0, i32 0
// OGCG:   call void @_ZN1SD1Ev(ptr {{.*}} %[[FA5_D]])
// OGCG:   ret void
