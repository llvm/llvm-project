// RUN: %clang_cc1 -no-enable-noundef-analysis %s -triple=x86_64-linux-gnu -fclangir -emit-cir -std=c++17 -fcxx-exceptions -fexceptions -o %t.cir
// RUN: FileCheck -check-prefixes=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -no-enable-noundef-analysis %s -triple=x86_64-linux-gnu -fclangir -emit-llvm -std=c++17 -fcxx-exceptions -fexceptions -o %t-cir.ll
// RUN: FileCheck -check-prefixes=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -no-enable-noundef-analysis %s -triple=x86_64-linux-gnu -emit-llvm -std=c++17 -fcxx-exceptions -fexceptions -o %t.ll
// RUN: FileCheck -check-prefixes=OGCG --input-file=%t.ll %s

struct S {
  S();
  ~S();
  operator bool();
};

S makeS();

void while_cond_cleanup(int n) {
  while (makeS())
    --n;
}

// CIR-LABEL: cir.func {{.*}} @_Z18while_cond_cleanupi
// CIR:   cir.while {
// CIR:     cir.call @_Z5makeSv()
// CIR:     cir.cleanup.scope {
// CIR:       cir.call @_ZN1ScvbEv(
// CIR:     } cleanup all {
// CIR:       cir.call @_ZN1SD1Ev({{.*}}) nothrow
// CIR:     }
// CIR:     cir.condition(
// CIR:   } do {

// LLVM-LABEL: define dso_local void @_Z18while_cond_cleanupi(i32 %0) {{.*}} personality ptr @__gxx_personality_v0 {
// LLVM:   %[[TMP:.*]] = alloca %struct.S
// LLVM:   call %struct.S @_Z5makeSv()
// LLVM:   invoke i1 @_ZN1ScvbEv(ptr {{.*}} %[[TMP]])
// LLVM:           to label %[[CONT:.*]] unwind label %[[UNWIND:.*]]
// LLVM: [[CONT]]:
// LLVM:   call void @_ZN1SD1Ev(ptr {{.*}} %[[TMP]])
// LLVM: [[UNWIND]]:
// LLVM:   landingpad { ptr, i32 }
// LLVM:          cleanup
// LLVM:   call void @_ZN1SD1Ev(ptr {{.*}} %[[TMP]])
// LLVM:   resume { ptr, i32 }
// LLVM:   br i1
// LLVM:   ret void

// OGCG-LABEL: define dso_local void @_Z18while_cond_cleanupi(i32 %n) {{.*}} personality ptr @__gxx_personality_v0 {
// OGCG: while.cond:
// OGCG:   %[[CALL:.*]] = invoke {{.*}} i1 @_ZN1ScvbEv(
// OGCG:           to label %[[CONT:.*]] unwind label %[[LPAD:.*]]
// OGCG: [[CONT]]:
// OGCG:   call void @_ZN1SD1Ev(
// OGCG:   br i1 %[[CALL]], label %[[BODY:.*]], label %[[END:.*]]
// OGCG: [[BODY]]:
// OGCG:   br label %while.cond
// OGCG: [[LPAD]]:
// OGCG:   landingpad { ptr, i32 }
// OGCG:          cleanup
// OGCG:   call void @_ZN1SD1Ev(
// OGCG: [[END]]:
// OGCG:   ret void

void do_while_cond_cleanup(int n) {
  do {
    --n;
  } while (makeS());
}

// CIR-LABEL: cir.func {{.*}} @_Z21do_while_cond_cleanupi
// CIR:   cir.do {
// CIR:     cir.yield
// CIR:   } while {
// CIR:     cir.call @_Z5makeSv()
// CIR:     cir.cleanup.scope {
// CIR:       cir.call @_ZN1ScvbEv(
// CIR:     } cleanup all {
// CIR:       cir.call @_ZN1SD1Ev({{.*}}) nothrow
// CIR:     }
// CIR:     cir.condition(

// LLVM-LABEL: define dso_local void @_Z21do_while_cond_cleanupi(i32 %0) {{.*}} personality ptr @__gxx_personality_v0 {
// LLVM:   %[[TMP:.*]] = alloca %struct.S
// LLVM:   call %struct.S @_Z5makeSv()
// LLVM:   invoke i1 @_ZN1ScvbEv(ptr {{.*}} %[[TMP]])
// LLVM:           to label %[[CONT:.*]] unwind label %[[UNWIND:.*]]
// LLVM: [[CONT]]:
// LLVM:   call void @_ZN1SD1Ev(ptr {{.*}} %[[TMP]])
// LLVM: [[UNWIND]]:
// LLVM:   landingpad { ptr, i32 }
// LLVM:          cleanup
// LLVM:   call void @_ZN1SD1Ev(ptr {{.*}} %[[TMP]])
// LLVM:   resume { ptr, i32 }
// LLVM:   br i1
// LLVM:   ret void

// OGCG-LABEL: define dso_local void @_Z21do_while_cond_cleanupi(i32 %n) {{.*}} personality ptr @__gxx_personality_v0 {
// OGCG: do.cond:
// OGCG:   %[[CALL:.*]] = invoke {{.*}} i1 @_ZN1ScvbEv(
// OGCG:           to label %[[CONT:.*]] unwind label %[[LPAD:.*]]
// OGCG: [[CONT]]:
// OGCG:   call void @_ZN1SD1Ev(
// OGCG:   br i1 %[[CALL]]
// OGCG:   ret void
// OGCG: [[LPAD]]:
// OGCG:   landingpad { ptr, i32 }
// OGCG:          cleanup
// OGCG:   call void @_ZN1SD1Ev(

void for_cond_cleanup(int n) {
  for (int i = 0; makeS(); ++i)
    --n;
}

// CIR-LABEL: cir.func {{.*}} @_Z16for_cond_cleanupi
// CIR:   cir.for : cond {
// CIR:     cir.call @_Z5makeSv()
// CIR:     cir.cleanup.scope {
// CIR:       cir.call @_ZN1ScvbEv(
// CIR:     } cleanup all {
// CIR:       cir.call @_ZN1SD1Ev({{.*}}) nothrow
// CIR:     }
// CIR:     cir.condition(
// CIR:   } body {

// LLVM-LABEL: define dso_local void @_Z16for_cond_cleanupi(i32 %0) {{.*}} personality ptr @__gxx_personality_v0 {
// LLVM:   %[[TMP:.*]] = alloca %struct.S
// LLVM:   call %struct.S @_Z5makeSv()
// LLVM:   invoke i1 @_ZN1ScvbEv(ptr {{.*}} %[[TMP]])
// LLVM:           to label %[[CONT:.*]] unwind label %[[UNWIND:.*]]
// LLVM: [[CONT]]:
// LLVM:   call void @_ZN1SD1Ev(ptr {{.*}} %[[TMP]])
// LLVM: [[UNWIND]]:
// LLVM:   landingpad { ptr, i32 }
// LLVM:          cleanup
// LLVM:   call void @_ZN1SD1Ev(ptr {{.*}} %[[TMP]])
// LLVM:   resume { ptr, i32 }
// LLVM:   br i1
// LLVM:   ret void

// OGCG-LABEL: define dso_local void @_Z16for_cond_cleanupi(i32 %n) {{.*}} personality ptr @__gxx_personality_v0 {
// OGCG: for.cond:
// OGCG:   %[[CALL:.*]] = invoke {{.*}} i1 @_ZN1ScvbEv(
// OGCG:           to label %[[CONT:.*]] unwind label %[[LPAD:.*]]
// OGCG: [[CONT]]:
// OGCG:   call void @_ZN1SD1Ev(
// OGCG:   br i1 %[[CALL]], label %[[BODY:.*]], label %[[END:.*]]
// OGCG: [[LPAD]]:
// OGCG:   landingpad { ptr, i32 }
// OGCG:          cleanup
// OGCG:   call void @_ZN1SD1Ev(
// OGCG: [[END]]:
// OGCG:   ret void

void for_step_cleanup(int n) {
  for (int i = 0; i < n; (void)makeS())
    --n;
}

// CIR-LABEL: cir.func {{.*}} @_Z16for_step_cleanupi
// CIR:   cir.for : cond {
// CIR:   } body {
// CIR:   } step {
// CIR:     cir.call @_Z5makeSv()
// CIR:     cir.cleanup.scope {
// CIR:     } cleanup all {
// CIR:       cir.call @_ZN1SD1Ev({{.*}}) nothrow
// CIR:     }
// CIR:   }

// LLVM-LABEL: define dso_local void @_Z16for_step_cleanupi(i32 %0) {{.*}} {
// LLVM:   %[[TMP:.*]] = alloca %struct.S
// LLVM:   call %struct.S @_Z5makeSv()
// LLVM:   call void @_ZN1SD1Ev(ptr {{.*}} %[[TMP]])
// LLVM:   br label
// LLVM:   ret void

// OGCG-LABEL: define dso_local void @_Z16for_step_cleanupi(i32 %n) {{.*}} {
// OGCG: for.inc:
// OGCG:   call void @_Z5makeSv(
// OGCG:   call void @_ZN1SD1Ev(
// OGCG:   br label %for.cond

struct EndSentinel {};

struct Iter {
  int operator*();
  Iter &operator++();
};

S operator!=(Iter, EndSentinel);

struct Range {
  Iter begin();
  EndSentinel end();
};

Range getRange();

void range_for_cond_cleanup() {
  for (int x : getRange()) {
    (void)x;
  }
}

// CIR-LABEL: cir.func {{.*}} @_Z22range_for_cond_cleanupv
// CIR:   cir.for : cond {
// CIR:     cir.call @_Zne4Iter11EndSentinel(
// CIR:     cir.cleanup.scope {
// CIR:       cir.call @_ZN1ScvbEv(
// CIR:     } cleanup all {
// CIR:       cir.call @_ZN1SD1Ev({{.*}}) nothrow
// CIR:     }
// CIR:     cir.condition(
// CIR:   } body {

// LLVM-LABEL: define dso_local void @_Z22range_for_cond_cleanupv() {{.*}} personality ptr @__gxx_personality_v0 {
// LLVM:   %[[TMP:.*]] = alloca %struct.S
// LLVM:   call %struct.S @_Zne4Iter11EndSentinel(
// LLVM:   invoke i1 @_ZN1ScvbEv(ptr {{.*}} %[[TMP]])
// LLVM:           to label %[[CONT:.*]] unwind label %[[UNWIND:.*]]
// LLVM: [[CONT]]:
// LLVM:   call void @_ZN1SD1Ev(ptr {{.*}} %[[TMP]])
// LLVM: [[UNWIND]]:
// LLVM:   landingpad { ptr, i32 }
// LLVM:          cleanup
// LLVM:   call void @_ZN1SD1Ev(ptr {{.*}} %[[TMP]])
// LLVM:   resume { ptr, i32 }
// LLVM:   br i1
// LLVM:   ret void

// OGCG-LABEL: define dso_local void @_Z22range_for_cond_cleanupv() {{.*}} personality ptr @__gxx_personality_v0 {
// OGCG: for.cond:
// OGCG:   %[[CALL:.*]] = invoke {{.*}} i1 @_ZN1ScvbEv(
// OGCG:           to label %[[CONT:.*]] unwind label %[[LPAD:.*]]
// OGCG: [[CONT]]:
// OGCG:   call void @_ZN1SD1Ev(
// OGCG:   br i1 %[[CALL]], label %[[BODY:.*]], label %[[END:.*]]
// OGCG: [[LPAD]]:
// OGCG:   landingpad { ptr, i32 }
// OGCG:          cleanup
// OGCG:   call void @_ZN1SD1Ev(
// OGCG: [[END]]:
// OGCG:   ret void

void while_body_temp_ref() {
  while (0)
    const S &op = 1 ? S() : S();
}

// CIR-LABEL: cir.func {{.*}} @_Z19while_body_temp_refv
// CIR:   cir.scope {
// CIR:     cir.while {
// CIR:     } do {
// CIR:       cir.if
// CIR:       cir.cleanup.scope {
// CIR:       } cleanup eh {
// CIR:         cir.call @_ZN1SD1Ev({{.*}}) nothrow
// CIR:       }
// CIR:       cir.cleanup.scope {
// CIR:       } cleanup all {
// CIR:         cir.call @_ZN1SD1Ev({{.*}}) nothrow
// CIR:       }
// CIR:     }
// CIR:   }
// CIR:   cir.return

// LLVM-LABEL: define dso_local void @_Z19while_body_temp_refv()
// LLVM:   %[[REF_TMP:.*]] = alloca %struct.S
// LLVM:   %[[OP:.*]] = alloca ptr
// LLVM:   %[[SPILL:.*]] = alloca ptr
// LLVM:   br label %{{.*}}
// LLVM:   br label %[[LOOP_COND:.*]]
// LLVM: [[LOOP_COND]]:
// LLVM:   br i1 false, label %[[BODY:.*]], label %[[AFTER:.*]]
// LLVM: [[BODY]]:
// LLVM:   br i1 true, label %[[TRUE:.*]], label %[[FALSE:.*]]
// LLVM: [[TRUE]]:
// LLVM:   call void @_ZN1SC1Ev(ptr {{.*}} %[[REF_TMP]])
// LLVM: [[FALSE]]:
// LLVM:   call void @_ZN1SC1Ev(ptr {{.*}} %[[REF_TMP]])
// LLVM:   store ptr %[[REF_TMP]], ptr %[[SPILL]]
// LLVM:   call void @_ZN1SD1Ev(ptr {{.*}} %[[REF_TMP]])
// LLVM:   %[[RELOAD:.*]] = load ptr, ptr %[[SPILL]]
// LLVM:   store ptr %[[RELOAD]], ptr %[[OP]]
// LLVM:   br label %[[LOOP_COND]]
// LLVM: [[AFTER]]:
// LLVM:   ret void

// OGCG-LABEL: define dso_local void @_Z19while_body_temp_refv()
// OGCG:   %[[OP:.*]] = alloca ptr
// OGCG:   %[[REF_TMP:.*]] = alloca %struct.S
// OGCG:   br label %[[WHILE_COND:.*]]
// OGCG: [[WHILE_COND]]:
// OGCG:   br i1 false, label %[[WHILE_BODY:.*]], label %[[WHILE_END:.*]]
// OGCG: [[WHILE_BODY]]:
// OGCG:   br i1 true, label %[[COND_TRUE:.*]], label %[[COND_FALSE:.*]]
// OGCG: [[COND_TRUE]]:
// OGCG:   call void @_ZN1SC1Ev(ptr {{.*}} %[[REF_TMP]])
// OGCG: [[COND_FALSE]]:
// OGCG:   call void @_ZN1SC1Ev(ptr {{.*}} %[[REF_TMP]])
// OGCG:   store ptr %[[REF_TMP]], ptr %[[OP]]
// OGCG:   call void @_ZN1SD1Ev(ptr {{.*}} %[[REF_TMP]])
// OGCG:   br label %[[WHILE_COND]]
// OGCG: [[WHILE_END]]:
// OGCG:   ret void

void for_body_temp_ref() {
  for (int i = 0; i < 0; ++i)
    const S &op = 1 ? S() : S();
}

// CIR-LABEL: cir.func {{.*}} @_Z17for_body_temp_refv
// CIR:   cir.scope {
// CIR:     cir.for : cond {
// CIR:     } body {
// CIR:       cir.if
// CIR:       cir.cleanup.scope {
// CIR:       } cleanup eh {
// CIR:         cir.call @_ZN1SD1Ev({{.*}}) nothrow
// CIR:       }
// CIR:       cir.cleanup.scope {
// CIR:       } cleanup all {
// CIR:         cir.call @_ZN1SD1Ev({{.*}}) nothrow
// CIR:       }
// CIR:     } step {
// CIR:     }
// CIR:   }
// CIR:   cir.return

// LLVM-LABEL: define dso_local void @_Z17for_body_temp_refv()
// LLVM:   %[[I:.*]] = alloca i32
// LLVM:   %[[REF_TMP:.*]] = alloca %struct.S
// LLVM:   %[[OP:.*]] = alloca ptr
// LLVM:   %[[SPILL:.*]] = alloca ptr
// LLVM:   br label %{{.*}}
// LLVM:   store i32 0, ptr %[[I]]
// LLVM:   br label %[[LOOP_COND:.*]]
// LLVM: [[LOOP_COND]]:
// LLVM:   %[[IVAL:.*]] = load i32, ptr %[[I]]
// LLVM:   %[[CMP:.*]] = icmp slt i32 %[[IVAL]], 0
// LLVM:   br i1 %[[CMP]], label %[[BODY:.*]], label %[[AFTER:.*]]
// LLVM: [[BODY]]:
// LLVM:   br i1 true, label %[[TRUE:.*]], label %[[FALSE:.*]]
// LLVM: [[TRUE]]:
// LLVM:   call void @_ZN1SC1Ev(ptr {{.*}} %[[REF_TMP]])
// LLVM: [[FALSE]]:
// LLVM:   call void @_ZN1SC1Ev(ptr {{.*}} %[[REF_TMP]])
// LLVM:   store ptr %[[REF_TMP]], ptr %[[SPILL]]
// LLVM:   call void @_ZN1SD1Ev(ptr {{.*}} %[[REF_TMP]])
// LLVM:   %[[RELOAD:.*]] = load ptr, ptr %[[SPILL]]
// LLVM:   store ptr %[[RELOAD]], ptr %[[OP]]
// LLVM:   %[[OLDI:.*]] = load i32, ptr %[[I]]
// LLVM:   %[[NEWI:.*]] = add nsw i32 %[[OLDI]], 1
// LLVM:   store i32 %[[NEWI]], ptr %[[I]]
// LLVM:   br label %[[LOOP_COND]]
// LLVM: [[AFTER]]:
// LLVM:   ret void

// OGCG-LABEL: define dso_local void @_Z17for_body_temp_refv()
// OGCG:   %[[I:.*]] = alloca i32
// OGCG:   %[[OP:.*]] = alloca ptr
// OGCG:   %[[REF_TMP:.*]] = alloca %struct.S
// OGCG:   store i32 0, ptr %[[I]]
// OGCG:   br label %[[FOR_COND:.*]]
// OGCG: [[FOR_COND]]:
// OGCG:   %[[IVAL:.*]] = load i32, ptr %[[I]]
// OGCG:   %[[CMP:.*]] = icmp slt i32 %[[IVAL]], 0
// OGCG:   br i1 %[[CMP]], label %[[FOR_BODY:.*]], label %[[FOR_END:.*]]
// OGCG: [[FOR_BODY]]:
// OGCG:   br i1 true, label %[[COND_TRUE:.*]], label %[[COND_FALSE:.*]]
// OGCG: [[COND_TRUE]]:
// OGCG:   call void @_ZN1SC1Ev(ptr {{.*}} %[[REF_TMP]])
// OGCG: [[COND_FALSE]]:
// OGCG:   call void @_ZN1SC1Ev(ptr {{.*}} %[[REF_TMP]])
// OGCG:   store ptr %[[REF_TMP]], ptr %[[OP]]
// OGCG:   call void @_ZN1SD1Ev(ptr {{.*}} %[[REF_TMP]])
// OGCG:   br label %[[FOR_INC:.*]]
// OGCG: [[FOR_INC]]:
// OGCG:   %[[OLDI:.*]] = load i32, ptr %[[I]]
// OGCG:   %[[NEWI:.*]] = add nsw i32 %[[OLDI]], 1
// OGCG:   store i32 %[[NEWI]], ptr %[[I]]
// OGCG:   br label %[[FOR_COND]]
// OGCG: [[FOR_END]]:
// OGCG:   ret void

void do_body_temp_ref() {
  do
    const S &op = 1 ? S() : S();
  while (0);
}

// CIR-LABEL: cir.func {{.*}} @_Z16do_body_temp_refv
// CIR:   cir.scope {
// CIR:     cir.do {
// CIR:       cir.if
// CIR:       cir.cleanup.scope {
// CIR:       } cleanup eh {
// CIR:         cir.call @_ZN1SD1Ev({{.*}}) nothrow
// CIR:       }
// CIR:       cir.cleanup.scope {
// CIR:       } cleanup all {
// CIR:         cir.call @_ZN1SD1Ev({{.*}}) nothrow
// CIR:       }
// CIR:     } while {
// CIR:     }
// CIR:   }
// CIR:   cir.return

// LLVM-LABEL: define dso_local void @_Z16do_body_temp_refv()
// LLVM:   %[[REF_TMP:.*]] = alloca %struct.S
// LLVM:   %[[OP:.*]] = alloca ptr
// LLVM:   %[[SPILL:.*]] = alloca ptr
// LLVM:   br label %{{.*}}
// LLVM:   br label %[[BODY:.*]]
// LLVM: [[LOOP_COND:.*]]:
// LLVM:   br i1 false, label %[[BODY]], label %[[AFTER:.*]]
// LLVM: [[BODY]]:
// LLVM:   br i1 true, label %[[TRUE:.*]], label %[[FALSE:.*]]
// LLVM: [[TRUE]]:
// LLVM:   call void @_ZN1SC1Ev(ptr {{.*}} %[[REF_TMP]])
// LLVM: [[FALSE]]:
// LLVM:   call void @_ZN1SC1Ev(ptr {{.*}} %[[REF_TMP]])
// LLVM:   store ptr %[[REF_TMP]], ptr %[[SPILL]]
// LLVM:   call void @_ZN1SD1Ev(ptr {{.*}} %[[REF_TMP]])
// LLVM:   %[[RELOAD:.*]] = load ptr, ptr %[[SPILL]]
// LLVM:   store ptr %[[RELOAD]], ptr %[[OP]]
// LLVM:   br label %[[LOOP_COND]]
// LLVM: [[AFTER]]:
// LLVM:   ret void

// OGCG-LABEL: define dso_local void @_Z16do_body_temp_refv()
// OGCG:   %[[OP:.*]] = alloca ptr
// OGCG:   %[[REF_TMP:.*]] = alloca %struct.S
// OGCG:   br label %[[DO_BODY:.*]]
// OGCG: [[DO_BODY]]:
// OGCG:   br i1 true, label %[[COND_TRUE:.*]], label %[[COND_FALSE:.*]]
// OGCG: [[COND_TRUE]]:
// OGCG:   call void @_ZN1SC1Ev(ptr {{.*}} %[[REF_TMP]])
// OGCG: [[COND_FALSE]]:
// OGCG:   call void @_ZN1SC1Ev(ptr {{.*}} %[[REF_TMP]])
// OGCG:   store ptr %[[REF_TMP]], ptr %[[OP]]
// OGCG:   call void @_ZN1SD1Ev(ptr {{.*}} %[[REF_TMP]])
// OGCG:   br label %[[DO_END:.*]]
// OGCG: [[DO_END]]:
// OGCG:   ret void
