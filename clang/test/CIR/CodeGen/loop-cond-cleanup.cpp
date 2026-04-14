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
