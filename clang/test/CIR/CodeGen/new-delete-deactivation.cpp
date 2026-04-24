// RUN: %clang_cc1 -no-enable-noundef-analysis %s -triple=x86_64-linux-gnu -fclangir -emit-cir -std=c++11 -fcxx-exceptions -fexceptions -o %t.cir
// RUN: FileCheck -check-prefixes=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -no-enable-noundef-analysis %s -triple=x86_64-linux-gnu -fclangir -emit-llvm -std=c++11 -fcxx-exceptions -fexceptions -o %t-cir.ll
// RUN: FileCheck -check-prefixes=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -no-enable-noundef-analysis %s -triple=x86_64-linux-gnu -emit-llvm -std=c++11 -fcxx-exceptions -fexceptions -o %t.ll
// RUN: FileCheck -check-prefixes=OGCG --input-file=%t.ll %s

struct A {
  A(int);
  ~A();
};

struct B {
  B();
  ~B();
  operator int();
  int x;
};

B makeB();

A *deact_simple() { return new A(makeB()); }

// CIR-LABEL: cir.func {{.*}} @_Z12deact_simplev() -> !cir.ptr<!rec_A> {
// CIR:   %[[RETVAL:.*]] = cir.alloca !cir.ptr<!rec_A>, !cir.ptr<!cir.ptr<!rec_A>>, ["__retval"]
// CIR:   %[[NEW_RESULT:.*]] = cir.alloca !cir.ptr<!rec_A>, !cir.ptr<!cir.ptr<!rec_A>>, ["__new_result"]
// CIR:   %[[TMP:.*]] = cir.alloca !rec_B, !cir.ptr<!rec_B>, ["ref.tmp0"]
// CIR:   %[[ACTIVE:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["cleanup.isactive"]
// CIR:   %[[PTR:.*]] = cir.call @_Znwm({{.*}}) {{{.*}}builtin}
// CIR:   cir.cleanup.scope {
// CIR:     %[[TRUE:.*]] = cir.const #true
// CIR:     cir.store %[[TRUE]], %[[ACTIVE]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR:     %[[MAKEB:.*]] = cir.call @_Z5makeBv() : () -> !rec_B
// CIR:     cir.store{{.*}} %[[MAKEB]], %[[TMP]] : !rec_B, !cir.ptr<!rec_B>
// CIR:     cir.cleanup.scope {
// CIR:       %[[CONV:.*]] = cir.call @_ZN1BcviEv(%[[TMP]])
// CIR:       cir.call @_ZN1AC1Ei({{.*}})
// CIR:       %[[FALSE:.*]] = cir.const #false
// CIR:       cir.store %[[FALSE]], %[[ACTIVE]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR:     } cleanup all {
// CIR:       cir.call @_ZN1BD1Ev(%[[TMP]]) nothrow
// CIR:     }
// CIR:   } cleanup eh {
// CIR:     %[[IS_ACTIVE:.*]] = cir.load{{.*}} %[[ACTIVE]] : !cir.ptr<!cir.bool>, !cir.bool
// CIR:     cir.if %[[IS_ACTIVE]] {
// CIR:       cir.call @_ZdlPv(%[[PTR]]) nothrow {builtin}
// CIR:     }
// CIR:   }

// LLVM-LABEL: define dso_local ptr @_Z12deact_simplev() {{.*}} personality ptr @__gxx_personality_v0 {
// LLVM:   %[[TMP:.*]] = alloca %struct.B
// LLVM:   %[[ACTIVE:.*]] = alloca i8
// LLVM:   %[[PTR:.*]] = call nonnull ptr @_Znwm(i64 1) #[[ATTR_BUILTIN_NEW:.*]]
// LLVM:   store i8 1, ptr %[[ACTIVE]]
// LLVM:   %[[MAKEB:.*]] = invoke %struct.B @_Z5makeBv()
// LLVM:           to label %[[INVOKE_CONT:.*]] unwind label %[[UNWIND_OUTER:.*]]
// LLVM: [[INVOKE_CONT]]:
// LLVM:   invoke void @_ZN1AC1Ei(ptr {{.*}} %[[PTR]], i32 {{.*}})
// LLVM:           to label %[[INVOKE_CONT2:.*]] unwind label %[[UNWIND_INNER:.*]]
// LLVM: [[INVOKE_CONT2]]:
// LLVM:   store i8 0, ptr %[[ACTIVE]]
// LLVM:   call void @_ZN1BD1Ev(ptr {{.*}} %[[TMP]]) #[[ATTR_NOUNWIND:.*]]
// LLVM: [[UNWIND_INNER]]:
// LLVM:   %[[EXN_INNER:.*]] = landingpad { ptr, i32 }
// LLVM:          cleanup
// LLVM:   call void @_ZN1BD1Ev(ptr {{.*}} %[[TMP]]) #[[ATTR_NOUNWIND]]
// LLVM: [[UNWIND_OUTER]]:
// LLVM:   %[[EXN_OUTER:.*]] = landingpad { ptr, i32 }
// LLVM:          cleanup
// LLVM: [[EH_CLEANUP:.*]]:
// LLVM:   %[[IS_ACTIVE_I8:.*]] = load i8, ptr %[[ACTIVE]]
// LLVM:   %[[IS_ACTIVE:.*]] = trunc i8 %[[IS_ACTIVE_I8]] to i1
// LLVM:   br i1 %[[IS_ACTIVE]], label %[[DO_DELETE:.*]], label %[[SKIP_DELETE:.*]]
// LLVM: [[DO_DELETE]]:
// LLVM:   call void @_ZdlPv(ptr %[[PTR]]) #[[ATTR_BUILTIN_DEL:.*]]

// OGCG-LABEL: define dso_local ptr @_Z12deact_simplev() {{.*}} personality ptr @__gxx_personality_v0 {
// OGCG: entry:
// OGCG:   %[[TMP:.*]] = alloca %struct.B
// OGCG:   %[[EXN_SLOT:.*]] = alloca ptr
// OGCG:   %[[EHSEL_SLOT:.*]] = alloca i32
// OGCG:   %[[ACTIVE:.*]] = alloca i1
// OGCG:   %[[PTR:.*]] = call {{.*}} ptr @_Znwm(i64 1) #[[OGCG_ATTR_BUILTIN_NEW:.*]]
// OGCG:   store i1 true, ptr %[[ACTIVE]]
// OGCG:   invoke void @_Z5makeBv(ptr {{.*}} %[[TMP]])
// OGCG:           to label %[[INVOKE_CONT:.*]] unwind label %[[LPAD_OUTER:.*]]
// OGCG: [[INVOKE_CONT]]:
// OGCG:   invoke void @_ZN1AC1Ei(ptr {{.*}} %[[PTR]], i32 {{.*}})
// OGCG:           to label %[[INVOKE_CONT2:.*]] unwind label %[[LPAD_INNER:.*]]
// OGCG: [[INVOKE_CONT2]]:
// OGCG:   store i1 false, ptr %[[ACTIVE]]
// OGCG:   call void @_ZN1BD1Ev(ptr {{.*}} %[[TMP]]) #[[OGCG_ATTR_NOUNWIND:.*]]
// OGCG: [[LPAD_INNER]]:
// OGCG:   call void @_ZN1BD1Ev(ptr {{.*}} %[[TMP]]) #[[OGCG_ATTR_NOUNWIND]]
// OGCG: [[EH_CLEANUP:.*]]:
// OGCG:   %[[IS_ACTIVE:.*]] = load i1, ptr %[[ACTIVE]]
// OGCG:   br i1 %[[IS_ACTIVE]], label %[[DO_DELETE:.*]], label %[[SKIP_DELETE:.*]]
// OGCG: [[DO_DELETE]]:
// OGCG:   call void @_ZdlPv(ptr %[[PTR]]) #[[OGCG_ATTR_BUILTIN_DEL:.*]]

A *deact_if(bool cond) {
  if (cond)
    return new A(makeB());
  return 0;
}

// CIR-LABEL: cir.func {{.*}} @_Z8deact_ifb
// CIR:   cir.if {{.*}} {
// CIR:     %[[ACTIVE:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["cleanup.isactive"]
// CIR:     %[[PTR:.*]] = cir.call @_Znwm({{.*}}) {{{.*}}builtin}
// CIR:     cir.cleanup.scope {
// CIR:       %[[TRUE:.*]] = cir.const #true
// CIR:       cir.store %[[TRUE]], %[[ACTIVE]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR:       cir.call @_ZN1AC1Ei({{.*}})
// CIR:       %[[FALSE:.*]] = cir.const #false
// CIR:       cir.store %[[FALSE]], %[[ACTIVE]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR:     } cleanup eh {
// CIR:       %[[IS_ACTIVE:.*]] = cir.load{{.*}} %[[ACTIVE]] : !cir.ptr<!cir.bool>, !cir.bool
// CIR:       cir.if %[[IS_ACTIVE]] {
// CIR:         cir.call @_ZdlPv(%[[PTR]]) nothrow {builtin}
// CIR:       }
// CIR:     }
// CIR:   }

// LLVM-LABEL: define dso_local ptr @_Z8deact_ifb(i1 %0) {{.*}} personality ptr @__gxx_personality_v0 {
// LLVM:   br i1 %{{.*}}, label %[[THEN:.*]], label %[[END:.*]]
// LLVM: [[THEN]]:
// LLVM:   %[[PTR:.*]] = call nonnull ptr @_Znwm(i64 1) #[[ATTR_BUILTIN_NEW]]
// LLVM:   store i8 1, ptr %[[ACTIVE:.*]]
// LLVM:   invoke void @_ZN1AC1Ei(ptr {{.*}} %[[PTR]], i32 {{.*}})
// LLVM:           to label %[[CONT:.*]] unwind label %[[UNWIND_INNER:.*]]
// LLVM: [[CONT]]:
// LLVM:   store i8 0, ptr %[[ACTIVE]]
// LLVM: [[UNWIND_INNER]]:
// LLVM:   landingpad { ptr, i32 }
// LLVM:          cleanup
// LLVM: [[EH_CLEANUP:.*]]:
// LLVM:   %[[ACTIVE_I8:.*]] = load i8, ptr %[[ACTIVE]]
// LLVM:   %[[ACTIVE_I1:.*]] = trunc i8 %[[ACTIVE_I8]] to i1
// LLVM:   br i1 %[[ACTIVE_I1]], label %[[DO_DELETE:.*]], label %[[SKIP_DELETE:.*]]
// LLVM: [[DO_DELETE]]:
// LLVM:   call void @_ZdlPv(ptr %[[PTR]])
// LLVM: [[END]]:

// OGCG-LABEL: define dso_local ptr @_Z8deact_ifb(i1 zeroext %cond) {{.*}} personality ptr @__gxx_personality_v0 {
// OGCG: if.then:
// OGCG:   %[[PTR:.*]] = call {{.*}} ptr @_Znwm(i64 1) #[[OGCG_ATTR_BUILTIN_NEW]]
// OGCG:   store i1 true, ptr %[[ACTIVE:.*]]
// OGCG:   invoke void @_ZN1AC1Ei(ptr {{.*}} %[[PTR]], i32 {{.*}})
// OGCG:           to label %[[CONT:.*]] unwind label %[[LPAD_INNER:.*]]
// OGCG: [[CONT]]:
// OGCG:   store i1 false, ptr %[[ACTIVE]]
// OGCG: [[LPAD_INNER]]:
// OGCG:   landingpad { ptr, i32 }
// OGCG:          cleanup
// OGCG: [[EH_CLEANUP:.*]]:
// OGCG:   %[[IS_ACTIVE:.*]] = load i1, ptr %[[ACTIVE]]
// OGCG:   br i1 %[[IS_ACTIVE]], label %[[DO_DELETE:.*]], label %[[SKIP_DELETE:.*]]
// OGCG: [[DO_DELETE]]:
// OGCG:   call void @_ZdlPv(ptr %[[PTR]])

A *deact_ternary(bool cond) { return (new A(makeB()), cond) ? nullptr : nullptr; }

// CIR-LABEL: cir.func {{.*}} @_Z13deact_ternaryb
// CIR:   %[[ACTIVE:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["cleanup.isactive"]
// CIR:   %[[PTR:.*]] = cir.call @_Znwm({{.*}}) {{{.*}}builtin}
// CIR:   cir.cleanup.scope {
// CIR:     %[[TRUE:.*]] = cir.const #true
// CIR:     cir.store %[[TRUE]], %[[ACTIVE]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR:     cir.call @_ZN1AC1Ei({{.*}})
// CIR:     %[[FALSE:.*]] = cir.const #false
// CIR:     cir.store %[[FALSE]], %[[ACTIVE]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR:   } cleanup eh {
// CIR:     %[[IS_ACTIVE:.*]] = cir.load{{.*}} %[[ACTIVE]] : !cir.ptr<!cir.bool>, !cir.bool
// CIR:     cir.if %[[IS_ACTIVE]] {
// CIR:       cir.call @_ZdlPv(%[[PTR]]) nothrow {builtin}
// CIR:     }
// CIR:   }

// LLVM-LABEL: define dso_local ptr @_Z13deact_ternaryb(i1 %0) {{.*}} personality ptr @__gxx_personality_v0 {
// LLVM:   %[[PTR:.*]] = call nonnull ptr @_Znwm(i64 1) #[[ATTR_BUILTIN_NEW]]
// LLVM:   store i8 1, ptr %[[ACTIVE:.*]]
// LLVM:   invoke void @_ZN1AC1Ei(ptr {{.*}} %[[PTR]], i32 {{.*}})
// LLVM:           to label %[[CONT:.*]] unwind label %[[UNWIND_INNER:.*]]
// LLVM: [[CONT]]:
// LLVM:   store i8 0, ptr %[[ACTIVE]]
// LLVM: [[UNWIND_INNER]]:
// LLVM:   landingpad { ptr, i32 }
// LLVM:          cleanup
// LLVM: [[EH_CLEANUP:.*]]:
// LLVM:   %[[ACTIVE_I8:.*]] = load i8, ptr %[[ACTIVE]]
// LLVM:   %[[ACTIVE_I1:.*]] = trunc i8 %[[ACTIVE_I8]] to i1
// LLVM:   br i1 %[[ACTIVE_I1]], label %[[DO_DELETE:.*]], label %[[SKIP_DELETE:.*]]
// LLVM: [[DO_DELETE]]:
// LLVM:   call void @_ZdlPv(ptr %[[PTR]])

// OGCG-LABEL: define dso_local ptr @_Z13deact_ternaryb(i1 zeroext %cond) {{.*}} personality ptr @__gxx_personality_v0 {
// OGCG: entry:
// OGCG:   %[[PTR:.*]] = call {{.*}} ptr @_Znwm(i64 1) #[[OGCG_ATTR_BUILTIN_NEW]]
// OGCG:   store i1 true, ptr %[[ACTIVE:.*]]
// OGCG:   invoke void @_ZN1AC1Ei(ptr {{.*}} %[[PTR]], i32 {{.*}})
// OGCG:           to label %[[CONT:.*]] unwind label %[[LPAD_INNER:.*]]
// OGCG: [[CONT]]:
// OGCG:   store i1 false, ptr %[[ACTIVE]]
// OGCG: [[LPAD_INNER]]:
// OGCG:   landingpad { ptr, i32 }
// OGCG:          cleanup
// OGCG: [[EH_CLEANUP:.*]]:
// OGCG:   %[[IS_ACTIVE:.*]] = load i1, ptr %[[ACTIVE]]
// OGCG:   br i1 %[[IS_ACTIVE]], label %[[DO_DELETE:.*]], label %[[SKIP_DELETE:.*]]
// OGCG: [[DO_DELETE]]:
// OGCG:   call void @_ZdlPv(ptr %[[PTR]])

A *deact_while_cond(int n) {
  while ((new A(makeB()), n > 0))
    --n;
  return 0;
}

// CIR-LABEL: cir.func {{.*}} @_Z16deact_while_condi
// CIR:   %[[ACTIVE:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["cleanup.isactive"]
// CIR:   cir.while {
// CIR:     %[[PTR:.*]] = cir.call @_Znwm({{.*}}) {{{.*}}builtin}
// CIR:     cir.cleanup.scope {
// CIR:       %[[TRUE:.*]] = cir.const #true
// CIR:       cir.store %[[TRUE]], %[[ACTIVE]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR:       cir.call @_ZN1AC1Ei({{.*}})
// CIR:       %[[FALSE:.*]] = cir.const #false
// CIR:       cir.store %[[FALSE]], %[[ACTIVE]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR:     } cleanup eh {
// CIR:       %[[IS_ACTIVE:.*]] = cir.load{{.*}} %[[ACTIVE]] : !cir.ptr<!cir.bool>, !cir.bool
// CIR:       cir.if %[[IS_ACTIVE]] {
// CIR:         cir.call @_ZdlPv(%[[PTR]]) nothrow {builtin}
// CIR:       }
// CIR:     }
// CIR:     cir.condition({{.*}})
// CIR:   } do {

// LLVM-LABEL: define dso_local ptr @_Z16deact_while_condi(i32 %0) {{.*}} personality ptr @__gxx_personality_v0 {
// LLVM:   %[[TMP:.*]] = alloca %struct.B
// LLVM:   %[[ACTIVE:.*]] = alloca i8
// LLVM:   br label %[[WHILE_COND:.*]]
// LLVM: [[WHILE_COND]]:
// LLVM:   %[[PTR:.*]] = call nonnull ptr @_Znwm(i64 1) #[[ATTR_BUILTIN_NEW]]
// LLVM:   store i8 1, ptr %[[ACTIVE]]
// LLVM:   invoke %struct.B @_Z5makeBv()
// LLVM:           to label %[[INVOKE_CONT:.*]] unwind label %[[UNWIND_OUTER:.*]]
// LLVM: [[INVOKE_CONT]]:
// LLVM:   invoke void @_ZN1AC1Ei(ptr {{.*}} %[[PTR]], i32 {{.*}})
// LLVM:           to label %[[INVOKE_CONT2:.*]] unwind label %[[UNWIND_INNER:.*]]
// LLVM: [[INVOKE_CONT2]]:
// LLVM:   store i8 0, ptr %[[ACTIVE]]
// LLVM:   br label %[[NORMAL_CLEANUP:.*]]
// LLVM: [[NORMAL_CLEANUP]]:
// LLVM:   call void @_ZN1BD1Ev(ptr {{.*}} %[[TMP]]) #[[ATTR_NOUNWIND]]
// LLVM: [[UNWIND_INNER]]:
// LLVM:   landingpad { ptr, i32 }
// LLVM:          cleanup
// LLVM:   call void @_ZN1BD1Ev(ptr {{.*}} %[[TMP]]) #[[ATTR_NOUNWIND]]
// LLVM: [[UNWIND_OUTER]]:
// LLVM:   landingpad { ptr, i32 }
// LLVM:          cleanup
// LLVM: [[EH_CLEANUP:.*]]:
// LLVM:   %[[IS_ACTIVE_I8:.*]] = load i8, ptr %[[ACTIVE]]
// LLVM:   %[[IS_ACTIVE:.*]] = trunc i8 %[[IS_ACTIVE_I8]] to i1
// LLVM:   br i1 %[[IS_ACTIVE]], label %[[DO_DELETE:.*]], label %[[SKIP_DELETE:.*]]
// LLVM: [[DO_DELETE]]:
// LLVM:   call void @_ZdlPv(ptr %[[PTR]]) #[[ATTR_BUILTIN_DEL]]
// LLVM: [[SKIP_DELETE]]:
// LLVM:   resume { ptr, i32 } {{.*}}

// OGCG-LABEL: define dso_local ptr @_Z16deact_while_condi(i32 %n) {{.*}} personality ptr @__gxx_personality_v0 {
// OGCG: while.cond:
// OGCG:   %[[PTR:.*]] = call {{.*}} ptr @_Znwm(i64 1) #[[OGCG_ATTR_BUILTIN_NEW]]
// OGCG:   store i1 true, ptr %[[ACTIVE:.*]]
// OGCG:   invoke void @_ZN1AC1Ei(ptr {{.*}} %[[PTR]], i32 {{.*}})
// OGCG:           to label %[[CONT:.*]] unwind label %[[LPAD_INNER:.*]]
// OGCG: [[CONT]]:
// OGCG:   store i1 false, ptr %[[ACTIVE]]
// OGCG:   br i1 %{{.*}}, label %[[BODY:.*]], label %[[END:.*]]
// OGCG: [[LPAD_INNER]]:
// OGCG:   landingpad { ptr, i32 }
// OGCG:          cleanup
// OGCG: [[EH_CLEANUP:.*]]:
// OGCG:   %[[IS_ACTIVE:.*]] = load i1, ptr %[[ACTIVE]]
// OGCG:   br i1 %[[IS_ACTIVE]], label %[[DO_DELETE:.*]], label %[[SKIP_DELETE:.*]]
// OGCG: [[DO_DELETE]]:
// OGCG:   call void @_ZdlPv(ptr %[[PTR]])

A *deact_switch(int kind) {
  switch (kind) {
  case 1:
    return new A(makeB());
  default:
    return 0;
  }
}

// CIR-LABEL: cir.func {{.*}} @_Z12deact_switchi
// CIR:   %[[ACTIVE:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["cleanup.isactive"]
// CIR:   cir.switch({{.*}}) {
// CIR:     cir.case(equal, [#cir.int<1> : !s32i]) {
// CIR:       %[[PTR:.*]] = cir.call @_Znwm({{.*}}) {{{.*}}builtin}
// CIR:       cir.cleanup.scope {
// CIR:         %[[TRUE:.*]] = cir.const #true
// CIR:         cir.store %[[TRUE]], %[[ACTIVE]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR:         cir.call @_ZN1AC1Ei({{.*}})
// CIR:         %[[FALSE:.*]] = cir.const #false
// CIR:         cir.store %[[FALSE]], %[[ACTIVE]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR:       } cleanup eh {
// CIR:         %[[IS_ACTIVE:.*]] = cir.load{{.*}} %[[ACTIVE]] : !cir.ptr<!cir.bool>, !cir.bool
// CIR:         cir.if %[[IS_ACTIVE]] {
// CIR:           cir.call @_ZdlPv(%[[PTR]]) nothrow {builtin}
// CIR:         }
// CIR:       }
// CIR:     }
// CIR:   }

// LLVM-LABEL: define dso_local ptr @_Z12deact_switchi(i32 %0) {{.*}} personality ptr @__gxx_personality_v0 {
// LLVM:   switch i32 %{{.*}}, label %[[DEFAULT:.*]] [
// LLVM:     i32 1, label %[[CASE1:.*]]
// LLVM:   ]
// LLVM: [[CASE1]]:
// LLVM:   %[[PTR:.*]] = call nonnull ptr @_Znwm(i64 1) #[[ATTR_BUILTIN_NEW]]
// LLVM:   store i8 1, ptr %[[ACTIVE:.*]]
// LLVM:   invoke void @_ZN1AC1Ei(ptr {{.*}} %[[PTR]], i32 {{.*}})
// LLVM:           to label %[[CONT:.*]] unwind label %[[UNWIND_INNER:.*]]
// LLVM: [[CONT]]:
// LLVM:   store i8 0, ptr %[[ACTIVE]]
// LLVM: [[UNWIND_INNER]]:
// LLVM:   landingpad { ptr, i32 }
// LLVM:          cleanup
// LLVM: [[EH_CLEANUP:.*]]:
// LLVM:   %[[ACTIVE_I8:.*]] = load i8, ptr %[[ACTIVE]]
// LLVM:   %[[ACTIVE_I1:.*]] = trunc i8 %[[ACTIVE_I8]] to i1
// LLVM:   br i1 %[[ACTIVE_I1]], label %[[DO_DELETE:.*]], label %[[SKIP_DELETE:.*]]
// LLVM: [[DO_DELETE]]:
// LLVM:   call void @_ZdlPv(ptr %[[PTR]])
// LLVM: [[DEFAULT]]:

// OGCG-LABEL: define dso_local ptr @_Z12deact_switchi(i32 %kind) {{.*}} personality ptr @__gxx_personality_v0 {
// OGCG: entry:
// OGCG:   switch i32 %{{.*}}, label %[[DEFAULT:.*]] [
// OGCG:     i32 1, label %[[CASE1:.*]]
// OGCG:   ]
// OGCG: [[CASE1]]:
// OGCG:   %[[PTR:.*]] = call {{.*}} ptr @_Znwm(i64 1) #[[OGCG_ATTR_BUILTIN_NEW]]
// OGCG:   store i1 true, ptr %[[ACTIVE:.*]]
// OGCG:   invoke void @_ZN1AC1Ei(ptr {{.*}} %[[PTR]], i32 {{.*}})
// OGCG:           to label %[[CONT:.*]] unwind label %[[LPAD_INNER:.*]]
// OGCG: [[CONT]]:
// OGCG:   store i1 false, ptr %[[ACTIVE]]
// OGCG: [[LPAD_INNER]]:
// OGCG:   landingpad { ptr, i32 }
// OGCG:          cleanup
// OGCG: [[EH_CLEANUP:.*]]:
// OGCG:   %[[IS_ACTIVE:.*]] = load i1, ptr %[[ACTIVE]]
// OGCG:   br i1 %[[IS_ACTIVE]], label %[[DO_DELETE:.*]], label %[[SKIP_DELETE:.*]]
// OGCG: [[DO_DELETE]]:
// OGCG:   call void @_ZdlPv(ptr %[[PTR]])

// LLVM-DAG: attributes #[[ATTR_BUILTIN_NEW]] = {{{.*}}builtin{{.*}}}
// LLVM-DAG: attributes #[[ATTR_BUILTIN_DEL]] = {{{.*}}builtin{{.*}}}
// LLVM-DAG: attributes #[[ATTR_NOUNWIND]] = {{{.*}}nounwind{{.*}}}
// OGCG-DAG: attributes #[[OGCG_ATTR_BUILTIN_NEW]] = {{{.*}}builtin{{.*}}}
// OGCG-DAG: attributes #[[OGCG_ATTR_BUILTIN_DEL]] = {{{.*}}builtin{{.*}}}
// OGCG-DAG: attributes #[[OGCG_ATTR_NOUNWIND]] = {{{.*}}nounwind{{.*}}}
