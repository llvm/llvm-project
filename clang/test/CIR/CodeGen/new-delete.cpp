// RUN: %clang_cc1 -no-enable-noundef-analysis %s -triple=x86_64-linux-gnu -fclangir -emit-cir -std=c++98 -fcxx-exceptions -fexceptions -o %t.cir
// RUN: FileCheck -check-prefixes=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -no-enable-noundef-analysis %s -triple=x86_64-linux-gnu -fclangir -emit-llvm -std=c++98 -fcxx-exceptions -fexceptions -o %t-cir.ll
// RUN: FileCheck -check-prefixes=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -no-enable-noundef-analysis %s -triple=x86_64-linux-gnu -emit-llvm -std=c++98 -fcxx-exceptions -fexceptions -o %t.ll
// RUN: FileCheck -check-prefixes=OGCG --input-file=%t.ll %s


struct A { A(int); ~A(); void *p; };

A *a() {
  return new A(5);
}

// CIR: cir.func {{.*}} @_Z1av() -> !cir.ptr<!rec_A> {
// CIR:   %[[RETVAL:.*]] = cir.alloca !cir.ptr<!rec_A>, !cir.ptr<!cir.ptr<!rec_A>>, ["__retval"]
// CIR:   %[[NEW_RESULT:.*]] = cir.alloca !cir.ptr<!rec_A>, !cir.ptr<!cir.ptr<!rec_A>>, ["__new_result"]
// CIR:   %[[ALLOC_SIZE:.*]] = cir.const #cir.int<8> : !u64i
// CIR:   %[[PTR:.*]] = cir.call @_Znwm(%[[ALLOC_SIZE]]) {{{.*}}builtin}
// CIR:   cir.cleanup.scope {
// CIR:     %[[PTR_A:.*]] = cir.cast bitcast %[[PTR]] : !cir.ptr<!void> -> !cir.ptr<!rec_A>
// CIR:     cir.store{{.*}} %[[PTR_A]], %[[NEW_RESULT]] : !cir.ptr<!rec_A>, !cir.ptr<!cir.ptr<!rec_A>>
// CIR:     %[[FIVE:.*]] = cir.const #cir.int<5> : !s32i
// CIR:     cir.call @_ZN1AC1Ei(%[[PTR_A]], %[[FIVE]])
// CIR:     cir.yield
// CIR:   } cleanup  eh {
// CIR:     cir.call @_ZdlPv(%[[PTR]]) nothrow {builtin} : (!cir.ptr<!void>) -> ()
// CIR:     cir.yield
// CIR:   }

// LLVM: define {{.*}} ptr @_Z1av() {{.*}} personality ptr @__gxx_personality_v0 {
// LLVM:   %[[RETVAL:.*]] = alloca ptr
// LLVM:   %[[NEW_RESULT:.*]] = alloca ptr
// LLVM:   %[[PTR:.*]] = call ptr @_Znwm(i64 8) #[[ATTR_BUILTIN_NEW:.*]]
// LLVM:   br label %[[EH_SCOPE:.*]]
// LLVM: [[EH_SCOPE]]:
// LLVM:   store ptr %[[PTR]], ptr %[[NEW_RESULT]]
// LLVM:   invoke void @_ZN1AC1Ei(ptr {{.*}} %[[PTR]], i32 5)
// LLVM:           to label %[[INVOKE_CONT:.*]] unwind label %[[UNWIND:.*]]
// LLVM: [[INVOKE_CONT]]:
// LLVM:   br label %[[EH_SCOPE_END:.*]]
// LLVM: [[UNWIND]]:
// LLVM:   %[[EXN:.*]] = landingpad { ptr, i32 }
// LLVM:          cleanup
// LLVM:   %[[EXN_PTR:.*]] = extractvalue { ptr, i32 } %[[EXN]], 0
// LLVM:   %[[TYPEID:.*]] = extractvalue { ptr, i32 } %[[EXN]], 1
// LLVM:   br label %[[EH_CLEANUP:.*]]
// LLVM: [[EH_CLEANUP]]:
// LLVM:   %[[EXN_PTR_PHI:.*]] = phi ptr [ %[[EXN_PTR]], %[[UNWIND]] ]
// LLVM:   %[[TYPEID_PHI:.*]] = phi i32 [ %[[TYPEID]], %[[UNWIND]] ]
// LLVM:   call void @_ZdlPv(ptr %[[PTR]]) #[[ATTR_BUILTIN_DEL:.*]]
// LLVM:   %[[EXN_INSERT:.*]] = insertvalue { ptr, i32 } poison, ptr %[[EXN_PTR_PHI]], 0
// LLVM:   %[[EXN_INSERT_2:.*]] = insertvalue { ptr, i32 } %[[EXN_INSERT]], i32 %[[TYPEID_PHI]], 1
// LLVM:   resume { ptr, i32 } %[[EXN_INSERT_2]]
// LLVM: [[EH_SCOPE_END]]:
// LLVM:   %[[LOAD:.*]] = load ptr, ptr %[[NEW_RESULT]]
// LLVM:   store ptr %[[LOAD]], ptr %[[RETVAL]]
// LLVM:   %[[RET:.*]] = load ptr, ptr %[[RETVAL]]
// LLVM:   ret ptr %[[RET]]

// OGCG: define {{.*}} ptr @_Z1av() {{.*}} personality ptr @__gxx_personality_v0 {
// OGCG:   %[[EXN_SLOT:.*]] = alloca ptr
// OGCG:   %[[EHSELECTOR_SLOT:.*]] = alloca i32
// OGCG:   %[[PTR:.*]] = call {{.*}} ptr @_Znwm(i64 8) #[[OGCG_ATTR_BUILTIN_NEW:.*]]
// OGCG:   invoke void @_ZN1AC1Ei(ptr {{.*}} %[[PTR]], i32 5)
// OGCG:           to label %[[INVOKE_CONT:.*]] unwind label %[[UNWIND:.*]]
// OGCG: [[INVOKE_CONT]]:
// OGCG:   ret ptr %[[PTR]]
// OGCG: [[UNWIND]]:
// OGCG:   %[[EXN:.*]] = landingpad { ptr, i32 }
// OGCG:          cleanup
// OGCG:   %[[EXN_PTR:.*]] = extractvalue { ptr, i32 } %[[EXN]], 0
// OGCG:   store ptr %[[EXN_PTR]], ptr %[[EXN_SLOT]]
// OGCG:   %[[TYPEID:.*]] = extractvalue { ptr, i32 } %[[EXN]], 1
// OGCG:   store i32 %[[TYPEID]], ptr %[[EHSELECTOR_SLOT]]
// OGCG:   call void @_ZdlPv(ptr %[[PTR]]) #[[OGCG_ATTR_BUILTIN_DEL:.*]]
// OGCG:   br label %[[EH_RESUME:.*]]
// OGCG: [[EH_RESUME]]:
// OGCG:   %[[EXN_PTR:.*]] = load ptr, ptr %[[EXN_SLOT]]
// OGCG:   %[[EHSELECTOR:.*]] = load i32, ptr %[[EHSELECTOR_SLOT]]
// OGCG:   %[[EXN_INSERT:.*]] = insertvalue { ptr, i32 } poison, ptr %[[EXN_PTR]], 0
// OGCG:   %[[EXN_INSERT_2:.*]] = insertvalue { ptr, i32 } %[[EXN_INSERT]], i32 %[[EHSELECTOR]], 1
// OGCG:   resume { ptr, i32 } %[[EXN_INSERT_2]]

A *b() {
  extern int foo();
  return new A(foo());
}

// CIR: cir.func {{.*}} @_Z1bv() -> !cir.ptr<!rec_A> {
// CIR:   %[[RETVAL:.*]] = cir.alloca !cir.ptr<!rec_A>, !cir.ptr<!cir.ptr<!rec_A>>, ["__retval"]
// CIR:   %[[NEW_RESULT:.*]] = cir.alloca !cir.ptr<!rec_A>, !cir.ptr<!cir.ptr<!rec_A>>, ["__new_result"]
// CIR:   %[[ALLOC_SIZE:.*]] = cir.const #cir.int<8> : !u64i
// CIR:   %[[PTR:.*]] = cir.call @_Znwm(%[[ALLOC_SIZE]]) {{{.*}}builtin}
// CIR:   cir.cleanup.scope {
// CIR:     %[[PTR_A:.*]] = cir.cast bitcast %[[PTR]] : !cir.ptr<!void> -> !cir.ptr<!rec_A>
// CIR:     cir.store{{.*}} %[[PTR_A]], %[[NEW_RESULT]] : !cir.ptr<!rec_A>, !cir.ptr<!cir.ptr<!rec_A>>
// CIR:     %[[FOO:.*]] = cir.call @_Z3foov() : () -> !s32i
// CIR:     cir.call @_ZN1AC1Ei(%[[PTR_A]], %[[FOO]])
// CIR:     cir.yield
// CIR:   } cleanup  eh {
// CIR:     cir.call @_ZdlPv(%[[PTR]]) nothrow {builtin} : (!cir.ptr<!void>) -> ()
// CIR:     cir.yield
// CIR:   }

// LLVM: define {{.*}} ptr @_Z1bv() {{.*}} personality ptr @__gxx_personality_v0 {
// LLVM:   %[[RETVAL:.*]] = alloca ptr
// LLVM:   %[[NEW_RESULT:.*]] = alloca ptr
// LLVM:   %[[PTR:.*]] = call ptr @_Znwm(i64 8) #[[ATTR_BUILTIN_NEW]]
// LLVM:   br label %[[EH_SCOPE:.*]]
// LLVM: [[EH_SCOPE]]:
// LLVM:   store ptr %[[PTR]], ptr %[[NEW_RESULT]]
// LLVM:   %[[FOO:.*]] = invoke i32 @_Z3foov()
// LLVM:           to label %[[INVOKE_CONT:.*]] unwind label %[[UNWIND:.*]]
// LLVM: [[INVOKE_CONT]]:
// LLVM:   invoke void @_ZN1AC1Ei(ptr {{.*}} %[[PTR]], i32 %[[FOO]])
// LLVM:           to label %[[INVOKE_CONT_2:.*]] unwind label %[[UNWIND:.*]]
// LLVM: [[INVOKE_CONT_2]]:
// LLVM:   br label %[[EH_SCOPE_END:.*]]
// LLVM: [[UNWIND]]:
// LLVM:   %[[EXN:.*]] = landingpad { ptr, i32 }
// LLVM:          cleanup
// LLVM:   %[[EXN_PTR:.*]] = extractvalue { ptr, i32 } %[[EXN]], 0
// LLVM:   %[[TYPEID:.*]] = extractvalue { ptr, i32 } %[[EXN]], 1
// LLVM:   br label %[[EH_CLEANUP:.*]]
// LLVM: [[EH_CLEANUP]]:
// LLVM:   %[[EXN_PTR_PHI:.*]] = phi ptr [ %[[EXN_PTR]], %[[UNWIND]] ]
// LLVM:   %[[TYPEID_PHI:.*]] = phi i32 [ %[[TYPEID]], %[[UNWIND]] ]
// LLVM:   call void @_ZdlPv(ptr %[[PTR]]) #[[ATTR_BUILTIN_DEL]]
// LLVM:   %[[EXN_INSERT:.*]] = insertvalue { ptr, i32 } poison, ptr %[[EXN_PTR_PHI]], 0
// LLVM:   %[[EXN_INSERT_2:.*]] = insertvalue { ptr, i32 } %[[EXN_INSERT]], i32 %[[TYPEID_PHI]], 1
// LLVM:   resume { ptr, i32 } %[[EXN_INSERT_2]]
// LLVM: [[EH_SCOPE_END]]:
// LLVM:   %[[LOAD:.*]] = load ptr, ptr %[[NEW_RESULT]]
// LLVM:   store ptr %[[LOAD]], ptr %[[RETVAL]]
// LLVM:   %[[RET:.*]] = load ptr, ptr %[[RETVAL]]
// LLVM:   ret ptr %[[RET]]

// OGCG: define {{.*}} ptr @_Z1bv() {{.*}} personality ptr @__gxx_personality_v0 {
// OGCG:   %[[EXN_SLOT:.*]] = alloca ptr
// OGCG:   %[[EHSELECTOR_SLOT:.*]] = alloca i32
// OGCG:   %[[PTR:.*]] = call {{.*}} ptr @_Znwm(i64 8) #[[OGCG_ATTR_BUILTIN_NEW]]
// OGCG:   %[[FOO:.*]] = invoke i32 @_Z3foov()
// OGCG:           to label %[[INVOKE_CONT:.*]] unwind label %[[UNWIND:.*]]
// OGCG: [[INVOKE_CONT]]:
// OGCG:   invoke void @_ZN1AC1Ei(ptr {{.*}} %[[PTR]], i32 %[[FOO]])
// OGCG:           to label %[[INVOKE_CONT_2:.*]] unwind label %[[UNWIND:.*]]
// OGCG: [[INVOKE_CONT_2]]:
// OGCG:   ret ptr %[[PTR]]
// OGCG: [[UNWIND]]:
// OGCG:   %[[EXN:.*]] = landingpad { ptr, i32 }
// OGCG:          cleanup
// OGCG:   %[[EXN_PTR:.*]] = extractvalue { ptr, i32 } %[[EXN]], 0
// OGCG:   store ptr %[[EXN_PTR]], ptr %[[EXN_SLOT]]
// OGCG:   %[[TYPEID:.*]] = extractvalue { ptr, i32 } %[[EXN]], 1
// OGCG:   store i32 %[[TYPEID]], ptr %[[EHSELECTOR_SLOT]]
// OGCG:   call void @_ZdlPv(ptr %[[PTR]]) #[[OGCG_ATTR_BUILTIN_DEL]]
// OGCG:   br label %[[EH_RESUME:.*]]
// OGCG: [[EH_RESUME]]:
// OGCG:   %[[EXN_PTR:.*]] = load ptr, ptr %[[EXN_SLOT]]
// OGCG:   %[[EHSELECTOR:.*]] = load i32, ptr %[[EHSELECTOR_SLOT]]
// OGCG:   %[[EXN_INSERT:.*]] = insertvalue { ptr, i32 } poison, ptr %[[EXN_PTR]], 0
// OGCG:   %[[EXN_INSERT_2:.*]] = insertvalue { ptr, i32 } %[[EXN_INSERT]], i32 %[[EHSELECTOR]], 1
// OGCG:   resume { ptr, i32 } %[[EXN_INSERT_2]]

// Class-specific operator new/delete should not get the 'builtin' attribute,
// since they are not replaceable global allocation functions.
struct B {
  B(int);
  ~B();
  void *operator new(__SIZE_TYPE__);
  void operator delete(void *) throw();
  int x;
};

B *c() {
  return new B(5);
}

// CIR: cir.func {{.*}} @_Z1cv() -> !cir.ptr<!rec_B> {
// CIR:   %[[RETVAL:.*]] = cir.alloca !cir.ptr<!rec_B>, !cir.ptr<!cir.ptr<!rec_B>>, ["__retval"]
// CIR:   %[[NEW_RESULT:.*]] = cir.alloca !cir.ptr<!rec_B>, !cir.ptr<!cir.ptr<!rec_B>>, ["__new_result"]
// CIR:   %[[ALLOC_SIZE:.*]] = cir.const #cir.int<4> : !u64i
// CIR:   %[[PTR:.*]] = cir.call @_ZN1BnwEm(%[[ALLOC_SIZE]]) : (!u64i) -> !cir.ptr<!void>
// CIR:   cir.cleanup.scope {
// CIR:     %[[PTR_B:.*]] = cir.cast bitcast %[[PTR]] : !cir.ptr<!void> -> !cir.ptr<!rec_B>
// CIR:     cir.store{{.*}} %[[PTR_B]], %[[NEW_RESULT]] : !cir.ptr<!rec_B>, !cir.ptr<!cir.ptr<!rec_B>>
// CIR:     %[[FIVE:.*]] = cir.const #cir.int<5> : !s32i
// CIR:     cir.call @_ZN1BC1Ei(%[[PTR_B]], %[[FIVE]])
// CIR:     cir.yield
// CIR:   } cleanup  eh {
// CIR:     cir.call @_ZN1BdlEPv(%[[PTR]]) nothrow : (!cir.ptr<!void>) -> ()
// CIR:     cir.yield
// CIR:   }

// LLVM: define {{.*}} ptr @_Z1cv() {{.*}} personality ptr @__gxx_personality_v0 {
// LLVM:   %[[RETVAL:.*]] = alloca ptr
// LLVM:   %[[NEW_RESULT:.*]] = alloca ptr
// LLVM:   %[[PTR:.*]] = call ptr @_ZN1BnwEm(i64 4)
// LLVM:   br label %[[EH_SCOPE:.*]]
// LLVM: [[EH_SCOPE]]:
// LLVM:   store ptr %[[PTR]], ptr %[[NEW_RESULT]]
// LLVM:   invoke void @_ZN1BC1Ei(ptr {{.*}} %[[PTR]], i32 5)
// LLVM:           to label %[[INVOKE_CONT:.*]] unwind label %[[UNWIND:.*]]
// LLVM: [[INVOKE_CONT]]:
// LLVM:   br label %[[EH_SCOPE_END:.*]]
// LLVM: [[UNWIND]]:
// LLVM:   %[[EXN:.*]] = landingpad { ptr, i32 }
// LLVM:          cleanup
// LLVM:   %[[EXN_PTR:.*]] = extractvalue { ptr, i32 } %[[EXN]], 0
// LLVM:   %[[TYPEID:.*]] = extractvalue { ptr, i32 } %[[EXN]], 1
// LLVM:   br label %[[EH_CLEANUP:.*]]
// LLVM: [[EH_CLEANUP]]:
// LLVM:   %[[EXN_PTR_PHI:.*]] = phi ptr [ %[[EXN_PTR]], %[[UNWIND]] ]
// LLVM:   %[[TYPEID_PHI:.*]] = phi i32 [ %[[TYPEID]], %[[UNWIND]] ]
// LLVM:   call void @_ZN1BdlEPv(ptr %[[PTR]])
// LLVM:   %[[EXN_INSERT:.*]] = insertvalue { ptr, i32 } poison, ptr %[[EXN_PTR_PHI]], 0
// LLVM:   %[[EXN_INSERT_2:.*]] = insertvalue { ptr, i32 } %[[EXN_INSERT]], i32 %[[TYPEID_PHI]], 1
// LLVM:   resume { ptr, i32 } %[[EXN_INSERT_2]]
// LLVM: [[EH_SCOPE_END]]:
// LLVM:   %[[LOAD:.*]] = load ptr, ptr %[[NEW_RESULT]]
// LLVM:   store ptr %[[LOAD]], ptr %[[RETVAL]]
// LLVM:   %[[RET:.*]] = load ptr, ptr %[[RETVAL]]
// LLVM:   ret ptr %[[RET]]

// OGCG: define {{.*}} ptr @_Z1cv() {{.*}} personality ptr @__gxx_personality_v0 {
// OGCG:   %[[EXN_SLOT:.*]] = alloca ptr
// OGCG:   %[[EHSELECTOR_SLOT:.*]] = alloca i32
// OGCG:   %[[PTR:.*]] = call ptr @_ZN1BnwEm(i64 4)
// OGCG:   invoke void @_ZN1BC1Ei(ptr {{.*}} %[[PTR]], i32 5)
// OGCG:           to label %[[INVOKE_CONT:.*]] unwind label %[[UNWIND:.*]]
// OGCG: [[INVOKE_CONT]]:
// OGCG:   ret ptr %[[PTR]]
// OGCG: [[UNWIND]]:
// OGCG:   %[[EXN:.*]] = landingpad { ptr, i32 }
// OGCG:          cleanup
// OGCG:   %[[EXN_PTR:.*]] = extractvalue { ptr, i32 } %[[EXN]], 0
// OGCG:   store ptr %[[EXN_PTR]], ptr %[[EXN_SLOT]]
// OGCG:   %[[TYPEID:.*]] = extractvalue { ptr, i32 } %[[EXN]], 1
// OGCG:   store i32 %[[TYPEID]], ptr %[[EHSELECTOR_SLOT]]
// OGCG:   call void @_ZN1BdlEPv(ptr %[[PTR]])
// OGCG:   br label %[[EH_RESUME:.*]]
// OGCG: [[EH_RESUME]]:
// OGCG:   %[[EXN_PTR2:.*]] = load ptr, ptr %[[EXN_SLOT]]
// OGCG:   %[[EHSELECTOR:.*]] = load i32, ptr %[[EHSELECTOR_SLOT]]
// OGCG:   %[[EXN_INSERT:.*]] = insertvalue { ptr, i32 } poison, ptr %[[EXN_PTR2]], 0
// OGCG:   %[[EXN_INSERT_2:.*]] = insertvalue { ptr, i32 } %[[EXN_INSERT]], i32 %[[EHSELECTOR]], 1
// OGCG:   resume { ptr, i32 } %[[EXN_INSERT_2]]

struct C {
  C();
  ~C();
};

C *test_new_delete_conditional(bool cond) {
  return cond ? new C : 0;
}

// CIR-LABEL: @_Z27test_new_delete_conditionalb
// CIR:   %[[CLEANUP_COND:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["cleanup.cond"]
// CIR:   %[[FALSE:.*]] = cir.const #false
// CIR:   cir.store %[[FALSE]], %[[CLEANUP_COND]]
// CIR:   %[[TERN_RESULT:.*]] = cir.ternary
// CIR:     %[[PTR_SAVE:.*]] = cir.alloca !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>, ["cond-cleanup.save"]
// CIR:     %[[SIZE_SAVE:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["cond-cleanup.save"]
// CIR:     %[[NEW_RESULT:.*]] = cir.alloca !cir.ptr<!rec_C>, !cir.ptr<!cir.ptr<!rec_C>>, ["__new_result"]
// CIR:     %[[ALLOC_SIZE:.*]] = cir.const #cir.int<1> : !u64i
// CIR:     %[[NEW_PTR:.*]] = cir.call @_Znwm(%[[ALLOC_SIZE]])
// CIR:     cir.store {{.*}}%[[NEW_PTR]], %[[PTR_SAVE]]
// CIR:     cir.store {{.*}}%[[ALLOC_SIZE]], %[[SIZE_SAVE]]
// CIR:     cir.cleanup.scope {
// CIR:       %[[TRUE:.*]] = cir.const #true
// CIR:       cir.store %[[TRUE]], %[[CLEANUP_COND]]
// CIR:       %[[NEW_AS_C:.*]] = cir.cast bitcast %[[NEW_PTR]] : !cir.ptr<!void> -> !cir.ptr<!rec_C>
// CIR:       cir.store{{.*}} %[[NEW_AS_C]], %[[NEW_RESULT]]
// CIR:       cir.call @_ZN1CC1Ev(%[[NEW_AS_C]])
// CIR:       cir.yield
// CIR:     } cleanup eh {
// CIR:       %[[FLAG:.*]] = cir.load {{.*}} %[[CLEANUP_COND]]
// CIR:       cir.if %[[FLAG]] {
// CIR:         %[[RESTORED_PTR:.*]] = cir.load {{.*}} %[[PTR_SAVE]]
// CIR:         cir.call @_ZdlPv(%[[RESTORED_PTR]])
// CIR:       }
// CIR:       cir.yield
// CIR:     }
// CIR:     %[[TRUE_RESULT:.*]] = cir.load{{.*}} %[[NEW_RESULT]]
// CIR:     cir.yield %[[TRUE_RESULT]] : !cir.ptr<!rec_C>
// CIR:   }, false {
// CIR:     %[[FALSE_RESULT:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!rec_C>
// CIR:     cir.yield %[[FALSE_RESULT]] : !cir.ptr<!rec_C>
// CIR:   }) : (!cir.bool) -> !cir.ptr<!rec_C>

// LLVM-LABEL: @_Z27test_new_delete_conditionalb
// LLVM:   store i8 0, ptr %[[CLEANUP_FLAG:.*]], align 1
// LLVM:   br i1 {{.*}}, label %[[TRUE_BB:.*]], label %[[FALSE_BB:.*]]
// LLVM: [[TRUE_BB]]:
// LLVM:   %[[NEWP:.*]] = call ptr @_Znwm(i64 1) #[[ATTR_BUILTIN_NEW]]
// LLVM:   store ptr %[[NEWP]], ptr %[[SAVE_PTR:.*]], align 8
// LLVM:   store i8 1, ptr %[[CLEANUP_FLAG]], align 1
// LLVM:   invoke void @_ZN1CC1Ev(ptr {{.*}}%[[NEWP]])
// LLVM:     to label %{{.*}} unwind label %[[LPAD:.*]]
// LLVM: [[LPAD]]:
// LLVM:   landingpad { ptr, i32 }
// LLVM-NEXT:   cleanup
// LLVM:   %[[IS_ACTIVE:.*]] = trunc i8 %{{.*}} to i1
// LLVM:   br i1 %[[IS_ACTIVE]], label %[[DO_DEL:.*]], label %[[SKIP_DEL:.*]]
// LLVM: [[DO_DEL]]:
// LLVM:   %[[LOAD_PTR:.*]] = load ptr, ptr %[[SAVE_PTR]], align 8
// LLVM:   call void @_ZdlPv(ptr %[[LOAD_PTR]]) #[[ATTR_BUILTIN_DEL]]
// LLVM:   resume

// OGCG-LABEL: @_Z27test_new_delete_conditionalb
// OGCG:   store i1 false, ptr %[[OG_FLAG:.*]], align 1
// OGCG:   br i1 {{.*}}, label %[[OG_TRUE:.*]], label %[[OG_FALSE:.*]]
// OGCG: [[OG_TRUE]]:
// OGCG:   %[[OG_NEWP:.*]] = call noalias nonnull ptr @_Znwm(i64 1) #[[OGCG_ATTR_BUILTIN_NEW]]
// OGCG:   store ptr %[[OG_NEWP]], ptr %[[OG_SAVE:.*]], align 8
// OGCG:   store i1 true, ptr %[[OG_FLAG]], align 1
// OGCG:   invoke void @_ZN1CC1Ev(ptr {{.*}}%[[OG_NEWP]])
// OGCG:     to label %{{.*}} unwind label %[[OG_LPAD:.*]]
// OGCG: [[OG_LPAD]]:
// OGCG:   landingpad { ptr, i32 }
// OGCG-NEXT:   cleanup
// OGCG:   %[[OG_ACTIVE:.*]] = load i1, ptr %[[OG_FLAG]], align 1
// OGCG:   br i1 %[[OG_ACTIVE]], label %[[OG_DO_DEL:.*]], label %[[OG_SKIP_DEL:.*]]
// OGCG: [[OG_DO_DEL]]:
// OGCG:   %[[OG_LOAD:.*]] = load ptr, ptr %[[OG_SAVE]], align 8
// OGCG:   call void @_ZdlPv(ptr %[[OG_LOAD]]) #[[OGCG_ATTR_BUILTIN_DEL]]
// OGCG:   resume

void *operator new(unsigned long, int);
void operator delete(void *, int);

C *test_new_delete_conditional_with_placement(bool cond, int tag) {
  return cond ? new (tag) C : 0;
}

// CIR-LABEL: @_Z42test_new_delete_conditional_with_placementbi
// CIR:   %[[CLEANUP_COND:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["cleanup.cond"]
// CIR:   %[[TERN_RESULT:.*]] = cir.ternary
// CIR:     %[[PTR_SAVE:.*]] = cir.alloca !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>, ["cond-cleanup.save"]
// CIR:     %[[SIZE_SAVE:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["cond-cleanup.save"]
// CIR:     %[[TAG_SAVE:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["cond-cleanup.save"]
// CIR:     %[[NEW_RESULT:.*]] = cir.alloca !cir.ptr<!rec_C>, !cir.ptr<!cir.ptr<!rec_C>>, ["__new_result"]
// CIR:     %[[ONE:.*]] = cir.const #cir.int<1> : !u64i
// CIR:     %[[TAG_VAL:.*]] = cir.load{{.*}}
// CIR:     %[[NEW_PTR:.*]] = cir.call @_Znwmi(%[[ONE]], %[[TAG_VAL]])
// CIR:     cir.store{{.*}} %[[NEW_PTR]], %[[PTR_SAVE]]
// CIR:     cir.store{{.*}} %[[ONE]], %[[SIZE_SAVE]]
// CIR:     cir.cleanup.scope {
// CIR:       cir.store{{.*}} %[[TAG_VAL]], %[[TAG_SAVE]]
// CIR:       %[[TRUE:.*]] = cir.const #true
// CIR:       cir.store %[[TRUE]], %[[CLEANUP_COND]]
// CIR:       %[[NEW_AS_C:.*]] = cir.cast bitcast %[[NEW_PTR]] : !cir.ptr<!void> -> !cir.ptr<!rec_C>
// CIR:       cir.store{{.*}} %[[NEW_AS_C]], %[[NEW_RESULT]]
// CIR:       cir.call @_ZN1CC1Ev(%[[NEW_AS_C]])
// CIR:     } cleanup eh {
// CIR:       %[[FLAG:.*]] = cir.load{{.*}} %[[CLEANUP_COND]]
// CIR:       cir.if %[[FLAG]] {
// CIR:         %[[RESTORED_PTR:.*]] = cir.load {{.*}} %[[PTR_SAVE]]
// CIR:         %[[RESTORED_TAG:.*]] = cir.load {{.*}} %[[TAG_SAVE]]
// CIR:         cir.call @_ZdlPvi(%[[RESTORED_PTR]], %[[RESTORED_TAG]])
// CIR:       }
// CIR:       cir.yield
// CIR:     }
// CIR:     %[[TRUE_RESULT:.*]] = cir.load{{.*}} %[[NEW_RESULT]]
// CIR:     cir.yield %[[TRUE_RESULT]] : !cir.ptr<!rec_C>
// CIR:   }, false {
// CIR:     %[[FALSE_RESULT:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!rec_C>
// CIR:     cir.yield %[[FALSE_RESULT]] : !cir.ptr<!rec_C>
// CIR:   }) : (!cir.bool) -> !cir.ptr<!rec_C>

// LLVM-LABEL: @_Z42test_new_delete_conditional_with_placementbi
// LLVM:   store i8 0, ptr %[[PL_FLAG:.*]], align 1
// LLVM:   br i1 {{.*}}, label %[[PL_TRUE:.*]], label %[[PL_FALSE:.*]]
// LLVM: [[PL_TRUE]]:
// LLVM:   %[[PL_TAG:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM:   %[[PL_NEWP:.*]] = call ptr @_Znwmi(i64 1, i32 %[[PL_TAG]])
// LLVM:   store ptr %[[PL_NEWP]], ptr %[[PL_SAVE_PTR:.*]], align 8
// LLVM:   store i32 %[[PL_TAG]], ptr %[[PL_SAVE_TAG:.*]], align 4
// LLVM:   store i8 1, ptr %[[PL_FLAG]], align 1
// LLVM:   invoke void @_ZN1CC1Ev(ptr {{.*}}%[[PL_NEWP]])
// LLVM:     to label %{{.*}} unwind label %[[PL_LPAD:.*]]
// LLVM: [[PL_LPAD]]:
// LLVM:   landingpad { ptr, i32 }
// LLVM-NEXT:   cleanup
// LLVM:   %[[PL_ACTIVE:.*]] = trunc i8 %{{.*}} to i1
// LLVM:   br i1 %[[PL_ACTIVE]], label %[[PL_DO_DEL:.*]], label %[[PL_SKIP_DEL:.*]]
// LLVM: [[PL_DO_DEL]]:
// LLVM:   %[[PL_LOAD_PTR:.*]] = load ptr, ptr %[[PL_SAVE_PTR]], align 8
// LLVM:   %[[PL_LOAD_TAG:.*]] = load i32, ptr %[[PL_SAVE_TAG]], align 4
// LLVM:   invoke void @_ZdlPvi(ptr %[[PL_LOAD_PTR]], i32 %[[PL_LOAD_TAG]])

// OGCG-LABEL: @_Z42test_new_delete_conditional_with_placementbi
// OGCG:   store i1 false, ptr %[[OGP_FLAG:.*]], align 1
// OGCG:   br i1 {{.*}}, label %[[OGP_TRUE:.*]], label %[[OGP_FALSE:.*]]
// OGCG: [[OGP_TRUE]]:
// OGCG:   %[[OGP_TAG:.*]] = load i32, ptr %{{.*}}, align 4
// OGCG:   %[[OGP_NEWP:.*]] = call ptr @_Znwmi(i64 1, i32 %[[OGP_TAG]])
// OGCG:   store ptr %[[OGP_NEWP]], ptr %[[OGP_SAVE_PTR:.*]], align 8
// OGCG:   store i32 %[[OGP_TAG]], ptr %[[OGP_SAVE_TAG:.*]], align 4
// OGCG:   store i1 true, ptr %[[OGP_FLAG]], align 1
// OGCG:   invoke void @_ZN1CC1Ev(ptr {{.*}}%[[OGP_NEWP]])
// OGCG:     to label %{{.*}} unwind label %[[OGP_LPAD:.*]]
// OGCG: [[OGP_LPAD]]:
// OGCG:   landingpad { ptr, i32 }
// OGCG-NEXT:   cleanup
// OGCG:   %[[OGP_ACTIVE:.*]] = load i1, ptr %[[OGP_FLAG]], align 1
// OGCG:   br i1 %[[OGP_ACTIVE]], label %[[OGP_DO_DEL:.*]], label %[[OGP_SKIP_DEL:.*]]
// OGCG: [[OGP_DO_DEL]]:
// OGCG:   %[[OGP_LOAD_PTR:.*]] = load ptr, ptr %[[OGP_SAVE_PTR]], align 8
// OGCG:   %[[OGP_LOAD_TAG:.*]] = load i32, ptr %[[OGP_SAVE_TAG]], align 4
// OGCG:   invoke void @_ZdlPvi(ptr %[[OGP_LOAD_PTR]], i32 %[[OGP_LOAD_TAG]])

// LLVM-DAG: attributes #[[ATTR_BUILTIN_NEW]] = {{{.*}}builtin{{.*}}}
// LLVM-DAG: attributes #[[ATTR_BUILTIN_DEL]] = {{{.*}}builtin{{.*}}}
// OGCG-DAG: attributes #[[OGCG_ATTR_BUILTIN_NEW]] = {{{.*}}builtin{{.*}}}
// OGCG-DAG: attributes #[[OGCG_ATTR_BUILTIN_DEL]] = {{{.*}}builtin{{.*}}}
