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

// LLVM-DAG: attributes #[[ATTR_BUILTIN_NEW]] = {{{.*}}builtin{{.*}}}
// LLVM-DAG: attributes #[[ATTR_BUILTIN_DEL]] = {{{.*}}builtin{{.*}}}
// OGCG-DAG: attributes #[[OGCG_ATTR_BUILTIN_NEW]] = {{{.*}}builtin{{.*}}}
// OGCG-DAG: attributes #[[OGCG_ATTR_BUILTIN_DEL]] = {{{.*}}builtin{{.*}}}
