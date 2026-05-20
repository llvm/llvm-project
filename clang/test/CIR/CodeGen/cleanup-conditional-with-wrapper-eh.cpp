// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fexceptions -fcxx-exceptions -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fexceptions -fcxx-exceptions -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fexceptions -fcxx-exceptions -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG

struct Base { ~Base(); };

namespace std {
  template<typename T>
  struct unique_ptr {
    unique_ptr(T*);
    ~unique_ptr();
  };
};

struct Wrapper {
  std::unique_ptr<Base> ptr;
  Wrapper();
  explicit Wrapper(std::unique_ptr<Base> p);
  static Wrapper empty();
};

bool flag;
Base* getSource();

// The use of unique_ptr here forces the creation of a temporary aggregate
// in the true branch of the conditional, which must be conditionally destroyed
// in the cleanup. Without the use of unique_ptr, the object returned by
// getSource would be passed directly to Wrapper, which uses a function-level
// alloca.
//
// The temporary aggregate must be hoisted out of the cleanup scope in order
// to properly dominate the cleanup region.
Wrapper makeWrapper() {
  return flag
    ? Wrapper(std::unique_ptr<Base>(getSource()))
    : Wrapper::empty();
}

// CIR: cir.func {{.*}} @_Z11makeWrapperv() -> !rec_Wrapper
// CIR:   %[[RETVAL:.*]] = cir.alloca !rec_Wrapper, !cir.ptr<!rec_Wrapper>, ["__retval"]
// CIR:   %[[CLEANUP_COND:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["cleanup.cond"]
// CIR:   %[[AGG_TMP0:.*]] = cir.alloca !rec_std3A3Aunique_ptr3CBase3E, !cir.ptr<!rec_std3A3Aunique_ptr3CBase3E>, ["agg.tmp0"]
// CIR:   cir.cleanup.scope {
// CIR:     %[[FLAG:.*]] = cir.load{{.*}} %{{.*}}
// CIR:     %[[FALSE:.*]] = cir.const #false
// CIR:     cir.store %[[FALSE]], %[[CLEANUP_COND]]
// CIR:     cir.if %[[FLAG]] {
// CIR:       %[[SOURCE:.*]] = cir.call @_Z9getSourcev()
// CIR:       cir.call @_ZNSt10unique_ptrI4BaseEC1EPS0_(%[[AGG_TMP0]], %[[SOURCE]])
// CIR:       %[[TRUE:.*]] = cir.const #true
// CIR:       cir.store %[[TRUE]], %[[CLEANUP_COND]]
// CIR:       %[[AGG_TMP0_LOAD:.*]] = cir.load{{.*}} %[[AGG_TMP0]]
// CIR:       cir.call @_ZN7WrapperC1ESt10unique_ptrI4BaseE(%[[RETVAL]], %[[AGG_TMP0_LOAD]])
// CIR:     } else {
// CIR:       %[[EMPTY:.*]] = cir.call @_ZN7Wrapper5emptyEv()
// CIR:       cir.store{{.*}} %[[EMPTY]], %[[RETVAL]] : !rec_Wrapper, !cir.ptr<!rec_Wrapper>
// CIR:     }
// CIR:     cir.yield
// CIR:   } cleanup all {
// CIR:     %[[SHOULD_CLEANUP:.*]] = cir.load{{.*}} %[[CLEANUP_COND]]
// CIR:     cir.if %[[SHOULD_CLEANUP]] {
// CIR:       cir.call @_ZNSt10unique_ptrI4BaseED1Ev(%[[AGG_TMP0]])
// CIR:     }
// CIR:     cir.yield
// CIR:   }
// CIR:   %[[RET:.*]] = cir.load %[[RETVAL]]
// CIR:   cir.return %[[RET]] : !rec_Wrapper

// LLVM: define {{.*}} %struct.Wrapper @_Z11makeWrapperv()
// LLVM:   %[[RETVAL:.*]] = alloca %struct.Wrapper
// LLVM:   %[[CLEANUP_COND:.*]] = alloca i8
// LLVM:   %[[AGG_TMP0:.*]] = alloca %"struct.std::unique_ptr<Base>"
// LLVM:   br label %[[INIT:.*]]
// LLVM: [[INIT]]:
// LLVM:   br i1 %{{.*}}, label %[[CONSTRUCT_TRUE:.*]], label %[[CONSTRUCT_FALSE:.*]]
// LLVM: [[CONSTRUCT_TRUE]]:

// Note: There is a difference here between OGCG and the CIR->LLVM path. OGCG
//       generates calls rather than invokes for getSource and the unique_ptr
//       because the temporary hasn't been constructed yet and therefore doesn't
//       need to be cleaned up. CIR generates invokes, but because we haven't
//       set the cleanup active flag yet, the EH cleanup will resume without
//       doing anything, so this is effectively equivalent to the OGCG behavior.
//       Curiously, OGCG generates an invoke for the Wrapper::empty() call,
//       even though that also doesn't activate the cleanup.

// LLVM:   %[[SOURCE:.*]] = invoke {{.*}} ptr @_Z9getSourcev()
// LLVM:                       to label %[[INVOKE_CONTINUE:.*]] unwind label %[[INVOKE_CLEANUP:.*]]
// LLVM: [[INVOKE_CONTINUE]]:
// LLVM:   invoke void @_ZNSt10unique_ptrI4BaseEC1EPS0_(ptr {{.*}} %[[AGG_TMP0]], ptr {{.*}} %[[SOURCE]])
// LLVM:                       to label %[[INVOKE_CONTINUE_2:.*]] unwind label %[[INVOKE_CLEANUP:.*]]
// LLVM: [[INVOKE_CONTINUE_2]]:
// LLVM:   store i8 1, ptr %[[CLEANUP_COND]]
// LLVM:   %[[AGG_TMP0_LOAD:.*]] = load %"struct.std::unique_ptr<Base>", ptr %[[AGG_TMP0]]
// LLVM:   invoke void @_ZN7WrapperC1ESt10unique_ptrI4BaseE(ptr {{.*}} %[[RETVAL]], %"struct.std::unique_ptr<Base>" %[[AGG_TMP0_LOAD]])
// LLVM:                       to label %[[INVOKE_CONTINUE_3:.*]] unwind label %[[INVOKE_CLEANUP:.*]]
// LLVM: [[INVOKE_CONTINUE_3]]:
// LLVM:   br label %[[CONSTRUCT_CONTINUE:.*]]
// LLVM: [[CONSTRUCT_FALSE]]:
// LLVM:   %[[EMPTY:.*]] = invoke %struct.Wrapper @_ZN7Wrapper5emptyEv()
// LLVM:                       to label %[[INVOKE_CONTINUE_4:.*]] unwind label %[[INVOKE_CLEANUP:.*]]
// LLVM: [[INVOKE_CONTINUE_4]]:
// LLVM:   store %struct.Wrapper %[[EMPTY]], ptr %[[RETVAL]]
// LLVM:   br label %[[CONSTRUCT_DONE:.*]]
// LLVM: [[CONSTRUCT_DONE]]:
// LLVM:   %[[CLEANUP_FLAG:.*]] = load i8, ptr %[[CLEANUP_COND]]
// LLVM:   %[[SHOULD_CLEANUP:.*]] = trunc i8 %[[CLEANUP_FLAG]] to i1
// LLVM:   br i1 %[[SHOULD_CLEANUP]], label %[[NORMAL_CLEANUP:.*]], label %[[CLEANUP_DONE:.*]]
// LLVM: [[NORMAL_CLEANUP]]:
// LLVM:   call void @_ZNSt10unique_ptrI4BaseED1Ev(ptr {{.*}} %[[AGG_TMP0]])
// LLVM:   br label %[[CLEANUP_DONE]]
// LLVM: [[CLEANUP_DONE]]:
// LLVM:   br label %[[EXIT_CLEANUP_SCOPE:.*]]
// LLVM: [[EXIT_CLEANUP_SCOPE]]:
// LLVM:   br label %[[DONE:.*]]
// LLVM: [[INVOKE_CLEANUP]]:
// LLVM:   landingpad { ptr, i32 }
// LLVM:     cleanup
// LLVM:   %[[CLEANUP_FLAG:.*]] = load i8, ptr %[[CLEANUP_COND]]
// LLVM:   %[[SHOULD_CLEANUP:.*]] = trunc i8 %[[CLEANUP_FLAG]] to i1
// LLVM:   br i1 %[[SHOULD_CLEANUP]], label %[[EH_CLEANUP:.*]], label %[[CLENAUP_DONE:.*]]
// LLVM: [[EH_CLEANUP]]:
// LLVM:   call void @_ZNSt10unique_ptrI4BaseED1Ev(ptr {{.*}} %[[AGG_TMP0]])
// LLVM:   br label %[[CLEANUP_DONE:.*]]
// LLVM: [[CLEANUP_DONE]]:
// LLVM:   resume
// LLVM: [[DONE]]:
// LLVM:   %[[RET:.*]] = load %struct.Wrapper, ptr %[[RETVAL]]
// LLVM:   ret %struct.Wrapper %[[RET]]
  
// OGCG: define {{.*}} void @_Z11makeWrapperv(ptr{{.*}} sret(%struct.Wrapper) {{.*}} %[[RETVAL:.*]])
// OGCG:   %[[RESULT_PTR:.*]] = alloca ptr
// OGCG:   %[[AGG_TMP:.*]] = alloca %"struct.std::unique_ptr"
// OGCG:   %[[CLEANUP_COND:.*]] = alloca i1
// OGCG:   store ptr %[[RETVAL]], ptr %[[RESULT_PTR]]
// OGCG:   %[[FLAG:.*]] = load i8, ptr @flag, align 1
// OGCG:   %[[LOADEDV:.*]] = icmp ne i8 %[[FLAG]], 0
// OGCG:   store i1 false, ptr %[[CLEANUP_COND]]
// OGCG:   br i1 %[[LOADEDV]], label %[[COND_TRUE:.*]], label %[[COND_FALSE:.*]]
// OGCG: [[COND_TRUE]]:
// OGCG:   %[[SOURCE:.*]] = call {{.*}} ptr @_Z9getSourcev()
// OGCG:   call void @_ZNSt10unique_ptrI4BaseEC1EPS0_(ptr {{.*}} %[[AGG_TMP]], {{.*}} %[[SOURCE]])
// OGCG:   store i1 true, ptr %[[CLEANUP_COND]]
// OGCG:   invoke void @_ZN7WrapperC1ESt10unique_ptrI4BaseE(ptr {{.*}} %[[RETVAL]], ptr {{.*}} %[[AGG_TMP]])
// OGCG:           to label %[[INVOKE_CONTINUE:.*]] unwind label %[[INVOKE_CLEANUP:.*]]
// OGCG: [[INVOKE_CONTINUE]]:
// OGCG:   br label %[[COND_END:.*]]
// OGCG: [[COND_FALSE]]:
// OGCG:   invoke void @_ZN7Wrapper5emptyEv(ptr {{.*}} %[[RETVAL]])
// OGCG:           to label %[[INVOKE_CONTINUE_2:.*]] unwind label %[[INVOKE_CLEANUP:.*]]
// OGCG: [[INVOKE_CONTINUE_2]]:
// OGCG:   br label %[[COND_END]]
// OGCG: [[COND_END]]:
// OGCG:   %[[CLEANUP_IS_ACTIVE:.*]] = load i1, ptr %[[CLEANUP_COND]]
// OGCG:   br i1 %[[CLEANUP_IS_ACTIVE]], label %[[CLEANUP_ACTION:.*]], label %[[DONE:.*]]
// OGCG: [[CLEANUP_ACTION]]:
// OGCG:   call void @_ZNSt10unique_ptrI4BaseED1Ev(ptr {{.*}} %[[AGG_TMP]])
// OGCG:   br label %[[DONE]]
// OGCG: [[DONE]]:
// OGCG:   ret void
// OGCG: [[INVOKE_CLEANUP]]:
// OGCG:   landingpad { ptr, i32 }
// OGCG:     cleanup
// OGCG:   %[[CLEANUP_IS_ACTIVE:.*]] = load i1, ptr %[[CLEANUP_COND]]
// OGCG:   br i1 %[[CLEANUP_IS_ACTIVE]], label %[[CLEANUP_ACTION:.*]], label %[[EH_DONE:.*]]
// OGCG: [[CLEANUP_ACTION]]:
// OGCG:   call void @_ZNSt10unique_ptrI4BaseED1Ev(ptr {{.*}} %[[AGG_TMP]])
// OGCG:   br label %[[EH_DONE]]
// OGCG: [[EH_DONE]]:
// OGCG:   resume
