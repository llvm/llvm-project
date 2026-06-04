// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
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
// CIR:   } cleanup normal {
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
// LLVM:   %[[SOURCE:.*]] = call {{.*}} ptr @_Z9getSourcev()
// LLVM:   call void @_ZNSt10unique_ptrI4BaseEC1EPS0_(ptr {{.*}} %[[AGG_TMP0]], ptr {{.*}} %[[SOURCE]])
// LLVM:   store i8 1, ptr %[[CLEANUP_COND]]
// LLVM:   %[[AGG_TMP0_LOAD:.*]] = load %"struct.std::unique_ptr<Base>", ptr %[[AGG_TMP0]]
// LLVM:   call void @_ZN7WrapperC1ESt10unique_ptrI4BaseE(ptr {{.*}} %[[RETVAL]], %"struct.std::unique_ptr<Base>" %[[AGG_TMP0_LOAD]])
// LLVM:   br label %[[CONSTRUCT_CONTINUE:.*]]
// LLVM: [[CONSTRUCT_FALSE]]:
// LLVM:   %[[EMPTY:.*]] = call %struct.Wrapper @_ZN7Wrapper5emptyEv()
// LLVM:   store %struct.Wrapper %[[EMPTY]], ptr %[[RETVAL]]
// LLVM:   br label %[[CONSTRUCT_DONE:.*]]
// LLVM: [[CONSTRUCT_DONE]]:
// LLVM:   %[[CLEANUP_FLAG:.*]] = load i8, ptr %[[CLEANUP_COND]]
// LLVM:   %[[SHOULD_CLEANUP:.*]] = trunc i8 %[[CLEANUP_FLAG]] to i1
// LLVM:   br i1 %[[SHOULD_CLEANUP]], label %[[NORMAL_CLEANUP:.*]], label %[[DONE:.*]]
// LLVM: [[NORMAL_CLEANUP]]:
// LLVM:   call void @_ZNSt10unique_ptrI4BaseED1Ev(ptr {{.*}} %[[AGG_TMP0]])
// LLVM:   br label %[[DONE:.*]]
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
// OGCG:   call void @_ZN7WrapperC1ESt10unique_ptrI4BaseE(ptr {{.*}} %[[RETVAL]], ptr {{.*}} %[[AGG_TMP]])
// OGCG:   br label %[[COND_END:.*]]
// OGCG: [[COND_FALSE]]:
// OGCG:   call void @_ZN7Wrapper5emptyEv(ptr {{.*}} %[[RETVAL]])
// OGCG:   br label %[[COND_END:.*]]
// OGCG: [[COND_END]]:
// OGCG:   %[[CLEANUP_IS_ACTIVE:.*]] = load i1, ptr %[[CLEANUP_COND]]
// OGCG:   br i1 %[[CLEANUP_IS_ACTIVE]], label %[[CLEANUP_ACTION:.*]], label %[[DONE:.*]]
// OGCG: [[CLEANUP_ACTION]]:
// OGCG:   call void @_ZNSt10unique_ptrI4BaseED1Ev(ptr {{.*}} %[[AGG_TMP]])
// OGCG:   br label %[[DONE]]
// OGCG: [[DONE]]:
// OGCG:   ret void
