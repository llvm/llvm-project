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

struct APInt {
  ~APInt();
  APInt uadd_sat();
};

struct APFixedPoint {
  void add(int x) const;
};

// A conditional expression whose two arms both materialize into the same
// aggregate temporary pushes two deferred conditional cleanups that share
// the same underlying alloca.
void APFixedPoint::add(int x) const {
  APInt ThisVal;
  if (x)
    x ? ThisVal : ThisVal.uadd_sat();
}

// CIR: cir.func {{.*}} @_ZNK12APFixedPoint3addEi(%{{.*}}: !cir.ptr<!rec_APFixedPoint>{{.*}}, %{{.*}}: !s32i{{.*}})
// CIR:   %[[X_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init]
// CIR:   %[[THISVAL:.*]] = cir.alloca !rec_APInt, !cir.ptr<!rec_APInt>, ["ThisVal"]
// CIR:   %[[CLEANUP_COND_TRUE:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["cleanup.cond"]
// CIR:   %[[CLEANUP_COND_FALSE:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["cleanup.cond"]
// CIR:   cir.cleanup.scope {
// CIR:     cir.scope {
// CIR:       %[[X:.*]] = cir.load{{.*}} %[[X_ADDR]]
// CIR:       %[[X_BOOL:.*]] = cir.cast int_to_bool %[[X]]
// CIR:       cir.if %[[X_BOOL]] {
// CIR:         %[[AGG_TMP:.*]] = cir.alloca !rec_APInt, !cir.ptr<!rec_APInt>, ["agg.tmp.ensured"]
// CIR:         cir.cleanup.scope {
// CIR:           %[[X2:.*]] = cir.load{{.*}} %[[X_ADDR]]
// CIR:           %[[X2_BOOL:.*]] = cir.cast int_to_bool %[[X2]]
// CIR:           %[[FALSE:.*]] = cir.const #false
// CIR:           cir.store{{.*}} %[[FALSE]], %[[CLEANUP_COND_TRUE]]
// CIR:           %[[FALSE:.*]] = cir.const #false
// CIR:           cir.store{{.*}} %[[FALSE]], %[[CLEANUP_COND_FALSE]]
// CIR:           cir.if %[[X2_BOOL]] {
// CIR:             %[[TRUE:.*]] = cir.const #true
// CIR:             cir.store %[[TRUE]], %[[CLEANUP_COND_TRUE]]
// CIR:           } else {
// CIR:             %[[CALL_RES:.*]] = cir.call @_ZN5APInt8uadd_satEv(%[[THISVAL]])
// CIR:             cir.store{{.*}} %[[CALL_RES]], %[[AGG_TMP]]
// CIR:             %[[TRUE:.*]] = cir.const #true
// CIR:             cir.store %[[TRUE]], %[[CLEANUP_COND_FALSE]]
// CIR:           }
// CIR:         } cleanup all {
// CIR:           %[[F_FLAG:.*]] = cir.load{{.*}} %[[CLEANUP_COND_FALSE]]
// CIR:           cir.if %[[F_FLAG]] {
// CIR:             cir.call @_ZN5APIntD1Ev(%[[AGG_TMP]])
// CIR:           }
// CIR:           %[[T_FLAG:.*]] = cir.load{{.*}} %[[CLEANUP_COND_TRUE]]
// CIR:           cir.if %[[T_FLAG]] {
// CIR:             cir.call @_ZN5APIntD1Ev(%[[AGG_TMP]])
// CIR:           }
// CIR:         }
// CIR:       }
// CIR:     }
// CIR:   } cleanup all {
// CIR:     cir.call @_ZN5APIntD1Ev(%[[THISVAL]])
// CIR:   }
// CIR:   cir.return

// LLVM: define {{.*}} void @_ZNK12APFixedPoint3addEi(ptr {{.*}} %{{.*}}, i32 {{.*}} %{{.*}}){{.*}} personality ptr @__gxx_personality_v0
// LLVM:   %[[AGG_TMP:.*]] = alloca %struct.APInt
// LLVM:   %[[THIS_ADDR:.*]] = alloca ptr
// LLVM:   %[[X_ADDR:.*]] = alloca i32
// LLVM:   %[[THISVAL:.*]] = alloca %struct.APInt
// LLVM:   %[[CLEANUP_COND_TRUE:.*]] = alloca i8
// LLVM:   %[[CLEANUP_COND_FALSE:.*]] = alloca i8
// LLVM:   br i1 %{{.*}}, label %[[OUTER_TRUE:.*]], label %[[OUTER_END:.*]]
// LLVM: [[OUTER_TRUE]]:
// LLVM:   br i1 %{{.*}}, label %[[COND_TRUE:.*]], label %[[COND_FALSE:.*]]
// LLVM: [[COND_TRUE]]:
// LLVM:   store i8 1, ptr %[[CLEANUP_COND_TRUE]]
// LLVM: [[COND_FALSE]]:
// LLVM:   %[[CALL_RES:.*]] = invoke %struct.APInt @_ZN5APInt8uadd_satEv(ptr {{.*}} %[[THISVAL]])
// LLVM:           to label %[[INVOKE_CONT:.*]] unwind label %[[LPAD:.*]]
// LLVM: [[INVOKE_CONT]]:
// LLVM:   store %struct.APInt %[[CALL_RES]], ptr %[[AGG_TMP]]
// LLVM:   store i8 1, ptr %[[CLEANUP_COND_FALSE]]
// LLVM:   %[[FF:.*]] = load i8, ptr %[[CLEANUP_COND_FALSE]]
// LLVM:   %[[FF_B:.*]] = trunc i8 %[[FF]] to i1
// LLVM:   br i1 %[[FF_B]], label %[[CLEANUP_F:.*]], label %[[AFTER_F:.*]]
// LLVM: [[CLEANUP_F]]:
// LLVM:   call void @_ZN5APIntD1Ev(ptr {{.*}} %[[AGG_TMP]])
// LLVM: [[AFTER_F]]:
// LLVM:   %[[TF:.*]] = load i8, ptr %[[CLEANUP_COND_TRUE]]
// LLVM:   %[[TF_B:.*]] = trunc i8 %[[TF]] to i1
// LLVM:   br i1 %[[TF_B]], label %[[CLEANUP_T:.*]], label %[[AFTER_T:.*]]
// LLVM: [[CLEANUP_T]]:
// LLVM:   call void @_ZN5APIntD1Ev(ptr {{.*}} %[[AGG_TMP]])
// LLVM: [[AFTER_T]]:
// LLVM: [[LPAD]]:
// LLVM:   landingpad { ptr, i32 }
// LLVM:     cleanup
// LLVM:   %[[FF_EH:.*]] = load i8, ptr %[[CLEANUP_COND_FALSE]]
// LLVM:   %[[FF_EH_B:.*]] = trunc i8 %[[FF_EH]] to i1
// LLVM:   br i1 %[[FF_EH_B]], label %[[EH_CLEANUP_F:.*]], label %[[EH_AFTER_F:.*]]
// LLVM: [[EH_CLEANUP_F]]:
// LLVM:   call void @_ZN5APIntD1Ev(ptr {{.*}} %[[AGG_TMP]])
// LLVM: [[EH_AFTER_F]]:
// LLVM:   %[[TF_EH:.*]] = load i8, ptr %[[CLEANUP_COND_TRUE]]
// LLVM:   %[[TF_EH_B:.*]] = trunc i8 %[[TF_EH]] to i1
// LLVM:   br i1 %[[TF_EH_B]], label %[[EH_CLEANUP_T:.*]], label %[[EH_AFTER_T:.*]]
// LLVM: [[EH_CLEANUP_T]]:
// LLVM:   call void @_ZN5APIntD1Ev(ptr {{.*}} %[[AGG_TMP]])
// LLVM: [[EH_AFTER_T]]:
// LLVM:   call void @_ZN5APIntD1Ev(ptr {{.*}} %[[THISVAL]])
// LLVM:   resume

// OGCG: define {{.*}} void @_ZNK12APFixedPoint3addEi(ptr {{.*}} %[[THIS:.*]], i32 {{.*}} %[[X:.*]]){{.*}} personality ptr @__gxx_personality_v0
// OGCG:   %[[THIS_ADDR:.*]] = alloca ptr
// OGCG:   %[[X_ADDR:.*]] = alloca i32
// OGCG:   %[[THISVAL:.*]] = alloca %struct.APInt
// OGCG:   %[[AGG_TMP:.*]] = alloca %struct.APInt
// OGCG:   %[[CLEANUP_COND_TRUE:.*]] = alloca i1
// OGCG:   %[[EXN_SLOT:.*]] = alloca ptr
// OGCG:   %[[EHSEL_SLOT:.*]] = alloca i32
// OGCG:   %[[CLEANUP_COND_FALSE:.*]] = alloca i1
// OGCG:   br i1 %{{.*}}, label %[[IF_THEN:.*]], label %[[IF_END:.*]]
// OGCG: [[IF_THEN]]:
// OGCG:   br i1 %{{.*}}, label %[[COND_TRUE:.*]], label %[[COND_FALSE:.*]]
// OGCG: [[COND_TRUE]]:
// OGCG:   store i1 true, ptr %[[CLEANUP_COND_TRUE]]
// OGCG: [[COND_FALSE]]:
// OGCG:   invoke void @_ZN5APInt8uadd_satEv(ptr {{.*}} sret(%struct.APInt) {{.*}} %[[AGG_TMP]], ptr {{.*}} %[[THISVAL]])
// OGCG:           to label %[[INVOKE_CONT:.*]] unwind label %[[LPAD:.*]]
// OGCG: [[INVOKE_CONT]]:
// OGCG:   store i1 true, ptr %[[CLEANUP_COND_FALSE]]
// OGCG: [[COND_END:.*]]:
// OGCG:   %[[FF:.*]] = load i1, ptr %[[CLEANUP_COND_FALSE]]
// OGCG:   br i1 %[[FF]], label %[[CLEANUP_F:.*]], label %[[AFTER_F:.*]]
// OGCG: [[CLEANUP_F]]:
// OGCG:   call void @_ZN5APIntD1Ev(ptr {{.*}} %[[AGG_TMP]])
// OGCG:   br label %[[AFTER_F]]
// OGCG: [[AFTER_F]]:
// OGCG:   %[[TF:.*]] = load i1, ptr %[[CLEANUP_COND_TRUE]]
// OGCG:   br i1 %[[TF]], label %[[CLEANUP_T:.*]], label %[[AFTER_T:.*]]
// OGCG: [[CLEANUP_T]]:
// OGCG:   call void @_ZN5APIntD1Ev(ptr {{.*}} %[[AGG_TMP]])
// OGCG:   br label %[[AFTER_T]]
// OGCG: [[AFTER_T]]:
// OGCG:   br label %[[IF_END]]
// OGCG: [[LPAD]]:
// OGCG:   landingpad { ptr, i32 }
// OGCG:     cleanup
// OGCG:   %[[TF_EH:.*]] = load i1, ptr %[[CLEANUP_COND_TRUE]]
// OGCG:   br i1 %[[TF_EH]], label %[[EH_CLEANUP_T:.*]], label %[[EH_AFTER_T:.*]]
// OGCG: [[EH_CLEANUP_T]]:
// OGCG:   call void @_ZN5APIntD1Ev(ptr {{.*}} %[[AGG_TMP]])
// OGCG:   br label %[[EH_AFTER_T]]
// OGCG: [[EH_AFTER_T]]:
// OGCG:   call void @_ZN5APIntD1Ev(ptr {{.*}} %[[THISVAL]])
// OGCG:   br label %[[EH_RESUME:.*]]
// OGCG: [[IF_END]]:
// OGCG:   call void @_ZN5APIntD1Ev(ptr {{.*}} %[[THISVAL]])
// OGCG:   ret void
// OGCG: [[EH_RESUME]]:
// OGCG:   resume

struct Path { ~Path(); } g_path;

struct Iter {
  ~Iter();
  operator bool();
};

struct Entry {
  Entry();
  Entry(Path);
};

// A conditional expression whose condition itself produces a temporary that
// needs cleanup (here, the Iter() temporary destroyed by ~Iter) nests the
// deferred-conditional cleanup of a temporary in one of the conditional's
// arms inside that condition's cleanup scope. The alloca for the
// conditionally-destroyed Path temporary must be hoisted out of the outer
// (full-expr) cleanup scope, even though in the freshly emitted IR its
// direct parent cleanup scope is the inner one created for the Iter
// temporary.
void makeEntry() {
  Iter() ? Entry() : g_path;
}

// CIR: cir.func {{.*}} @_Z9makeEntryv()
// CIR:   %[[REF_TMP:.*]] = cir.alloca !rec_Iter, !cir.ptr<!rec_Iter>, ["ref.tmp0"]
// CIR:   %[[CLEANUP_COND:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["cleanup.cond"]
// CIR:   %[[AGG_TMP0:.*]] = cir.alloca !rec_Path, !cir.ptr<!rec_Path>, ["agg.tmp0"]
// CIR:   cir.cleanup.scope {
// CIR:     cir.cleanup.scope {
// CIR:       %[[CALL:.*]] = cir.call @_ZN4ItercvbEv(%[[REF_TMP]])
// CIR:       cir.if %[[CALL]] {
// CIR:         %[[ENSURED_T:.*]] = cir.alloca !rec_Entry, !cir.ptr<!rec_Entry>, ["agg.tmp.ensured"]
// CIR:         cir.call @_ZN5EntryC1Ev(%[[ENSURED_T]])
// CIR:       } else {
// CIR:         %[[ENSURED_F:.*]] = cir.alloca !rec_Entry, !cir.ptr<!rec_Entry>, ["agg.tmp.ensured"]
// CIR:         %{{.*}} = cir.get_global @g_path
// CIR:         %[[TRUE:.*]] = cir.const #true
// CIR:         cir.store %[[TRUE]], %[[CLEANUP_COND]]
// CIR:         %[[PATH_LOAD:.*]] = cir.load{{.*}} %[[AGG_TMP0]]
// CIR:         cir.call @_ZN5EntryC1E4Path(%[[ENSURED_F]], %[[PATH_LOAD]])
// CIR:       }
// CIR:       cir.yield
// CIR:     } cleanup all {
// CIR:       cir.call @_ZN4IterD1Ev(%[[REF_TMP]])
// CIR:       cir.yield
// CIR:     }
// CIR:     cir.yield
// CIR:   } cleanup all {
// CIR:     %[[FLAG:.*]] = cir.load{{.*}} %[[CLEANUP_COND]]
// CIR:     cir.if %[[FLAG]] {
// CIR:       cir.call @_ZN4PathD1Ev(%[[AGG_TMP0]])
// CIR:     }
// CIR:     cir.yield
// CIR:   }
// CIR:   cir.return

// LLVM: define {{.*}} void @_Z9makeEntryv(){{.*}} personality ptr @__gxx_personality_v0
// LLVM:   %[[ENSURED_T:.*]] = alloca %struct.Entry
// LLVM:   %[[ENSURED_F:.*]] = alloca %struct.Entry
// LLVM:   %[[REF_TMP:.*]] = alloca %struct.Iter
// LLVM:   %[[CLEANUP_COND:.*]] = alloca i8
// LLVM:   %[[AGG_TMP0:.*]] = alloca %struct.Path
// LLVM:   %[[CALL:.*]] = invoke {{.*}} i1 @_ZN4ItercvbEv(ptr {{.*}} %[[REF_TMP]])
// LLVM:                     to label %[[CALL_CONT:.*]] unwind label %[[LPAD:.*]]
// LLVM: [[CALL_CONT]]:
// LLVM:   br i1 %[[CALL]], label %[[TRUE_BB:.*]], label %[[FALSE_BB:.*]]
// LLVM: [[TRUE_BB]]:
// LLVM:   invoke void @_ZN5EntryC1Ev(ptr {{.*}} %[[ENSURED_T]])
// LLVM:                     to label %[[TRUE_CONT:.*]] unwind label %[[LPAD]]
// LLVM: [[TRUE_CONT]]:
// LLVM:   br label %[[COND_END:.*]]
// LLVM: [[FALSE_BB]]:
// LLVM:   store i8 1, ptr %[[CLEANUP_COND]]
// LLVM:   %[[PATH_LOAD:.*]] = load %struct.Path, ptr %[[AGG_TMP0]]
// LLVM:   invoke void @_ZN5EntryC1E4Path(ptr {{.*}} %[[ENSURED_F]], %struct.Path %[[PATH_LOAD]])
// LLVM:                     to label %[[FALSE_CONT:.*]] unwind label %[[LPAD]]
// LLVM: [[FALSE_CONT]]:
// LLVM:   br label %[[COND_END]]
// LLVM: [[COND_END]]:
// LLVM:   br label %[[AFTER_INNER:.*]]
// LLVM: [[AFTER_INNER]]:
// LLVM:   call void @_ZN4IterD1Ev(ptr {{.*}} %[[REF_TMP]])
// LLVM:   br label %[[INNER_DONE:.*]]
// LLVM: [[INNER_DONE]]:
// LLVM:   br label %[[NORMAL_OUTER:.*]]
// LLVM: [[LPAD]]:
// LLVM:   landingpad { ptr, i32 }
// LLVM:     cleanup
// LLVM:   call void @_ZN4IterD1Ev(ptr {{.*}} %[[REF_TMP]])
// LLVM:   br label %[[EH_OUTER:.*]]
// LLVM: [[NORMAL_OUTER]]:
// LLVM:   store i8 0, ptr %[[CLEANUP_COND]]
// LLVM:   br label %[[CHECK_FLAG:.*]]
// LLVM: [[CHECK_FLAG]]:
// LLVM:   %[[FLAG_BYTE:.*]] = load i8, ptr %[[CLEANUP_COND]]
// LLVM:   %[[FLAG:.*]] = trunc i8 %[[FLAG_BYTE]] to i1
// LLVM:   br i1 %[[FLAG]], label %[[DO_PATH_DTOR:.*]], label %[[DONE_PATH:.*]]
// LLVM: [[DO_PATH_DTOR]]:
// LLVM:   call void @_ZN4PathD1Ev(ptr {{.*}} %[[AGG_TMP0]])
// LLVM:   br label %[[DONE_PATH]]
// LLVM: [[DONE_PATH]]:
// LLVM:   br label %[[BEFORE_RET:.*]]
// LLVM: [[BEFORE_RET]]:
// LLVM:   br label %[[DONE:.*]]
// LLVM: [[EH_OUTER]]:
// LLVM:   %[[EH_FLAG_BYTE:.*]] = load i8, ptr %[[CLEANUP_COND]]
// LLVM:   %[[EH_FLAG:.*]] = trunc i8 %[[EH_FLAG_BYTE]] to i1
// LLVM:   br i1 %[[EH_FLAG]], label %[[EH_PATH_DTOR:.*]], label %[[EH_RESUME:.*]]
// LLVM: [[EH_PATH_DTOR]]:
// LLVM:   call void @_ZN4PathD1Ev(ptr {{.*}} %[[AGG_TMP0]])
// LLVM:   br label %[[EH_RESUME]]
// LLVM: [[EH_RESUME]]:
// LLVM:   resume
// LLVM: [[DONE]]:
// LLVM:   ret void

// OGCG: define {{.*}} void @_Z9makeEntryv(){{.*}} personality ptr @__gxx_personality_v0
// OGCG:   %[[REF_TMP:.*]] = alloca %struct.Iter
// OGCG:   %[[ENSURED_T:.*]] = alloca %struct.Entry
// OGCG:   %[[ENSURED_F:.*]] = alloca %struct.Entry
// OGCG:   %[[AGG_TMP:.*]] = alloca %struct.Path
// OGCG:   %[[CLEANUP_COND:.*]] = alloca i1
// OGCG:   store i1 false, ptr %[[CLEANUP_COND]]
// OGCG:   %[[CALL:.*]] = invoke {{.*}} i1 @_ZN4ItercvbEv(ptr {{.*}} %[[REF_TMP]])
// OGCG:           to label %[[INVOKE_CONT:.*]] unwind label %[[LPAD:.*]]
// OGCG: [[INVOKE_CONT]]:
// OGCG:   br i1 %[[CALL]], label %[[COND_TRUE:.*]], label %[[COND_FALSE:.*]]
// OGCG: [[COND_TRUE]]:
// OGCG:   invoke void @_ZN5EntryC1Ev(ptr {{.*}} %[[ENSURED_T]])
// OGCG:           to label %[[TRUE_CONT:.*]] unwind label %[[LPAD]]
// OGCG: [[TRUE_CONT]]:
// OGCG:   br label %[[COND_END:.*]]
// OGCG: [[COND_FALSE]]:
// OGCG:   store i1 true, ptr %[[CLEANUP_COND]]
// OGCG:   invoke void @_ZN5EntryC1E4Path(ptr {{.*}} %[[ENSURED_F]], ptr {{.*}} %[[AGG_TMP]])
// OGCG:           to label %[[FALSE_CONT:.*]] unwind label %[[LPAD2:.*]]
// OGCG: [[FALSE_CONT]]:
// OGCG:   br label %[[COND_END]]
// OGCG: [[COND_END]]:
// OGCG:   %[[IS_ACTIVE:.*]] = load i1, ptr %[[CLEANUP_COND]]
// OGCG:   br i1 %[[IS_ACTIVE]], label %[[CLEANUP_ACTION:.*]], label %[[CLEANUP_DONE:.*]]
// OGCG: [[CLEANUP_ACTION]]:
// OGCG:   call void @_ZN4PathD1Ev(ptr {{.*}} %[[AGG_TMP]])
// OGCG:   br label %[[CLEANUP_DONE]]
// OGCG: [[CLEANUP_DONE]]:
// OGCG:   call void @_ZN4IterD1Ev(ptr {{.*}} %[[REF_TMP]])
// OGCG:   ret void
// OGCG: [[LPAD]]:
// OGCG:   landingpad { ptr, i32 }
// OGCG:     cleanup
// OGCG:   br label %[[EHCLEANUP:.*]]
// OGCG: [[LPAD2]]:
// OGCG:   landingpad { ptr, i32 }
// OGCG:     cleanup
// OGCG:   %[[EH_IS_ACTIVE:.*]] = load i1, ptr %[[CLEANUP_COND]]
// OGCG:   br i1 %[[EH_IS_ACTIVE]], label %[[EH_CLEANUP_ACTION:.*]], label %[[EH_CLEANUP_DONE:.*]]
// OGCG: [[EH_CLEANUP_ACTION]]:
// OGCG:   call void @_ZN4PathD1Ev(ptr {{.*}} %[[AGG_TMP]])
// OGCG:   br label %[[EH_CLEANUP_DONE]]
// OGCG: [[EH_CLEANUP_DONE]]:
// OGCG:   br label %[[EHCLEANUP]]
// OGCG: [[EHCLEANUP]]:
// OGCG:   call void @_ZN4IterD1Ev(ptr {{.*}} %[[REF_TMP]])
// OGCG:   br label %[[EH_RESUME:.*]]
// OGCG: [[EH_RESUME]]:
// OGCG:   resume
