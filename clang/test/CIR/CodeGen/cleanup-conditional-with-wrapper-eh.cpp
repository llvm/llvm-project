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

// CIR: cir.func {{.*}} @_Z11makeWrapperv(%[[RETVAL:.*]]: !cir.ptr<!rec_Wrapper> {llvm.align = 1 : i64, llvm.dead_on_unwind, llvm.noalias, llvm.sret = !rec_Wrapper, llvm.writable}{{.*}})
// CIR:   %[[CLEANUP_COND:.*]] = cir.alloca "cleanup.cond" {{.*}} : !cir.ptr<!cir.bool>
// CIR:   %[[AGG_TMP0:.*]] = cir.alloca "agg.tmp0" {{.*}} : !cir.ptr<!rec_std3A3Aunique_ptr3CBase3E>
// CIR:   %[[FLAG:.*]] = cir.load{{.*}} %{{.*}}
// CIR:   cir.cleanup.scope {
// CIR:     %[[FALSE:.*]] = cir.const #false
// CIR:     cir.store %[[FALSE]], %[[CLEANUP_COND]]
// CIR:     cir.if %[[FLAG]] {
// CIR:       %[[SOURCE:.*]] = cir.call @_Z9getSourcev()
// CIR:       cir.call @_ZNSt10unique_ptrI4BaseEC1EPS0_(%[[AGG_TMP0]], %[[SOURCE]])
// CIR:       %[[TRUE:.*]] = cir.const #true
// CIR:       cir.store %[[TRUE]], %[[CLEANUP_COND]]
// CIR:       cir.call @_ZN7WrapperC1ESt10unique_ptrI4BaseE(%[[RETVAL]], %[[AGG_TMP0]]) : ({{.*}}, !cir.ptr<!rec_std3A3Aunique_ptr3CBase3E> {llvm.align = 1 : i64, llvm.dereferenceable = 1 : i64, llvm.nofreeobj, llvm.noundef}) -> ()
// CIR:     } else {
// CIR:       cir.call @_ZN7Wrapper5emptyEv(%[[RETVAL]])
// CIR:     }
// CIR:     cir.yield
// CIR:   } cleanup all {
// CIR:     %[[SHOULD_CLEANUP:.*]] = cir.load{{.*}} %[[CLEANUP_COND]]
// CIR:     cir.if %[[SHOULD_CLEANUP]] {
// CIR:       cir.call @_ZNSt10unique_ptrI4BaseED1Ev(%[[AGG_TMP0]])
// CIR:     }
// CIR:     cir.yield
// CIR:   }
// CIR:   cir.return

// LLVM: define {{.*}} void @_Z11makeWrapperv(ptr {{.*}} sret(%struct.Wrapper) {{.*}} %[[RETVAL:.*]])
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
// LLVM:   invoke void @_ZN7WrapperC1ESt10unique_ptrI4BaseE(ptr {{.*}} %[[RETVAL]], ptr nofreeobj noundef align 1 dereferenceable(1) %[[AGG_TMP0]])
// LLVM:                       to label %[[INVOKE_CONTINUE_3:.*]] unwind label %[[INVOKE_CLEANUP:.*]]
// LLVM: [[INVOKE_CONTINUE_3]]:
// LLVM:   br label %[[CONSTRUCT_CONTINUE:.*]]
// LLVM: [[CONSTRUCT_FALSE]]:
// LLVM:   invoke void @_ZN7Wrapper5emptyEv(ptr {{.*}} sret(%struct.Wrapper) {{.*}} %[[RETVAL]])
// LLVM:                       to label %[[INVOKE_CONTINUE_4:.*]] unwind label %[[INVOKE_CLEANUP:.*]]
// LLVM: [[INVOKE_CONTINUE_4]]:
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
// LLVM:   ret void
  
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
// OGCG:   invoke void @_ZN7WrapperC1ESt10unique_ptrI4BaseE(ptr {{.*}} %[[RETVAL]], ptr nofreeobj noundef align 1 dereferenceable(1) %[[AGG_TMP]])
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
// CIR:   %[[X_ADDR:.*]] = cir.alloca "x" {{.*}} init : !cir.ptr<!s32i>
// CIR:   %[[THISVAL:.*]] = cir.alloca "ThisVal" {{.*}} : !cir.ptr<!rec_APInt>
// CIR:   %[[CLEANUP_COND_TRUE:.*]] = cir.alloca "cleanup.cond" {{.*}} : !cir.ptr<!cir.bool>
// CIR:   %[[CLEANUP_COND_FALSE:.*]] = cir.alloca "cleanup.cond" {{.*}} : !cir.ptr<!cir.bool>
// CIR:   cir.cleanup.scope {
// CIR:     cir.scope {
// CIR:       %[[X:.*]] = cir.load{{.*}} %[[X_ADDR]]
// CIR:       %[[X_BOOL:.*]] = cir.cast int_to_bool %[[X]]
// CIR:       cir.if %[[X_BOOL]] {
// CIR:         %[[AGG_TMP:.*]] = cir.alloca "agg.tmp.ensured" {{.*}} : !cir.ptr<!rec_APInt>
// CIR:         %[[X2:.*]] = cir.load{{.*}} %[[X_ADDR]]
// CIR:         %[[X2_BOOL:.*]] = cir.cast int_to_bool %[[X2]]
// CIR:         cir.cleanup.scope {
// CIR:           %[[FALSE:.*]] = cir.const #false
// CIR:           cir.store{{.*}} %[[FALSE]], %[[CLEANUP_COND_TRUE]]
// CIR:           %[[FALSE:.*]] = cir.const #false
// CIR:           cir.store{{.*}} %[[FALSE]], %[[CLEANUP_COND_FALSE]]
// CIR:           cir.if %[[X2_BOOL]] {
// CIR:             %[[TRUE:.*]] = cir.const #true
// CIR:             cir.store %[[TRUE]], %[[CLEANUP_COND_TRUE]]
// CIR:           } else {
// CIR:             cir.call @_ZN5APInt8uadd_satEv(%[[AGG_TMP]], %[[THISVAL]]) : (!cir.ptr<!rec_APInt> {llvm.align = 1 : i64, llvm.dead_on_unwind, llvm.sret = !rec_APInt, llvm.writable}, {{.*}}) -> ()
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
// LLVM:   invoke void @_ZN5APInt8uadd_satEv(ptr {{.*}} sret(%struct.APInt) {{.*}} %[[AGG_TMP]], ptr {{.*}} %[[THISVAL]])
// LLVM:           to label %[[INVOKE_CONT:.*]] unwind label %[[LPAD:.*]]
// LLVM: [[INVOKE_CONT]]:
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
// needs cleanup, here the Iter() temporary destroyed by ~Iter. Iter() is
// constructed before the conditional, so its cleanup scope is the outer one
// and the conditionally-destroyed Path temporary is destroyed first, on both
// the normal and the unwind path.
void makeEntry() {
  Iter() ? Entry() : g_path;
}

// CIR: cir.func {{.*}} @_Z9makeEntryv()
// CIR:   %[[REF_TMP:.*]] = cir.alloca "ref.tmp0" {{.*}} : !cir.ptr<!rec_Iter>
// CIR:   %[[CLEANUP_COND:.*]] = cir.alloca "cleanup.cond" {{.*}} : !cir.ptr<!cir.bool>
// CIR:   cir.cleanup.scope {
// CIR:     %[[AGG_TMP0:.*]] = cir.alloca "agg.tmp0" {{.*}} : !cir.ptr<!rec_Path>
// CIR:     %[[CALL:.*]] = cir.call @_ZN4ItercvbEv(%[[REF_TMP]])
// CIR:     cir.cleanup.scope {
// CIR:       %[[FALSE:.*]] = cir.const #false
// CIR:       cir.store %[[FALSE]], %[[CLEANUP_COND]]
// CIR:       cir.if %[[CALL]] {
// CIR:         %[[ENSURED_T:.*]] = cir.alloca "agg.tmp.ensured" {{.*}} : !cir.ptr<!rec_Entry>
// CIR:         cir.call @_ZN5EntryC1Ev(%[[ENSURED_T]])
// CIR:       } else {
// CIR:         %[[ENSURED_F:.*]] = cir.alloca "agg.tmp.ensured" {{.*}} : !cir.ptr<!rec_Entry>
// CIR:         %{{.*}} = cir.get_global @g_path
// CIR:         %[[TRUE:.*]] = cir.const #true
// CIR:         cir.store %[[TRUE]], %[[CLEANUP_COND]]
// CIR:         cir.call @_ZN5EntryC1E4Path(%[[ENSURED_F]], %[[AGG_TMP0]]) : ({{.*}}, !cir.ptr<!rec_Path> {llvm.align = 1 : i64, llvm.dereferenceable = 1 : i64, llvm.nofreeobj, llvm.noundef}) -> ()
// CIR:       }
// CIR:       cir.yield
// CIR:     } cleanup all {
// CIR:       %[[FLAG:.*]] = cir.load{{.*}} %[[CLEANUP_COND]]
// CIR:       cir.if %[[FLAG]] {
// CIR:         cir.call @_ZN4PathD1Ev(%[[AGG_TMP0]])
// CIR:       }
// CIR:       cir.yield
// CIR:     }
// CIR:     cir.yield
// CIR:   } cleanup all {
// CIR:     cir.call @_ZN4IterD1Ev(%[[REF_TMP]])
// CIR:     cir.yield
// CIR:   }
// CIR:   cir.return

// LLVM: define {{.*}} void @_Z9makeEntryv(){{.*}} personality ptr @__gxx_personality_v0
// LLVM:   %[[AGG_TMP0:.*]] = alloca %struct.Path
// LLVM:   %[[ENSURED_T:.*]] = alloca %struct.Entry
// LLVM:   %[[ENSURED_F:.*]] = alloca %struct.Entry
// LLVM:   %[[REF_TMP:.*]] = alloca %struct.Iter
// LLVM:   %[[CLEANUP_COND:.*]] = alloca i8
// LLVM:   %[[CALL:.*]] = invoke {{.*}} i1 @_ZN4ItercvbEv(ptr {{.*}} %[[REF_TMP]])
// LLVM:                     to label %[[CALL_CONT:.*]] unwind label %[[LPAD_ITER:.*]]
// LLVM: [[CALL_CONT]]:
// The flag is cleared after Iter is constructed but before the conditional,
// so it is initialized on the arms' unwind paths too.
// LLVM:   store i8 0, ptr %[[CLEANUP_COND]]
// LLVM:   br i1 %[[CALL]], label %[[TRUE_BB:.*]], label %[[FALSE_BB:.*]]
// LLVM: [[TRUE_BB]]:
// LLVM:   invoke void @_ZN5EntryC1Ev(ptr {{.*}} %[[ENSURED_T]])
// LLVM:                     to label %[[TRUE_CONT:.*]] unwind label %[[LPAD_PATH:.*]]
// LLVM: [[TRUE_CONT]]:
// LLVM:   br label %[[COND_END:.*]]
// LLVM: [[FALSE_BB]]:
// LLVM:   store i8 1, ptr %[[CLEANUP_COND]]
// LLVM:   invoke void @_ZN5EntryC1E4Path(ptr {{.*}} %[[ENSURED_F]], ptr nofreeobj noundef align 1 dereferenceable(1) %[[AGG_TMP0]])
// LLVM:                     to label %[[FALSE_CONT:.*]] unwind label %[[LPAD_PATH]]
// LLVM: [[FALSE_CONT]]:
// LLVM:   br label %[[COND_END]]
// Normal path destroys the conditional Path first.
// LLVM: [[COND_END]]:
// LLVM:   %[[FLAG_BYTE:.*]] = load i8, ptr %[[CLEANUP_COND]]
// LLVM:   %[[FLAG:.*]] = trunc i8 %[[FLAG_BYTE]] to i1
// LLVM:   br i1 %[[FLAG]], label %[[DO_PATH_DTOR:.*]], label %[[DONE_PATH:.*]]
// LLVM: [[DO_PATH_DTOR]]:
// LLVM:   call void @_ZN4PathD1Ev(ptr {{.*}} %[[AGG_TMP0]])
// LLVM:   br label %[[DONE_PATH]]
// Unwinding out of an arm destroys Path the same way.
// LLVM: [[LPAD_PATH]]:
// LLVM:   landingpad { ptr, i32 }
// LLVM:     cleanup
// LLVM:   %[[EH_FLAG_BYTE:.*]] = load i8, ptr %[[CLEANUP_COND]]
// LLVM:   %[[EH_FLAG:.*]] = trunc i8 %[[EH_FLAG_BYTE]] to i1
// LLVM:   br i1 %[[EH_FLAG]], label %[[EH_PATH_DTOR:.*]], label %{{.*}}
// LLVM: [[EH_PATH_DTOR]]:
// LLVM:   call void @_ZN4PathD1Ev(ptr {{.*}} %[[AGG_TMP0]])
// Iter is destroyed after Path, on the normal path and from its own
// landingpad.
// LLVM:   call void @_ZN4IterD1Ev(ptr {{.*}} %[[REF_TMP]])
// LLVM: [[LPAD_ITER]]:
// LLVM:   landingpad { ptr, i32 }
// LLVM:     cleanup
// LLVM:   call void @_ZN4IterD1Ev(ptr {{.*}} %[[REF_TMP]])
// LLVM:   resume
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
// OGCG:   invoke void @_ZN5EntryC1E4Path(ptr {{.*}} %[[ENSURED_F]], ptr nofreeobj noundef align 1 dereferenceable(1) %[[AGG_TMP]])
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

struct AltArg { AltArg(); ~AltArg(); };
struct Arg { Arg(); Arg(const AltArg &); ~Arg(); };
struct Extra { Extra(); ~Extra(); };
void consume(const Arg &, const Extra &);

// A conditional argument followed by an unconditional one.
//
// Both arms of the conditional produce an Arg, the true arm directly and the
// false arm through the converting constructor from AltArg, so ~Arg runs on
// every path that leaves the conditional and needs no active flag. AltArg is
// built only on the false arm, so it is the one that gets the flag.
//
// The arms are emitted in the body of the conditional's scope, outside the
// nested scope that holds ~Arg. Unwinding out of an arm has entered only the
// conditional's scope, so it runs the guarded ~AltArg and nothing else.
//
// Arg is constructed before Extra, so the destructors run Extra, then Arg,
// then the guarded AltArg. This is the mirror of makeEntry above, where the
// conditionally destroyed temporary is the one constructed last.
void callCondArg(bool c) {
  consume(c ? Arg() : AltArg(), Extra());
}


// CIR: cir.func {{.*}} @_Z11callCondArgb
// CIR:   %[[ARG:.*]] = cir.alloca "ref.tmp0" {{.*}} : !cir.ptr<!rec_Arg>
// CIR:   %[[ALT:.*]] = cir.alloca "ref.tmp1" {{.*}} : !cir.ptr<!rec_AltArg>
// CIR:   %[[ACTIVE:.*]] = cir.alloca "cleanup.cond" {{.*}} : !cir.ptr<!cir.bool>
// CIR:   %[[EXTRA:.*]] = cir.alloca "ref.tmp2" {{.*}} : !cir.ptr<!rec_Extra>
// The conditional scope opens at the conditional and is outermost, since the
// AltArg temporary it guards is the first one constructed.
// CIR:   cir.cleanup.scope {
// CIR:     %[[FALSE:.*]] = cir.const #false
// CIR:     cir.store %[[FALSE]], %[[ACTIVE]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR:     cir.if %{{.*}} {
// CIR:       cir.call @_ZN3ArgC1Ev(%[[ARG]])
// CIR:     } else {
// CIR:       cir.call @_ZN6AltArgC1Ev(%[[ALT]])
// CIR:       %[[TRUE:.*]] = cir.const #true
// CIR:       cir.store %[[TRUE]], %[[ACTIVE]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR:       cir.call @_ZN3ArgC1ERK6AltArg(%[[ARG]], %[[ALT]])
// CIR:     }
// Arg is built on both arms, so its cleanup is unconditional and gets a
// plain scope nested inside the conditional one.
// CIR:     cir.cleanup.scope {
// CIR:       cir.call @_ZN5ExtraC1Ev(%[[EXTRA]])
// Extra is constructed last, so its scope is innermost and it dies first.
// CIR:       cir.cleanup.scope {
// CIR:         cir.call @_Z7consumeRK3ArgRK5Extra(%[[ARG]], %[[EXTRA]])
// CIR:         cir.yield
// CIR:       } cleanup all {
// CIR:         cir.call @_ZN5ExtraD1Ev(%[[EXTRA]])
// CIR:         cir.yield
// CIR:       }
// CIR:       cir.yield
// CIR:     } cleanup all {
// CIR:       cir.call @_ZN3ArgD1Ev(%[[ARG]])
// CIR:       cir.yield
// CIR:     }
// CIR:     cir.yield
// CIR:   } cleanup all {
// CIR:     %[[IS_ACTIVE:.*]] = cir.load{{.*}} %[[ACTIVE]]
// CIR:     cir.if %[[IS_ACTIVE]] {
// CIR:       cir.call @_ZN6AltArgD1Ev(%[[ALT]])
// CIR:     }
// CIR:     cir.yield
// CIR:   }

// LLVM: define {{.*}} void @_Z11callCondArgb({{.*}} personality ptr @__gxx_personality_v0
// LLVM:   %[[ARG:.*]] = alloca %struct.Arg
// LLVM:   %[[ALT:.*]] = alloca %struct.AltArg
// LLVM:   %[[ACTIVE:.*]] = alloca i8
// LLVM:   %[[EXTRA:.*]] = alloca %struct.Extra
// LLVM:   store i8 0, ptr %[[ACTIVE]]
// LLVM:   br i1 %{{.*}}, label %[[TRUE_BB:.*]], label %[[FALSE_BB:.*]]
// Every unwind edge out of the conditional lands on one pad, which runs only
// the guarded ~AltArg. No Arg exists on any of those edges, and the flag is
// set only where AltArg has been built.
//
// The two bare constructors unwind here but are plain calls in OGCG below.
// The conditional's scope is opened before its arms are emitted, so the whole
// conditional unwinds to it, while OGCG starts using invoke only once the
// AltArg cleanup is live. The flag is false on those two edges, so the pad
// reaches the resume without running a destructor.
// LLVM: [[TRUE_BB]]:
// LLVM:   invoke void @_ZN3ArgC1Ev(ptr {{.*}} %[[ARG]])
// LLVM:           to label %[[TRUE_CONT:.*]] unwind label %[[LPAD_ARMS:.*]]
// LLVM: [[TRUE_CONT]]:
// LLVM:   br label %[[COND_END:.*]]
// LLVM: [[FALSE_BB]]:
// LLVM:   invoke void @_ZN6AltArgC1Ev(ptr {{.*}} %[[ALT]])
// LLVM:           to label %[[ALT_CONT:.*]] unwind label %[[LPAD_ARMS]]
// LLVM: [[ALT_CONT]]:
// LLVM:   store i8 1, ptr %[[ACTIVE]]
// LLVM:   invoke void @_ZN3ArgC1ERK6AltArg(ptr {{.*}} %[[ARG]], ptr {{.*}} %[[ALT]])
// LLVM:           to label %[[CONV_CONT:.*]] unwind label %[[LPAD_ARMS]]
// LLVM: [[CONV_CONT]]:
// LLVM:   br label %[[COND_END]]
// LLVM: [[COND_END]]:
// LLVM:   invoke void @_ZN5ExtraC1Ev(ptr {{.*}} %[[EXTRA]])
// LLVM:           to label %[[EXTRA_CONT:.*]] unwind label %[[LPAD_EXTRA:.*]]
// LLVM: [[EXTRA_CONT]]:
// LLVM:   invoke void @_Z7consumeRK3ArgRK5Extra(ptr {{.*}} %[[ARG]], ptr {{.*}} %[[EXTRA]])
// LLVM:           to label %[[CALL_CONT:.*]] unwind label %[[LPAD_CONSUME:.*]]
// Normal path starts with Extra, the last temporary constructed.
// LLVM: [[CALL_CONT]]:
// LLVM:   call void @_ZN5ExtraD1Ev(ptr {{.*}} %[[EXTRA]])
// Unwinding from consume destroys Extra, then joins Arg's cleanup.
// LLVM: [[LPAD_CONSUME]]:
// LLVM:   landingpad { ptr, i32 }
// LLVM:     cleanup
// LLVM:   call void @_ZN5ExtraD1Ev(ptr {{.*}} %[[EXTRA]])
// LLVM:   br label %[[EH_ARG:.*]]
// Normal path continues with Arg.
// LLVM:   call void @_ZN3ArgD1Ev(ptr {{.*}} %[[ARG]])
// Unwinding from Extra's constructor skips ~Extra and joins the same pad.
// LLVM: [[LPAD_EXTRA]]:
// LLVM:   landingpad { ptr, i32 }
// LLVM:     cleanup
// LLVM:   br label %[[EH_ARG]]
// LLVM: [[EH_ARG]]:
// LLVM:   call void @_ZN3ArgD1Ev(ptr {{.*}} %[[ARG]])
// LLVM:   br label %[[EH_ALT:.*]]
// Normal path ends with the guarded AltArg.
// LLVM:   %[[BYTE:.*]] = load i8, ptr %[[ACTIVE]]
// LLVM:   %[[BOOL:.*]] = trunc i8 %[[BYTE]] to i1
// LLVM:   br i1 %[[BOOL]], label %[[ALT_DTOR:.*]], label %{{.*}}
// LLVM: [[ALT_DTOR]]:
// LLVM:   call void @_ZN6AltArgD1Ev(ptr {{.*}} %[[ALT]])
// The arms' pad skips ~Extra and ~Arg and joins the guarded AltArg directly.
// LLVM: [[LPAD_ARMS]]:
// LLVM:   landingpad { ptr, i32 }
// LLVM:     cleanup
// LLVM:   br label %[[EH_ALT]]
// LLVM: [[EH_ALT]]:
// LLVM:   %[[EH_BYTE:.*]] = load i8, ptr %[[ACTIVE]]
// LLVM:   %[[EH_BOOL:.*]] = trunc i8 %[[EH_BYTE]] to i1
// LLVM:   br i1 %[[EH_BOOL]], label %[[EH_ALT_DTOR:.*]], label %{{.*}}
// LLVM: [[EH_ALT_DTOR]]:
// LLVM:   call void @_ZN6AltArgD1Ev(ptr {{.*}} %[[ALT]])
// LLVM:   resume

// OGCG: define {{.*}} void @_Z11callCondArgb({{.*}} personality ptr @__gxx_personality_v0
// OGCG:   %[[ARG:.*]] = alloca %struct.Arg
// OGCG:   %[[ALT:.*]] = alloca %struct.AltArg
// OGCG:   %[[ACTIVE:.*]] = alloca i1
// OGCG:   %[[EXTRA:.*]] = alloca %struct.Extra
// OGCG:   store i1 false, ptr %[[ACTIVE]]
// OGCG:   br i1 %{{.*}}, label %[[COND_TRUE:.*]], label %[[COND_FALSE:.*]]
// Nothing is live yet when either arm starts, so those constructors need no
// unwind edge. Only the conversion, which runs with AltArg live, invokes.
// The CIR pipeline invokes all three, as noted above.
// OGCG: [[COND_TRUE]]:
// OGCG:   call void @_ZN3ArgC1Ev(ptr {{.*}} %[[ARG]])
// OGCG:   br label %[[COND_END:.*]]
// OGCG: [[COND_FALSE]]:
// OGCG:   call void @_ZN6AltArgC1Ev(ptr {{.*}} %[[ALT]])
// OGCG:   store i1 true, ptr %[[ACTIVE]]
// OGCG:   invoke void @_ZN3ArgC1ERK6AltArg(ptr {{.*}} %[[ARG]], ptr {{.*}} %[[ALT]])
// OGCG:           to label %[[CONV_CONT:.*]] unwind label %[[LPAD_CONV:.*]]
// OGCG: [[CONV_CONT]]:
// OGCG:   br label %[[COND_END]]
// OGCG: [[COND_END]]:
// OGCG:   invoke void @_ZN5ExtraC1Ev(ptr {{.*}} %[[EXTRA]])
// OGCG:           to label %[[EXTRA_CONT:.*]] unwind label %[[LPAD_EXTRA:.*]]
// OGCG: [[EXTRA_CONT]]:
// OGCG:   invoke void @_Z7consumeRK3ArgRK5Extra(ptr {{.*}} %[[ARG]], ptr {{.*}} %[[EXTRA]])
// OGCG:           to label %[[CALL_CONT:.*]] unwind label %[[LPAD_CONSUME:.*]]
// Normal path: Extra, then Arg, then the guarded AltArg.
// OGCG: [[CALL_CONT]]:
// OGCG:   call void @_ZN5ExtraD1Ev(ptr {{.*}} %[[EXTRA]])
// OGCG:   call void @_ZN3ArgD1Ev(ptr {{.*}} %[[ARG]])
// OGCG:   %[[IS_ACTIVE:.*]] = load i1, ptr %[[ACTIVE]]
// OGCG:   br i1 %[[IS_ACTIVE]], label %[[ALT_DTOR:.*]], label %[[DONE:.*]]
// OGCG: [[ALT_DTOR]]:
// OGCG:   call void @_ZN6AltArgD1Ev(ptr {{.*}} %[[ALT]])
// OGCG: [[DONE]]:
// OGCG:   ret void
// The conversion's pad runs only the guarded ~AltArg, since Arg is not
// constructed there.
// OGCG: [[LPAD_CONV]]:
// OGCG:   landingpad { ptr, i32 }
// OGCG:     cleanup
// OGCG:   br label %[[EH_ALT:.*]]
// Unwinding from Extra's constructor skips ~Extra and joins Arg's cleanup.
// OGCG: [[LPAD_EXTRA]]:
// OGCG:   landingpad { ptr, i32 }
// OGCG:     cleanup
// OGCG:   br label %[[EH_ARG:.*]]
// Unwinding from consume destroys Extra first, then joins the same pad.
// OGCG: [[LPAD_CONSUME]]:
// OGCG:   landingpad { ptr, i32 }
// OGCG:     cleanup
// OGCG:   call void @_ZN5ExtraD1Ev(ptr {{.*}} %[[EXTRA]])
// OGCG:   br label %[[EH_ARG]]
// OGCG: [[EH_ARG]]:
// OGCG:   call void @_ZN3ArgD1Ev(ptr {{.*}} %[[ARG]])
// OGCG:   br label %[[EH_ALT]]
// OGCG: [[EH_ALT]]:
// OGCG:   %[[EH_IS_ACTIVE:.*]] = load i1, ptr %[[ACTIVE]]
// OGCG:   br i1 %[[EH_IS_ACTIVE]], label %[[EH_ALT_DTOR:.*]], label %{{.*}}
// OGCG: [[EH_ALT_DTOR]]:
// OGCG:   call void @_ZN6AltArgD1Ev(ptr {{.*}} %[[ALT]])
// OGCG:   resume

struct Mid { Mid(); ~Mid(); };
void take(const Arg &, const Mid &, const Arg &);

// Two conditionals in one full expression with an unconditional temporary
// between them. Each conditional opens its own cleanup scope where it begins,
// so the second nests inside the first, and Mid's scope sits between them.
//
// Construction runs AltArg1 (false arm only), Arg1, Mid, AltArg2 (false arm
// only), Arg2, so destruction runs Arg2, AltArg2, Mid, Arg1, AltArg1. That is
// five nested scopes closing innermost first.
void twoConditionals(bool c1, bool c2) {
  take(c1 ? Arg() : AltArg(), Mid(), c2 ? Arg() : AltArg());
}

// CIR: cir.func {{.*}} @_Z15twoConditionalsbb
// CIR:   %[[ARG1:.*]] = cir.alloca "ref.tmp0" {{.*}} : !cir.ptr<!rec_Arg>
// CIR:   %[[ALT1:.*]] = cir.alloca "ref.tmp1" {{.*}} : !cir.ptr<!rec_AltArg>
// CIR:   %[[FLAG1:.*]] = cir.alloca "cleanup.cond" {{.*}} : !cir.ptr<!cir.bool>
// CIR:   %[[MID:.*]] = cir.alloca "ref.tmp2" {{.*}} : !cir.ptr<!rec_Mid>
// CIR:   %[[ARG2:.*]] = cir.alloca "ref.tmp3" {{.*}} : !cir.ptr<!rec_Arg>
// CIR:   %[[ALT2:.*]] = cir.alloca "ref.tmp4" {{.*}} : !cir.ptr<!rec_AltArg>
// CIR:   %[[FLAG2:.*]] = cir.alloca "cleanup.cond" {{.*}} : !cir.ptr<!cir.bool>
// Scope 1: the first conditional, holding the guarded ~AltArg1.
// CIR:   cir.cleanup.scope {
// CIR:     %[[FALSE1:.*]] = cir.const #false
// CIR:     cir.store %[[FALSE1]], %[[FLAG1]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR:     cir.if %{{.*}} {
// CIR:       cir.call @_ZN3ArgC1Ev(%[[ARG1]])
// CIR:     } else {
// CIR:       cir.call @_ZN6AltArgC1Ev(%[[ALT1]])
// CIR:       %[[TRUE1:.*]] = cir.const #true
// CIR:       cir.store %[[TRUE1]], %[[FLAG1]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR:       cir.call @_ZN3ArgC1ERK6AltArg(%[[ARG1]], %[[ALT1]])
// CIR:     }
// Scope 2: Arg1, built on both arms and so destroyed unconditionally.
// CIR:     cir.cleanup.scope {
// CIR:       cir.call @_ZN3MidC1Ev(%[[MID]])
// Scope 3: Mid, constructed between the two conditionals.
// CIR:       cir.cleanup.scope {
// Scope 4: the second conditional, holding the guarded ~AltArg2.
// CIR:         cir.cleanup.scope {
// CIR:           %[[FALSE2:.*]] = cir.const #false
// CIR:           cir.store %[[FALSE2]], %[[FLAG2]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR:           cir.if %{{.*}} {
// CIR:             cir.call @_ZN3ArgC1Ev(%[[ARG2]])
// CIR:           } else {
// CIR:             cir.call @_ZN6AltArgC1Ev(%[[ALT2]])
// CIR:             %[[TRUE2:.*]] = cir.const #true
// CIR:             cir.store %[[TRUE2]], %[[FLAG2]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR:             cir.call @_ZN3ArgC1ERK6AltArg(%[[ARG2]], %[[ALT2]])
// CIR:           }
// Scope 5: Arg2, the last temporary constructed and the first destroyed.
// CIR:           cir.cleanup.scope {
// CIR:             cir.call @_Z4takeRK3ArgRK3MidS1_(%[[ARG1]], %[[MID]], %[[ARG2]])
// CIR:             cir.yield
// CIR:           } cleanup all {
// CIR:             cir.call @_ZN3ArgD1Ev(%[[ARG2]])
// CIR:             cir.yield
// CIR:           }
// CIR:           cir.yield
// CIR:         } cleanup all {
// CIR:           %[[IS2:.*]] = cir.load{{.*}} %[[FLAG2]]
// CIR:           cir.if %[[IS2]] {
// CIR:             cir.call @_ZN6AltArgD1Ev(%[[ALT2]])
// CIR:           }
// CIR:           cir.yield
// CIR:         }
// CIR:         cir.yield
// CIR:       } cleanup all {
// CIR:         cir.call @_ZN3MidD1Ev(%[[MID]])
// CIR:         cir.yield
// CIR:       }
// CIR:       cir.yield
// CIR:     } cleanup all {
// CIR:       cir.call @_ZN3ArgD1Ev(%[[ARG1]])
// CIR:       cir.yield
// CIR:     }
// CIR:     cir.yield
// CIR:   } cleanup all {
// CIR:     %[[IS1:.*]] = cir.load{{.*}} %[[FLAG1]]
// CIR:     cir.if %[[IS1]] {
// CIR:       cir.call @_ZN6AltArgD1Ev(%[[ALT1]])
// CIR:     }
// CIR:     cir.yield
// CIR:   }

// LLVM: define {{.*}} void @_Z15twoConditionalsbb({{.*}} personality ptr @__gxx_personality_v0
// LLVM:   %[[ARG1:.*]] = alloca %struct.Arg
// LLVM:   %[[ALT1:.*]] = alloca %struct.AltArg
// LLVM:   %[[FLAG1:.*]] = alloca i8
// LLVM:   %[[MID:.*]] = alloca %struct.Mid
// LLVM:   %[[ARG2:.*]] = alloca %struct.Arg
// LLVM:   %[[ALT2:.*]] = alloca %struct.AltArg
// LLVM:   %[[FLAG2:.*]] = alloca i8
// LLVM:   store i8 0, ptr %[[FLAG1]]
// LLVM:   br i1 %{{.*}}, label %[[T1:.*]], label %[[F1:.*]]
// Both arms of the first conditional share one pad.
// LLVM: [[T1]]:
// LLVM:   invoke void @_ZN3ArgC1Ev(ptr {{.*}} %[[ARG1]])
// LLVM:           to label %{{.*}} unwind label %[[LPAD_ARMS1:.*]]
// LLVM: [[F1]]:
// LLVM:   invoke void @_ZN6AltArgC1Ev(ptr {{.*}} %[[ALT1]])
// LLVM:           to label %{{.*}} unwind label %[[LPAD_ARMS1]]
// LLVM:   store i8 1, ptr %[[FLAG1]]
// LLVM:   invoke void @_ZN3ArgC1ERK6AltArg(ptr {{.*}} %[[ARG1]], ptr {{.*}} %[[ALT1]])
// LLVM:           to label %{{.*}} unwind label %[[LPAD_ARMS1]]
// LLVM:   invoke void @_ZN3MidC1Ev(ptr {{.*}} %[[MID]])
// LLVM:           to label %{{.*}} unwind label %[[LPAD_MID:.*]]
// LLVM:   store i8 0, ptr %[[FLAG2]]
// LLVM:   br i1 %{{.*}}, label %[[T2:.*]], label %[[F2:.*]]
// Both arms of the second conditional share a different pad.
// LLVM: [[T2]]:
// LLVM:   invoke void @_ZN3ArgC1Ev(ptr {{.*}} %[[ARG2]])
// LLVM:           to label %{{.*}} unwind label %[[LPAD_ARMS2:.*]]
// LLVM: [[F2]]:
// LLVM:   invoke void @_ZN6AltArgC1Ev(ptr {{.*}} %[[ALT2]])
// LLVM:           to label %{{.*}} unwind label %[[LPAD_ARMS2]]
// LLVM:   store i8 1, ptr %[[FLAG2]]
// LLVM:   invoke void @_ZN3ArgC1ERK6AltArg(ptr {{.*}} %[[ARG2]], ptr {{.*}} %[[ALT2]])
// LLVM:           to label %{{.*}} unwind label %[[LPAD_ARMS2]]
// LLVM:   invoke void @_Z4takeRK3ArgRK3MidS1_(ptr {{.*}} %[[ARG1]], ptr {{.*}} %[[MID]], ptr {{.*}} %[[ARG2]])
// LLVM:           to label %{{.*}} unwind label %[[LPAD_TAKE:.*]]
// Normal path starts with Arg2, the last temporary constructed.
// LLVM:   call void @_ZN3ArgD1Ev(ptr {{.*}} %[[ARG2]])
// Unwinding from take destroys Arg2 and joins the second conditional's pad.
// LLVM: [[LPAD_TAKE]]:
// LLVM:   landingpad { ptr, i32 }
// LLVM:     cleanup
// LLVM:   call void @_ZN3ArgD1Ev(ptr {{.*}} %[[ARG2]])
// LLVM:   br label %[[EH_ALT2:.*]]
// Normal path continues with the guarded AltArg2.
// LLVM:   %[[B2:.*]] = load i8, ptr %[[FLAG2]]
// LLVM:   %[[C2:.*]] = trunc i8 %[[B2]] to i1
// LLVM:   br i1 %[[C2]], label %[[ALT2_DTOR:.*]], label %{{.*}}
// LLVM: [[ALT2_DTOR]]:
// LLVM:   call void @_ZN6AltArgD1Ev(ptr {{.*}} %[[ALT2]])
// The second conditional's arms skip ~Arg2 and enter at the same level.
// LLVM: [[LPAD_ARMS2]]:
// LLVM:   landingpad { ptr, i32 }
// LLVM:     cleanup
// LLVM:   br label %[[EH_ALT2]]
// LLVM: [[EH_ALT2]]:
// LLVM:   %[[EB2:.*]] = load i8, ptr %[[FLAG2]]
// LLVM:   %[[EC2:.*]] = trunc i8 %[[EB2]] to i1
// LLVM:   br i1 %[[EC2]], label %[[EH_ALT2_DTOR:.*]], label %{{.*}}
// LLVM: [[EH_ALT2_DTOR]]:
// LLVM:   call void @_ZN6AltArgD1Ev(ptr {{.*}} %[[ALT2]])
// Normal path then Mid, and the unwind chain reaches Mid next as well.
// LLVM:   call void @_ZN3MidD1Ev(ptr {{.*}} %[[MID]])
// LLVM:   call void @_ZN3MidD1Ev(ptr {{.*}} %[[MID]])
// LLVM:   br label %[[EH_ARG1:.*]]
// Unwinding from Mid's constructor skips ~Mid and joins Arg1's cleanup.
// LLVM: [[LPAD_MID]]:
// LLVM:   landingpad { ptr, i32 }
// LLVM:     cleanup
// LLVM:   br label %[[EH_ARG1]]
// LLVM: [[EH_ARG1]]:
// LLVM:   call void @_ZN3ArgD1Ev(ptr {{.*}} %[[ARG1]])
// LLVM:   br label %[[EH_ALT1:.*]]
// Normal path ends with the guarded AltArg1.
// LLVM:   %[[B1:.*]] = load i8, ptr %[[FLAG1]]
// LLVM:   %[[C1:.*]] = trunc i8 %[[B1]] to i1
// LLVM:   br i1 %[[C1]], label %[[ALT1_DTOR:.*]], label %{{.*}}
// LLVM: [[ALT1_DTOR]]:
// LLVM:   call void @_ZN6AltArgD1Ev(ptr {{.*}} %[[ALT1]])
// The first conditional's arms enter the outermost level directly.
// LLVM: [[LPAD_ARMS1]]:
// LLVM:   landingpad { ptr, i32 }
// LLVM:     cleanup
// LLVM:   br label %[[EH_ALT1]]
// LLVM: [[EH_ALT1]]:
// LLVM:   %[[EB1:.*]] = load i8, ptr %[[FLAG1]]
// LLVM:   %[[EC1:.*]] = trunc i8 %[[EB1]] to i1
// LLVM:   br i1 %[[EC1]], label %[[EH_ALT1_DTOR:.*]], label %{{.*}}
// LLVM: [[EH_ALT1_DTOR]]:
// LLVM:   call void @_ZN6AltArgD1Ev(ptr {{.*}} %[[ALT1]])
// LLVM:   resume

// OGCG: define {{.*}} void @_Z15twoConditionalsbb({{.*}} personality ptr @__gxx_personality_v0
// OGCG:   %[[ARG1:.*]] = alloca %struct.Arg
// OGCG:   %[[ALT1:.*]] = alloca %struct.AltArg
// OGCG:   %[[FLAG1:.*]] = alloca i1
// OGCG:   %[[MID:.*]] = alloca %struct.Mid
// OGCG:   %[[ARG2:.*]] = alloca %struct.Arg
// OGCG:   %[[ALT2:.*]] = alloca %struct.AltArg
// OGCG:   %[[FLAG2:.*]] = alloca i1
// OGCG:   store i1 false, ptr %[[FLAG1]]
// OGCG:   br i1 %{{.*}}, label %[[T1:.*]], label %[[F1:.*]]
// Nothing is live yet in the first conditional, so its arms use plain calls.
// The second conditional's arms invoke, because Arg1 and Mid are live by
// then. The CIR pipeline invokes both, since each conditional's scope is
// opened before its arms are emitted.
// OGCG: [[T1]]:
// OGCG:   call void @_ZN3ArgC1Ev(ptr {{.*}} %[[ARG1]])
// OGCG: [[F1]]:
// OGCG:   call void @_ZN6AltArgC1Ev(ptr {{.*}} %[[ALT1]])
// OGCG:   store i1 true, ptr %[[FLAG1]]
// OGCG:   invoke void @_ZN3ArgC1ERK6AltArg(ptr {{.*}} %[[ARG1]], ptr {{.*}} %[[ALT1]])
// OGCG:           to label %{{.*}} unwind label %[[LPAD_CONV1:.*]]
// OGCG:   invoke void @_ZN3MidC1Ev(ptr {{.*}} %[[MID]])
// OGCG:           to label %{{.*}} unwind label %[[LPAD_MID:.*]]
// OGCG:   store i1 false, ptr %[[FLAG2]]
// OGCG:   br i1 %{{.*}}, label %[[T2:.*]], label %[[F2:.*]]
// OGCG: [[T2]]:
// OGCG:   invoke void @_ZN3ArgC1Ev(ptr {{.*}} %[[ARG2]])
// OGCG:           to label %{{.*}} unwind label %[[LPAD_ARMS2:.*]]
// OGCG: [[F2]]:
// OGCG:   invoke void @_ZN6AltArgC1Ev(ptr {{.*}} %[[ALT2]])
// OGCG:           to label %{{.*}} unwind label %[[LPAD_ARMS2]]
// OGCG:   store i1 true, ptr %[[FLAG2]]
// OGCG:   invoke void @_ZN3ArgC1ERK6AltArg(ptr {{.*}} %[[ARG2]], ptr {{.*}} %[[ALT2]])
// OGCG:           to label %{{.*}} unwind label %[[LPAD_CONV2:.*]]
// OGCG:   invoke void @_Z4takeRK3ArgRK3MidS1_(ptr {{.*}} %[[ARG1]], ptr {{.*}} %[[MID]], ptr {{.*}} %[[ARG2]])
// OGCG:           to label %{{.*}} unwind label %[[LPAD_TAKE:.*]]
// Normal path: Arg2, AltArg2, Mid, Arg1, AltArg1.
// OGCG:   call void @_ZN3ArgD1Ev(ptr {{.*}} %[[ARG2]])
// OGCG:   %[[IS2:.*]] = load i1, ptr %[[FLAG2]]
// OGCG:   br i1 %[[IS2]], label %[[ALT2_DTOR:.*]], label %[[DONE2:.*]]
// OGCG: [[ALT2_DTOR]]:
// OGCG:   call void @_ZN6AltArgD1Ev(ptr {{.*}} %[[ALT2]])
// OGCG: [[DONE2]]:
// OGCG:   call void @_ZN3MidD1Ev(ptr {{.*}} %[[MID]])
// OGCG:   call void @_ZN3ArgD1Ev(ptr {{.*}} %[[ARG1]])
// OGCG:   %[[IS1:.*]] = load i1, ptr %[[FLAG1]]
// OGCG:   br i1 %[[IS1]], label %[[ALT1_DTOR:.*]], label %[[DONE1:.*]]
// OGCG: [[ALT1_DTOR]]:
// OGCG:   call void @_ZN6AltArgD1Ev(ptr {{.*}} %[[ALT1]])
// OGCG: [[DONE1]]:
// OGCG:   ret void
// The unwind chain enters at the level matching what is already built.
// OGCG: [[LPAD_CONV1]]:
// OGCG:   landingpad { ptr, i32 }
// OGCG:     cleanup
// OGCG:   br label %[[EH_ALT1:.*]]
// OGCG: [[LPAD_MID]]:
// OGCG:   landingpad { ptr, i32 }
// OGCG:     cleanup
// OGCG:   br label %[[EH_ARG1:.*]]
// OGCG: [[LPAD_ARMS2]]:
// OGCG:   landingpad { ptr, i32 }
// OGCG:     cleanup
// OGCG:   br label %[[EH_MID:.*]]
// OGCG: [[LPAD_CONV2]]:
// OGCG:   landingpad { ptr, i32 }
// OGCG:     cleanup
// OGCG:   br label %[[EH_ALT2:.*]]
// OGCG: [[LPAD_TAKE]]:
// OGCG:   landingpad { ptr, i32 }
// OGCG:     cleanup
// OGCG:   call void @_ZN3ArgD1Ev(ptr {{.*}} %[[ARG2]])
// OGCG:   br label %[[EH_ALT2]]
// OGCG: [[EH_ALT2]]:
// OGCG:   %[[EIS2:.*]] = load i1, ptr %[[FLAG2]]
// OGCG:   br i1 %[[EIS2]], label %[[EH_ALT2_DTOR:.*]], label %{{.*}}
// OGCG: [[EH_ALT2_DTOR]]:
// OGCG:   call void @_ZN6AltArgD1Ev(ptr {{.*}} %[[ALT2]])
// OGCG: [[EH_MID]]:
// OGCG:   call void @_ZN3MidD1Ev(ptr {{.*}} %[[MID]])
// OGCG:   br label %[[EH_ARG1]]
// OGCG: [[EH_ARG1]]:
// OGCG:   call void @_ZN3ArgD1Ev(ptr {{.*}} %[[ARG1]])
// OGCG:   br label %[[EH_ALT1]]
// OGCG: [[EH_ALT1]]:
// OGCG:   %[[EIS1:.*]] = load i1, ptr %[[FLAG1]]
// OGCG:   br i1 %[[EIS1]], label %[[EH_ALT1_DTOR:.*]], label %{{.*}}
// OGCG: [[EH_ALT1_DTOR]]:
// OGCG:   call void @_ZN6AltArgD1Ev(ptr {{.*}} %[[ALT1]])
// OGCG:   resume
