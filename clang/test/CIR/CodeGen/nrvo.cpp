// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fno-elide-constructors -fclangir -emit-cir %s -o %t-noelide.cir
// RUN: FileCheck --input-file=%t-noelide.cir %s --check-prefix=CIR-NOELIDE
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG

// There are no LLVM and OGCG tests with -fno-elide-constructors because the
// lowering isn't of interest for this test. We just need to see that the
// copy constructor is elided without -fno-elide-constructors but not with it.

struct S {
  S();
  int a;
  int b;
};

struct S f1() {
  S s;
  return s;
}


// CIR:      cir.func{{.*}} @_Z2f1v() -> !u64i
// CIR-NEXT:   %[[COERCE:.*]] = cir.alloca "coerce" {{.*}} : !cir.ptr<!rec_S>
// CIR-NEXT:   %[[RETVAL:.*]] = cir.alloca "__retval" {{.*}} init : !cir.ptr<!rec_S>
// CIR-NEXT:   cir.call @_ZN1SC1Ev(%[[RETVAL]]) : (!cir.ptr<!rec_S> {{.*}}) -> ()
// CIR-NEXT:   %[[RET:.*]] = cir.load %[[RETVAL]] : !cir.ptr<!rec_S>, !rec_S
// CIR-NEXT:   cir.store %[[RET]], %[[COERCE]] : !rec_S, !cir.ptr<!rec_S>
// CIR-NEXT:   %[[SLOT:.*]] = cir.cast bitcast %[[COERCE]] : !cir.ptr<!rec_S> -> !cir.ptr<!u64i>
// CIR-NEXT:   %[[COERCED:.*]] = cir.load %[[SLOT]] : !cir.ptr<!u64i>, !u64i
// CIR-NEXT:   cir.return %[[COERCED]]

// CIR-NOELIDE:      cir.func{{.*}} @_Z2f1v() -> !u64i
// CIR-NOELIDE-NEXT:   %[[COERCE:.*]] = cir.alloca "coerce" {{.*}} : !cir.ptr<!rec_S>
// CIR-NOELIDE-NEXT:   %[[RETVAL:.*]] = cir.alloca "__retval" {{.*}} : !cir.ptr<!rec_S>
// CIR-NOELIDE-NEXT:   %[[S:.*]] = cir.alloca "s" {{.*}} init : !cir.ptr<!rec_S>
// CIR-NOELIDE-NEXT:   cir.call @_ZN1SC1Ev(%[[S]]) : (!cir.ptr<!rec_S> {{.*}}) -> ()
// CIR-NOELIDE-NEXT:   cir.copy %[[S]] align(4) to %[[RETVAL]] align(4) : !cir.ptr<!rec_S>
// CIR-NOELIDE-NEXT:   %[[RET:.*]] = cir.load %[[RETVAL]] : !cir.ptr<!rec_S>, !rec_S
// CIR-NOELIDE-NEXT:   cir.store %[[RET]], %[[COERCE]] : !rec_S, !cir.ptr<!rec_S>
// CIR-NOELIDE-NEXT:   %[[SLOT:.*]] = cir.cast bitcast %[[COERCE]] : !cir.ptr<!rec_S> -> !cir.ptr<!u64i>
// CIR-NOELIDE-NEXT:   %[[COERCED:.*]] = cir.load %[[SLOT]] : !cir.ptr<!u64i>, !u64i
// CIR-NOELIDE-NEXT:   cir.return %[[COERCED]]

// LLVM:      define{{.*}} i64 @_Z2f1v()
// LLVM-NEXT:   %[[COERCE:.*]] = alloca %struct.S
// LLVM-NEXT:   %[[RETVAL:.*]] = alloca %struct.S
// LLVM-NEXT:   call void @_ZN1SC1Ev(ptr {{.*}} %[[RETVAL]])
// LLVM-NEXT:   %[[TMP:.*]] = load %struct.S, ptr %[[RETVAL]]
// LLVM-NEXT:   store %struct.S %[[TMP]], ptr %[[COERCE]]
// LLVM-NEXT:   %[[RET:.*]] = load i64, ptr %[[COERCE]]
// LLVM-NEXT:   ret i64 %[[RET]]

// OGCG:      define{{.*}} i64 @_Z2f1v()
// OGCG-NEXT: entry:
// OGCG-NEXT:   %[[RETVAL:.*]] = alloca %struct.S
// OGCG-NEXT:   call void @_ZN1SC1Ev(ptr {{.*}} %[[RETVAL]])
// OGCG-NEXT:   %[[RET:.*]] = load i64, ptr %[[RETVAL]]
// OGCG-NEXT:   ret i64 %[[RET]]

struct NonTrivial {
  ~NonTrivial();
};

void maybeThrow();

NonTrivial test_nrvo() {
  NonTrivial result;
  maybeThrow();
  return result;
}

// TODO(cir): Handle normal cleanup properly.


// CIR: cir.func {{.*}} @_Z9test_nrvov(%[[RESULT:.*]]: !cir.ptr<!rec_NonTrivial> {{.*}}llvm.sret = !rec_NonTrivial{{.*}})
// CIR:   %[[NRVO_FLAG:.*]] = cir.alloca "nrvo" {{.*}} : !cir.ptr<!cir.bool>
// CIR:   %[[FALSE:.*]] = cir.const #false
// CIR:   cir.store{{.*}} %[[FALSE]], %[[NRVO_FLAG]]
// CIR:   cir.cleanup.scope {
// CIR:     cir.call @_Z10maybeThrowv() : () -> ()
// CIR:     %[[TRUE:.*]] = cir.const #true
// CIR:     cir.store{{.*}} %[[TRUE]], %[[NRVO_FLAG]]
// CIR:     cir.return
// CIR:   } cleanup normal {
// CIR:     %[[NRVO_FLAG_VAL:.*]] = cir.load{{.*}} %[[NRVO_FLAG]]
// CIR:     %[[NOT_NRVO_VAL:.*]] = cir.not %[[NRVO_FLAG_VAL]]
// CIR:     cir.if %[[NOT_NRVO_VAL]] {
// CIR:       cir.call @_ZN10NonTrivialD1Ev(%[[RESULT]])
// CIR:     }
// CIR:     cir.yield
// CIR:   }
//
// TODO(cir): This is unreachable, but it really shouldn't be here. This is an
//            artifact of us falling through to emitImplicitReturn().
// CIR:   cir.trap

// LLVM: define {{.*}} void @_Z9test_nrvov(ptr dead_on_unwind noalias writable sret(%struct.NonTrivial) align 1 %[[RESULT:.*]])
// LLVM:   %[[NRVO_FLAG:.*]] = alloca i8
// LLVM:   store i8 0, ptr %[[NRVO_FLAG]]
// LLVM:   call void @_Z10maybeThrowv()
// LLVM:   store i8 1, ptr %[[NRVO_FLAG]]
// LLVM:   %[[NRVO_VAL:.*]] = load i8, ptr %[[NRVO_FLAG]]
// LLVM:   %[[NRVO_VAL_TRUNC:.*]] = trunc i8 %[[NRVO_VAL]] to i1
// LLVM:   %[[NOT_NRVO_VAL:.*]] = xor i1 %[[NRVO_VAL_TRUNC]], true
// LLVM:   br i1 %[[NOT_NRVO_VAL]], label %[[NRVO_UNUSED:.*]], label %[[NRVO_USED:.*]]
// LLVM: [[NRVO_UNUSED]]:
// LLVM:   call void @_ZN10NonTrivialD1Ev(ptr {{.*}} %[[RESULT]])
// LLVM:   br label %[[NRVO_USED]]
// LLVM: [[NRVO_USED]]:
// LLVM:   ret void

// OGCG: define {{.*}} void @_Z9test_nrvov(ptr {{.*}} sret(%struct.NonTrivial) {{.*}} %[[RESULT:.*]])
// OGCG:   %[[RESULT_ADDR:.*]] = alloca ptr
// OGCG:   %[[NRVO_FLAG:.*]] = alloca i1, align 1
// OGCG:   store ptr %[[RESULT]], ptr %[[RESULT_ADDR]]
// OGCG:   store i1 false, ptr %[[NRVO_FLAG]]
// OGCG:   call void @_Z10maybeThrowv()
// OGCG:   store i1 true, ptr %[[NRVO_FLAG]]
// OGCG:   %[[NRVO_VAL:.*]] = load i1, ptr %[[NRVO_FLAG]]
// OGCG:   br i1 %[[NRVO_VAL]], label %[[SKIPDTOR:.*]], label %[[NRVO_UNUSED:.*]]
// OGCG: [[NRVO_UNUSED]]:
// OGCG:   call void @_ZN10NonTrivialD1Ev(ptr {{.*}} %[[RESULT]])
// OGCG:   br label %[[SKIPDTOR]]
// OGCG: [[SKIPDTOR]]:
// OGCG:   ret void
