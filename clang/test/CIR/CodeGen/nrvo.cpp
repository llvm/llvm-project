// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fno-elide-constructors -fclangir -emit-cir %s -o %t-noelide.cir
// RUN: FileCheck --input-file=%t-noelide.cir %s --check-prefix=CIR-NOELIDE
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefixes=LLVM,LLVMCIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefixes=LLVM,OGCG

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
// CIR-NEXT:   %[[RETVAL:.*]] = cir.alloca "__retval" {{.*}} init : !cir.ptr<!rec_S>
// CIR-NEXT:   cir.call @_ZN1SC1Ev(%[[RETVAL]]) : (!cir.ptr<!rec_S> {{.*}}) -> ()
// CIR-NEXT:   %[[SLOT:.*]] = cir.cast bitcast %[[RETVAL]] : !cir.ptr<!rec_S> -> !cir.ptr<!u64i>
// CIR-NEXT:   %[[COERCED:.*]] = cir.load align(4) %[[SLOT]] : !cir.ptr<!u64i>, !u64i
// CIR-NEXT:   cir.return %[[COERCED]]

// CIR-NOELIDE:      cir.func{{.*}} @_Z2f1v() -> !u64i
// CIR-NOELIDE-NEXT:   %[[RETVAL:.*]] = cir.alloca "__retval" {{.*}} : !cir.ptr<!rec_S>
// CIR-NOELIDE-NEXT:   %[[S:.*]] = cir.alloca "s" {{.*}} init : !cir.ptr<!rec_S>
// CIR-NOELIDE-NEXT:   cir.call @_ZN1SC1Ev(%[[S]]) : (!cir.ptr<!rec_S> {{.*}}) -> ()
// CIR-NOELIDE-NEXT:   cir.copy %[[S]] align(4) to %[[RETVAL]] align(4) : !cir.ptr<!rec_S>
// CIR-NOELIDE-NEXT:   %[[SLOT:.*]] = cir.cast bitcast %[[RETVAL]] : !cir.ptr<!rec_S> -> !cir.ptr<!u64i>
// CIR-NOELIDE-NEXT:   %[[COERCED:.*]] = cir.load align(4) %[[SLOT]] : !cir.ptr<!u64i>, !u64i
// CIR-NOELIDE-NEXT:   cir.return %[[COERCED]]

// LLVM:      define{{.*}} i64 @_Z2f1v()
// OGCG-NEXT: entry:
// LLVM-NEXT:   %[[RETVAL:.*]] = alloca %struct.S
// LLVM-NEXT:   call void @_ZN1SC1Ev(ptr {{.*}} %[[RETVAL]])
// LLVM-NEXT:   %[[RET:.*]] = load i64, ptr %[[RETVAL]], align 4
// LLVM-NEXT:   ret i64 %[[RET]]

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
// OGCG:   %[[RESULT_ADDR:.*]] = alloca ptr
// LLVMCIR:   %[[NRVO_FLAG:.*]] = alloca i8
// OGCG:   %[[NRVO_FLAG:.*]] = alloca i1, align 1
// OGCG:   store ptr %[[RESULT]], ptr %[[RESULT_ADDR]]
// LLVMCIR:   store i8 0, ptr %[[NRVO_FLAG]]
// OGCG:   store i1 false, ptr %[[NRVO_FLAG]]
// LLVM:   call void @_Z10maybeThrowv()
// LLVMCIR:   store i8 1, ptr %[[NRVO_FLAG]]
// OGCG:   store i1 true, ptr %[[NRVO_FLAG]]
// LLVMCIR:   %[[NRVO_VAL:.*]] = load i8, ptr %[[NRVO_FLAG]]
// LLVMCIR:   %[[NRVO_VAL_TRUNC:.*]] = trunc i8 %[[NRVO_VAL]] to i1
// LLVMCIR:   %[[NOT_NRVO_VAL:.*]] = xor i1 %[[NRVO_VAL_TRUNC]], true
// LLVMCIR:   br i1 %[[NOT_NRVO_VAL]], label %[[NRVO_UNUSED:.*]], label %[[NRVO_DONE:.*]]
// OGCG:   %[[NRVO_VAL:.*]] = load i1, ptr %[[NRVO_FLAG]]
// OGCG:   br i1 %[[NRVO_VAL]], label %[[NRVO_DONE:.*]], label %[[NRVO_UNUSED:.*]]
// LLVM: [[NRVO_UNUSED]]:
// LLVM-NEXT:   call void @_ZN10NonTrivialD1Ev(ptr noundef nonnull align 1 dereferenceable(1) %[[RESULT]])
// LLVM-NEXT:   br label %[[NRVO_DONE]]
// LLVM: [[NRVO_DONE]]:
// LLVM:   ret void
