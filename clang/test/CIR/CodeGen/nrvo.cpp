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

// CIR:      cir.func{{.*}} @_Z2f1v() -> !rec_S {
// CIR-NEXT:   %[[RETVAL:.*]] = cir.alloca !rec_S, !cir.ptr<!rec_S>, ["__retval", init]
// CIR-NEXT:   cir.call @_ZN1SC1Ev(%[[RETVAL]]) : (!cir.ptr<!rec_S>) -> ()
// CIR-NEXT:   %[[RET:.*]] = cir.load %[[RETVAL]] : !cir.ptr<!rec_S>, !rec_S
// CIR-NEXT:   cir.return %[[RET]]

// CIR-NOELIDE:      cir.func{{.*}} @_Z2f1v() -> !rec_S {
// CIR-NOELIDE-NEXT:   %[[RETVAL:.*]] = cir.alloca !rec_S, !cir.ptr<!rec_S>, ["__retval"]
// CIR-NOELIDE-NEXT:   %[[S:.*]] = cir.alloca !rec_S, !cir.ptr<!rec_S>, ["s", init]
// CIR-NOELIDE-NEXT:   cir.call @_ZN1SC1Ev(%[[S]]) : (!cir.ptr<!rec_S>) -> ()
// CIR-NOELIDE-NEXT:   cir.call @_ZN1SC1EOS_(%[[RETVAL]], %[[S]]){{.*}} : (!cir.ptr<!rec_S>, !cir.ptr<!rec_S>) -> ()
// CIR-NOELIDE-NEXT:   %[[RET:.*]] = cir.load %[[RETVAL]] : !cir.ptr<!rec_S>, !rec_S
// CIR-NOELIDE-NEXT:   cir.return %[[RET]]

// FIXME: Update this when calling convetnion lowering is implemented.
// LLVM:      define{{.*}} %struct.S @_Z2f1v()
// LLVM-NEXT:   %[[RETVAL:.*]] = alloca %struct.S
// LLVM-NEXT:   call void @_ZN1SC1Ev(ptr %[[RETVAL]])
// LLVM-NEXT:   %[[RET:.*]] = load %struct.S, ptr %[[RETVAL]]
// LLVM-NEXT:   ret %struct.S %[[RET]]

// OGCG:      define{{.*}} i64 @_Z2f1v()
// OGCG-NEXT: entry:
// OGCG-NEXT:   %[[RETVAL:.*]] = alloca %struct.S
// OGCG-NEXT:   call void @_ZN1SC1Ev(ptr {{.*}} %[[RETVAL]])
// OGCG-NEXT:   %[[RET:.*]] = load i64, ptr %[[RETVAL]]
// OGCG-NEXT:   ret i64 %[[RET]]
