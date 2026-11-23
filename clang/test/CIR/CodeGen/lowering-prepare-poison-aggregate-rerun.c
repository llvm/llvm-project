// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: cir-opt %t.cir -o %t.once.cir
// RUN: cir-opt %t.cir --cir-lowering-prepare -o %t.twice.cir
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck %s --input-file=%t-cir.ll --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -x cir -emit-llvm %t.twice.cir -o %t-rerun.ll
// RUN: FileCheck %s --input-file=%t-rerun.ll --check-prefix=LLVM
// RUN: diff %t.once.cir %t.twice.cir

// CallConvLowering runs after LoweringPrepare and stores a poison value into
// the local copy of an ignored empty-record parameter. Running LoweringPrepare
// again on the emitted CIR must leave that store alone instead of rewriting it
// into a copy from a new constant global, so the LLVM IR is the same either way.

typedef struct {} Empty;

void take_empty(Empty e) { (void)e; }

// LLVM-NOT: @__const.
// LLVM: define dso_local void @take_empty()
// LLVM:   %[[E:.*]] = alloca %struct.Empty, align 1
// LLVM:   store %struct.Empty poison, ptr %[[E]], align 1
// LLVM-NOT: llvm.memcpy
// LLVM:   ret void
