// Check code generation
// RUN: %clang_cc1 -verify -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=51 -emit-llvm %s -o - | FileCheck %s --check-prefix=IR

// Check same results after serialization round-trip
// RUN: %clang_cc1 -verify -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=51 -emit-pch -o %t %s
// RUN: %clang_cc1 -verify -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=51 -include-pch %t -emit-llvm %s -o - | FileCheck %s --check-prefix=IR
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

// placeholder for loop body code.
extern "C" void body(...) {}


// IR-LABEL: @func(
// IR-NEXT:  [[ENTRY:.*]]:
// IR-NEXT:    %[[I:.+]] = alloca i32, align 4
// IR-NEXT:    store i32 7, ptr %[[I]], align 4
// IR-NEXT:    br label %[[FOR_COND:.+]]
// IR-EMPTY:
// IR-NEXT:  [[FOR_COND]]:
// IR-NEXT:    %[[TMP0:.+]] = load i32, ptr %[[I]], align 4
// IR-NEXT:    %[[CMP:.+]] = icmp slt i32 %[[TMP0]], 17
// IR-NEXT:    br i1 %[[CMP]], label %[[FOR_BODY:.+]], label %[[FOR_END:.+]]
// IR-EMPTY:
// IR-NEXT:  [[FOR_BODY]]:
// IR-NEXT:    %[[TMP1:.+]] = load i32, ptr %[[I]], align 4
// IR-NEXT:    call void (...) @body(i32 noundef %[[TMP1]])
// IR-NEXT:    br label %[[FOR_INC:.+]]
// IR-EMPTY:
// IR-NEXT:  [[FOR_INC]]:
// IR-NEXT:    %[[TMP2:.+]] = load i32, ptr %[[I]], align 4
// IR-NEXT:    %[[ADD:.+]] = add nsw i32 %[[TMP2]], 3
// IR-NEXT:    store i32 %[[ADD]], ptr %[[I]], align 4
// IR-NEXT:    br label %[[FOR_COND]], !llvm.loop ![[LOOP2:[0-9]+]]
// IR-EMPTY:
// IR-NEXT:  [[FOR_END]]:
// IR-NEXT:    ret void
// IR-NEXT:  }
extern "C" void func() {
  #pragma omp unroll full
  for (int i = 7; i < 17; i += 3)
    body(i);
}

#endif /* HEADER */


// IR: ![[LOOP2]] = distinct !{![[LOOP2]], ![[LOOPPROP3:[0-9]+]], ![[LOOPPROP4:[0-9]+]]}
// IR: ![[LOOPPROP3]] = !{!"llvm.loop.mustprogress"}
// IR: ![[LOOPPROP4]] = !{!"llvm.loop.unroll.full"}
