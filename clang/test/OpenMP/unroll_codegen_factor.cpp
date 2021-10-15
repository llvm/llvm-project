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
// IR-NEXT:    %[[START_ADDR:.+]] = alloca i32, align 4
// IR-NEXT:    %[[END_ADDR:.+]] = alloca i32, align 4
// IR-NEXT:    %[[STEP_ADDR:.+]] = alloca i32, align 4
// IR-NEXT:    %[[I:.+]] = alloca i32, align 4
// IR-NEXT:    store i32 %[[START:.+]], i32* %[[START_ADDR]], align 4
// IR-NEXT:    store i32 %[[END:.+]], i32* %[[END_ADDR]], align 4
// IR-NEXT:    store i32 %[[STEP:.+]], i32* %[[STEP_ADDR]], align 4
// IR-NEXT:    %[[TMP0:.+]] = load i32, i32* %[[START_ADDR]], align 4
// IR-NEXT:    store i32 %[[TMP0]], i32* %[[I]], align 4
// IR-NEXT:    br label %[[FOR_COND:.+]]
// IR-EMPTY:
// IR-NEXT:  [[FOR_COND]]:
// IR-NEXT:    %[[TMP1:.+]] = load i32, i32* %[[I]], align 4
// IR-NEXT:    %[[TMP2:.+]] = load i32, i32* %[[END_ADDR]], align 4
// IR-NEXT:    %[[CMP:.+]] = icmp slt i32 %[[TMP1]], %[[TMP2]]
// IR-NEXT:    br i1 %[[CMP]], label %[[FOR_BODY:.+]], label %[[FOR_END:.+]]
// IR-EMPTY:
// IR-NEXT:  [[FOR_BODY]]:
// IR-NEXT:    %[[TMP3:.+]] = load i32, i32* %[[START_ADDR]], align 4
// IR-NEXT:    %[[TMP4:.+]] = load i32, i32* %[[END_ADDR]], align 4
// IR-NEXT:    %[[TMP5:.+]] = load i32, i32* %[[STEP_ADDR]], align 4
// IR-NEXT:    %[[TMP6:.+]] = load i32, i32* %[[I]], align 4
// IR-NEXT:    call void (...) @body(i32 noundef %[[TMP3]], i32 noundef %[[TMP4]], i32 noundef %[[TMP5]], i32 noundef %[[TMP6]])
// IR-NEXT:    br label %[[FOR_INC:.+]]
// IR-EMPTY:
// IR-NEXT:  [[FOR_INC]]:
// IR-NEXT:    %[[TMP7:.+]] = load i32, i32* %[[STEP_ADDR]], align 4
// IR-NEXT:    %[[TMP8:.+]] = load i32, i32* %[[I]], align 4
// IR-NEXT:    %[[ADD:.+]] = add nsw i32 %[[TMP8]], %[[TMP7]]
// IR-NEXT:    store i32 %[[ADD]], i32* %[[I]], align 4
// IR-NEXT:    br label %[[FOR_COND]], !llvm.loop ![[LOOP2:[0-9]+]]
// IR-EMPTY:
// IR-NEXT:  [[FOR_END]]:
// IR-NEXT:    ret void
// IR-NEXT:  }
extern "C" void func(int start, int end, int step) {
  #pragma omp unroll partial(4)
  for (int i = start; i < end; i+=step)
    body(start, end, step, i);
}

#endif /* HEADER */


// IR: ![[LOOP2]] = distinct !{![[LOOP2]], ![[LOOPPROP3:[0-9]+]], ![[LOOPPROP4:[0-9]+]], ![[LOOPPROP5:[0-9]+]]}
// IR: ![[LOOPPROP3]] = !{!"llvm.loop.mustprogress"}
// IR: ![[LOOPPROP4]] = !{!"llvm.loop.unroll.count", i32 4}
// IR: ![[LOOPPROP5]] = !{!"llvm.loop.unroll.enable"}
