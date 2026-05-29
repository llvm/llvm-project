; REQUIRES: x86-registered-target
; REQUIRES: reproducer-reduction
;
; RUN: rm -rf %t && mkdir -p %t
; RUN: split-file %s %t
; RUN: cd %t
;
; RUN: env LLVM_DISABLE_SYMBOLIZATION=0 %reduce-clang-crash --clang %clang --opt opt --llvm-reduce llvm-reduce --opt-arg=-enable-matrix --opt-arg=-verify-matrix-shapes --auto crash_middleend.sh reduce-clang-crash.ll | FileCheck %s
; RUN: FileCheck --check-prefix=REDUCED %s < %t/reduced.ll
;
; CHECK: Found Middle/Backend failure
; CHECK-NEXT: Checking opt for failure
; CHECK-NEXT: Found MiddleEnd Crash
; CHECK-NEXT: Writing interestingness test...
; CHECK-EMPTY:
; CHECK-NEXT: Creating the interestingness test...
; CHECK-NEXT: Starting llvm-reduce with opt test case
; CHECK-EMPTY:
; CHECK-NEXT: Running llvm-reduce tool...
; CHECK-NEXT: Done Reducing IR file.
;
; REDUCED: phi <16 x float>

//--- reduce-clang-crash.ll
define <16 x float> @foo(i1 %c, ptr %p1, ptr %p2) {
entry:
  br i1 %c, label %bb1, label %bb2

bb1:
  %a = call <16 x float> @llvm.matrix.column.major.load.v16f32.p0(ptr %p1, i64 4, i1 false, i32 4, i32 4)
  br label %exit

bb2:
  %b = call <16 x float> @llvm.matrix.column.major.load.v16f32.p0(ptr %p2, i64 2, i1 false, i32 2, i32 8)
  br label %exit

exit:
  %y = phi <16 x float> [ %a, %bb1 ], [ %b, %bb2 ]
  ret <16 x float> %y
}

declare <16 x float> @llvm.matrix.column.major.load.v16f32.p0(ptr, i64, i1, i32, i32)

//--- crash_middleend.sh
clang -cc1 -triple x86_64-unknown-linux-gnu -O2 -emit-obj -mllvm -enable-matrix -mllvm -verify-matrix-shapes -x ir reduce-clang-crash.ll
