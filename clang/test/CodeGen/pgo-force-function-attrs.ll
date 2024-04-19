; RUN: %clang_cc1 -O2 -mllvm -pgo-cold-func-opt=optsize -mllvm -enable-pgo-force-function-attrs -fprofile-sample-use=%S/Inputs/pgo-sample.prof %s -emit-llvm -o - | FileCheck %s --check-prefix=OPTSIZE
; Check that no profile means no optsize
; RUN: %clang_cc1 -O2 -mllvm -pgo-cold-func-opt=optsize -mllvm -enable-pgo-force-function-attrs %s -emit-llvm -o - | FileCheck %s --check-prefix=NONE
; Check that no -pgo-cold-func-opt=optsize means no optsize
; RUN: %clang_cc1 -O2 -mllvm -enable-pgo-force-function-attrs -fprofile-sample-use=%S/Inputs/pgo-sample.prof %s -emit-llvm -o - | FileCheck %s --check-prefix=NONE

; NONE-NOT: optsize
; OPTSIZE: optsize

define void @f() cold {
  ret void
}
