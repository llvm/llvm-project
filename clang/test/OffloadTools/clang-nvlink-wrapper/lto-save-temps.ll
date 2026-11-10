; REQUIRES: nvptx-registered-target
;
; RUN: llvm-as %s -o %t.bc
; RUN: mkdir -p %t.dir
; RUN: cd %t.dir && clang-nvlink-wrapper --dry-run --save-temps --plugin-opt=lto-partitions=2 -arch sm_90 %t.bc -o multi 2>&1 | FileCheck %s --check-prefix=MULTI
; RUN: cd %t.dir && clang-nvlink-wrapper --dry-run --save-temps -arch sm_90 %t.bc -o single 2>&1 | FileCheck %s --check-prefix=SINGLE
;
; MULTI: ptxas{{.*}} -c multi.s {{.*}} -o multi.cubin
; MULTI: ptxas{{.*}} -c multi.1.s {{.*}} -o multi.1.cubin
; MULTI: nvlink{{.*}} -o multi multi.cubin multi.1.cubin
; SINGLE: ptxas{{.*}} -c single.s {{.*}} -o single.cubin
; SINGLE: nvlink{{.*}} -o single single.cubin

target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define i32 @foo(i32 %x) {
  %r = add i32 %x, 1
  ret i32 %r
}

define i32 @bar(i32 %x) {
  %r = mul i32 %x, 3
  ret i32 %r
}
