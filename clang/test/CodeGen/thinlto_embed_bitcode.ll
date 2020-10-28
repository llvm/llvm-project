; REQUIRES: x86-registered-target

; check the -lto-embed-bitcode=post-thinlto-merge does not perform optimizations
; we expect 't1' - i.e start-lib1.ll's products - have both foo and bar defined,
; but the bar call is still made from foo.
; RUN: opt -module-summary %p/Inputs/start-lib1.ll -o %t1.bc
; RUN: opt -module-summary %p/Inputs/start-lib2.ll -o %t2.bc
; RUN: llvm-lto -thinlto -o %t.o %t1.bc %t2.bc

; For the optimized case, we expect the inlining of foo into bar to happen.
; RUN: %clang -target x86_64-unknown-linux-gnu -O2 -o %t-opt.o -x ir %t1.bc -c -fthinlto-index=%t.o.thinlto.bc -mllvm -lto-embed-bitcode=optimized
; RUN: llvm-readelf -S %t-opt.o | FileCheck %s --check-prefixes=CHECK-ELF,CHECK-NO-CMD
; RUN: llvm-objcopy --dump-section=.llvmbc=%t-embedded.bc %t-opt.o /dev/null
; RUN: llvm-dis %t-embedded.bc -o - | FileCheck %s --check-prefixes=CHECK,CHECK-OPT

; For the post-merge case, perform the embedded bitcode extraction, then
; round-trip through compilation and ensure the objects are the same.
; RUN: %clang -target x86_64-unknown-linux-gnu -O2 -o %t.o -x ir %t1.bc -c -fthinlto-index=%t.o.thinlto.bc -mllvm -lto-embed-bitcode=post-merge-pre-opt
; RUN: llvm-readelf -S %t.o | FileCheck %s --check-prefixes=CHECK-ELF,CHECK-CMD
; RUN: llvm-objcopy --dump-section=.llvmbc=%t-embedded.bc %t.o /dev/null
; RUN: llvm-dis %t-embedded.bc -o - | FileCheck %s --check-prefixes=CHECK,CHECK-NOOPT
; We should only need the index and the post-thinlto merged module to generate 
; the exact same .o as we originally did.
; RUN: rm %t1.bc %t2.bc
; RUN: %clang -target x86_64-unknown-linux-gnu -O2 -o %t-redo.o -x ir %t-embedded.bc -c -fthinlto-index=%t.o.thinlto.bc -mllvm -lto-embed-bitcode=post-merge-pre-opt -mllvm -thinlto-assume-merged
; RUN: diff %t-redo.o %t.o

; CHECK-ELF:      .text   PROGBITS 0000000000000000 [[#%x,OFF:]] [[#%x,SIZE:]] 00 AX 0
; CHECK-ELF-NEXT: .llvmbc PROGBITS 0000000000000000 [[#%x,OFF:]] [[#%x,SIZE:]] 00    0
; CHECK-ELF-CMD:  .llvmcmd
; CHECK-ELF-NO-CMD-NOT: .llvmcmd

; CHECK:          define void @foo() 
; CHECK-OPT-NEXT:   ret void
; CHECK-NOOPT-NEXT: call void @bar()
; CHECK-NOOPT: define available_externally void @bar()
; CHECK-NOOPT-NEXT: ret void
