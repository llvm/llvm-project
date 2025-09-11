# REQUIRES: aarch64
# RUN: rm -rf %t; split-file %s %t

# This test verifies ICF deterministically chooses root functions based on symbol names
# regardless of input object file order, with all three identical functions being folded together
# and _a chosen as the root due to lexicographic ordering.

# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/a.s -o %t/a.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/b.s -o %t/b.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/c.s -o %t/c.o

# RUN: %lld -dylib -arch arm64 -lSystem -o %t/abc --icf=all -map %t/abc.txt %t/a.o %t/b.o %t/c.o
# RUN: %lld -dylib -arch arm64 -lSystem -o %t/bac --icf=all -map %t/bac.txt %t/b.o %t/a.o %t/c.o
# RUN: %lld -dylib -arch arm64 -lSystem -o %t/cba --icf=all -map %t/cba.txt %t/c.o %t/b.o %t/a.o

# RUN: cat %t/abc.txt | FileCheck %s
# RUN: cat %t/bac.txt | FileCheck %s
# RUN: cat %t/cba.txt | FileCheck %s

# CHECK: Symbols:
# CHECK: [[#%X,ADDR:]] 0x00000008  {{.*}} _a
# CHECK-NEXT: [[#ADDR]] 0x00000000 {{.*}} _b
# CHECK-NEXT: [[#ADDR]] 0x00000000 {{.*}} _c

#--- a.s
.section __TEXT,__text,regular,pure_instructions
  .globl _a
_a:
  mov x0, 100
  ret

#--- b.s
.section __TEXT,__text,regular,pure_instructions
  .globl _b
_b:
  mov x0, 100
  ret

#--- c.s
.section __TEXT,__text,regular,pure_instructions
  .globl _c
_c:
  mov x0, 100
  ret
