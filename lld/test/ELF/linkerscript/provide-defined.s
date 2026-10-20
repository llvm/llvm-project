# REQUIRES: x86
## Test the GC behavior when the PROVIDE symbol is defined by a relocatable file.

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 a.s -o a.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 b.s -o b.o
# RUN: llvm-as c.ll -o c.bc
# RUN: ld.lld -T a.t --gc-sections a.o b.o -o a
# RUN: llvm-readelf -s a | FileCheck %s

# RUN: ld.lld -T a.t -shared a.o b.o c.bc -o a.so
# RUN: llvm-readelf -s -r a.so | FileCheck %s --check-prefix=DSO

# CHECK:     1: {{.*}}               0 NOTYPE  GLOBAL DEFAULT     1 _start
# CHECK-NEXT:2: {{.*}}               0 NOTYPE  WEAK   DEFAULT     2 f3
# CHECK-NOT: {{.}}

# DSO:     .rela.plt
# DSO-NOT: f5
# DSO:     Symbol table '.dynsym'
# DSO-NOT: f5
# DSO:     Symbol table '.symtab'
# DSO:     {{.*}}               0 NOTYPE  LOCAL  HIDDEN  [[#]] f5

#--- a.s
.global _start, f1, f2, bar
.weak f3
_start:
  call f3

.section .text.f1,"ax"; f1:
.section .text.f2,"ax"; f2: # referenced by another relocatable file
.section .text.f3,"ax"; f3: # live
.section .text.bar,"ax"; bar:

.comm comm,4,4

#--- b.s
  call f2

#--- c.ll
target triple = "x86_64-unknown-linux-gnu"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

declare void @f5()

define void @f3() {
  call void @f5()
  ret void
}

#--- a.t
SECTIONS {
  . = . + SIZEOF_HEADERS;
  PROVIDE(f1 = bar+1);
  PROVIDE(f2 = bar+2);
  PROVIDE(f3 = bar+3);
  PROVIDE(f4 = comm+4);
  PROVIDE_HIDDEN(f5 = bar+5);
}
