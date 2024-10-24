# REQUIRES: x86
## Test the GC behavior when the PROVIDE symbol is defined by a relocatable file.

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 a.s -o a.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 b.s -o b.o
# RUN: ld.lld -T a.t --gc-sections a.o b.o -o a
# RUN: llvm-readelf -s a | FileCheck %s

# CHECK:     1: {{.*}}               0 NOTYPE  GLOBAL DEFAULT     1 _start
# CHECK-NEXT:2: {{.*}}               0 NOTYPE  GLOBAL DEFAULT     2 f3
# CHECK-NOT: {{.}}

#--- a.s
.global _start, f1, f2, f3, bar
_start:
  call f3

.section .text.f1,"ax"; f1:
.section .text.f2,"ax"; f2: # referenced by another relocatable file
.section .text.f3,"ax"; f3: # live
.section .text.bar,"ax"; bar:

.comm comm,4,4

#--- b.s
  call f2

#--- a.t
SECTIONS {
  . = . + SIZEOF_HEADERS;
  PROVIDE(f1 = bar+1);
  PROVIDE(f2 = bar+2);
  PROVIDE(f3 = bar+3);
  PROVIDE(f4 = comm+4);
}
