# REQUIRES: x86
# RUN: rm -rf %t && split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/first.s -o %t/first-ro.o
# RUN: llvm-objcopy --set-section-flags .eh_frame=alloc,data \
# RUN:   %t/first-ro.o %t/first.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/second.s -o %t/second-ro.o
# RUN: llvm-objcopy --set-section-flags .eh_frame=alloc,data \
# RUN:   %t/second-ro.o %t/second.o
# RUN: ld.lld -shared -z notext --apply-dynamic-relocs \
# RUN:   --check-dynamic-relocations \
# RUN:   %t/first.o %t/second.o -o %t/out.so
# RUN: llvm-readelf -S -r %t/out.so | FileCheck %s

# CHECK: .eh_frame PROGBITS [[#%x,EH:]]
# CHECK: {{0*}}[[#%x,EH+0x12]] {{.*}} R_X86_64_64 {{.*}} external1 + 10
# CHECK: {{0*}}[[#%x,EH+0x4a]] {{.*}} R_X86_64_64 {{.*}} external2 + 20

#--- first.s
.set personality1, external1+0x10
.text
.globl first
first:
  .cfi_startproc
  .cfi_personality 0, personality1
  ret
  .cfi_endproc

#--- second.s
.set personality2, external2+0x20
.text
.globl second
second:
  .cfi_startproc
  .cfi_personality 0, personality2
  ret
  .cfi_endproc
