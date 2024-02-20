# REQUIRES: x86
# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/a.s -o %t/a.o
# RUN: not ld.lld %t/a.o -T %t/lds -o /dev/null 2>&1 | FileCheck %s

# CHECK: error: {{.*}}a.o:(.text+{{.*}}): relocation R_X86_64_PC32 out of range: {{.*}}; R_X86_64_PC32 should not reference a section marked SHF_X86_64_LARGE

#--- a.s
.text
.globl _start
.type _start, @function
_start:
  movq hello(%rip), %rax

.section ldata,"awl",@progbits
.type   hello, @object
.globl  hello
hello:
.long   1

#--- lds
SECTIONS {
  .text 0x100000 : { *(.text) }
  ldata 0x80200000 : { *(ldata) }
}
