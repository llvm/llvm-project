# REQUIRES: x86
## SHF_GNU_RETAIN is a generic feature defined in the OS specific range. The
## flag marks a section as a GC root.

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: ld.lld --gc-sections --print-gc-sections %t.o -o %t | count 0
# RUN: llvm-readobj -hS %t | FileCheck %s
# RUN: ld.lld -r -e _start --gc-sections --print-gc-sections %t.o -o %t.ro | count 0
# RUN: llvm-readobj -hS %t.ro | FileCheck %s

## SHF_GNU_RETAIN has no significance in executables/shared objects. Multiple
## OSABI values can benefit from this flag. Test that we don't change EI_OSABI,
## even for relocatable output.
# CHECK:       OS/ABI: SystemV (0x0)

# CHECK:      Name: .retain
# CHECK-NEXT: Type: SHT_PROGBITS
# CHECK-NEXT: Flags [
# CHECK-NEXT:   SHF_ALLOC
# CHECK-NEXT:   SHF_GNU_RETAIN
# CHECK-NEXT: ]

# RUN: llvm-mc -filetype=obj -triple=x86_64 --defsym NONALLOC=1 %s -o %t1.o
# RUN: not ld.lld --gc-sections %t1.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR

# ERR: error: {{.*}}.o:(.nonalloc): sh_link points to discarded section {{.*}}.o:(.discard)

.global _start
_start:

.section .retain,"aR",@progbits
.quad .foo

.section .foo,"a",@progbits
.quad 0

.ifdef NONALLOC
.section .discard,"a",@progbits

## With SHF_GNU_RETAIN, .nonalloc is retained while its linked-to section
## .discard is discarded, so there will be an error.
.section .nonalloc,"oR",@progbits,.discard
.quad .text
.endif
