# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
# RUN: ld.lld -z retpolineplt -z now %t.o -o %t
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s

#0x2011a9+5 + 34 = 0x2011d0 (foo@plt)
# CHECK:      <_start>:
# CHECK-NEXT:  2011a9:       callq   0x2011d0

#Static IPLT header due to -z retpolineplt
# CHECK:       00000000002011b0 <.plt>:
# CHECK-NEXT:  2011b0:       callq   0x2011c0 <.plt+0x10>
# CHECK-NEXT:  2011b5:       pause
# CHECK-NEXT:  2011b7:       lfence
#foo@plt
# CHECK:       2011d0:       movq    4105(%rip), %r11
# CHECK-NEXT:  2011d7:       jmp     0x2011b0 <.plt>

.type foo STT_GNU_IFUNC
.globl foo
foo:
  ret

.globl _start
_start:
  call foo
