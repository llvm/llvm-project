# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t
# RUN: ld.lld -shared --gc-sections -o %t1 %t
# RUN: llvm-readelf --file-headers --symbols %t1 \
# RUN:   | FileCheck %s
# CHECK: Entry point address:               0x1238
# CHECK: 0000000000001238     0 FUNC    LOCAL  HIDDEN     5 _start
# CHECK: 000000000000123e     0 FUNC    LOCAL  HIDDEN     5 internal
# CHECK: 000000000000123d     0 FUNC    GLOBAL DEFAULT    5 foobar

.section .text.start,"ax"
.globl _start
.type _start,%function
.hidden _start
_start:
  jmp internal

.section .text.foobar,"ax"
.globl foobar
.type foobar,%function
foobar:
  ret

.section .text.internal,"ax"
.globl internal
.hidden internal
.type internal,%function
internal:
	ret
