# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t
# RUN: ld.lld %t --gc-sections --print-gc-sections -o %t2 2>&1 | FileCheck -check-prefix=PRINT %s
# RUN: ld.lld %t --gc-sections --print-gc-sections=- -o %t2 2>&1 | FileCheck -check-prefix=PRINT %s
# RUN: ld.lld %t --gc-sections --print-gc-sections=%t.txt -o %t2
# RUN: FileCheck --check-prefix=PRINT %s --input-file=%t.txt

# PRINT:      removing unused section {{.*}}:(.text.x)
# PRINT-NEXT: removing unused section {{.*}}:(.text.y)

# RUN: rm %t.txt
# RUN: ld.lld %t --gc-sections --print-gc-sections --print-gc-sections=%t.txt --no-print-gc-sections -o %t2 >& %t.log
# RUN: not ls %t.txt
# RUN: echo >> %t.log
# RUN: FileCheck -check-prefix=NOPRINT %s < %t.log

# NOPRINT-NOT: removing

# RUN: not ld.lld %t --gc-sections --print-gc-sections=/ -o %t2 2>&1 | FileCheck --check-prefix=ERR %s

# ERR: error: cannot open --print-gc-sections= file /: {{.*}}

.globl _start
.protected a, x, y
_start:
 call a

.section .text.a,"ax",@progbits
a:
 nop

.section .text.x,"ax",@progbits
x:
 nop

.section .text.y,"ax",@progbits
y:
 nop
