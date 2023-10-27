# REQUIRES: avr

# RUN: rm -rf %t && split-file %s %t && cd %t

# RUN: llvm-mc -filetype=obj -triple=avr -mcpu=atmega328 avr-pcrel-7.s -o avr-pcrel-7.o
# RUN: not ld.lld avr-pcrel-7.o -o /dev/null -Ttext=0x1000 --defsym=callee0=0x1040 --defsym=callee1=0x1044 --defsym=callee2=0x100f 2>&1 | \
# RUN:     FileCheck %s --check-prefix=PCREL7
# RUN: llvm-mc -filetype=obj -triple=avr -mcpu=atmega328 avr-pcrel-13.s -o avr-pcrel-13.o
# RUN: not ld.lld avr-pcrel-13.o -o /dev/null -Ttext=0x1000 --defsym=callee0=0x2000 --defsym=callee1=0x2004 --defsym=callee2=0x100f 2>&1 | \
# RUN:     FileCheck %s --check-prefix=PCREL13
# RUN: llvm-mc -filetype=obj -triple=avr -mcpu=atmega328 avr-abs.s -o avr-abs.o
# RUN: not ld.lld avr-abs.o -o /dev/null -Ttext=0x1000 --defsym=callee0=0x1009 --defsym=callee1=0x1010 2>&1 | \
# RUN:     FileCheck %s --check-prefix=ABS

#--- avr-pcrel-7.s

.section .LDI,"ax",@progbits

.globl __start
__start:

# PCREL7-NOT: callee0
# PCREL7:     error: {{.*}} relocation R_AVR_7_PCREL out of range: {{.*}} is not in [-64, 63]; references 'callee1'
# PCREL7:     error: {{.*}} improper alignment for relocation R_AVR_7_PCREL: {{.*}} is not aligned to 2 bytes
brne callee0
breq callee1
brlt callee2

#--- avr-pcrel-13.s

.section .LDI,"ax",@progbits

.globl __start
__start:

# PCREL13-NOT: callee0
# PCREL13:     error: {{.*}} relocation R_AVR_13_PCREL out of range: {{.*}} is not in [-4096, 4095]; references 'callee1'
# PCREL13:     error: {{.*}} improper alignment for relocation R_AVR_13_PCREL: {{.*}} is not aligned to 2 bytes
rjmp  callee0
rcall callee1
rjmp  callee2

#--- avr-abs.s

.section .LDI,"ax",@progbits

.globl __start
__start:

# ABS:     error: {{.*}} improper alignment for relocation R_AVR_CALL: 0x1009 is not aligned to 2 bytes
# ABS-NOT: 0x1010
call callee0
jmp  callee1
