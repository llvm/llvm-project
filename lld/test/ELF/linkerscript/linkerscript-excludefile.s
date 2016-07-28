# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t1
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux \
# RUN:   %p/Inputs/include.s -o %t2
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux \
# RUN:   %p/Inputs/notinclude.s -o %t3.notinclude

# RUN: echo "SECTIONS {} " > %t.script
# RUN: ld.lld -o %t --script %t.script %t1 %t2 %t3.notinclude
# RUN: llvm-objdump -d %t | \
# RUN:   FileCheck %s

# CHECK: Disassembly of section .text:
# CHECK: _start:
# CHECK:      120:       48 c7 c0 3c 00 00 00    movq    $60, %rax
# CHECK:      127:       48 c7 c7 2a 00 00 00    movq    $42, %rdi
# CHECK:      12e:       00 00   addb    %al, (%rax)
# CHECK: _potato:
# CHECK:      130:       90      nop
# CHECK:      131:       90      nop
# CHECK:      132:       00 00   addb    %al, (%rax)
# CHECK: tomato:
# CHECK:      134:       b8 01 00 00 00  movl    $1, %eax

# RUN: echo "SECTIONS { .patatino : \
# RUN: { KEEP(*(EXCLUDE_FILE(*notinclude) .text)) } }" \
# RUN:  > %t.script
# RUN: ld.lld -o %t2 --script %t.script %t1 %t2 %t3.notinclude
# RUN: llvm-objdump -d %t2 | \
# RUN:   FileCheck %s --check-prefix=EXCLUDE

# EXCLUDE: Disassembly of section .patatino:
# EXCLUDE: _start:
# EXCLUDE:      120:       48 c7 c0 3c 00 00 00    movq    $60, %rax
# EXCLUDE:      127:       48 c7 c7 2a 00 00 00    movq    $42, %rdi
# EXCLUDE:      12e:       00 00   addb    %al, (%rax)
# EXCLUDE: _potato:
# EXCLUDE:      130:       90      nop
# EXCLUDE:      131:       90      nop
# EXCLUDE: Disassembly of section .text:
# EXCLUDE: tomato:
# EXCLUDE:      134:       b8 01 00 00 00  movl    $1, %eax

.section .text
.globl _start
_start:
    mov $60, %rax
    mov $42, %rdi
