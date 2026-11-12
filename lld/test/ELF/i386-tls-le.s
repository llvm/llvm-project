# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=i686 %s -o %t.o
# RUN: ld.lld %t.o -o %t
# RUN: ld.lld %t.o -pie -o %t.pie
# RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t | FileCheck %s --check-prefix=DIS
# RUN: llvm-readobj -r %t | FileCheck %s --check-prefix=RELOC
# RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t.pie | FileCheck %s --check-prefix=DIS
# RUN: llvm-readobj -r %t.pie | FileCheck %s --check-prefix=RELOC

## Reject local-exec TLS relocations for -shared.
# RUN: not ld.lld -shared %t.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR --implicit-check-not=error:

# ERR: error: relocation R_386_TLS_LE_32 against var cannot be used with -shared
# ERR: error: relocation R_386_TLS_LE_32 against var1 cannot be used with -shared
# ERR: error: relocation R_386_TLS_LE against var cannot be used with -shared
# ERR: error: relocation R_386_TLS_LE against var1 cannot be used with -shared

.section ".tdata", "awT", @progbits
.globl var
.globl var1
var:
.long 0
var1:
.long 1

.section test, "awx"
.global _start
_start:
 movl $var@tpoff, %edx
 movl %gs:0, %ecx
 subl %edx, %eax
 movl $var1@tpoff, %edx
 movl %gs:0, %ecx
 subl %edx, %eax

 movl %gs:0, %ecx
 leal var@ntpoff(%ecx), %eax
 movl %gs:0, %ecx
 leal var1@ntpoff+123(%ecx), %eax

# DIS:      Disassembly of section test:
# DIS-EMPTY:
# DIS-NEXT: <_start>:
# DIS-NEXT:   movl    $8, %edx
# DIS-NEXT:   movl    %gs:0, %ecx
# DIS-NEXT:   subl    %edx, %eax
# DIS-NEXT:   movl    $4, %edx
# DIS-NEXT:   movl    %gs:0, %ecx
# DIS-NEXT:   subl    %edx, %eax
# DIS-NEXT:   movl    %gs:0, %ecx
# DIS-NEXT:   leal    -8(%ecx), %eax
# DIS-NEXT:   movl    %gs:0, %ecx
# DIS-NEXT:   leal    119(%ecx), %eax

# RELOC: Relocations [
# RELOC-NEXT: ]
