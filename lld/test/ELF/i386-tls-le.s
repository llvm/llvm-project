# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=i686 %s -o %t.o
# RUN: ld.lld %t.o -o %t
# RUN: ld.lld %t.o -shared -o %t.so
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s --check-prefix=DIS
# RUN: llvm-readobj -r %t | FileCheck %s --check-prefix=RELOC
# RUN: llvm-objdump -d --no-show-raw-insn %t.so | FileCheck %s --check-prefix=DISSHARED
# RUN: llvm-readobj -r %t.so | FileCheck %s --check-prefix=RELOCSHARED

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
# DIS-NEXT: 402134:       movl    $8, %edx
# DIS-NEXT: 402139:       movl    %gs:0, %ecx
# DIS-NEXT: 402140:       subl    %edx, %eax
# DIS-NEXT: 402142:       movl    $4, %edx
# DIS-NEXT: 402147:       movl    %gs:0, %ecx
# DIS-NEXT: 40214e:       subl    %edx, %eax
# DIS-NEXT: 402150:       movl    %gs:0, %ecx
# DIS-NEXT: 402157:       leal    -8(%ecx), %eax
# DIS-NEXT: 40215d:       movl    %gs:0, %ecx
# DIS-NEXT: 402164:       leal    119(%ecx), %eax

# RELOC: Relocations [
# RELOC-NEXT: ]

# DISSHARED: Disassembly of section test:
# DISSHARED-EMPTY:
# DISSHARED-NEXT: <_start>:
# DISSHARED-NEXT: 2218:       movl    $0, %edx
# DISSHARED-NEXT: 221d:       movl    %gs:0, %ecx
# DISSHARED-NEXT: 2224:       subl    %edx, %eax
# DISSHARED-NEXT: 2226:       movl    $0, %edx
# DISSHARED-NEXT: 222b:       movl    %gs:0, %ecx
# DISSHARED-NEXT: 2232:       subl    %edx, %eax
# DISSHARED-NEXT: 2234:       movl    %gs:0, %ecx
# DISSHARED-NEXT: 223b:       leal    (%ecx), %eax
# DISSHARED-NEXT: 2241:       movl    %gs:0, %ecx
# DISSHARED-NEXT: 2248:       leal    123(%ecx), %eax

# RELOCSHARED:      Relocations [
# RELOCSHARED-NEXT: Section (5) .rel.dyn {
# RELOCSHARED-NEXT:   0x2219 R_386_TLS_TPOFF32 var
# RELOCSHARED-NEXT:   0x223D R_386_TLS_TPOFF var
# RELOCSHARED-NEXT:   0x2227 R_386_TLS_TPOFF32 var1
# RELOCSHARED-NEXT:   0x224A R_386_TLS_TPOFF var1
# RELOCSHARED-NEXT:  }
# RELOCSHARED-NEXT: ]
