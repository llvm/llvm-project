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
# DIS-NEXT: _start:
# DIS-NEXT: 401000:       movl    $8, %edx
# DIS-NEXT: 401005:       movl    %gs:0, %ecx
# DIS-NEXT: 40100c:       subl    %edx, %eax
# DIS-NEXT: 40100e:       movl    $4, %edx
# DIS-NEXT: 401013:       movl    %gs:0, %ecx
# DIS-NEXT: 40101a:       subl    %edx, %eax
# DIS-NEXT: 40101c:       movl    %gs:0, %ecx
# DIS-NEXT: 401023:       leal    -8(%ecx), %eax
# DIS-NEXT: 401029:       movl    %gs:0, %ecx
# DIS-NEXT: 401030:       leal    119(%ecx), %eax

# RELOC: Relocations [
# RELOC-NEXT: ]

# DISSHARED: Disassembly of section test:
# DISSHARED-EMPTY:
# DISSHARED-NEXT: _start:
# DISSHARED-NEXT: 1000:       movl    $0, %edx
# DISSHARED-NEXT: 1005:       movl    %gs:0, %ecx
# DISSHARED-NEXT: 100c:       subl    %edx, %eax
# DISSHARED-NEXT: 100e:       movl    $0, %edx
# DISSHARED-NEXT: 1013:       movl    %gs:0, %ecx
# DISSHARED-NEXT: 101a:       subl    %edx, %eax
# DISSHARED-NEXT: 101c:       movl    %gs:0, %ecx
# DISSHARED-NEXT: 1023:       leal    (%ecx), %eax
# DISSHARED-NEXT: 1029:       movl    %gs:0, %ecx
# DISSHARED-NEXT: 1030:       leal    123(%ecx), %eax

# RELOCSHARED:      Relocations [
# RELOCSHARED-NEXT: Section (5) .rel.dyn {
# RELOCSHARED-NEXT:   0x1001 R_386_TLS_TPOFF32 var 0x0
# RELOCSHARED-NEXT:   0x1025 R_386_TLS_TPOFF var 0x0
# RELOCSHARED-NEXT:   0x100F R_386_TLS_TPOFF32 var1 0x0
# RELOCSHARED-NEXT:   0x1032 R_386_TLS_TPOFF var1 0x0
# RELOCSHARED-NEXT:  }
# RELOCSHARED-NEXT: ]
