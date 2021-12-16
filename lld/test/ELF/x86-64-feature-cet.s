# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %p/Inputs/x86-64-cet1.s -o %t1.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %p/Inputs/x86-64-cet2.s -o %t2.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %p/Inputs/x86-64-cet3.s -o %t3.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %p/Inputs/x86-64-cet4.s -o %t4.o

# RUN: ld.lld -e func1 %t.o %t1.o -o %t
# RUN: llvm-readelf -n %t | FileCheck --check-prefix=CET --match-full-lines %s

# RUN: ld.lld -e func1 %t.o %t2.o -o %t
# RUN: llvm-readelf -n %t | FileCheck --check-prefix=CET --match-full-lines %s

# CET: Properties: x86 feature: IBT, SHSTK

# RUN: ld.lld -e func1 %t.o %t3.o -o %t
# RUN: llvm-readelf -S %t | FileCheck --check-prefix=NOCET %s

# NOCET:     Section Headers
# NOCET-NOT: .note.gnu.property

# RUN: ld.lld -e func1 %t.o %t3.o -o %t -z force-ibt 2>&1 \
# RUN:   | FileCheck --check-prefix=WARN %s
# WARN: {{.*}}.o: -z force-ibt: file does not have GNU_PROPERTY_X86_FEATURE_1_IBT property

# RUN:not ld.lld -e func1 %t.o %t3.o -o /dev/null -z cet-report=something 2>&1 \
# RUN:   | FileCheck --check-prefix=REPORT_INVALID %s
# REPORT_INVALID: error: -z cet-report= parameter something is not recognized
# REPORT_INVALID-EMPTY:

# RUN: ld.lld -e func1 %t.o %t3.o -o /dev/null  -z force-ibt -z cet-report=warning 2>&1 \
# RUN:   | FileCheck --check-prefix=REPORT_FORCE %s
# REPORT_FORCE: {{.*}}.o: -z cet-report: file does not have GNU_PROPERTY_X86_FEATURE_1_IBT property
# REPORT_FORCE: {{.*}}.o: -z cet-report: file does not have GNU_PROPERTY_X86_FEATURE_1_SHSTK property
# REPORT_FORCE-EMPTY:

# RUN: ld.lld -e func1 %t.o %t3.o -o /dev/null -z cet-report=warning 2>&1 \
# RUN:   | FileCheck --check-prefix=CET_REPORT_WARN %s
# CET_REPORT_WARN: {{.*}}.o: -z cet-report: file does not have GNU_PROPERTY_X86_FEATURE_1_IBT property
# CET_REPORT_WARN: {{.*}}.o: -z cet-report: file does not have GNU_PROPERTY_X86_FEATURE_1_SHSTK property
# CET_REPORT_WARN-EMPTY:

# RUN: not ld.lld -e func1 %t.o %t3.o -o /dev/null -z cet-report=error 2>&1 \
# RUN:   | FileCheck --check-prefix=CET_REPORT_ERROR %s
# CET_REPORT_ERROR: {{.*}}.o: -z cet-report: file does not have GNU_PROPERTY_X86_FEATURE_1_IBT property
# CET_REPORT_ERROR: {{.*}}.o: -z cet-report: file does not have GNU_PROPERTY_X86_FEATURE_1_SHSTK property
# CET_REPORT_ERROR-EMPTY:

# RUN: ld.lld -e func1 %t.o %t4.o -o %t
# RUN: llvm-readelf -n %t | FileCheck --check-prefix=NOSHSTK %s

# Check .note.gnu.protery without property SHSTK.
# NOSHSTK: Properties: x86 feature: IBT{{$}}

# RUN: ld.lld -shared %t1.o -soname=so -o %t1.so
# RUN: ld.lld -e func1 %t.o %t1.so -o %t
# RUN: llvm-readelf -n %t | FileCheck --check-prefix=CET --match-full-lines %s
# RUN: llvm-readelf -x .got.plt %t | FileCheck --check-prefix=GOTPLT %s
# RUN: llvm-objdump -d --no-show-raw-insn --print-imm-hex %t | FileCheck --check-prefix=DISASM %s

# GOTPLT:      Hex dump of section '.got.plt':
# GOTPLT-NEXT: 203480 80232000 00000000 00000000 00000000
# GOTPLT-NEXT: 203490 00000000 00000000 50132000 00000000
# GOTPLT-NEXT: 2034a0 00000000 00000000

# DISASM:      Disassembly of section .text:
# DISASM:      0000000000201330 <func1>:
# DISASM-NEXT: 201330:       callq   0x201360 <func2+0x201360>
# DISASM-NEXT: 201335:       callq   0x201370 <func2+0x201370>
# DISASM-NEXT:               retq

# DISASM:      Disassembly of section .plt:
# DISASM:      0000000000201340 <.plt>:
# DISASM-NEXT: 201340:       pushq   0x2142(%rip)
# DISASM-NEXT:               jmpq    *0x2144(%rip)
# DISASM-NEXT:               nopl    (%rax)
# DISASM-NEXT:               endbr64
# DISASM-NEXT:               pushq   $0x0
# DISASM-NEXT:               jmp     0x201340 <.plt>
# DISASM-NEXT:               nop

# DISASM:      Disassembly of section .plt.sec:
# DISASM:      0000000000201360 <.plt.sec>:
# DISASM-NEXT: 201360:       endbr64
# DISASM-NEXT:               jmpq    *0x212e(%rip)
# DISASM-NEXT:               nopw    (%rax,%rax)

# DISASM:      Disassembly of section .iplt:
# DISASM:      0000000000201370 <.iplt>:
# DISASM-NEXT: 201370:       endbr64
# DISASM-NEXT:               jmpq    *0x2126(%rip)
# DISASM-NEXT:               nopw    (%rax,%rax)

.section ".note.gnu.property", "a"
.long 4
.long 0x10
.long 0x5
.asciz "GNU"

.long 0xc0000002 # GNU_PROPERTY_X86_FEATURE_1_AND
.long 4
.long 3          # GNU_PROPERTY_X86_FEATURE_1_IBT and SHSTK
.long 0

.text
.globl func1
.type func1,@function
func1:
  call func2
  call ifunc
  ret

.type ifunc,@gnu_indirect_function
ifunc:
  ret
