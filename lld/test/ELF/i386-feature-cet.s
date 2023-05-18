# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=i386 %s -o %t.o
# RUN: llvm-mc -filetype=obj -triple=i386 %p/Inputs/i386-cet1.s -o %t1.o
# RUN: llvm-mc -filetype=obj -triple=i386 %p/Inputs/i386-cet2.s -o %t2.o
# RUN: llvm-mc -filetype=obj -triple=i386 %p/Inputs/i386-cet3.s -o %t3.o
# RUN: llvm-mc -filetype=obj -triple=i386 %p/Inputs/i386-cet4.s -o %t4.o

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

# RUN: not ld.lld -e func1 %t.o %t3.o -o /dev/null -z cet-report=something 2>&1 \
# RUN:   | FileCheck --check-prefix=REPORT_INVALID %s
# REPORT_INVALID: error: -z cet-report= parameter something is not recognized
# REPORT_INVALID-EMPTY:

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
# GOTPLT-NEXT: 0x004032d0 50224000 00000000 00000000 20124000
# GOTPLT-NEXT: 0x004032e0 0b124000

# DISASM:      Disassembly of section .text:
# DISASM:      00401200 <func1>:
# DISASM-NEXT: 401200:       calll   0x401230 <func2+0x401230>
# DISASM-NEXT: 401205:       calll   0x401240 <ifunc>
# DISASM-NEXT:               retl

# DISASM:      Disassembly of section .plt:
# DISASM:      00401210 <.plt>:
# DISASM-NEXT: 401210:       pushl   0x4032d4
# DISASM-NEXT:               jmpl    *0x4032d8
# DISASM-NEXT:               nop
# DISASM-NEXT:               nop
# DISASM-NEXT:               nop
# DISASM-NEXT:               nop
# DISASM-NEXT:               endbr32
# DISASM-NEXT:               pushl   $0x0
# DISASM-NEXT:               jmp     0x401210 <.plt>
# DISASM-NEXT:               nop

# DISASM:      Disassembly of section .plt.sec:
# DISASM:      00401230 <.plt.sec>:
# DISASM-NEXT: 401230:       endbr32
# DISASM-NEXT:               jmpl    *0x4032dc
# DISASM-NEXT:               nopw    (%eax,%eax)

# DISASM:      Disassembly of section .iplt:
# DISASM:      00401240 <ifunc>:
# DISASM-NEXT: 401240:       endbr32
# DISASM-NEXT:               jmpl    *0x4032e0
# DISASM-NEXT:               nopw    (%eax,%eax)

.section ".note.gnu.property", "a"
.long 4
.long 0xc
.long 0x5
.asciz "GNU"

.long 0xc0000002 # GNU_PROPERTY_X86_FEATURE_1_AND
.long 4
.long 3          # GNU_PROPERTY_X86_FEATURE_1_IBT and SHSTK

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
