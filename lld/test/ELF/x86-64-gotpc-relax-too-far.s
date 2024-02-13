# REQUIRES: x86
# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/a.s -o %t/a.o
# RUN: ld.lld -T %t/lds1 %t/a.o -o %t/bin
# RUN: llvm-objdump --no-print-imm-hex -d %t/bin | FileCheck --check-prefix=DISASM %s
# RUN: llvm-readelf -S %t/bin | FileCheck --check-prefixes=GOT %s
# RUN: ld.lld -T %t/lds2 %t/a.o -o %t/bin2
# RUN: llvm-readelf -S %t/bin2 | FileCheck --check-prefixes=UNNECESSARY-GOT %s

# DISASM:      <_foo>:
# DISASM-NEXT: movl    2097146(%rip), %eax
# DISASM:      <_start>:
# DISASM-NEXT: movl    1048578(%rip), %eax
# DISASM-NEXT: movq    1048571(%rip), %rax
# DISASM-NEXT: leaq    2147483641(%rip), %rax
# DISASM-NEXT: leal    2147483635(%rip), %eax

# In our implementation, .got is retained even if all GOT-generating relocations are optimized.
# Make sure .got still exists with the right size.
# UNNECESSARY-GOT: .got PROGBITS 0000000000300000 101020 000000 00 WA 0 0 8
#             GOT: .got PROGBITS 0000000000300000 102000 000010 00 WA 0 0 8

#--- a.s
.section .text.foo,"ax"
.globl _foo
.type _foo, @function
_foo:
  movl __start_data@GOTPCREL(%rip), %eax  # out of range

.section .text,"ax"
.globl _start
.type _start, @function
_start:
  movl __stop_data@GOTPCREL(%rip), %eax  # out of range
  movq __stop_data@GOTPCREL(%rip), %rax  # out of range
  movq __stop_data@GOTPCREL(%rip), %rax  # in range
  movl __stop_data@GOTPCREL(%rip), %eax  # in range

.section data,"aw",@progbits
.space 13

#--- lds1
SECTIONS {
  .text.foo 0x100000 : { *(.text.foo) }
  .text 0x200000 : { *(.text) }
  .got 0x300000 : { *(.got) }
  data 0x80200000 : { *(data) }
}
#--- lds2
SECTIONS {
  .text.foo 0x100000 : { *(.text.foo) }
  .text 0x200000 : { *(.text) }
  .got 0x300000 : { *(.got) }
  data 0x400000 : { *(data) }
}
