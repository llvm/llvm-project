# REQUIRES: x86
# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/a.s -o %t/a.o
# RUN: ld.lld -T %t/lds1 %t/a.o -o %t/bin
# RUN: llvm-objdump -d %t/bin | FileCheck --check-prefix=DISASM %s
# RUN: llvm-readelf -S %t/bin | FileCheck --check-prefixes=GOT %s
# RUN: ld.lld -T %t/lds2 %t/a.o -o %t/bin2
# RUN: llvm-objdump -d %t/bin2 | FileCheck --check-prefix=DISASM %s
# RUN: llvm-readelf -S %t/bin2 | FileCheck --check-prefixes=GOT %s
# RUN: ld.lld -T %t/lds3 %t/a.o -o %t/bin3
# RUN: llvm-readelf -S %t/bin3 | FileCheck --check-prefixes=UNNECESSARY-GOT %s
# RUN: ld.lld -T %t/lds4 %t/a.o -o %t/bin4
# RUN: llvm-objdump -d %t/bin4 | FileCheck --check-prefix=DISASM4 %s

# DISASM:      <_foo>:
# DISASM-NEXT: movl    0x1ffffa(%rip), %eax
# DISASM-NEXT: addq    0x1ffffb(%rip), %rax
# DISASM-NEXT: addq    $0x7fffffff, %rax
# DISASM:      <_start>:
# DISASM-NEXT: movl    0x10000a(%rip), %eax
# DISASM-NEXT: movq    0x100003(%rip), %rax
# DISASM-NEXT: leaq    0x7ffffff9(%rip), %rax
# DISASM-NEXT: leal    0x7ffffff3(%rip), %eax

# DISASM4:      <_foo>:
# DISASM4-NEXT: leal    0x7fffeffa(%rip), %eax
# DISASM4-NEXT: addq    0x1ff3(%rip), %rax
# DISASM4-NEXT: addq    $0x7fffffff, %rax

# In our implementation, .got is retained even if all GOT-generating relocations are optimized.
# Make sure .got still exists with the right size.
# UNNECESSARY-GOT: .got PROGBITS 0000000000300000 101020 000000 00 WA 0 0 8
#             GOT: .got PROGBITS 0000000000300000 102000 000018 00 WA 0 0 8

#--- a.s
.section .text.foo,"ax"
.globl _foo
.type _foo, @function
_foo:
  movl __start_data@GOTPCREL(%rip), %eax  # out of range
  addq foo_1@GOTPCREL(%rip), %rax         # out of range
  addq foo@GOTPCREL(%rip), %rax           # in range

.section .text,"ax"
.globl _start
.type _start, @function
_start:
  movl __stop_data@GOTPCREL(%rip), %eax  # out of range
  movq __stop_data@GOTPCREL(%rip), %rax  # out of range
  movq __stop_data@GOTPCREL(%rip), %rax  # in range
  movl __stop_data@GOTPCREL(%rip), %eax  # in range

.section foo,"aw",@progbits
.space 1
foo_1:
.space 1

.section data,"aw",@progbits
.space 13

#--- lds1
SECTIONS {
  .text.foo 0x100000 : { *(.text.foo) }
  .text 0x200000 : { *(.text) }
  .got 0x300000 : { *(.got) }
  foo 0x7fffffff : { *(foo) }
  data 0x80200000 : { *(data) }
}
#--- lds2
SECTIONS {
  .text.foo 0x100000 : { *(.text.foo) }
  .text 0x1ff000 : { . = . + 0x1000 ; *(.text) }
  .got 0x300000 : { *(.got) }
  foo 0x7fffffff : { *(foo) }
  data 0x80200000 : { *(data) }
}
#--- lds3
SECTIONS {
  .text.foo 0x100000 : { *(.text.foo) }
  .text 0x200000 : { *(.text) }
  .got 0x300000 : { *(.got) }
  data 0x400000 : { *(data) }
}

#--- lds4
## Max VA difference < 0x80000000
SECTIONS {
  .text.foo 0x02000 : { *(.text.foo) }
  .text 0x3000 : { *(.text) }
  .got 0x4000 : { *(.got) }
  foo 0x7fffffff : { *(foo) }
  data 0x80001000 : { *(data) }
}
