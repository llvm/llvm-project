## gdb-index-a.elf and gdb-index-b.elf are a test.o and test2.o renamed,
## were generated in this way:
## test.cpp:
##  int main() { return 0; }
## test2.cpp:
##  int main2() { return 0; }
## Compiled with:
## gcc -gsplit-dwarf -c test.cpp test2.cpp
## gcc version 5.3.1 20160413
## Info about gdb-index: https://sourceware.org/gdb/onlinedocs/gdb/Index-Section-Format.html

# REQUIRES: x86
# RUN: ld.lld --gdb-index -e main %p/Inputs/gdb-index-a.elf %p/Inputs/gdb-index-b.elf -o %t
# RUN: llvm-dwarfdump -debug-dump=gdb_index %t | FileCheck %s
# RUN: llvm-objdump -d %t | FileCheck %s --check-prefix=DISASM

# DISASM:       Disassembly of section .text:
# DISASM:       main:
# DISASM-CHECK:     11000: 55 pushq %rbp
# DISASM-CHECK:     11001: 48 89 e5 movq %rsp, %rbp
# DISASM-CHECK:     11004: b8 00 00 00 00 movl $0, %eax
# DISASM-CHECK:     11009: 5d popq %rbp
# DISASM-CHECK:     1100a: c3 retq
# DISASM:       _Z5main2v:
# DISASM-CHECK:     1100b: 55 pushq %rbp
# DISASM-CHECK:     1100c: 48 89 e5  movq %rsp, %rbp
# DISASM-CHECK:     1100f: b8 00 00 00 00 movl $0, %eax
# DISASM-CHECK:     11014: 5d popq %rbp
# DISASM-CHECK:     11015: c3 retq

# CHECK:      .gnu_index contents:
# CHECK-NEXT:    Version = 7
# CHECK:       CU list offset = 0x18, has 2 entries:
# CHECK-NEXT:    0: Offset = 0x0, Length = 0x34
# CHECK-NEXT:    1: Offset = 0x34, Length = 0x34
# CHECK:       Address area offset = 0x38, has 2 entries:
# CHECK-NEXT:    Low address = 0x201000, High address = 0x20100b, CU index = 0
# CHECK-NEXT:    Low address = 0x20100b, High address = 0x201016, CU index = 1
# CHECK:       Symbol table offset = 0x60, size = 0, filled slots:
# CHECK:       Constant pool offset = 0x60, has 0 CU vectors:
