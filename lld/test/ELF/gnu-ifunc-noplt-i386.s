// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=i686-pc-freebsd %S/Inputs/shared2-x86-64.s -o %t1.o
// RUN: ld.lld %t1.o --shared --soname=t.so -o %t.so
// RUN: llvm-mc -filetype=obj -triple=i686-pc-freebsd %s -o %t.o
// RUN: ld.lld -z ifunc-noplt -z notext --hash-style=sysv %t.so %t.o -o %tout
// RUN: llvm-objdump -d --no-show-raw-insn --print-imm-hex %tout | FileCheck %s --check-prefix=DISASM
// RUN: llvm-readobj -r --dynamic-table %tout | FileCheck %s

// Check that we emitted relocations for the ifunc calls
// CHECK: Relocations [
// CHECK-NEXT:   Section (4) .rel.dyn {
// CHECK-NEXT:     0x4011EF R_386_PLT32 foo
// CHECK-NEXT:     0x4011F4 R_386_PLT32 bar
// CHECK-NEXT:   }
// CHECK-NEXT:   Section (5) .rel.plt {
// CHECK-NEXT:     0x4032D4 R_386_JUMP_SLOT bar2
// CHECK-NEXT:     0x4032D8 R_386_JUMP_SLOT zed2
// CHECK-NEXT:   }

// Check that ifunc call sites still require relocation
// DISASM: Disassembly of section .text:
// DISASM-EMPTY:
// DISASM-NEXT: 004011ec <foo>:
// DISASM-NEXT:   retl
// DISASM-EMPTY:
// DISASM-NEXT: 004011ed <bar>:
// DISASM-NEXT:   retl
// DISASM-EMPTY:
// DISASM-NEXT: 004011ee <_start>:
// DISASM-NEXT:   calll	0x4011ef <_start+0x1>
// DISASM-NEXT:   calll	0x4011f4 <_start+0x6>
// DISASM-NEXT:   calll	0x401220 <bar2@plt>
// DISASM-NEXT:   calll	0x401230 <zed2@plt>
// DISASM-EMPTY:
// DISASM-NEXT: Disassembly of section .plt:
// DISASM-EMPTY:
// DISASM-NEXT: 00401210 <.plt>:
// DISASM-NEXT:   pushl 0x4032cc
// DISASM-NEXT:   jmpl *0x4032d0
// DISASM-NEXT:   nop
// DISASM-NEXT:   nop
// DISASM-NEXT:   nop
// DISASM-NEXT:   nop
// DISASM-EMPTY:
// DISASM-NEXT: 00401220 <bar2@plt>:
// DISASM-NEXT:   jmpl	*0x4032d4
// DISASM-NEXT:   pushl	$0x0
// DISASM-NEXT:   jmp	0x401210 <.plt>
// DISASM-EMPTY:
// DISASM-NEXT: 00401230 <zed2@plt>:
// DISASM-NEXT:   jmpl	*0x4032d8
// DISASM-NEXT:   pushl	$0x8
// DISASM-NEXT:   jmp	0x401210 <.plt>

.text
.type foo STT_GNU_IFUNC
.globl foo
foo:
 ret

.type bar STT_GNU_IFUNC
.globl bar
bar:
 ret

.globl _start
_start:
 call foo@plt
 call bar@plt
 call bar2@plt
 call zed2@plt
