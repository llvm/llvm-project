// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=i686-pc-linux %S/Inputs/shared2-x86-64.s -o %t1.o
// RUN: ld.lld %t1.o --shared --soname=t.so -o %t.so
// RUN: llvm-mc -filetype=obj -triple=i686-pc-linux %s -o %t.o
// RUN: ld.lld --hash-style=sysv %t.so %t.o -o %tout
// RUN: llvm-objdump -d --no-show-raw-insn --print-imm-hex %tout | FileCheck %s --check-prefix=DISASM
// RUN: llvm-objdump -s %tout | FileCheck %s --check-prefix=GOTPLT
// RUN: llvm-readobj -r --dynamic-table %tout | FileCheck %s

/// Check that the PLTRELSZ tag does not include the IRELATIVE relocations
// CHECK: DynamicSection [
// CHECK:  0x00000012 RELSZ                24 (bytes)
// CHECK:  0x00000002 PLTRELSZ             16 (bytes)

/// Check that the IRELATIVE relocations are placed to the .rel.dyn section after
/// other regular relocations (e.g. GLOB_DAT).
// CHECK: Relocations [
// CHECK-NEXT:   Section (4) .rel.dyn {
// CHECK-NEXT:     0x4022C8 R_386_GLOB_DAT bar3 0x0
// CHECK-NEXT:     0x4032E0 R_386_IRELATIVE - 0x0
// CHECK-NEXT:     0x4032E4 R_386_IRELATIVE - 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Section (5) .rel.plt {
// CHECK-NEXT:     0x4032D8 R_386_JUMP_SLOT bar2 0x0
// CHECK-NEXT:     0x4032DC R_386_JUMP_SLOT zed2 0x0
// CHECK-NEXT:   }

// Check that IRELATIVE .got.plt entries point to ifunc resolver and not
// back to the plt entry + 6.
// GOTPLT: Contents of section .got.plt:
// GOTPLT:       4032cc 50224000 00000000 00000000 16124000
// GOTPLT-NEXT:  4032dc 26124000 dc114000 dd114000
//                                  ^        ^-- <bar> (0x4011dd)
//                                  -- <foo> (0x4011dcd)

/// Check that we have 2 PLT sections: one regular .plt section and one
/// .iplt section for ifunc entries.
// DISASM: Disassembly of section .text:
// DISASM-EMPTY:
// DISASM-NEXT: <foo>:
// DISASM-NEXT:    4011dc:       retl
// DISASM:      <bar>:
// DISASM-NEXT:    4011dd:       retl
// DISASM:      <_start>:
// DISASM-NEXT:    4011de:       calll   0x401230
// DISASM-NEXT:                  calll   0x401240
// DISASM-NEXT:                  calll   0x401210 <bar2@plt>
// DISASM-NEXT:                  calll   0x401220 <zed2@plt>
// DISASM-NEXT:                  movl    -0x1004(%eax), %eax
// DISASM-EMPTY:
// DISASM-NEXT: Disassembly of section .plt:
// DISASM-EMPTY:
// DISASM-NEXT: <.plt>:
// DISASM-NEXT:    401200:       pushl   0x4032d0
// DISASM-NEXT:                  jmpl    *0x4032d4
// DISASM-NEXT:                  nop
// DISASM-NEXT:                  nop
// DISASM-NEXT:                  nop
// DISASM-NEXT:                  nop
// DISASM-EMPTY:
// DISASM-NEXT:   <bar2@plt>:
// DISASM-NEXT:    401210:       jmpl    *0x4032d8
// DISASM-NEXT:                  pushl   $0x0
// DISASM-NEXT:                  jmp     0x401200 <.plt>
// DISASM-EMPTY:
// DISASM-NEXT:   <zed2@plt>:
// DISASM-NEXT:    401220:       jmpl    *0x4032dc
// DISASM-NEXT:                  pushl   $0x8
// DISASM-NEXT:                  jmp     0x401200 <.plt>
// DISASM-EMPTY:
// DISASM-NEXT: Disassembly of section .iplt:
// DISASM-EMPTY:
// DISASM-NEXT: <.iplt>:
// DISASM-NEXT:                  jmpl    *0x4032e0
// DISASM-NEXT:                  pushl   $0x0
// DISASM-NEXT:                  jmp     0x401200 <.plt>
// DISASM-NEXT:                  jmpl    *0x4032e4
// DISASM-NEXT:                  pushl   $0x8
// DISASM-NEXT:                  jmp     0x401200 <.plt>

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
 movl bar3@GOT(%eax), %eax
