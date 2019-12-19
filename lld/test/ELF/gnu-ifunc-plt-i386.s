// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=i686-pc-linux %S/Inputs/shared2-x86-64.s -o %t1.o
// RUN: ld.lld %t1.o --shared --soname=t.so -o %t.so
// RUN: llvm-mc -filetype=obj -triple=i686-pc-linux %s -o %t.o
// RUN: ld.lld --hash-style=sysv %t.so %t.o -o %tout
// RUN: llvm-objdump -d --no-show-raw-insn %tout | FileCheck %s --check-prefix=DISASM
// RUN: llvm-objdump -s %tout | FileCheck %s --check-prefix=GOTPLT
// RUN: llvm-readobj -r --dynamic-table %tout | FileCheck %s

// Check that the IRELATIVE relocations are after the JUMP_SLOT in the plt
// CHECK: Relocations [
// CHECK-NEXT:   Section (4) .rel.dyn {
// CHECK-NEXT:     0x4032AC R_386_IRELATIVE
// CHECK-NEXT:     0x4032B0 R_386_IRELATIVE
// CHECK-NEXT:   }
// CHECK-NEXT:   Section (5) .rel.plt {
// CHECK-NEXT:     0x4032A4 R_386_JUMP_SLOT bar2
// CHECK-NEXT:     0x4032A8 R_386_JUMP_SLOT zed2
// CHECK-NEXT:   }

// Check that IRELATIVE .got.plt entries point to ifunc resolver and not
// back to the plt entry + 6.
// GOTPLT: Contents of section .got.plt:
// GOTPLT:       403298 20224000 00000000 00000000 e6114000
// GOTPLT-NEXT:  4032a8 f6114000 b4114000 b5114000

// Check that the PLTRELSZ tag does not include the IRELATIVE relocations
// CHECK: DynamicSection [
// CHECK:  0x00000012 RELSZ                16 (bytes)
// CHECK:  0x00000002 PLTRELSZ             16 (bytes)

// Check that a PLT header is written and the ifunc entries appear last
// DISASM: Disassembly of section .text:
// DISASM-EMPTY:
// DISASM-NEXT: foo:
// DISASM-NEXT:    4011b4:       retl
// DISASM:      bar:
// DISASM-NEXT:    4011b5:       retl
// DISASM:      _start:
// DISASM-NEXT:    4011b6:       calll   69 <zed2+0x401200>
// DISASM-NEXT:                  calll   80 <zed2+0x401210>
// DISASM-NEXT:                  calll   27 <bar2@plt>
// DISASM-NEXT:                  calll   38 <zed2@plt>
// DISASM-EMPTY:
// DISASM-NEXT: Disassembly of section .plt:
// DISASM-EMPTY:
// DISASM-NEXT: .plt:
// DISASM-NEXT:    4011d0:       pushl   4207260
// DISASM-NEXT:                  jmpl    *4207264
// DISASM-NEXT:                  nop
// DISASM-NEXT:                  nop
// DISASM-NEXT:                  nop
// DISASM-NEXT:                  nop
// DISASM-EMPTY:
// DISASM-NEXT:   bar2@plt:
// DISASM-NEXT:    4011e0:       jmpl    *4207268
// DISASM-NEXT:                  pushl   $0
// DISASM-NEXT:                  jmp     -32 <.plt>
// DISASM-EMPTY:
// DISASM-NEXT:   zed2@plt:
// DISASM-NEXT:    4011f0:       jmpl    *4207272
// DISASM-NEXT:                  pushl   $8
// DISASM-NEXT:                  jmp     -48 <.plt>
// DISASM-EMPTY:
// DISASM-NEXT: Disassembly of section .iplt:
// DISASM-EMPTY:
// DISASM-NEXT: .iplt:
// DISASM-NEXT:                  jmpl    *4207276
// DISASM-NEXT:                  pushl   $0
// DISASM-NEXT:                  jmp     -64 <.plt>
// DISASM-NEXT:                  jmpl    *4207280
// DISASM-NEXT:                  pushl   $8
// DISASM-NEXT:                  jmp     -80 <.plt>

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
