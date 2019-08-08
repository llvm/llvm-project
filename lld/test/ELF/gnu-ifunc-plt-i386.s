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
// CHECK-NEXT:     0x403014 R_386_IRELATIVE
// CHECK-NEXT:     0x403018 R_386_IRELATIVE
// CHECK-NEXT:   }
// CHECK-NEXT:   Section (5) .rel.plt {
// CHECK-NEXT:     0x40300C R_386_JUMP_SLOT bar2
// CHECK-NEXT:     0x403010 R_386_JUMP_SLOT zed2
// CHECK-NEXT:   }

// Check that IRELATIVE .got.plt entries point to ifunc resolver and not
// back to the plt entry + 6.
// GOTPLT: Contents of section .got.plt:
// GOTPLT:       403000 00204000 00000000 00000000 36104000
// GOTPLT-NEXT:  403010 46104000 00104000 01104000

// Check that the PLTRELSZ tag does not include the IRELATIVE relocations
// CHECK: DynamicSection [
// CHECK:  0x00000012 RELSZ                16 (bytes)
// CHECK:  0x00000002 PLTRELSZ             16 (bytes)

// Check that a PLT header is written and the ifunc entries appear last
// DISASM: Disassembly of section .text:
// DISASM-EMPTY:
// DISASM-NEXT: foo:
// DISASM-NEXT:    401000:       retl
// DISASM:      bar:
// DISASM-NEXT:    401001:       retl
// DISASM:      _start:
// DISASM-NEXT:    401002:       calll   73 <zed2@plt+0x10>
// DISASM-NEXT:                  calll   84 <zed2@plt+0x20>
// DISASM-NEXT:                  calll   31 <bar2@plt>
// DISASM-NEXT:                  calll   42 <zed2@plt>
// DISASM-EMPTY:
// DISASM-NEXT: Disassembly of section .plt:
// DISASM-EMPTY:
// DISASM-NEXT: .plt:
// DISASM-NEXT:    401020:       pushl   4206596
// DISASM-NEXT:                  jmpl    *4206600
// DISASM-NEXT:                  nop
// DISASM-NEXT:                  nop
// DISASM-NEXT:                  nop
// DISASM-NEXT:                  nop
// DISASM-EMPTY:
// DISASM-NEXT:   bar2@plt:
// DISASM-NEXT:    401030:       jmpl    *4206604
// DISASM-NEXT:                  pushl   $0
// DISASM-NEXT:                  jmp     -32 <.plt>
// DISASM-EMPTY:
// DISASM-NEXT:   zed2@plt:
// DISASM-NEXT:    401040:       jmpl    *4206608
// DISASM-NEXT:                  pushl   $8
// DISASM-NEXT:                  jmp     -48 <.plt>
// DISASM-NEXT:                  jmpl    *4206612
// DISASM-NEXT:                  pushl   $48
// DISASM-NEXT:                  jmp     -32 <zed2@plt>
// DISASM-NEXT:                  jmpl    *4206616
// DISASM-NEXT:                  pushl   $56
// DISASM-NEXT:                  jmp     -48 <zed2@plt>

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
