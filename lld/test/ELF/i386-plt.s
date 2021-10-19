// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=i686-unknown-linux %s -o %t.o
// RUN: llvm-mc -filetype=obj -triple=i686-unknown-linux %p/Inputs/shared.s -o %t2.o
// RUN: ld.lld -shared %t2.o -soname=t2.so -o %t2.so
// RUN: ld.lld %t.o %t2.so -o %t
// RUN: llvm-readobj -S -r %t | FileCheck %s
// RUN: llvm-objdump -d --no-show-raw-insn --print-imm-hex %t | FileCheck --check-prefix=DISASM %s
// RUN: ld.lld -shared %t.o %t2.so -o %t
// RUN: llvm-readobj -S -r %t | FileCheck --check-prefix=CHECKSHARED %s
// RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck --check-prefix=DISASMSHARED %s
// RUN: ld.lld -pie %t.o %t2.so -o %t
// RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck --check-prefix=DISASMPIE %s

// CHECK:      Name: .plt
// CHECK-NEXT: Type: SHT_PROGBITS
// CHECK-NEXT: Flags [
// CHECK-NEXT:   SHF_ALLOC
// CHECK-NEXT:   SHF_EXECINSTR
// CHECK-NEXT: ]
// CHECK-NEXT: Address: 0x4011E0
// CHECK-NEXT: Offset:
// CHECK-NEXT: Size: 48
// CHECK-NEXT: Link: 0
// CHECK-NEXT: Info: 0
// CHECK-NEXT: AddressAlignment: 16

// CHECK:      Name: .got.plt
// CHECK-NEXT: Type: SHT_PROGBITS
// CHECK-NEXT: Flags [
// CHECK-NEXT:   SHF_ALLOC
// CHECK-NEXT:   SHF_WRITE
// CHECK-NEXT: ]
// CHECK-NEXT: Address: 0x403278
// CHECK-NEXT: Offset: 0x278
// CHECK-NEXT: Size: 20
// CHECK-NEXT: Link: 0
// CHECK-NEXT: Info: 0
// CHECK-NEXT: AddressAlignment: 4
// CHECK-NEXT: EntrySize: 0

// First 3 slots of .got.plt are reserved.
// &.got.plt[3] = 0x403278 + 12 = 0x403284
// &.got.plt[4] = 0x403278 + 16 = 0x403288
// CHECK:      Relocations [
// CHECK-NEXT:   Section ({{.*}}) .rel.plt {
// CHECK-NEXT:     0x403284 R_386_JUMP_SLOT bar
// CHECK-NEXT:     0x403288 R_386_JUMP_SLOT zed
// CHECK-NEXT:   }
// CHECK-NEXT: ]

// DISASM:       <local>:
// DISASM-NEXT:  4011bc:
// DISASM-NEXT:  4011be:
// DISASM:       <_start>:
// DISASM-NEXT: 4011c0:       jmp 0x4011f0 <bar@plt>
// DISASM-NEXT: 4011c5:       jmp 0x4011f0 <bar@plt>
// DISASM-NEXT: 4011ca:       jmp 0x401200 <zed@plt>
// DISASM-NEXT: 4011cf:       jmp 0x4011bc <local>

// DISASM:      Disassembly of section .plt:
// DISASM-EMPTY:
// DISASM-NEXT: <.plt>:
/// Push .got.plt[1], then jump to .got.plt[2]
// DISASM-NEXT: 4011e0:       pushl 0x40327c
// DISASM-NEXT:               jmpl *0x403280
// DISASM-NEXT:               nop
// DISASM-NEXT:               nop
// DISASM-NEXT:               nop
// DISASM-NEXT:               nop
// DISASM-EMPTY:
// DISASM-NEXT: <bar@plt>:
/// .got.plt[3] = 0x403278 + 12 = 0x403284
// DISASM-NEXT: 4011f0:       jmpl *0x403284
// DISASM-NEXT:               pushl $0x0
// DISASM-NEXT:               jmp 0x4011e0 <.plt>
// DISASM-EMPTY:
// DISASM-NEXT: <zed@plt>:
/// .got.plt[4] = 0x403278 + 16 = 0x403288
// DISASM-NEXT: 401200:       jmpl *0x403288
// DISASM-NEXT:               pushl $0x8
// DISASM-NEXT:               jmp 0x4011e0 <.plt>

// CHECKSHARED:        Name: .plt
// CHECKSHARED-NEXT:   Type: SHT_PROGBITS
// CHECKSHARED-NEXT:   Flags [
// CHECKSHARED-NEXT:     SHF_ALLOC
// CHECKSHARED-NEXT:     SHF_EXECINSTR
// CHECKSHARED-NEXT:   ]
// CHECKSHARED-NEXT:   Address: 0x1200
// CHECKSHARED-NEXT:   Offset: 0x200
// CHECKSHARED-NEXT:   Size: 48
// CHECKSHARED-NEXT:   Link: 0
// CHECKSHARED-NEXT:   Info: 0
// CHECKSHARED-NEXT:   AddressAlignment: 16
// CHECKSHARED-NEXT:   EntrySize: 0
// CHECKSHARED-NEXT:   }
// CHECKSHARED:        Name: .got.plt
// CHECKSHARED-NEXT:   Type: SHT_PROGBITS
// CHECKSHARED-NEXT:   Flags [
// CHECKSHARED-NEXT:     SHF_ALLOC
// CHECKSHARED-NEXT:     SHF_WRITE
// CHECKSHARED-NEXT:   ]
// CHECKSHARED-NEXT:   Address: 0x3290
// CHECKSHARED-NEXT:   Offset: 0x290
// CHECKSHARED-NEXT:   Size: 20
// CHECKSHARED-NEXT:   Link: 0
// CHECKSHARED-NEXT:   Info: 0
// CHECKSHARED-NEXT:   AddressAlignment: 4
// CHECKSHARED-NEXT:   EntrySize: 0
// CHECKSHARED-NEXT:   }

// 0x3000 + got.plt.reserved(12) = 0x300C
// 0x3000 + got.plt.reserved(12) + 4 = 0x3010
// CHECKSHARED:        Relocations [
// CHECKSHARED-NEXT:     Section ({{.*}}) .rel.plt {
// CHECKSHARED-NEXT:       0x329C R_386_JUMP_SLOT bar
// CHECKSHARED-NEXT:       0x32A0 R_386_JUMP_SLOT zed
// CHECKSHARED-NEXT:     }
// CHECKSHARED-NEXT:   ]

// DISASMSHARED:      <local>:
// DISASMSHARED-NEXT: 11e0:
// DISASMSHARED-NEXT: 11e2:
// DISASMSHARED:      <_start>:
// DISASMSHARED-NEXT: 11e4:       jmp 0x1210 <bar@plt>
// DISASMSHARED-NEXT:             jmp 0x1210 <bar@plt>
// DISASMSHARED-NEXT:             jmp 0x1220 <zed@plt>
// DISASMSHARED-NEXT:             jmp 0x11e0 <local>
// DISASMSHARED-EMPTY:
// DISASMSHARED-NEXT: Disassembly of section .plt:
// DISASMSHARED-EMPTY:
// DISASMSHARED-NEXT: <.plt>:
// DISASMSHARED-NEXT: 1200:       pushl 4(%ebx)
// DISASMSHARED-NEXT:             jmpl *8(%ebx)
// DISASMSHARED-NEXT:             nop
// DISASMSHARED-NEXT:             nop
// DISASMSHARED-NEXT:             nop
// DISASMSHARED-NEXT:             nop
// DISASMSHARED:      <bar@plt>:
// DISASMSHARED-NEXT: 1210:       jmpl *12(%ebx)
// DISASMSHARED-NEXT:             pushl $0
// DISASMSHARED-NEXT:             jmp 0x1200 <.plt>
// DISASMSHARED:      <zed@plt>:
// DISASMSHARED-NEXT: 1220:       jmpl *16(%ebx)
// DISASMSHARED-NEXT:             pushl $8
// DISASMSHARED-NEXT:             jmp 0x1200 <.plt>

// DISASMPIE:      Disassembly of section .plt:
// DISASMPIE-EMPTY:
// DISASMPIE-NEXT: <.plt>:
// DISASMPIE-NEXT: 11e0:       pushl 4(%ebx)
// DISASMPIE-NEXT:             jmpl *8(%ebx)
// DISASMPIE-NEXT:             nop
// DISASMPIE-NEXT:             nop
// DISASMPIE-NEXT:             nop
// DISASMPIE-NEXT:             nop
// DISASMPIE-EMPTY:
// DISASMPIE-NEXT: <bar@plt>:
// DISASMPIE-NEXT: 11f0:       jmpl *12(%ebx)
// DISASMPIE-NEXT:             pushl $0
// DISASMPIE-NEXT:             jmp 0x11e0 <.plt>
// DISASMPIE-EMPTY:
// DISASMPIE-NEXT: <zed@plt>:
// DISASMPIE-NEXT: 1200:       jmpl *16(%ebx)
// DISASMPIE-NEXT:             pushl $8
// DISASMPIE-NEXT:             jmp 0x11e0 <.plt>

local:
.long 0

.global _start
_start:
  jmp bar@PLT
  jmp bar@PLT
  jmp zed@PLT
  jmp local@plt
