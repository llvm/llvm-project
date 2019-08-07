// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=i686-unknown-linux %s -o %t.o
// RUN: llvm-mc -filetype=obj -triple=i686-unknown-linux %p/Inputs/shared.s -o %t2.o
// RUN: ld.lld -shared %t2.o -soname=t2.so -o %t2.so
// RUN: ld.lld %t.o %t2.so -o %t
// RUN: llvm-readobj -S -r %t | FileCheck %s
// RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck --check-prefix=DISASM %s
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
// CHECK-NEXT: Address: 0x401020
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
// CHECK-NEXT: Address: 0x403000
// CHECK-NEXT: Offset: 0x3000
// CHECK-NEXT: Size: 20
// CHECK-NEXT: Link: 0
// CHECK-NEXT: Info: 0
// CHECK-NEXT: AddressAlignment: 4
// CHECK-NEXT: EntrySize: 0

// 0x12000 + got.plt.reserved(12) = 0x1200C
// 0x12000 + got.plt.reserved(12) + 4 = 0x12010
// CHECK:      Relocations [
// CHECK-NEXT:   Section ({{.*}}) .rel.plt {
// CHECK-NEXT:     0x40300C R_386_JUMP_SLOT bar 0x0
// CHECK-NEXT:     0x403010 R_386_JUMP_SLOT zed 0x0
// CHECK-NEXT:   }
// CHECK-NEXT: ]

// Unfortunately FileCheck can't do math, so we have to check for explicit
// values:

// 16 is the size of PLT[0]
// (0x401010 + 16) - (0x401000 + 1) - 4 = 27
// (0x401010 + 16) - (0x401005 + 1) - 4 = 22
// (0x401020 + 16) - (0x40100a + 1) - 4 = 33

// DISASM:       local:
// DISASM-NEXT:  401000:
// DISASM-NEXT:  401002:
// DISASM:       _start:
// 0x401013 + 5 - 24 = 0x401000
// DISASM-NEXT: 401004:       jmp 39 <bar@plt>
// DISASM-NEXT: 401009:       jmp 34 <bar@plt>
// DISASM-NEXT: 40100e:       jmp 45 <zed@plt>
// DISASM-NEXT: 401013:       jmp -24 <local>

// 0x401010 - 0x40102b - 5 = -32
// 0x401010 - 0x40103b - 5 = -48
// 4206596 = 0x403004 = .got.plt (0x403000) + 4
// 4206600 = 0x403008 = .got.plt (0x403000) + 8
// 4206604 = 0x40300C = .got.plt (0x403000) + got.plt.reserved(12)
// 4206608 = 0x403010 = .got.plt (0x403000) + got.plt.reserved(12) + 4
// DISASM:      Disassembly of section .plt:
// DISASM-EMPTY:
// DISASM-NEXT: .plt:
// DISASM-NEXT: 401020:       pushl 4206596
// DISASM-NEXT:               jmpl *4206600
// DISASM-NEXT:               nop
// DISASM-NEXT:               nop
// DISASM-NEXT:               nop
// DISASM-NEXT:               nop
// DISASM-EMPTY:
// DISASM-NEXT: bar@plt:
// DISASM-NEXT: 401030:       jmpl *4206604
// DISASM-NEXT:               pushl $0
// DISASM-NEXT:               jmp -32 <.plt>
// DISASM-EMPTY:
// DISASM-NEXT: zed@plt:
// DISASM-NEXT: 401040:       jmpl *4206608
// DISASM-NEXT:               pushl $8
// DISASM-NEXT:               jmp -48 <.plt>

// CHECKSHARED:        Name: .plt
// CHECKSHARED-NEXT:   Type: SHT_PROGBITS
// CHECKSHARED-NEXT:   Flags [
// CHECKSHARED-NEXT:     SHF_ALLOC
// CHECKSHARED-NEXT:     SHF_EXECINSTR
// CHECKSHARED-NEXT:   ]
// CHECKSHARED-NEXT:   Address: 0x1020
// CHECKSHARED-NEXT:   Offset: 0x1020
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
// CHECKSHARED-NEXT:   Address: 0x3000
// CHECKSHARED-NEXT:   Offset: 0x3000
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
// CHECKSHARED-NEXT:       0x300C R_386_JUMP_SLOT bar 0x0
// CHECKSHARED-NEXT:       0x3010 R_386_JUMP_SLOT zed 0x0
// CHECKSHARED-NEXT:     }
// CHECKSHARED-NEXT:   ]

// DISASMSHARED:      local:
// DISASMSHARED-NEXT: 1000:
// DISASMSHARED-NEXT: 1002:
// DISASMSHARED:      _start:
// 0x1013 + 5 - 24 = 0x1000
// DISASMSHARED-NEXT: 1004:       jmp 39 <bar@plt>
// DISASMSHARED-NEXT:             jmp 34 <bar@plt>
// DISASMSHARED-NEXT:             jmp 45 <zed@plt>
// DISASMSHARED-NEXT:             jmp -24 <local>
// DISASMSHARED-EMPTY:
// DISASMSHARED-NEXT: Disassembly of section .plt:
// DISASMSHARED-EMPTY:
// DISASMSHARED-NEXT: .plt:
// DISASMSHARED-NEXT: 1020:       pushl 4(%ebx)
// DISASMSHARED-NEXT:             jmpl *8(%ebx)
// DISASMSHARED-NEXT:             nop
// DISASMSHARED-NEXT:             nop
// DISASMSHARED-NEXT:             nop
// DISASMSHARED-NEXT:             nop
// DISASMSHARED:      bar@plt:
// DISASMSHARED-NEXT: 1030:       jmpl *12(%ebx)
// DISASMSHARED-NEXT:             pushl $0
// DISASMSHARED-NEXT:             jmp -32 <.plt>
// DISASMSHARED:      zed@plt:
// DISASMSHARED-NEXT: 1040:       jmpl *16(%ebx)
// DISASMSHARED-NEXT:             pushl $8
// DISASMSHARED-NEXT:             jmp -48 <.plt>

// DISASMPIE:      Disassembly of section .plt:
// DISASMPIE-EMPTY:
// DISASMPIE-NEXT: .plt:
// DISASMPIE-NEXT: 1020:       pushl 4(%ebx)
// DISASMPIE-NEXT:             jmpl *8(%ebx)
// DISASMPIE-NEXT:             nop
// DISASMPIE-NEXT:             nop
// DISASMPIE-NEXT:             nop
// DISASMPIE-NEXT:             nop
// DISASMPIE-EMPTY:
// DISASMPIE-NEXT: bar@plt:
// DISASMPIE-NEXT: 1030:       jmpl *12(%ebx)
// DISASMPIE-NEXT:             pushl $0
// DISASMPIE-NEXT:             jmp -32 <.plt>
// DISASMPIE-EMPTY:
// DISASMPIE-NEXT: zed@plt:
// DISASMPIE-NEXT: 1040:       jmpl *16(%ebx)
// DISASMPIE-NEXT:             pushl $8
// DISASMPIE-NEXT:             jmp -48 <.plt>

local:
.long 0

.global _start
_start:
  jmp bar@PLT
  jmp bar@PLT
  jmp zed@PLT
  jmp local@plt
