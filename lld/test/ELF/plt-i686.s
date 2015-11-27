// RUN: llvm-mc -filetype=obj -triple=i686-unknown-linux %s -o %t.o
// RUN: llvm-mc -filetype=obj -triple=i686-unknown-linux %p/Inputs/shared.s -o %t2.o
// RUN: ld.lld -shared %t2.o -o %t2.so
// RUN: ld.lld %t.o %t2.so -o %t
// RUN: llvm-readobj -s -r %t | FileCheck --check-prefix=CHECK %s
// RUN: llvm-objdump -d %t | FileCheck --check-prefix=DISASM %s
// RUN: ld.lld -shared %t.o %t2.so -o %t
// RUN: llvm-readobj -s -r %t | FileCheck --check-prefix=CHECKSHARED %s
// RUN: llvm-objdump -d %t | FileCheck --check-prefix=DISASMSHARED %s

// REQUIRES: x86

// CHECK:      Name: .plt
// CHECK-NEXT: Type: SHT_PROGBITS
// CHECK-NEXT: Flags [
// CHECK-NEXT:   SHF_ALLOC
// CHECK-NEXT:   SHF_EXECINSTR
// CHECK-NEXT: ]
// CHECK-NEXT: Address: 0x11010
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
// CHECK-NEXT: Address: 0x12058
// CHECK-NEXT: Offset: 0x2058
// CHECK-NEXT: Size: 20
// CHECK-NEXT: Link: 0
// CHECK-NEXT: Info: 0
// CHECK-NEXT: AddressAlignment: 4
// CHECK-NEXT: EntrySize: 0

// 0x12058 + got.plt.reserved(12) = 0x12064
// 0x12058 + got.plt.reserved(12) + 4 = 0x12068
// CHECK:      Relocations [
// CHECK-NEXT:   Section ({{.*}}) .rel.plt {
// CHECK-NEXT:     0x12064 R_386_JUMP_SLOT bar 0x0
// CHECK-NEXT:     0x12068 R_386_JUMP_SLOT zed 0x0
// CHECK-NEXT:   }
// CHECK-NEXT: ]

// Unfortunately FileCheck can't do math, so we have to check for explicit
// values:

// 16 is the size of PLT[0]
// (0x11010 + 16) - (0x11000 + 1) - 4 = 27
// (0x11010 + 16) - (0x11005 + 1) - 4 = 22
// (0x11020 + 16) - (0x1100a + 1) - 4 = 33

// DISASM:      _start:
// DISASM-NEXT:   11000:  e9 1b 00 00 00  jmp  27
// DISASM-NEXT:   11005:  e9 16 00 00 00  jmp  22
// DISASM-NEXT:   1100a:  e9 21 00 00 00  jmp  33

// 0x11010 - 0x1102b - 5 = -32
// 0x11010 - 0x1103b - 5 = -48
// 73820 = 0x1205C = .got.plt (0x12058) + 4
// 73824 = 0x12060 = .got.plt (0x12058) + 8
// 73828 = 0x12064 = .got.plt (0x12058) + got.plt.reserved(12)
// 73832 = 0x12068 = .got.plt (0x12058) + got.plt.reserved(12) + 4
// DISASM:      Disassembly of section .plt:
// DISASM-NEXT: .plt:
// DISASM-NEXT:    11010: ff 35 5c 20 01 00 pushl 73820
// DISASM-NEXT:    11016: ff 25 60 20 01 00 jmpl *73824
// DISASM-NEXT:    1101c: 90 nop
// DISASM-NEXT:    1101d: 90 nop
// DISASM-NEXT:    1101e: 90 nop
// DISASM-NEXT:    1101f: 90 nop
// DISASM-NEXT:    11020: ff 25 64 20 01 00 jmpl *73828
// DISASM-NEXT:    11026: 68 00 00 00 00 pushl $0
// DISASM-NEXT:    1102b: e9 e0 ff ff ff jmp -32 <.plt>
// DISASM-NEXT:    11030: ff 25 68 20 01 00 jmpl *73832
// DISASM-NEXT:    11036: 68 08 00 00 00 pushl $8
// DISASM-NEXT:    1103b: e9 d0 ff ff ff jmp -48 <.plt>

// CHECKSHARED:        Name: .plt
// CHECKSHARED-NEXT:   Type: SHT_PROGBITS
// CHECKSHARED-NEXT:   Flags [
// CHECKSHARED-NEXT:     SHF_ALLOC
// CHECKSHARED-NEXT:     SHF_EXECINSTR
// CHECKSHARED-NEXT:   ]
// CHECKSHARED-NEXT:   Address: 0x1010
// CHECKSHARED-NEXT:   Offset: 0x1010
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
// CHECKSHARED-NEXT:   Address: 0x2058
// CHECKSHARED-NEXT:   Offset: 0x2058
// CHECKSHARED-NEXT:   Size: 20
// CHECKSHARED-NEXT:   Link: 0
// CHECKSHARED-NEXT:   Info: 0
// CHECKSHARED-NEXT:   AddressAlignment: 4
// CHECKSHARED-NEXT:   EntrySize: 0
// CHECKSHARED-NEXT:   }

// 0x2058 + got.plt.reserved(12) = 0x2064
// 0x2058 + got.plt.reserved(12) + 4 = 0x2068
// CHECKSHARED:        Relocations [
// CHECKSHARED-NEXT:     Section ({{.*}}) .rel.plt {
// CHECKSHARED-NEXT:       0x2064 R_386_JUMP_SLOT bar 0x0
// CHECKSHARED-NEXT:       0x2068 R_386_JUMP_SLOT zed 0x0
// CHECKSHARED-NEXT:     }
// CHECKSHARED-NEXT:   ]

// DISASMSHARED:       _start:
// DISASMSHARED-NEXT:  1000: e9 1b 00 00 00 jmp 27
// DISASMSHARED-NEXT:  1005: e9 16 00 00 00 jmp 22
// DISASMSHARED-NEXT:  100a: e9 21 00 00 00 jmp 33
// DISASMSHARED-NEXT:  Disassembly of section .plt:
// DISASMSHARED-NEXT:  .plt:
// DISASMSHARED-NEXT:  1010: ff b3 04 00 00 00  pushl 4(%ebx)
// DISASMSHARED-NEXT:  1016: ff a3 08 00 00 00  jmpl *8(%ebx)
// DISASMSHARED-NEXT:  101c: 90 nop
// DISASMSHARED-NEXT:  101d: 90 nop
// DISASMSHARED-NEXT:  101e: 90 nop
// DISASMSHARED-NEXT:  101f: 90 nop
// DISASMSHARED-NEXT:  1020: ff a3 0c 00 00 00  jmpl *12(%ebx)
// DISASMSHARED-NEXT:  1026: 68 00 00 00 00     pushl $0
// DISASMSHARED-NEXT:  102b: e9 e0 ff ff ff     jmp -32 <.plt>
// DISASMSHARED-NEXT:  1030: ff a3 10 00 00 00  jmpl *16(%ebx)
// DISASMSHARED: 1036: 68 08 00 00 00     pushl $8
// DISASMSHARED: 103b: e9 d0 ff ff ff     jmp -48 <.plt>

.global _start
_start:
  jmp bar@PLT
  jmp bar@PLT
  jmp zed@PLT
