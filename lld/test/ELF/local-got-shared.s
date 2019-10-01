// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
// RUN: ld.lld %t.o -o %t -shared
// RUN: llvm-readobj -S -r -d %t | FileCheck %s
// RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck --check-prefix=DISASM %s

bar:
	call foo@gotpcrel

        .hidden foo
        .global foo
foo:
        nop

// 0x22E0 - 0x1228 - 5 = 4275
// DISASM:      bar:
// DISASM-NEXT:   1228:       callq 4275

// DISASM:      foo:
// DISASM-NEXT:   122d:       nop

// CHECK:      Name: .got
// CHECK-NEXT: Type: SHT_PROGBITS
// CHECK-NEXT: Flags [
// CHECK-NEXT:   SHF_ALLOC
// CHECK-NEXT:   SHF_WRITE
// CHECK-NEXT: ]
// CHECK-NEXT: Address: 0x22E0
// CHECK-NEXT: Offset:
// CHECK-NEXT: Size: 8

// CHECK:      Relocations [
// CHECK-NEXT:   Section ({{.*}}) .rela.dyn {
// CHECK-NEXT:     0x22E0 R_X86_64_RELATIVE - 0x122D
// CHECK-NEXT:   }
// CHECK-NEXT: ]
// CHECK:      0x000000006FFFFFF9 RELACOUNT            1
