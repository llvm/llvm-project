// REQUIRES: x86-registered-target
/// cc1as option for --reloc-section-sym. See llvm/test/MC/ELF/reloc-section-sym.s

// RUN: %clang -cc1as -triple x86_64 %s -filetype obj -o %t.default.o
// RUN: %clang -cc1as -triple x86_64 %s -filetype obj --reloc-section-sym=internal -o %t.internal.o
// RUN: %clang -cc1as -triple x86_64 %s -filetype obj --reloc-section-sym=none -o %t.none.o
// RUN: llvm-readelf -rs %t.default.o | FileCheck %s --check-prefix=DEFAULT
// RUN: llvm-readelf -rs %t.internal.o | FileCheck %s --check-prefix=INTERNAL
// RUN: llvm-readelf -rs %t.none.o | FileCheck %s --check-prefix=NONE

.text
  nop
local:
  nop
.Ltemp:
  nop

.section .text1,"ax"
  call local
  call .Ltemp

// DEFAULT:       R_X86_64_PLT32 {{.*}} .text - 3
// DEFAULT-NEXT:  R_X86_64_PLT32 {{.*}} .text - 2
// INTERNAL:      R_X86_64_PLT32 {{.*}} local - 4
// INTERNAL-NEXT: R_X86_64_PLT32 {{.*}} .text - 2
// NONE:          R_X86_64_PLT32 {{.*}} local - 4
// NONE-NEXT:     R_X86_64_PLT32 {{.*}} .Ltemp - 4
