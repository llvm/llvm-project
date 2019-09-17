// REQUIRES: x86

/// Test we correctly wrap PLT calls.

// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t

// RUN: ld.lld -o %t2 %t -wrap foo -shared
// RUN: llvm-readobj -S -r %t2 | FileCheck %s
// RUN: llvm-objdump -d %t2 | FileCheck --check-prefix=DISASM %s

// CHECK:      Relocations [
// CHECK-NEXT:   Section ({{.*}}) .rela.plt {
// CHECK-NEXT:     R_X86_64_JUMP_SLOT __wrap_foo 0x0
// CHECK-NEXT:     R_X86_64_JUMP_SLOT _start 0x0
// CHECK-NEXT:   }
// CHECK-NEXT: ]

// DISASM:      _start:
// DISASM-NEXT:   jmp {{.*}} <__wrap_foo@plt>
// DISASM-NEXT:   jmp {{.*}} <__wrap_foo@plt>
// DISASM-NEXT:   jmp {{.*}} <_start@plt>

.global foo
foo:
  nop

.global __wrap_foo
__wrap_foo:
  nop

.global _start
_start:
  jmp foo@plt
  jmp __wrap_foo@plt
  jmp _start@plt
