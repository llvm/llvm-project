# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t
# RUN: not ld.lld %t -o /dev/null
# RUN: ld.lld %t --noinhibit-exec -o %t2
# RUN: llvm-objdump -d %t2 | FileCheck %s
# RUN: llvm-readobj -r %t2 | FileCheck %s --check-prefix=RELOC

# CHECK: Disassembly of section .text:
# CHECK-EMPTY:
# CHECK-NEXT: _start
# CHECK-NEXT: 201120: {{.*}} callq 0x0

# RELOC:      Relocations [
# RELOC-NEXT: ]

# next code will not link without noinhibit-exec flag
# because of undefined symbol _bar
.globl _start
_start:
  call _bar
