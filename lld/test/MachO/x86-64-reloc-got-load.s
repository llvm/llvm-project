# REQUIRES: x86

## Check that we perform relaxation for GOT_LOAD relocations to defined symbols.
## Note: GOT_LOAD relocations to dylib symbols are already tested in dylink.s.

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o
# RUN: lld -flavor darwinnew -o %t %t.o
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s
# CHECK: leaq	[[#]](%rip), %rax  # {{.*}} <_foo>

.globl _main, _foo

_main:
  movq _foo@GOTPCREL(%rip), %rax
  ret

_foo:
  .space 0
