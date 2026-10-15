# REQUIRES: aarch64

## Both segment planners extend the same external symbol and start numbering
## at zero. Only the extender outside __TEXT gets a segment prefix.
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %s -o %t.o
# RUN: %lld -arch arm64 -o %t %t.o -segprot __HIB rx rx -segprot __LAST rx rx -arm64-islands-max-hops 0
# RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t | FileCheck %s

# CHECK-LABEL: Disassembly of section __TEXT,__text:
# CHECK-LABEL: <_main>:
# CHECK-NEXT: bl 0x[[#%x, FIRST:]] <_target.thunk.0>
# CHECK-NEXT: ret
# CHECK: [[#FIRST]] <_target.thunk.0>:
# CHECK-NEXT: adrp x16,
# CHECK-NEXT: add x16, x16,
# CHECK-NEXT: br x16
# CHECK-LABEL: Disassembly of section __HIB,__text:
# CHECK-LABEL: <_caller>:
# CHECK-NEXT: bl 0x[[#%x, SECOND:]] <__HIB._target.thunk.0>
# CHECK-NEXT: ret
# CHECK: [[#SECOND]] <__HIB._target.thunk.0>:
# CHECK-NEXT: adrp x16,
# CHECK-NEXT: add x16, x16,
# CHECK-NEXT: br x16
# CHECK-LABEL: Disassembly of section __LAST,__text:
# CHECK-LABEL: <_target>:
# CHECK-NEXT: ret

.text
.globl _main
.p2align 2
_main:
  bl _target
  ret
_space1:
.space 0x8100000

.section __HIB,__text,regular,pure_instructions
.globl _caller
.p2align 2
_caller:
  bl _target
  ret
_space2:
.space 0x8100000

.section __LAST,__text,regular,pure_instructions
.globl _target
.p2align 2
_target:
  ret

.subsections_via_symbols
