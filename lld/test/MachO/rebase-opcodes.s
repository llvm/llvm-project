# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o
# RUN: %lld -dylib %t.o -o %t.dylib
# RUN: obj2yaml %t.dylib | FileCheck %s

## Test that:
## 1/ Consecutive rebases are encoded as REBASE_OPCODE_DO_REBASE_IMM_TIMES.
## 2/ Gaps smaller than 15 words are encoded as REBASE_OPCODE_ADD_ADDR_IMM_SCALED.
## 3/ Gaps larger than that become REBASE_OPCODE_ADD_ADDR_ULEB.
## FIXME: The last rebase could be transformed into a REBASE_OPCODE_DO_REBASE_ADD_ADDR_ULEB.

# CHECK: RebaseOpcodes:
# CHECK-NEXT: Opcode:          REBASE_OPCODE_SET_TYPE_IMM
# CHECK-NEXT: Imm:             1
# CHECK-NEXT: Opcode:          REBASE_OPCODE_SET_SEGMENT_AND_OFFSET_ULEB
# CHECK-NEXT: Imm:             1
# CHECK-NEXT: ExtraData:       [ 0x0 ]
# CHECK-NEXT: Opcode:          REBASE_OPCODE_DO_REBASE_IMM_TIMES
# CHECK-NEXT: Imm:             1
# CHECK-NEXT: Opcode:          REBASE_OPCODE_ADD_ADDR_IMM_SCALED
# CHECK-NEXT: Imm:             14
# CHECK-NEXT: Opcode:          REBASE_OPCODE_DO_REBASE_IMM_TIMES
# CHECK-NEXT: Imm:             3
# CHECK-NEXT: Opcode:          REBASE_OPCODE_ADD_ADDR_ULEB
# CHECK-NEXT: Imm:             0
# CHECK-NEXT: ExtraData:       [ 0x78 ]
# CHECK-NEXT: Opcode:          REBASE_OPCODE_DO_REBASE_IMM_TIMES
# CHECK-NEXT: Imm:             1
# CHECK-NEXT: Opcode:          REBASE_OPCODE_DONE
# CHECK-NEXT: Imm:             0


.text
.globl _foo
_foo:

.data
.quad _foo
.space 112
.quad _foo
.quad _foo
.quad _foo
.space 120
.quad _foo
