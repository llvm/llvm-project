# REQUIRES: ppc

# RUN: llvm-mc -filetype=obj -triple=powerpc64le-unknown-linux %s -o %t.o
# RUN: ld.lld --defsym callee=0x12010010 --defsym tail_callee=0x12010020 \
# RUN:   -z separate-code %t.o -o %t
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s
# RUN: ld.lld --defsym callee=0x12010010 --defsym tail_callee=0x12010020 \
# RUN:   -z separate-code %t.o -o %t
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s
# RUN: ld.lld --defsym callee=0xE010014 --defsym tail_callee=0xE010024 \
# RUN:   -z separate-code %t.o -o %t
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck --check-prefix=NEGOFFSET  %s
# RUN: ld.lld --defsym callee=0x12010018 --defsym tail_callee=0x12010028 \
# RUN:   -z separate-code %t.o -o %t
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck --check-prefix=THUNK %s
# RUN: llvm-readelf --sections %t | FileCheck --check-prefix=BRANCHLT %s
# RUN: not ld.lld --defsym callee=0x1001002D --defsym tail_callee=0x1001002F \
# RUN:   -z separate-code %t.o -o %t 2>&1 | FileCheck --check-prefix=MISSALIGNED %s

# RUN: llvm-mc -filetype=obj -triple=powerpc64-unknown-linux %s -o %t.o
# RUN: ld.lld --defsym callee=0x12010010 --defsym tail_callee=0x12010020 \
# RUN:   -z separate-code %t.o -o %t
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s
# RUN: ld.lld --defsym callee=0x12010010 --defsym tail_callee=0x12010020 \
# RUN:   -z separate-code %t.o -o %t
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s
# RUN: ld.lld --defsym callee=0xE010014 --defsym tail_callee=0xE010024 \
# RUN:   -z separate-code %t.o -o %t
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck --check-prefix=NEGOFFSET  %s
# RUN: ld.lld --defsym callee=0x12010018 --defsym tail_callee=0x12010028 \
# RUN:   -z separate-code %t.o -o %t
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck --check-prefix=THUNK %s
# RUN: llvm-readelf --sections %t | FileCheck --check-prefix=BRANCHLT %s
# RUN: not ld.lld --defsym callee=0x1001002D --defsym tail_callee=0x1001002F \
# RUN:   -z separate-code %t.o -o %t 2>&1 | FileCheck --check-prefix=MISSALIGNED %s

# MISSALIGNED: ld.lld: error: {{.*}}.o:(.text+0x14): improper alignment for relocation R_PPC64_REL24: 0x19 is not aligned to 4 bytes
# MISSALIGNED: ld.lld: error: {{.*}}.o:(.text+0x24): improper alignment for relocation R_PPC64_REL24: 0xB is not aligned to 4 bytes

        .global test
        .p2align        4
        .type   test,@function
test:
.Lgep:
        addis 2, 12, .TOC.-.Lgep@ha
        addi  2, 2,  .TOC.-.Lgep@l
.Llep:
        .localentry test, .Llep-.Lgep
        mflr 0
        std 0, 16(1)
        stdu 1, 32(1)
        bl callee
        addi 1, 1, 32
        ld 0, 16(1)
        mtlr 0
        b tail_callee

# Check that we are branching to the definitions, and not range-extending
# thunks.
# CHECK-LABEL: test
# CHECK:  10010014:       bl 0x12010010
# CHECK:  10010024:       b  0x12010020

# NEGOFFSET-LABEL: test
# NEGOFFSET:  10010014:       bl 0xe010014
# NEGOFFSET:  10010024:       b  0xe010024

# THUNK-LABEL: <test>:
# THUNK: 10010014:       bl 0x10010030
# THUNK: 10010024:       b 0x10010050

# .branch_lt[0]
# THUNK-LABEL: <__long_branch_callee>:
# THUNK-NEXT: 10010030:       addis 12, 2, 1
# THUNK-NEXT:                 ld 12, -32760(12)
# THUNK-NEXT:                 mtctr 12
# THUNK-NEXT:                 bctr

# .branch_lt[1]
# THUNK-LABEL: <__long_branch_tail_callee>:
# THUNK-NEXT: 10010050:       addis 12, 2, 1
# THUNK-NEXT:                 ld 12, -32752(12)
# THUNK-NEXT:                 mtctr 12
# THUNK-NEXT:                 bctr

# The offset from the TOC to the .branch_lt section  is (-1 << 16) - 32768.
#                Name             Type            Address          Off    Size
# BRANCHLT:     .got              PROGBITS        0000000010020000 020000 000008
# BRANCHLT:     .branch_lt        PROGBITS        0000000010030008 020008 000010
# BRANCHLT-NOT: .plt
