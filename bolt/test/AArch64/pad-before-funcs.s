# Test checks that --pad-before-funcs is working as expected.
# It should be able to introduce a configurable offset for the _start symbol.
# It should reject requests which don't obey the code alignment requirement.

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q -Wl,--section-start=.text=0x4000
# RUN: llvm-bolt %t.exe -o %t.bolt.0 --pad-funcs-before=_start:0
# RUN: llvm-bolt %t.exe -o %t.bolt.4 --pad-funcs-before=_start:4
# RUN: llvm-bolt %t.exe -o %t.bolt.8 --pad-funcs-before=_start:8

# RUN: not llvm-bolt %t.exe -o %t.bolt.8 --pad-funcs-before=_start:1 2>&1 | FileCheck --check-prefix=CHECK-BAD-ALIGN %s

# CHECK-BAD-ALIGN: user-requested 1 padding bytes before function _start(*2) is not a multiple of the minimum function alignment (4).

# RUN: llvm-objdump --section=.text --disassemble %t.bolt.0 | FileCheck --check-prefix=CHECK-0 %s
# RUN: llvm-objdump --section=.text --disassemble %t.bolt.4 | FileCheck --check-prefix=CHECK-4 %s
# RUN: llvm-objdump --section=.text --disassemble %t.bolt.8 | FileCheck --check-prefix=CHECK-8 %s

# Trigger relocation mode in bolt.
.reloc 0, R_AARCH64_NONE

.section .text
.globl _start

# CHECK-0: 0000000000400000 <_start>
# CHECK-4: 0000000000400004 <_start>
# CHECK-8: 0000000000400008 <_start>
_start:
    ret
