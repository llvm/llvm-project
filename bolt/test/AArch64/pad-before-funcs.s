# Test checks that --pad-before-funcs is working as expected.
# It should be able to introduce a configurable offset for the _start symbol.
# It should reject requests which don't obey the code alignment requirement.

# Tests check inserting padding before _start; and additionally a test where
# padding is inserted after start. In each case, check that the following
# symbol ends up in the expected place as well.


# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q -Wl,--image-base=0x3000,--section-start=.text=0x4000
# RUN: llvm-bolt %t.exe -o %t.bolt.0 --pad-funcs-before=_start:0
# RUN: llvm-bolt %t.exe -o %t.bolt.4 --pad-funcs-before=_start:4
# RUN: llvm-bolt %t.exe -o %t.bolt.8 --pad-funcs-before=_start:8
# RUN: llvm-bolt %t.exe -o %t.bolt.4.4 --pad-funcs-before=_start:4 --pad-funcs=_start:4
# RUN: llvm-bolt %t.exe -o %t.bolt.4.8 --pad-funcs-before=_start:4 --pad-funcs=_start:8

# RUN: not llvm-bolt %t.exe -o %t.bolt.1 --pad-funcs-before=_start:1 2>&1 | FileCheck --check-prefix=CHECK-BAD-ALIGN %s

# CHECK-BAD-ALIGN: user-requested 1 padding bytes before function _start(*2) is not a multiple of the minimum function alignment (4).

# RUN: llvm-objdump --section=.text --disassemble %t.bolt.0 | FileCheck --check-prefix=CHECK-0 %s
# RUN: llvm-objdump --section=.text --disassemble %t.bolt.4 | FileCheck --check-prefix=CHECK-4 %s
# RUN: llvm-objdump --section=.text --disassemble %t.bolt.8 | FileCheck --check-prefix=CHECK-8 %s
# RUN: llvm-objdump --section=.text --disassemble %t.bolt.4.4 | FileCheck --check-prefix=CHECK-4-4 %s
# RUN: llvm-objdump --section=.text --disassemble %t.bolt.4.8 | FileCheck --check-prefix=CHECK-4-8 %s

# Trigger relocation mode in bolt.
.reloc 0, R_AARCH64_NONE

.section .text

# CHECK-0: 0000000000400000 <_start>
# CHECK-4: 0000000000400004 <_start>
# CHECK-4-4: 0000000000400004 <_start>
# CHECK-8: 0000000000400008 <_start>
.globl _start
_start:
    ret

# CHECK-0: 0000000000400004 <_subsequent>
# CHECK-4: 0000000000400008 <_subsequent>
# CHECK-4-4: 000000000040000c <_subsequent>
# CHECK-4-8: 0000000000400010 <_subsequent>
# CHECK-8: 000000000040000c <_subsequent>
.globl _subsequent
_subsequent:
    ret
