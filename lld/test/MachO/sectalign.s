# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o

# RUN: not %lld -dylib -o %t %t.o -sectalign __TEXT __text asdf 2>&1 \
# RUN:     | FileCheck --check-prefix=NONUM -DNUM=asdf %s
# RUN: not %lld -dylib -o %t %t.o -sectalign __TEXT __text 0x0X4 2>&1 \
# RUN:     | FileCheck --check-prefix=NONUM -DNUM=0x0X4 %s
# NONUM: error: -sectalign: failed to parse '[[NUM]]' as number

# RUN: not %lld -dylib -o %t %t.o -sectalign __TEXT __text 16 2>&1 \
# RUN:     | FileCheck --check-prefix=NOPOW -DNUM=16 %s
# RUN: not %lld -dylib -o %t %t.o -sectalign __TEXT __text 0x16 2>&1 \
# RUN:     | FileCheck --check-prefix=NOPOW -DNUM=0x16 %s
# RUN: not %lld -dylib -o %t %t.o -sectalign __TEXT __text 0 2>&1 \
# RUN:     | FileCheck --check-prefix=NOPOW -DNUM=0 %s
# NOPOW:  error: -sectalign: '[[NUM]]' (in base 16) not a power of two

## __DATA_CONST tests that the flag applies to names after section renaming.
# RUN: %lld -dylib -o %t %t.o -sectalign __TEXT __text 20 \
# RUN:                        -sectalign __DATA_CONST __const 0x40
# RUN: llvm-readobj --section-headers %t \
# RUN:     | FileCheck -DSECT=__text -DSEG=__TEXT -DALIGN=5 %s
# RUN: llvm-readobj --section-headers %t \
# RUN:     | FileCheck -DSECT=__const -DSEG=__DATA_CONST -DALIGN=6 %s

# RUN: %lld -dylib -o %t %t.o -rename_section __TEXT __text __TxT __foo \
# RUN:                        -sectalign __TxT __foo 0x40
# RUN: llvm-readobj --section-headers %t \
# RUN:     | FileCheck -DSECT=__foo -DSEG=__TxT -DALIGN=6 %s

# CHECK:      Name: [[SECT]]
# CHECK-NEXT: Segment: [[SEG]]
# CHECK-NEXT: Address:
# CHECK-NEXT: Size:
# CHECK-NEXT: Offset:
# CHECK-NEXT: Alignment: [[ALIGN]]

.section __TEXT,__text
.space 1

.section __DATA,__const
.space 1
