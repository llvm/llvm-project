# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %p/Inputs/shared.s -o %t2.o
# RUN: ld.lld -shared -soname=t2 %t2.o -o %t2.so

## -z mark-plt emits DT_X86_64_PLT, DT_X86_64_PLTSZ, DT_X86_64_PLTENT.
## PLT layout: 16-byte header + 1 entry * 16 bytes = 0x20 bytes total.
# RUN: ld.lld %t.o %t2.so -z mark-plt -z now -o %t
# RUN: llvm-readelf --dynamic-table %t | FileCheck %s --check-prefix=MARK
# RUN: llvm-readelf --relocs %t | FileCheck %s --check-prefix=RELA

## Without -z mark-plt: no tags and zero addend on JUMP_SLOT.
# RUN: ld.lld %t.o %t2.so -z now -o %t.nomark
# RUN: llvm-readelf --dynamic-table %t.nomark | FileCheck %s --check-prefix=NOMARK
# RUN: llvm-readelf --relocs %t.nomark | FileCheck %s --check-prefix=NORELA

## -z mark-plt also works for shared libraries.
# RUN: ld.lld -shared %t.o %t2.so -z mark-plt -z now -o %t.so
# RUN: llvm-readelf --dynamic-table %t.so | FileCheck %s --check-prefix=MARK

# MARK:     0x0000000070000000 (X86_64_PLT)   {{0x[0-9a-f]+}}
# MARK:     0x0000000070000001 (X86_64_PLTSZ)  0x20
# MARK:     0x0000000070000003 (X86_64_PLTENT) 0x10

# NOMARK-NOT: (X86_64_PLT)
# NOMARK-NOT: (X86_64_PLTSZ)
# NOMARK-NOT: (X86_64_PLTENT)

## With -z mark-plt, R_X86_64_JUMP_SLOT addend is the PLT entry VA (non-zero).
# RELA:     R_X86_64_JUMP_SLOT {{.*}} bar + {{[1-9a-f][0-9a-f]+}}

## Without -z mark-plt, R_X86_64_JUMP_SLOT addend is 0.
# NORELA:   R_X86_64_JUMP_SLOT {{.*}} bar + 0

.globl _start
_start:
  call bar@plt
