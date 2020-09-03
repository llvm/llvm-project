# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o

# RUN: ld.lld --dynamic-linker foo %t.o -o %t
# RUN: llvm-readelf --program-headers --section-headers %t | FileCheck --implicit-check-not=.bss %s

# RUN: ld.lld --dynamic-linker=foo %t.o -o %t
# RUN: llvm-readelf --program-headers --section-headers %t | FileCheck --implicit-check-not=.bss %s

# CHECK: [Requesting program interpreter: foo]

# RUN: ld.lld %t.o -o %t
# RUN: llvm-readelf -S -l %t | FileCheck --check-prefix=NO %s

# RUN: ld.lld --dynamic-linker foo --no-dynamic-linker %t.o -o %t
# RUN: llvm-readelf -S -l %t | FileCheck --check-prefix=NO %s

## {clang,gcc} -nostdlib -r passes --dynamic-linker, and the expected behavior is to ignore it.
# RUN: ld.lld -r --dynamic-linker foo %t.o -o %t
# RUN: llvm-readelf -S -l %t | FileCheck --check-prefix=NO %s

# NO-NOT: .interp
# NO-NOT: PT_INTERP

.globl _start
_start:
  nop
