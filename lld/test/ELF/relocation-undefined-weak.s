# REQUIRES: x86

## On some targets (e.g. ARM, AArch64, and PPC), PC relative relocations to
## weak undefined symbols resolve to special positions. On many others
## the target symbols as treated as VA 0. Absolute relocations are always 
## resolved as VA 0.

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: ld.lld %t.o -o %t

# RUN: llvm-objdump -d --no-show-raw-insn --print-imm-hex %t | FileCheck %s --check-prefix=TEXT
# TEXT: 201158: movl $0x1, -0x201162(%rip)

# RUN: llvm-readelf -r --hex-dump=.data %t | FileCheck %s --check-prefix=DATA
# DATA: Hex dump of section '.data':
# DATA-NEXT: {{.*}} 00000000 00000000 
# DATA-EMPTY:

# RUN: llvm-readelf -r %t | FileCheck %s --check-prefix=NORELCS
# NORELCS: no relocations

.global _start
_start:
  movl $1, sym1(%rip)

.data
.weak sym1
.quad sym1
