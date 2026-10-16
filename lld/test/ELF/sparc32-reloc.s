# REQUIRES: sparc
# RUN: llvm-mc -filetype=obj -triple=sparc %s -o %t.o
# RUN: ld.lld -m elf32_sparc -Ttext=0xf0004000 %t.o -o %t
# RUN: llvm-objdump -d -j .text --no-show-raw-insn --no-print-imm-hex %t | FileCheck %s
# RUN: llvm-readelf -x .data %t | FileCheck %s --check-prefix=DATA

## The relocated value is sign-extended to the ABI width, so on the 32-bit ABI
## an address with bit 31 set arrives with every high bit set. The unsigned
## range checks must read it as a 32-bit value rather than an overflow.
# CHECK:      sethi 3932176, %g1
# CHECK-NEXT: or %g1, 8, %g1

# DATA: 0x{{[0-9a-f]+}} f0004008

## EM_SPARC is the 32-bit big-endian ABI, so an object claiming it with another
## class or byte order is malformed.
# RUN: yaml2obj %S/Inputs/sparc-elf64.yaml -o %t64.o
# RUN: not ld.lld %t64.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=BADCLASS

# BADCLASS: error: {{.*}}64.o is incompatible

.globl _start
_start:
  sethi %hi(_start+8), %g1
  or %g1, %lo(_start+8), %g1

.data
.word _start + 8
