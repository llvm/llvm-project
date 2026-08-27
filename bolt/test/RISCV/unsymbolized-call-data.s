// Test that data following AUIPC is not decoded as the JALR half of a
// relocation-free call pair.

// RUN: llvm-mc -triple riscv64 -mattr=-relax -filetype=obj -o %t.o %s
// RUN: ld.lld --no-relax --emit-relocs -e _start -o %t %t.o
// RUN: llvm-readelf --symbols %t | FileCheck --check-prefix=MAPPING %s
// RUN: llvm-bolt --print-cfg --print-only=_start -o %t.bolt %t \
// RUN:     | FileCheck --check-prefix=BOLT %s

// MAPPING: $x
// MAPPING: $d
// MAPPING: $x

// BOLT-LABEL: Binary Function "_start" after building cfg {
// BOLT:       auipc ra, 0x0
// BOLT-NOT:   auipc ra, {{.*}}_start

  .text
  .option norvc
  .option norelax

  .globl _start
  .type _start,@function
_start:
  auipc ra, 0
  // This data has the encoding of "jalr ra, 0(ra)". The $d mapping symbol
  // must prevent the symbolizer from treating it as an instruction.
  .word 0x000080e7
  ret
  .size _start, .-_start

  // Retain a relocation so BOLT processes the executable in relocation mode.
  .globl relocated_call
  .type relocated_call,@function
relocated_call:
  call _start
  ret
  .size relocated_call, .-relocated_call
