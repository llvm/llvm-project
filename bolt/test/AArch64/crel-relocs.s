## Check that BOLT accepts relocation mode when code relocations are stored
## in SHT_CREL.

# RUN: %clang %cflags -Wno-unused-command-line-argument -fuse-ld=lld \
# RUN:   -Wl,--emit-relocs -Wa,--allow-experimental-crel,--crel %s -o %t.exe
# RUN: llvm-readelf -S %t.exe | FileCheck %s --check-prefix=SECTIONS
# RUN: llvm-bolt %t.exe -o %t.bolt -relocs --print-cfg \
# RUN:   --print-relocations 2>&1 | FileCheck %s
# RUN: llvm-readelf -S %t.bolt | FileCheck %s --check-prefix=OUTPUT

# SECTIONS: .crel.text

# CHECK: BOLT-INFO: enabling relocation mode
# CHECK: R_AARCH64_ADR_PREL_PG_HI21
# CHECK: R_AARCH64_ADD_ABS_LO12_NC

# OUTPUT: .text
# OUTPUT-NOT: .crel.text

  .text
  .globl _start
  .type _start, %function
_start:
  adrp x0, target
  add x0, x0, :lo12:target
  ret
  .size _start, .-_start

  .globl target
  .type target, %object
  .data
target:
  .xword 0
  .size target, .-target
