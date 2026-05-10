## Check that BOLT accepts relocation mode when code relocations are stored
## in SHT_CREL.

# RUN: %clang %cflags -Wno-unused-command-line-argument -fuse-ld=lld \
# RUN:   -Wl,--emit-relocs -Wa,--allow-experimental-crel,--crel %s -o %t.exe
# RUN: llvm-readelf -S %t.exe | FileCheck %s --check-prefix=SECTIONS
# RUN: llvm-bolt %t.exe -o %t.bolt -relocs 2>&1 | FileCheck %s
# RUN: llvm-readelf -S %t.bolt | FileCheck %s --check-prefix=OUTPUT

# SECTIONS: .crel.text

# CHECK: BOLT-INFO: enabling relocation mode

# OUTPUT: .text
# OUTPUT-NOT: .crel.text

  .text
  .globl _start
  .type _start, %function
_start:
  bl target
  ret
  .size _start, .-_start

  .globl target
  .type target, %function
target:
  ret
  .size target, .-target
