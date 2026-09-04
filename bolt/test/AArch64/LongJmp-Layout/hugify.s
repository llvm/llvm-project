## Check the extra alignment between the main and cold code sections inserted
## for --hugify.

# REQUIRES: system-linux, asserts, bolt-runtime, target=aarch64{{.*}}

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags %t.o -o %t.exe -nostdlib -Wl,-q
# RUN: link_fdata --no-lbr %s %t.exe %t.fdata
# RUN: llvm-bolt %t.exe -o %t.bolt --data %t.fdata --lite=0 --hugify \
# RUN:   --debug-only=longjmp 2>&1 | FileCheck %s
# RUN: llvm-readelf -S %t.bolt | FileCheck %s --check-prefix=HUGIFY-SECTIONS

# CHECK: BOLT-DEBUG: LongJmp: layout iteration 1
# CHECK: BOLT-DEBUG: LongJmp layout starts at 0x
# CHECK: BOLT-DEBUG: LongJmp layout: section .text starts at 0x
# CHECK: BOLT-DEBUG: LongJmp layout: main fragment
# CHECK: BOLT-DEBUG: LongJmp layout: basic block
# CHECK: BOLT-DEBUG: LongJmp layout: section .text ends at 0x

# HUGIFY-SECTIONS: .text             PROGBITS {{[0-9a-f]+}}00000
# HUGIFY-SECTIONS: .text.cold        PROGBITS {{[0-9a-f]+}}00000

  .text
  .globl _start
  .type _start, %function
_start:
.entry_start:
# FDATA: 1 _start #.entry_start# 10
  bl hot
  ret
  .size _start, .-_start

  .globl hot
  .type hot, %function
hot:
.entry_hot:
# FDATA: 1 hot #.entry_hot# 10
  ret
  .size hot, .-hot

  .globl cold_function
  .type cold_function, %function
cold_function:
  add x0, x0, #1
  ret
  .size cold_function, .-cold_function

  .reloc 0, R_AARCH64_NONE
