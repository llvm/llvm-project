# REQUIRES: mips
# Check R_MIPS_64 relocation calculation.

# RUN: llvm-mc -filetype=obj -triple=mips64-unknown-linux %s -o %t.o
# RUN: ld.lld -shared %t.o -o %t.so
# RUN: llvm-objdump -s -t %t.so | FileCheck -check-prefix=SYM %s
# RUN: llvm-readelf -r -s --dynamic-table -A %t.so | FileCheck %s

  .data
  .globl v2
v1:
  .quad v2+8 # R_MIPS_64 target v2 addend 8
v2:
  .quad v1   # R_MIPS_64 target v1 addend 0

# SYM: SYMBOL TABLE:
# SYM: 00000000[[V1:[0-9a-f]+]]  .data  00000000 v1

# SYM: Contents of section .data:
# SYM-NEXT:  {{.*}} 00000000 00000008 00000000 [[V1]]

# CHECK: Relocation section
# CHECK: [[V2:[0-9a-f]+]]  {{.*}} R_MIPS_REL32/R_MIPS_64/R_MIPS_NONE
# CHECK: [[V1:[0-9a-f]+]]  {{.*}} R_MIPS_REL32/R_MIPS_64/R_MIPS_NONE [[V2]] v2

# CHECK: Symbol table '.symtab'
# CHECK: {{.*}}: [[V1]]  {{.*}}  v1
# CHECK: {{.*}}: [[V2]]  {{.*}}  v2

# CHECK: Dynamic section
# CHECK: (RELSZ)   32 (bytes)
# CHECK: (RELENT)  16 (bytes)

# CHECK: Primary GOT:
# CHECK:  Global entries:
# CHECK:   {{.*}} -32736(gp) [[V2]] [[V2]] {{.*}} v2
