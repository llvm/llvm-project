# REQUIRES: mips
# Check R_MIPS_64 relocation calculation.

# RUN: llvm-mc -filetype=obj -triple=mips64-unknown-linux %s -o %t-be.o
# RUN: ld.lld -shared %t-be.o -o %t-be.so
# RUN: llvm-objdump -s -t %t-be.so | FileCheck --check-prefixes=SYM,SYM-BE %s
# RUN: llvm-readelf --dynamic-table -r -s -A %t-be.so | FileCheck %s

# RUN: llvm-mc -filetype=obj -triple=mips64el-unknown-linux %s -o %t-el.o
# RUN: ld.lld -shared %t-el.o -o %t-el.so
# RUN: llvm-objdump -s -t %t-el.so | FileCheck --check-prefixes=SYM,SYM-EL %s
# RUN: llvm-readelf --dynamic-table -r -s -A %t-el.so | FileCheck %s

  .data
  .globl v2
v1:
  .quad v2+8 # R_MIPS_64 target v2 addend 8
v2:
  .quad v1   # R_MIPS_64 target v1 addend 0

# SYM: SYMBOL TABLE:
# SYM: 00000000000203b0 l .data  0000000000000000 v1

# SYM-BE: Contents of section .data:
# SYM-BE-NEXT:  {{.*}} 00000000 00000008 00000000 000203b0

# SYM-EL: Contents of section .data:
# SYM-EL-NEXT:  {{.*}} 08000000 00000000 b0030200 00000000

# CHECK: Dynamic section
# CHECK: (RELSZ)   32 (bytes)
# CHECK: (RELENT)  16 (bytes)

# CHECK: Relocation section
# CHECK:      [[V1:[0-9a-f]+]]  {{.*}} R_MIPS_REL32/R_MIPS_64/R_MIPS_NONE [[V2:[0-9a-f]+]] v2
# CHECK-NEXT: [[V2]]            {{.*}} R_MIPS_REL32/R_MIPS_64/R_MIPS_NONE {{$}}

# CHECK: Symbol table '.symtab'
# CHECK: {{.*}}: [[V1]]  {{.*}}  v1
# CHECK: {{.*}}: [[V2]]  {{.*}}  v2

# CHECK: Primary GOT:
# CHECK:  Global entries:
# CHECK:   {{.*}} -32736(gp) [[V2]] [[V2]] {{.*}} v2
