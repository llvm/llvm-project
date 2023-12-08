# REQUIRES: mips
# Check the primary GOT cannot be made to overflow

# RUN: llvm-mc -filetype=obj -triple=mips64-unknown-linux \
# RUN:         %p/Inputs/mips-64-got-load.s -o %t1.so.o
# RUN: llvm-mc -filetype=obj -triple=mips64-unknown-linux %s -o %t2.so.o
# RUN: ld.lld -shared -mips-got-size 32 %t1.so.o %t2.so.o -o %t-sgot.so
# RUN: ld.lld -shared -mips-got-size 24 %t1.so.o %t2.so.o -o %t-mgot.so
# RUN: llvm-readelf -s -A %t-sgot.so | FileCheck -check-prefixes=SYM,SGOT %s
# RUN: llvm-readelf -s -A %t-mgot.so | FileCheck -check-prefixes=SYM,MGOT %s

# SYM: Symbol table '.symtab'
# SYM: {{.*}}: [[LOC1:[0-9a-f]+]]  {{.*}} local1
# SYM: {{.*}}: [[LOC2:[0-9a-f]+]]  {{.*}} local2

# SGOT:      Primary GOT:
# SGOT-NEXT:  Canonical gp value:
# SGOT-EMPTY:
# SGOT-NEXT:  Reserved entries:
# SGOT-NEXT:  Address     Access          Initial Purpose
# SGOT-NEXT:   {{.*}} -32752(gp) 0000000000000000 Lazy resolver
# SGOT-NEXT:   {{.*}} -32744(gp) 8000000000000000 Module pointer (GNU extension)
# SGOT-EMPTY:
# SGOT-NEXT:  Local entries:
# SGOT-NEXT:  Address     Access          Initial
# SGOT-NEXT:   {{.*}} -32736(gp) [[LOC1]]
# SGOT-NEXT:   {{.*}} -32728(gp) [[LOC2]]

# MGOT:      Primary GOT:
# MGOT-NEXT:  Canonical gp value:
# MGOT-EMPTY:
# MGOT-NEXT:  Reserved entries:
# MGOT-NEXT:  Address     Access          Initial Purpose
# MGOT-NEXT:   {{.*}} -32752(gp) 0000000000000000 Lazy resolver
# MGOT-NEXT:   {{.*}} -32744(gp) 8000000000000000 Module pointer (GNU extension)
# MGOT-EMPTY:
# MGOT-NEXT:  Local entries:
# MGOT-NEXT:  Address     Access          Initial
# MGOT-NEXT:   {{.*}} -32736(gp) [[LOC1]]
# MGOT-EMPTY:
# MGOT-NEXT:  Number of TLS and multi-GOT entries 1

  .text
  .global foo2
foo2:
  ld $2, %got_disp(local2)($gp)

  .bss
local2:
  .word 0
