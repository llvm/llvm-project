# Should print an expected message in case of conflict with an internally generated symbol.

# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux %s -o %t.o
# RUN: not ld.lld -shared %t.o -o %t.so 2>&1 | FileCheck %s

# CHECK: duplicate symbol: _gp in (internal) and {{.*}}

# REQUIRES: mips

  .globl  _gp
_gp = 0
