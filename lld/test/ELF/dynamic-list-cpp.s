# REQUIRES: x86

## Confirm both mangled and unmangled names may appear in
## the --dynamic-list file.

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o

# RUN: echo '{ _Z1fv; extern "C++" { "g()"; }; };' > %t.list
# RUN: ld.lld -pie --dynamic-list %t.list %t.o -o %t
# RUN: llvm-readelf --dyn-syms %t | FileCheck %s

# CHECK:      Symbol table '.dynsym' contains 3 entries:
# CHECK:      _Z1fv
# CHECK-NEXT: _Z1gv

.globl _Z1fv, _Z1gv
_Z1fv:
_Z1gv:
