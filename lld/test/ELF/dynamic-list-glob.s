# REQUIRES: x86
## --dynamic-list entries may be wildcards. Entries need not match a symbol.
## extern "C++" entries match demangled names, which for a non-mangled symbol
## is the name itself.

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o

# RUN: echo '{ [fb]o?1*; };' > %t.list
# RUN: ld.lld -pie --dynamic-list %t.list %t.o -o %t
# RUN: llvm-readelf --dyn-syms %t | FileCheck %s
# CHECK:      UND
# CHECK-NEXT: [[#]] boo1
# CHECK-NEXT: [[#]] foo1
# CHECK-NEXT: [[#]] foo11
# CHECK-NOT:  {{.}}

## Both mangled and unmangled names may appear.
# RUN: echo '{ _Z1fv; extern "C++" { "g()"; h*; }; };' > %t.list
# RUN: ld.lld -pie --dynamic-list %t.list %t.o -o %t
# RUN: llvm-readelf --dyn-syms %t | FileCheck %s --check-prefix=CXX
# CXX:      UND
# CXX-NEXT: [[#]] _Z1fv
# CXX-NEXT: [[#]] _Z1gv
# CXX-NEXT: [[#]] _Z1hv
# CXX-NEXT: [[#]] _Z2hhv
# CXX-NEXT: [[#]] hello
# CXX-NOT:  {{.}}

.globl _start, boo1, foo1, foo11, foo2, _Z1fv, _Z1gv, _Z1hv, _Z2hhv, _Z1iv, hello
_start:
boo1:
foo1:
foo11:
foo2:
_Z1fv:
_Z1gv:
_Z1hv:
_Z2hhv:
_Z1iv:
hello:
