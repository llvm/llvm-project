# REQUIRES: x86

# RUN: echo "{ global: foo*; bar*; local: *; };" > %t.script
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
# RUN: ld.lld -shared --version-script %t.script %t.o -o %t.so
# RUN: llvm-readelf --dyn-syms %t.so | FileCheck %s

.globl foo1, bar1, zed1, local
foo1:
bar1:
zed1:
local:

# CHECK:      foo1{{$}}
# CHECK-NEXT: bar1{{$}}
# CHECK-NOT:  {{.}}

# RUN: echo "{ global : local; local: *; };" > %t1.script
# RUN: ld.lld -shared --version-script %t1.script %t.o -o %t.so
# RUN: llvm-readelf --dyn-syms %t.so | FileCheck --check-prefix=LOCAL %s

# LOCAL:     local{{$}}
# LOCAL-NOT: {{.}}
