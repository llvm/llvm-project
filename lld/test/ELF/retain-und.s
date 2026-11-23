# REQUIRES: x86
## An empty --retain-symbols-file behaves like a `{ local: *; }` version script,
## keeping the R_X86_64_64 dynamic relocation to the weak undefined foo.

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: echo > %t.retain
# RUN: echo '{ local: *; };' > %t.script
# RUN: ld.lld -shared --version-script %t.script %t.o -o %t1.so
# RUN: ld.lld -shared --retain-symbols-file %t.retain %t.o -o %t2.so
# RUN: llvm-readelf -r %t1.so | FileCheck %s
# RUN: llvm-readelf -r %t2.so | FileCheck %s

# CHECK:      Relocation section '.rela.dyn' {{.+}} contains 1
# CHECK:      R_X86_64_64 {{.*}} foo + 0

.data
.quad foo
.weak foo
