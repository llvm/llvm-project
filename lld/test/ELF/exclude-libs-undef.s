# REQUIRES: x86
## Test we don't assign VER_NDX_LOCAL to an undefined symbol.
## If we do, an undefined weak will become non-preemptible,
## and we will report an error when an R_PLT_PC (optimized to R_PC)
## references the undefined weak (considered absolute).

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: rm -f %t.a
# RUN: llvm-ar rc %t.a %t.o
# RUN: ld.lld -shared --whole-archive --exclude-libs=ALL %t.a -o %t.so
# RUN: llvm-readelf --dyn-syms %t.so | FileCheck %s

# CHECK:     1: {{.*}} WEAK DEFAULT UND bar
# CHECK-NOT: 2:

.globl foo
.weak bar
foo:
  call bar
