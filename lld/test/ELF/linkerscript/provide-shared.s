# REQUIRES: x86
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 a.s -o a.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 b.s -o b.o
# RUN: ld.lld b.o -o b.so -shared -soname=b.so
# RUN: ld.lld -T a.t a.o b.so -o a
# RUN: llvm-readelf -s a | FileCheck  %s

# CHECK: 000000000000002a 0 NOTYPE GLOBAL DEFAULT ABS f1
# CHECK: 000000000000002a 0 NOTYPE GLOBAL DEFAULT ABS f2

#--- a.s
.global _start
_start:
.quad f1

#--- b.s
.global f1
.size f1, 8
.type f1, @object
f1:
.quad 42

.global f2
.data
.dc.a f2

#--- a.t
SECTIONS {
  . = . + SIZEOF_HEADERS;
  PROVIDE(f1 = 42);
  PROVIDE(f2 = 42);
}
