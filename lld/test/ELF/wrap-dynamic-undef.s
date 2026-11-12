# REQUIRES: x86
# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/a.s -o %t/a.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/def.s -o %t/def.o
# RUN: ld.lld %t/def.o -o %t/def.so -shared --soname=def
# RUN: ld.lld %t/a.o %t/def.so -o %t1 --wrap foo
# RUN: llvm-readelf --dyn-syms %t1 | FileCheck %s

# Test that the dynamic relocation uses foo. We used to produce a
# relocation with __real_foo.

# CHECK:      Symbol table '.dynsym' contains 2 entries:
# CHECK:      NOTYPE  LOCAL  DEFAULT  UND
# CHECK-NEXT: NOTYPE  GLOBAL DEFAULT  UND foo

# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/b.s -o %t/b.o
# RUN: ld.lld -shared --wrap foo %t/b.o -o %t2.so
# RUN: llvm-readelf --dyn-syms %t2.so | FileCheck %s --check-prefix=SYM2

# SYM2:      Symbol table '.dynsym' contains 4 entries:
# SYM2:      NOTYPE  LOCAL  DEFAULT   UND
# SYM2-NEXT: NOTYPE  WEAK   DEFAULT   UND foo
# SYM2-NEXT: NOTYPE  WEAK   DEFAULT   UND __wrap_foo
# SYM2-NEXT: NOTYPE  GLOBAL DEFAULT [[#]] _start

#--- a.s
.global _start
_start:
	callq	__real_foo@plt

#--- def.s
.globl foo
foo:

#--- b.s
.weak foo
.weak __real_foo
.global _start
_start:
  call __real_foo@plt
  call foo@plt
