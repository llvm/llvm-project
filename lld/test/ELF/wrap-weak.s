# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: ld.lld -shared -o %t.so %t.o -wrap foo

# RUN: llvm-readelf --dyn-syms %t.so | FileCheck %s

# CHECK:      Symbol table '.dynsym' contains 4 entries:
# CHECK:      NOTYPE  LOCAL  DEFAULT   UND
# CHECK-NEXT: NOTYPE  WEAK   DEFAULT   UND foo
# CHECK-NEXT: NOTYPE  GLOBAL DEFAULT [[#]] __wrap_foo
# CHECK-NEXT: NOTYPE  GLOBAL DEFAULT [[#]] _start

.global foo
.weak __real_foo

.global __wrap_foo
__wrap_foo:
  movq __real_foo@gotpcrel(%rip), %rax
  call __real_foo@plt

.global _start
_start:
  call foo@plt
