# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: echo '.globl bar; bar: call __real_foo' | llvm-mc -filetype=obj -triple=x86_64 - -o %t1.o
# RUN: ld.lld -shared -soname=t.so %t1.o -o %t.so

## --no-allow-shlib-undefined errors because __real_foo is not defined.
# RUN: not ld.lld %t.o %t.so -o /dev/null 2>&1 | FileCheck --check-prefix=ERR %s
# ERR: {{.*}}.so: undefined reference to __real_foo [--no-allow-shlib-undefined]

## --wrap=foo defines __real_foo.
# RUN: ld.lld %t.o %t.so --wrap=foo -o %t
# RUN: llvm-readelf --dyn-syms %t | FileCheck %s

## FIXME GNU ld does not export __wrap_foo
## The reference __real_foo from %t.so causes foo to be exported.
# CHECK:      Symbol table '.dynsym' contains 4 entries:
# CHECK:      NOTYPE  LOCAL  DEFAULT  UND
# CHECK-NEXT: NOTYPE  GLOBAL DEFAULT  UND bar
# CHECK-NEXT: NOTYPE  GLOBAL DEFAULT  UND __wrap_foo
# CHECK-NEXT: NOTYPE  GLOBAL DEFAULT    6 foo

.globl _start, foo
_start:
  call bar
foo:
