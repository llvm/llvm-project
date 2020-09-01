# REQUIRES: x86

# RUN: split-file %s %t.dir
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t.dir/main.s -o %t.o
# RUN: echo '.globl bar; bar: call __real_foo' | llvm-mc -filetype=obj -triple=x86_64 - -o %t1.o
# RUN: ld.lld -shared -soname=t.so %t1.o -o %t.so

## --no-allow-shlib-undefined errors because __real_foo is not defined.
# RUN: not ld.lld %t.o %t.so -o /dev/null 2>&1 | FileCheck --check-prefix=ERR %s
# ERR: {{.*}}.so: undefined reference to __real_foo [--no-allow-shlib-undefined]

## --wrap=foo defines __real_foo.
# RUN: ld.lld %t.o %t.so --wrap=foo -o %t
# RUN: llvm-readelf --dyn-syms %t | FileCheck %s

## The reference __real_foo from %t.so causes foo to be exported.
## __wrap_foo is not used, thus not exported.
# CHECK:      Symbol table '.dynsym' contains 3 entries:
# CHECK:      NOTYPE  LOCAL  DEFAULT  UND
# CHECK-NEXT: NOTYPE  GLOBAL DEFAULT  UND bar
# CHECK-NEXT: NOTYPE  GLOBAL DEFAULT    6 foo

# RUN: llvm-mc -filetype=obj -triple=x86_64 %t.dir/wrap.s -o %twrap.o
# RUN: ld.lld -shared --soname=fixed %twrap.o -o %twrap.so
# RUN: ld.lld %t.o %twrap.so --wrap bar -o %t1
# RUN: llvm-readelf --dyn-syms %t1 | FileCheck %s --check-prefix=DYNSYM
# RUN: llvm-objdump -d %t1 | FileCheck %s --check-prefix=ASM

# DYNSYM:      Symbol table '.dynsym' contains 2 entries:
# DYNSYM:      NOTYPE  LOCAL  DEFAULT  UND
# DYNSYM-NEXT: NOTYPE  GLOBAL DEFAULT  UND __wrap_bar

# ASM:      <_start>:
# ASM-NEXT:   callq {{.*}} <__wrap_bar@plt>

#--- main.s
.globl _start, foo
_start:
  call bar
foo:

#--- wrap.s
.globl __wrap_bar
__wrap_bar:
  retq
