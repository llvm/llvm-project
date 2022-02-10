# REQUIRES: x86

# RUN: rm -rf %t && split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/main.s -o %t/main.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/call-foo.s -o %t/call-foo.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/bar.s -o %t/bar.o
# RUN: ld.lld -shared -soname=t.so %t/bar.o -o %t/bar.so
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/wrap.s -o %t/wrap.o
# RUN: ld.lld -shared --soname=fixed %t/wrap.o -o %t/wrap.so

## foo is defined, then referenced in another object file.
# RUN: ld.lld -shared %t/main.o %t/call-foo.o --wrap foo -o %t1.so
# RUN: llvm-readelf -r %t1.so | FileCheck %s --check-prefix=CHECK1

# CHECK1:      R_X86_64_JUMP_SLOT 0000000000000000 bar + 0
# CHECK1-NEXT: R_X86_64_JUMP_SLOT 0000000000000000 __wrap_foo + 0

## --no-allow-shlib-undefined errors because __real_foo is not defined.
# RUN: not ld.lld %t/main.o %t/bar.so -o /dev/null 2>&1 | FileCheck --check-prefix=ERR %s
# ERR: {{.*}}.so: undefined reference to __real_foo [--no-allow-shlib-undefined]

## --wrap=foo defines __real_foo.
# RUN: ld.lld %t/main.o %t/bar.so --wrap=foo -o %t2
# RUN: llvm-readelf --dyn-syms %t2 | FileCheck %s --check-prefix=CHECK2

## See wrap-plt2.s why __wrap_foo is retained.
# CHECK2:      Symbol table '.dynsym' contains 3 entries:
# CHECK2:      NOTYPE  LOCAL  DEFAULT  UND
# CHECK2-NEXT: NOTYPE  GLOBAL DEFAULT  UND bar
# CHECK2-NEXT: NOTYPE  GLOBAL DEFAULT  UND __wrap_foo

## __wrap_bar is undefined.
# RUN: ld.lld -shared %t/main.o --wrap=bar -o %t3.so
# RUN: llvm-readelf -r --dyn-syms %t3.so | FileCheck %s --check-prefix=CHECK3
# CHECK3:      R_X86_64_JUMP_SLOT 0000000000000000 __wrap_bar + 0
# CHECK3:      Symbol table '.dynsym' contains 4 entries:
# CHECK3:      NOTYPE  LOCAL  DEFAULT  UND
# CHECK3-NEXT: NOTYPE  GLOBAL DEFAULT  UND __wrap_bar
# CHECK3-NEXT: NOTYPE  GLOBAL DEFAULT    6 _start
# CHECK3-NEXT: NOTYPE  GLOBAL DEFAULT    6 foo

## __wrap_bar is defined in %t/wrap.so.
# RUN: ld.lld -shared %t/main.o %t/wrap.so --wrap=bar -o %t4.so
# RUN: llvm-readelf -r --dyn-syms %t4.so | FileCheck %s --check-prefix=CHECK4
# CHECK4:      R_X86_64_JUMP_SLOT {{.*}} __wrap_bar + 0
# CHECK4:      Symbol table '.dynsym' contains 4 entries:
# CHECK4:      NOTYPE  LOCAL  DEFAULT  UND
# CHECK4-NEXT: NOTYPE  GLOBAL DEFAULT  UND __wrap_bar
# CHECK4-NEXT: NOTYPE  GLOBAL DEFAULT    6 _start
# CHECK4-NEXT: NOTYPE  GLOBAL DEFAULT    6 foo

# RUN: ld.lld %t/main.o %t/wrap.so --wrap bar -o %t1
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

#--- call-foo.s
  call foo

#--- bar.s
.globl bar
bar:
  call __real_foo

#--- wrap.s
.globl __wrap_bar
__wrap_bar:
  retq
