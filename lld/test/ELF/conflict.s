# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t1.o
# RUN: not ld.lld %t1.o %t1.o -o %t2 2>&1 | FileCheck -check-prefix=DEMANGLE %s

# DEMANGLE:    duplicate symbol: {{mul\(double, double\)|_Z3muldd}} in
# DEMANGLE:    duplicate symbol: foo in

# RUN: not ld.lld %t1.o %t1.o -o %t2 --no-demangle 2>&1 | \
# RUN:   FileCheck -check-prefix=NO_DEMANGLE %s

# NO_DEMANGLE: duplicate symbol: _Z3muldd in
# NO_DEMANGLE: duplicate symbol: foo in

# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %S/Inputs/conflict.s -o %t2.o
# RUN: llvm-ar rcs %t3.a %t2.o
# RUN: not ld.lld %t1.o %t3.a -u baz -o %t2 2>&1 | FileCheck -check-prefix=ARCHIVE %s

# ARCHIVE: duplicate symbol: foo in {{.*}}1.o and {{.*}}3.a({{.*}}2.o)

.globl _Z3muldd, foo
_Z3muldd:
foo:
  mov $60, %rax
  mov $42, %rdi
  syscall
