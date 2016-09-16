# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %p/Inputs/undef.s -o %t2.o
# RUN: llvm-ar rc %t2.a %t2.o
# RUN: not ld.lld %t.o %t2.a -o %t.exe 2>&1 | FileCheck %s
# RUN: not ld.lld -pie %t.o %t2.a -o %t.exe 2>&1 | FileCheck %s
# CHECK: undefined symbol: foo(int) in
# CHECK: undefined symbol: bar in
# CHECK: undefined symbol: foo in
# CHECK: undefined symbol: zed2 in {{.*}}2.a({{.*}}.o)

# RUN: not ld.lld %t.o %t2.a -o %t.exe -no-demangle 2>&1 | \
# RUN:   FileCheck -check-prefix=NO-DEMANGLE %s
# NO-DEMANGLE: undefined symbol: _Z3fooi in

  .globl _start
_start:
  call foo
  call bar
  call zed1
  call _Z3fooi
