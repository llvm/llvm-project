# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %p/Inputs/undef.s -o %t2.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %p/Inputs/undef-debug.s -o %t3.o
# RUN: llvm-ar rc %t2.a %t2.o
# RUN: not ld.lld %t.o %t2.a %t3.o -o %t.exe 2>&1 | FileCheck %s
# RUN: not ld.lld -pie %t.o %t2.a %t3.o -o %t.exe 2>&1 | FileCheck %s
# CHECK: error: undef.s:(.text+0x1): undefined symbol 'foo'
# CHECK: error: undef.s:(.text+0x6): undefined symbol 'bar'
# CHECK: error: undef.s:(.text+0x10): undefined symbol 'foo(int)'
# CHECK: error: {{.*}}2.a({{.*}}.o):(.text+0x0): undefined symbol 'zed2'
# CHECK: error: dir{{/|\\}}undef-debug.s:3: undefined symbol 'zed3'
# CHECK: error: dir{{/|\\}}undef-debug.s:7: undefined symbol 'zed4'
# CHECK: error: dir{{/|\\}}undef-debug.s:11: undefined symbol 'zed5'

# RUN: not ld.lld %t.o %t2.a -o %t.exe -no-demangle 2>&1 | \
# RUN:   FileCheck -check-prefix=NO-DEMANGLE %s
# NO-DEMANGLE: error: undef.s:(.text+0x10): undefined symbol '_Z3fooi'

.file "undef.s"

  .globl _start
_start:
  call foo
  call bar
  call zed1
  call _Z3fooi
