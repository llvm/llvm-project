# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: echo 'foo = _start;' > %t.script
# RUN: not ld.lld -shared -T %t.script %t.o -o /dev/null 2>&1 | FileCheck %s
# RUN: not ld.lld -shared --defsym 'foo = _start' %t.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK2

## Here we check that symbol 'foo' location is reported properly.

# CHECK: error: relocation R_X86_64_PC32 cannot be used against symbol 'foo'
# CHECK: >>> defined in {{.*}}.script:1
# CHECK: >>> referenced by {{.*}}.o:(.text+0x1)

# CHECK2: error: relocation R_X86_64_PC32 cannot be used against symbol 'foo'
# CHECK2: >>> defined in --defsym{{$}}
# CHECK2: >>> referenced by {{.*}}.o:(.text+0x1)

.text
.globl _start
_start:
  .byte 0xe8
  .long foo - .
