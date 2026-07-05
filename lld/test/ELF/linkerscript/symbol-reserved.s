# REQUIRES: x86

## Test linker synthesized symbols (e.g. __ehdr_start, _end) can be used in linker scripts.

# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t
# RUN: echo "PROVIDE_HIDDEN(newsym = __ehdr_start + 5);" > %t.script
# RUN: ld.lld -o %t1 %t.script %t
# RUN: llvm-objdump -t %t1 | FileCheck %s

# CHECK: 0000000000200005 l .text 0000000000000000 .hidden newsym

# RUN: ld.lld -o %t1.so %t.script %t -shared
# RUN: llvm-objdump -t %t1.so | FileCheck --check-prefix=SHARED %s

# SHARED: 0000000000000005 l .dynsym 0000000000000000 .hidden newsym

# RUN: echo "PROVIDE_HIDDEN(newsym = ALIGN(__ehdr_start, CONSTANT(MAXPAGESIZE)) + 5);" > %t.script
# RUN: ld.lld -o %t1 %t.script %t
# RUN: llvm-objdump -t %t1 | FileCheck --check-prefix=ALIGNED %s

# ALIGNED: 0000000000200005 l .text 0000000000000000 .hidden newsym

# RUN: echo "PROVIDE_HIDDEN(newsym = ALIGN(3, 8) + 10);" > %t.script
# RUN: ld.lld -o %t1 %t.script %t
# RUN: llvm-objdump -t %t1 | FileCheck --check-prefix=ALIGN-ADD %s
# ALIGN-ADD: 0000000000000012 l *ABS* 0000000000000000 .hidden newsym

# RUN: echo "PROVIDE_HIDDEN(newsym = ALIGN(11, 8) - 10);" > %t.script
# RUN: ld.lld -o %t1 %t.script %t
# RUN: llvm-objdump -t %t1 | FileCheck --check-prefix=ALIGN-SUB %s
# ALIGN-SUB: 0000000000000006 l *ABS* 0000000000000000 .hidden newsym

# RUN: echo "PROVIDE_HIDDEN(newsym = ALIGN(_end, CONSTANT(MAXPAGESIZE)) + 5);" > %t.script
# RUN: ld.lld -o %t1 %t %t.script
# RUN: llvm-objdump -t %t1 | FileCheck --check-prefix=RELATIVE %s
# RELATIVE: 0000000000202005 l .text 0000000000000000 .hidden newsym
# RELATIVE: 0000000000201127 g .text 0000000000000000 _end

# RUN: echo "PROVIDE_HIDDEN(newsym = ALIGN(_end, CONSTANT(MAXPAGESIZE)) + 5);" > %t.script
# RUN: ld.lld -o %t1 --script %p/Inputs/symbol-reserved.script %t %t.script
# RUN: llvm-objdump -t %t1 | FileCheck --check-prefix=RELATIVE-ADD %s
# RELATIVE-ADD: 0000000000001005 l .text 0000000000000000 .hidden newsym
# RELATIVE-ADD: 0000000000000007 l .text 0000000000000000 .hidden _end

# RUN: echo "PROVIDE_HIDDEN(newsym = ALIGN(_end, CONSTANT(MAXPAGESIZE)) - 5);" > %t.script
# RUN: ld.lld -o %t1 --script %p/Inputs/symbol-reserved.script %t %t.script
# RUN: llvm-objdump -t %t1 | FileCheck --check-prefix=RELATIVE-SUB %s
# RELATIVE-SUB: 0000000000000ffb l .text 0000000000000000 .hidden newsym
# RELATIVE-SUB: 0000000000000007 l .text 0000000000000000 .hidden _end

.global _start
_start:
  lea newsym(%rip),%rax
