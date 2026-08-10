# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t1.o
# RUN: not ld.lld --error-limit=1 %t1.o %t1.o %t1.o -o /dev/null 2>%t.output
# RUN: echo "END" >> %t.output
# RUN: FileCheck %s -input-file=%t.output

# CHECK:      error: duplicate symbol: _start
# CHECK-NEXT: >>> defined at {{.*}}1.o:(.text+0x0)
# CHECK-NEXT: >>> defined at {{.*}}1.o:(.text+0x0)
# CHECK-EMPTY:
# CHECK-NEXT: ld.lld: error: too many errors emitted, stopping now (use --error-limit=0 to see all errors)
## Ensure that there isn't an additional newline before the next message:
# CHECK-NEXT: END
.globl _start
_start:
  nop

.globl foo
foo:
  nop

.globl bar
bar:
  nop
