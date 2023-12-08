# REQUIRES: x86

## Don't create a canonical PLT if the symbol is defined as protected in a DSO,
## because thay may break pointer equality.

# RUN: echo '.globl foo; .protected foo; .type foo,@function; foo:' | \
# RUN:   llvm-mc -filetype=obj -triple=x86_64 - -o %t2.o
# RUN: ld.lld %t2.o -o %t2.so -shared -soname=so
# RUN: llvm-mc -triple x86_64-pc-linux -filetype=obj %s -o %t.o

# RUN: not ld.lld %t.o %t2.so -o /dev/null 2>&1 | FileCheck --check-prefix=ERR %s
# ERR: error: cannot preempt symbol: foo

# RUN: ld.lld --ignore-function-address-equality %t.o %t2.so -o %t
# RUN: llvm-readobj --dyn-symbols --relocations %t | FileCheck %s

# Check that we have a relocation and an undefined symbol with a non zero address

# CHECK: R_X86_64_JUMP_SLOT foo 0x0

# CHECK:      Name: foo
# CHECK-NEXT: Value: 0x20{{.*}}
# CHECK-NEXT: Size:
# CHECK-NEXT: Binding: Global
# CHECK-NEXT: Type: Function
# CHECK-NEXT: Other:
# CHECK-NEXT: Section: Undefined

.global _start
_start:
  .quad foo

