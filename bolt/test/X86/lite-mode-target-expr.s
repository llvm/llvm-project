## Check that llvm-bolt properly updates references in unoptimized code when
## such references are non-trivial expressions.

# RUN: %clang %cflags %s -o %t.exe -Wl,-q -no-pie
# RUN: llvm-bolt %t.exe -o %t.bolt --funcs=_start
# RUN: llvm-objdump -d --disassemble-symbols=_start %t.bolt > %t.out
# RUN: llvm-objdump -d --disassemble-symbols=cold %t.bolt >> %t.out
# RUN: FileCheck %s < %t.out

## _start() will be optimized and assigned a new address.
# CHECK: [[#%x,ADDR:]] <_start>:

## cold() is not optimized, but references to _start are updated.
# CHECK-LABEL: <cold>:
# CHECK-NEXT: movl $0x[[#ADDR - 1]], %ecx
# CHECK-NEXT: movl $0x[[#ADDR]], %ecx
# CHECK-NEXT: movl $0x[[#ADDR + 1]], %ecx

  .text
  .globl cold
  .type cold, %function
cold:
	movl $_start-1, %ecx
	movl $_start, %ecx
	movl $_start+1, %ecx
  ret
  .size cold, .-cold

  .globl _start
  .type _start, %function
_start:
	ret
  .size _start, .-_start
