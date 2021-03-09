# REQUIRES: x86
# RUN: mkdir -p %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %p/Inputs/libhello.s \
# RUN:   -o %t/libhello.o
# RUN: %lld -lSystem -dylib -install_name \
# RUN:   @executable_path/libhello.dylib %t/libhello.o -o %t/libhello.dylib
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t/test.o
# RUN: %lld -lSystem -o %t/test %t/test.o -L%t -lhello
# RUN: llvm-objdump --full-contents --rebase --bind %t/test | FileCheck %s --match-full-lines

## Check that the GOT references the cstrings. --full-contents displays the
## address offset and the contents at that address very similarly, so am using
## --match-full-lines to make sure we match on the right thing.
# CHECK:      Contents of section __TEXT,__cstring:
# CHECK-NEXT: 100000434 {{.*}}

## 1st 8 bytes refer to the start of __cstring + 0xe, 2nd 8 bytes refer to the
## start of __cstring
# CHECK:      Contents of section __DATA_CONST,__got:
# CHECK-NEXT: [[#%X,ADDR:]]  42040000 01000000 34040000 01000000 {{.*}}
# CHECK-NEXT: [[#ADDR + 16]] 00000000 00000000 {{.*}}

## Check that the rebase table is empty.
# CHECK:      Rebase table:
# CHECK-NEXT: segment      section  address         type

## Check that a non-locally-defined symbol is still bound at the correct offset:
# CHECK-NEXT: Bind table:
# CHECK-NEXT: segment      section  address         type     addend  dylib     symbol
# CHECK-NEXT: __DATA_CONST __got    0x[[#ADDR+16]]  pointer  0       libhello  _hello_its_me

# RUN: %lld -pie -lSystem -o %t/test %t/test.o -L%t -lhello
# RUN: llvm-objdump --macho --rebase --bind %t/test | FileCheck %s --check-prefix=PIE --match-full-lines
# PIE:      Rebase table:
# PIE-NEXT: segment      section  address          type
# PIE-NEXT: __DATA_CONST __got    0x[[#%X,ADDR:]]  pointer
# PIE-NEXT: __DATA_CONST __got    0x[[#ADDR + 8]]  pointer

# PIE-NEXT: Bind table:
# PIE-NEXT: segment      section  address          type     addend  dylib     symbol
# PIE-NEXT: __DATA_CONST __got    0x[[#ADDR+16]]   pointer  0       libhello  _hello_its_me

.globl _main

.text
_main:
  movl $0x2000004, %eax # write() syscall
  mov $1, %rdi # stdout
  movq _hello_its_me@GOTPCREL(%rip), %rsi
  mov $15, %rdx # length of str
  syscall

  movl $0x2000004, %eax # write() syscall
  mov $1, %rdi # stdout
## We use pushq/popq here instead of movq in order to avoid relaxation.
  pushq _hello_world@GOTPCREL(%rip)
  popq %rsi
  mov $13, %rdx # length of str
  syscall

  movl $0x2000004, %eax # write() syscall
  mov $1, %rdi # stdout
  pushq _goodbye_world@GOTPCREL(%rip)
  popq %rsi
  mov $15, %rdx # length of str
  syscall

  mov $0, %rax
  ret

.section __TEXT,__cstring
_hello_world:
  .asciz "Hello world!\n"

_goodbye_world:
  .asciz "Goodbye world!\n"
