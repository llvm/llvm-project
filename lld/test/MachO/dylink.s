# REQUIRES: x86
# RUN: mkdir -p %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %p/Inputs/libhello.s \
# RUN:   -o %t/libhello.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %p/Inputs/libgoodbye.s \
# RUN:   -o %t/libgoodbye.o
# RUN: lld -flavor darwinnew -arch x86_64 -dylib -install_name \
# RUN:   @executable_path/libhello.dylib %t/libhello.o -o %t/libhello.dylib
# RUN: lld -flavor darwinnew -arch x86_64 -dylib -install_name \
# RUN:   @executable_path/libgoodbye.dylib %t/libgoodbye.o -o %t/libgoodbye.dylib

## Make sure we are using the export trie and not the symbol table when linking
## against these dylibs.
# RUN: llvm-strip %t/libhello.dylib
# RUN: llvm-strip %t/libgoodbye.dylib
# RUN: llvm-nm %t/libhello.dylib 2>&1 | FileCheck %s --check-prefix=NOSYM
# RUN: llvm-nm %t/libgoodbye.dylib 2>&1 | FileCheck %s --check-prefix=NOSYM
# NOSYM: no symbols

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t/dylink.o
# RUN: lld -flavor darwinnew -arch x86_64 -o %t/dylink -Z -L%t -lhello -lgoodbye %t/dylink.o
# RUN: llvm-objdump --bind -d %t/dylink | FileCheck %s

# CHECK: movq [[#%u, HELLO_OFF:]](%rip), %rsi
# CHECK-NEXT: [[#%x, HELLO_RIP:]]:

# CHECK: movq [[#%u, HELLO_ITS_ME_OFF:]](%rip), %rsi
# CHECK-NEXT: [[#%x, HELLO_ITS_ME_RIP:]]:

# CHECK: movq [[#%u, GOODBYE_OFF:]](%rip), %rsi
# CHECK-NEXT: [[#%x, GOODBYE_RIP:]]:

# CHECK-LABEL: Bind table:
# CHECK-DAG: __DATA_CONST __got 0x{{0*}}[[#%x, HELLO_RIP + HELLO_OFF]]               pointer 0 libhello   _hello_world
# CHECK-DAG: __DATA_CONST __got 0x{{0*}}[[#%x, HELLO_ITS_ME_RIP + HELLO_ITS_ME_OFF]] pointer 0 libhello   _hello_its_me
# CHECK-DAG: __DATA_CONST __got 0x{{0*}}[[#%x, GOODBYE_RIP + GOODBYE_OFF]]           pointer 0 libgoodbye _goodbye_world

.section __TEXT,__text
.globl _main

_main:
  movl $0x2000004, %eax # write() syscall
  mov $1, %rdi # stdout
  movq _hello_world@GOTPCREL(%rip), %rsi
  mov $13, %rdx # length of str
  syscall

  movl $0x2000004, %eax # write() syscall
  mov $1, %rdi # stdout
  movq _hello_its_me@GOTPCREL(%rip), %rsi
  mov $15, %rdx # length of str
  syscall

  movl $0x2000004, %eax # write() syscall
  mov $1, %rdi # stdout
  movq _goodbye_world@GOTPCREL(%rip), %rsi
  mov $15, %rdx # length of str
  syscall
  mov $0, %rax
  ret
