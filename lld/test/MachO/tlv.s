# REQUIRES: x86
# RUN: mkdir -p %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t/test.o
# RUN: %lld -lSystem -o %t/test %t/test.o
# RUN: llvm-readobj --file-headers %t/test | FileCheck %s --check-prefix=HEADER
# RUN: llvm-objdump -D %t/test | FileCheck %s

# HEADER: MH_HAS_TLV_DESCRIPTORS

# CHECK:       Disassembly of section __TEXT,__text:
# CHECK-EMPTY:
# CHECK-NEXT:  <_main>:
# CHECK-NEXT:  leaq    {{.*}}(%rip), %rax  # {{.*}} <_foo>
# CHECK-NEXT:  leaq    {{.*}}(%rip), %rax  # {{.*}} <_bar>
# CHECK-NEXT:  retq
# CHECK-EMPTY:
# CHECK-NEXT:  Disassembly of section __DATA,__thread_data:
# CHECK-EMPTY:
# CHECK-NEXT:  <__thread_data>:
# CHECK-NEXT:  ef
# CHECK-NEXT:  be ad de be ba
# CHECK-NEXT:  fe ca
# CHECK-EMPTY:
# CHECK-NEXT:  Disassembly of section __DATA,__thread_vars:
# CHECK-EMPTY:
# CHECK-NEXT: <_foo>:
# CHECK-NEXT:          ...
# CHECK-EMPTY:
# CHECK-NEXT:  <_bar>:
# CHECK-NEXT:          ...
# CHECK-NEXT:  04 00
# CHECK-NEXT:  00 00
# CHECK-NEXT:  00 00
# CHECK-NEXT:  00 00

.globl _main
_main:
  mov _foo@TLVP(%rip), %rax
  mov _bar@TLVP(%rip), %rax
  ret

.section	__DATA,__thread_data,thread_local_regular
_foo$tlv$init:
  .long	0xdeadbeef
_bar$tlv$init:
  .long	0xcafebabe

.section	__DATA,__thread_vars,thread_local_variables
.globl	_foo, _bar
_foo:
  .quad	__tlv_bootstrap
  .quad	0
  .quad	_foo$tlv$init
_bar:
  .quad	__tlv_bootstrap
  .quad	0
  .quad	_bar$tlv$init
