# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o
# RUN: %lld -arch x86_64 -lSystem -o %t.out %t.o
# RUN: llvm-otool -vs __TEXT __objc_stubs %t.out | FileCheck %s

# CHECK: Contents of (__TEXT,__objc_stubs) section

# CHECK-NEXT: _objc_msgSend$foo:
# CHECK-NEXT: 00000001000004b8	movq	0x1b51(%rip), %rsi
# CHECK-NEXT: 00000001000004bf	jmpq	*0xb3b(%rip)

# CHECK-NEXT: _objc_msgSend$length:
# CHECK-NEXT: 00000001000004c5	movq	0x1b4c(%rip), %rsi
# CHECK-NEXT: 00000001000004cc	jmpq	*0xb2e(%rip)

# CHECK-EMPTY:

.section  __TEXT,__objc_methname,cstring_literals
lselref1:
  .asciz  "foo"
lselref2:
  .asciz  "bar"

.section  __DATA,__objc_selrefs,literal_pointers,no_dead_strip
.p2align  3
.quad lselref1
.quad lselref2

.text
.globl _objc_msgSend
_objc_msgSend:
  ret

.globl _main
_main:
  callq  _objc_msgSend$length
  callq  _objc_msgSend$foo
  callq  _objc_msgSend$foo
  ret
