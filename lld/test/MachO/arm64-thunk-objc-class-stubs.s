# REQUIRES: aarch64

## Thunk generation treats __objc_clsstubs like the other stubs sections when
## it is the only one. __text is 0x7fffff0 bytes, so only the class stubs that
## follow it put _main's calls out of range and make thunks necessary, while
## _goo's calls stay in range and do not use them.

# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %s -o %t.o
# RUN: %lld -arch arm64 -o %t %t.o
# RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t \
# RUN:   | FileCheck %s --implicit-check-not=.thunk.

# CHECK-LABEL: <_main>:
# CHECK-NEXT:    bl 0x[[#%x,THUNK_A:]] <_objc_msgSendClass$a$_OBJC_CLASS_$_Foo.thunk.0>
# CHECK-NEXT:    bl 0x[[#%x,THUNK_B:]] <_objc_msgSendClass$b$_OBJC_CLASS_$_Foo.thunk.0>
# CHECK-NEXT:    ret
# CHECK:       [[#THUNK_A]] <_objc_msgSendClass$a$_OBJC_CLASS_$_Foo.thunk.0>:
# CHECK:       [[#THUNK_B]] <_objc_msgSendClass$b$_OBJC_CLASS_$_Foo.thunk.0>:

# CHECK-LABEL: <_goo>:
# CHECK-NEXT:    bl 0x[[#%x,STUB_A:]] <_objc_msgSendClass$a$_OBJC_CLASS_$_Foo>
# CHECK-NEXT:    bl 0x[[#%x,STUB_B:]] <_objc_msgSendClass$b$_OBJC_CLASS_$_Foo>
# CHECK-NEXT:    ret

# CHECK-LABEL: Disassembly of section __TEXT,__objc_clsstubs:
# CHECK:       [[#STUB_A]] <_objc_msgSendClass$a$_OBJC_CLASS_$_Foo>:
# CHECK:       [[#STUB_B]] <_objc_msgSendClass$b$_OBJC_CLASS_$_Foo>:

.text
.globl _main
.p2align 2
_main:
  bl _objc_msgSendClass$a$_OBJC_CLASS_$_Foo
  bl _objc_msgSendClass$b$_OBJC_CLASS_$_Foo
  ret

## 0x7fffff0 bytes of __text in total.
_spacer:
.space 0x7fffff0-28

.globl _goo
.p2align 2
_goo:
  bl _objc_msgSendClass$a$_OBJC_CLASS_$_Foo
  bl _objc_msgSendClass$b$_OBJC_CLASS_$_Foo
  ret

.globl _objc_msgSend
.p2align 2
_objc_msgSend:
  ret

.data
.globl _OBJC_CLASS_$_Foo
_OBJC_CLASS_$_Foo:
  .quad 0

.subsections_via_symbols
