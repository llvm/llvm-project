# REQUIRES: aarch64

# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %s -o %t.o
# RUN: %lld -arch arm64 -U _extern_sym -o %t %t.o
# RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t | FileCheck %s --implicit-check-not=.thunk.

# CHECK-LABEL: Disassembly of section __TEXT,__text:

# CHECK-LABEL: <_main>:
# CHECK-NEXT:    bl
# CHECK-NEXT:    bl 0x[[#%x,THUNK:]] <_extern_sym.thunk.0>
# CHECK-NEXT:    bl 0x[[#%x,THUNK_FOO:]] <_objc_msgSend$foo.thunk.0>
# CHECK-NEXT:    bl 0x[[#%x,THUNK_BAR:]] <_objc_msgSend$bar.thunk.0>
# CHECK-NEXT:    ret

# CHECK-LABEL: <_foo>:
# CHECK-NEXT:    bl 0x[[#%x,THUNK:]] <_extern_sym.thunk.0>
# CHECK-NEXT:    bl 0x[[#%x,THUNK_FOO:]] <_objc_msgSend$foo.thunk.0>
# CHECK-NEXT:    bl 0x[[#%x,THUNK_BAR:]] <_objc_msgSend$bar.thunk.0>
# CHECK-NEXT:    ret

# CHECK: [[#THUNK]] <_extern_sym.thunk.0>:
# CHECK: [[#THUNK_FOO]] <_objc_msgSend$foo.thunk.0>:
# CHECK: [[#THUNK_BAR]] <_objc_msgSend$bar.thunk.0>:

# CHECK-LABEL: Disassembly of section __TEXT,__stubs:
# CHECK-LABEL: Disassembly of section __TEXT,__objc_stubs:

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

.text

.globl _main
_main:
  bl _foo
  bl _extern_sym
  bl _objc_msgSend$foo
  bl _objc_msgSend$bar
  ret

_spacer0:
.space 0x4000000-8

.globl _foo
_foo:
  bl _extern_sym
  bl _objc_msgSend$foo
  bl _objc_msgSend$bar
  ret

_spacer1:
.space 0x8000000

.globl _goo
_goo:
  bl _extern_sym
  bl _objc_msgSend$foo
  bl _objc_msgSend$bar
  ret

.subsections_via_symbols
