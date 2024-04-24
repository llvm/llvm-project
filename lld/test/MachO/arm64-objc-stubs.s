# REQUIRES: aarch64

# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %s -o %t.o
# RUN: %lld -arch arm64 -lSystem -o %t.out %t.o
# RUN: llvm-otool -vs __TEXT __objc_stubs %t.out | FileCheck %s
# RUN: %lld -arch arm64 -lSystem -o %t.out %t.o -dead_strip
# RUN: llvm-otool -vs __TEXT __objc_stubs %t.out | FileCheck %s
# RUN: %lld -arch arm64 -lSystem -o %t.out %t.o -objc_stubs_fast
# RUN: llvm-otool -vs __TEXT __objc_stubs %t.out | FileCheck %s
# RUN: llvm-otool -l %t.out | FileCheck %s --check-prefix=FASTALIGN
# RUN: %lld -arch arm64 -lSystem -o %t.out %t.o -objc_stubs_small
# RUN: llvm-otool -vs __TEXT __objc_stubs  %t.out | FileCheck %s --check-prefix=SMALL
# RUN: llvm-otool -l %t.out | FileCheck %s --check-prefix=SMALLALIGN

# CHECK: Contents of (__TEXT,__objc_stubs) section

# CHECK-NEXT: _objc_msgSend$foo:
# CHECK-NEXT: adrp    x1, 8 ; 0x100008000
# CHECK-NEXT: ldr     x1, [x1, #0x10]
# CHECK-NEXT: adrp    x16, 4 ; 0x100004000
# CHECK-NEXT: ldr     x16, [x16]
# CHECK-NEXT: br      x16
# CHECK-NEXT: brk     #0x1
# CHECK-NEXT: brk     #0x1
# CHECK-NEXT: brk     #0x1

# CHECK-NEXT: _objc_msgSend$length:
# CHECK-NEXT: adrp    x1, 8 ; 0x100008000
# CHECK-NEXT: ldr     x1, [x1, #0x18]
# CHECK-NEXT: adrp    x16, 4 ; 0x100004000
# CHECK-NEXT: ldr     x16, [x16]
# CHECK-NEXT: br      x16
# CHECK-NEXT: brk     #0x1
# CHECK-NEXT: brk     #0x1
# CHECK-NEXT: brk     #0x1

# CHECK-EMPTY:

# FASTALIGN:       sectname __objc_stubs
# FASTALIGN-NEXT:   segname __TEXT
# FASTALIGN-NEXT:      addr
# FASTALIGN-NEXT:      size
# FASTALIGN-NEXT:    offset
# FASTALIGN-NEXT:     align 2^5 (32)

# SMALL: _objc_msgSend$foo:
# SMALL-NEXT: adrp    x1, 4 ; 0x100004000
# SMALL-NEXT: ldr     x1, [x1, #0x10]
# SMALL-NEXT: b

# SMALL-NEXT: _objc_msgSend$length:
# SMALL-NEXT: adrp    x1, 4 ; 0x100004000
# SMALL-NEXT: ldr     x1, [x1, #0x18]
# SMALL-NEXT: b

# SMALLALIGN:       sectname __objc_stubs
# SMALLALIGN-NEXT:   segname __TEXT
# SMALLALIGN-NEXT:      addr
# SMALLALIGN-NEXT:      size
# SMALLALIGN-NEXT:    offset
# SMALLALIGN-NEXT:     align 2^2 (4)

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
  bl  _objc_msgSend$length
  bl  _objc_msgSend$foo
  bl  _objc_msgSend$foo
  ret
