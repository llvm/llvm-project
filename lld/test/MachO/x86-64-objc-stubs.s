# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o
# RUN: %lld -arch x86_64 -lSystem -o %t.out %t.o
# RUN: llvm-objdump --macho --section-headers %t.out > %t.txt
# RUN: llvm-otool -vs __DATA __objc_selrefs %t.out >> %t.txt
# RUN: llvm-otool -vs __TEXT __objc_stubs %t.out >> %t.txt
# RUN: FileCheck %s < %t.txt
# RUN: %no-fatal-warnings-lld -arch x86_64 -lSystem -o %t.out %t.o -objc_stubs_small 2>&1 | FileCheck %s --check-prefix=WARNING

# WARNING: warning: -objc_stubs_small is not yet implemented, defaulting to -objc_stubs_fast

# RUN: %lld -arch x86_64 -lSystem -o %t-icfsafe.out --icf=safe %t.o
# RUN: llvm-otool -vs __DATA __objc_selrefs %t-icfsafe.out | FileCheck %s --check-prefix=ICF
# RUN: %lld -arch x86_64 -lSystem -o %t-icfall.out --icf=all %t.o
# RUN: llvm-otool -vs __DATA __objc_selrefs %t-icfall.out | FileCheck %s --check-prefix=ICF

# CHECK: Sections:
# CHECK: __got            {{[0-9a-f]*}} [[#%x, GOTSTART:]] DATA
# CHECK: __objc_selrefs   {{[0-9a-f]*}} [[#%x, SELSTART:]] DATA

# CHECK: Contents of (__DATA,__objc_selrefs) section

# CHECK-NEXT: {{[0-9a-f]*}}  __TEXT:__objc_methname:foo
# CHECK-NEXT: {{[0-9a-f]*}}  __TEXT:__objc_methname:bar
# CHECK-NEXT: [[#%x, FOOSELREF:]]  __TEXT:__objc_methname:foo
# CHECK-NEXT: [[#%x, LENGTHSELREF:]]  __TEXT:__objc_methname:length

# ICF: Contents of (__DATA,__objc_selrefs) section

# ICF-NEXT: {{[0-9a-f]*}}  __TEXT:__objc_methname:foo
# ICF-NEXT: {{[0-9a-f]*}}  __TEXT:__objc_methname:bar
# ICF-NEXT: {{[0-9a-f]*}}  __TEXT:__objc_methname:length
# ICF-EMPTY:

# CHECK: Contents of (__TEXT,__objc_stubs) section

# CHECK-NEXT: _objc_msgSend$foo:
# CHECK-NEXT: [[#%x, PC1:]]
# CHECK-SAME: movq    0x[[#%x, FOOSELREF - PC1 - 7]](%rip), %rsi
# CHECK-NEXT: [[#%x, PC2:]]
# CHECK-SAME: jmpq    *0x[[#%x, GOTSTART - PC2 - 6]](%rip)

# CHECK-NEXT: _objc_msgSend$length:
# CHECK-NEXT: [[#%x, PC3:]]
# CHECK-SAME: movq    0x[[#%x, LENGTHSELREF - PC3 - 7]](%rip), %rsi
# CHECK-NEXT: [[#%x, PC4:]]
# CHECK-SAME: jmpq    *0x[[#%x, GOTSTART - PC4 - 6]](%rip)

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
