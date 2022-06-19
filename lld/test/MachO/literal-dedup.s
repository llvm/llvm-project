# REQUIRES: x86
# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/test.s -o %t/test.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/qux.s -o %t/qux.o
# RUN: %lld -dylib --deduplicate-literals %t/test.o %t/qux.o -o %t/test
# RUN: llvm-objdump --macho --section="__TEXT,__literals" --section="__DATA,ptrs" --syms %t/test | FileCheck %s
# RUN: llvm-readobj --section-headers %t/test | FileCheck %s --check-prefix=HEADER

## Make sure literal deduplication can be overridden or that the later flag wins.
# RUN: %lld -dylib --deduplicate-literals -no_deduplicate %t/test.o %t/qux.o -o %t/no-dedup-test
# RUN: llvm-objdump --macho --section="__TEXT,__literals" --section="__DATA,ptrs" %t/no-dedup-test | FileCheck %s --check-prefix=NO-DEDUP

# RUN: %lld -dylib -no_deduplicate --deduplicate-literals %t/test.o %t/qux.o -o %t/test
# RUN: llvm-objdump --macho --section="__TEXT,__literals" --section="__DATA,ptrs" --syms %t/test | FileCheck %s
# RUN: llvm-readobj --section-headers %t/test | FileCheck %s --check-prefix=HEADER

# NO-DEDUP-NOT:  Contents of (__TEXT,__literals) section
# NO-DEDUP:      Contents of (__DATA,ptrs) section
# NO-DEDUP-NEXT: __TEXT:__literal16:0xdeadbeef 0xdeadbeef 0xdeadbeef 0xdeadbeef
# NO-DEDUP-NEXT: __TEXT:__literal16:0xdeadbeef 0xdeadbeef 0xdeadbeef 0xdeadbeef
# NO-DEDUP-NEXT: __TEXT:__literal16:0xfeedface 0xfeedface 0xfeedface 0xfeedface
# NO-DEDUP-NEXT: __TEXT:__literal16:0xdeadbeef 0xdeadbeef 0xdeadbeef 0xdeadbeef
# NO-DEDUP-NEXT: __TEXT:__literal8:0xdeadbeef 0xdeadbeef
# NO-DEDUP-NEXT: __TEXT:__literal8:0xdeadbeef 0xdeadbeef
# NO-DEDUP-NEXT: __TEXT:__literal8:0xfeedface 0xfeedface
# NO-DEDUP-NEXT: __TEXT:__literal8:0xdeadbeef 0xdeadbeef
# NO-DEDUP-NEXT: __TEXT:__literal4:0xdeadbeef
# NO-DEDUP-NEXT: __TEXT:__literal4:0xdeadbeef
# NO-DEDUP-NEXT: __TEXT:__literal4:0xfeedface
# NO-DEDUP-NEXT: __TEXT:__literal4:0xdeadbeef

# CHECK:      Contents of (__TEXT,__literals) section
# CHECK-NEXT: [[#%.16x,DEADBEEF16:]] ef be ad de ef be ad de ef be ad de ef be ad de
# CHECK-NEXT: [[#%.16x,FEEDFACE16:]] ce fa ed fe ce fa ed fe ce fa ed fe ce fa ed fe
# CHECK-NEXT: [[#%.16x,DEADBEEF8:]]  ef be ad de ef be ad de ce fa ed fe ce fa ed fe
# CHECK-NEXT: [[#%.16x,DEADBEEF4:]]  ef be ad de ce fa ed fe
# CHECK-NEXT: Contents of (__DATA,ptrs) section
# CHECK-NEXT: 0000000000001000  0x[[#%x,DEADBEEF16]]
# CHECK-NEXT: 0000000000001008  0x[[#%x,DEADBEEF16]]
# CHECK-NEXT: 0000000000001010  0x[[#%x,FEEDFACE16]]
# CHECK-NEXT: 0000000000001018  0x[[#%x,DEADBEEF16]]
# CHECK-NEXT: 0000000000001020  0x[[#%x,DEADBEEF8]]
# CHECK-NEXT: 0000000000001028  0x[[#%x,DEADBEEF8]]
# CHECK-NEXT: 0000000000001030  0x[[#%x,DEADBEEF8 + 8]]
# CHECK-NEXT: 0000000000001038  0x[[#%x,DEADBEEF8]]
# CHECK-NEXT: 0000000000001040  0x[[#%x,DEADBEEF4]]
# CHECK-NEXT: 0000000000001048  0x[[#%x,DEADBEEF4]]
# CHECK-NEXT: 0000000000001050  0x[[#%x,DEADBEEF4 + 4]]
# CHECK-NEXT: 0000000000001058  0x[[#%x,DEADBEEF4]]

## Make sure the symbol addresses are correct too.
# CHECK:     SYMBOL TABLE:
# CHECK-DAG: [[#DEADBEEF16]] g     O __TEXT,__literals _qux16
# CHECK-DAG: [[#DEADBEEF8]]  g     O __TEXT,__literals _qux8
# CHECK-DAG: [[#DEADBEEF4]]  g     O __TEXT,__literals _qux4

## Make sure we set the right alignment and flags.
# HEADER:        Name: __literals
# HEADER-NEXT:   Segment: __TEXT
# HEADER-NEXT:   Address:
# HEADER-NEXT:   Size:
# HEADER-NEXT:   Offset:
# HEADER-NEXT:   Alignment: 4
# HEADER-NEXT:   RelocationOffset:
# HEADER-NEXT:   RelocationCount: 0
# HEADER-NEXT:   Type: Regular
# HEADER-NEXT:   Attributes [ (0x0)
# HEADER-NEXT:   ]
# HEADER-NEXT:   Reserved1: 0x0
# HEADER-NEXT:   Reserved2: 0x0
# HEADER-NEXT:   Reserved3: 0x0

#--- test.s
.literal4
.p2align 2
L._foo4:
  .long 0xdeadbeef
L._bar4:
  .long 0xdeadbeef
L._baz4:
  .long 0xfeedface

.literal8
L._foo8:
  .quad 0xdeadbeefdeadbeef
L._bar8:
  .quad 0xdeadbeefdeadbeef
L._baz8:
  .quad 0xfeedfacefeedface

.literal16
L._foo16:
  .quad 0xdeadbeefdeadbeef
  .quad 0xdeadbeefdeadbeef
L._bar16:
  .quad 0xdeadbeefdeadbeef
  .quad 0xdeadbeefdeadbeef
L._baz16:
  .quad 0xfeedfacefeedface
  .quad 0xfeedfacefeedface

.section __DATA,ptrs,literal_pointers
.quad L._foo16
.quad L._bar16
.quad L._baz16
.quad _qux16

.quad L._foo8
.quad L._bar8
.quad L._baz8
.quad _qux8

.quad L._foo4
.quad L._bar4
.quad L._baz4
.quad _qux4

#--- qux.s
.globl _qux4, _qux8, _qux16

.literal4
.p2align 2
_qux4:
  .long 0xdeadbeef

.literal8
_qux8:
  .quad 0xdeadbeefdeadbeef

.literal16
_qux16:
  .quad 0xdeadbeefdeadbeef
  .quad 0xdeadbeefdeadbeef
