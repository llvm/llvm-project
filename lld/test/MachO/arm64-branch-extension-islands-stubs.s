# REQUIRES: aarch64

# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/islands.s -o %t/islands.o

## Ordinary and ObjC stubs beyond direct branch range are reached through two
## islands each. Generated _objc_msgSend$foo.island.* symbols belong to __text,
## not __objc_stubs, despite sharing the ObjC stub's name prefix.
# RUN: %lld -arch arm64 -dylib -U _external -U _objc_msgSend -o %t/islands %t/islands.o \
# RUN:   -arm64-islands-max-hops 2 --verbose 2> %t/islands.log
# RUN: FileCheck %s --check-prefix=ISLANDS-LOG --input-file=%t/islands.log
# RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t/islands \
# RUN:   | FileCheck %s --check-prefix=ISLANDS --implicit-check-not=.thunk.

# ISLANDS-LOG: branch extender for __TEXT:
# ISLANDS-LOG-SAME: total extenders = 4

# ISLANDS-LABEL: Disassembly of section __TEXT,__text:
# ISLANDS-LABEL: <_main>:
# ISLANDS-NEXT: bl 0x[[#%x, ISLAND0:]] <_external.island.0>
# ISLANDS-NEXT: bl 0x[[#%x, OBJC_ISLAND0:]] <_objc_msgSend$foo.island.0>
# ISLANDS-NEXT: ret
# ISLANDS-DAG: [[#ISLAND0]]: b 0x[[#%x, ISLAND1:]] <_external.island.1>
# ISLANDS-DAG: [[#OBJC_ISLAND0]]: b 0x[[#%x, OBJC_ISLAND1:]] <_objc_msgSend$foo.island.1>
# ISLANDS-DAG: [[#ISLAND1]]: b 0x[[#%x, STUB:]] <{{.*}}>
# ISLANDS-DAG: [[#OBJC_ISLAND1]]: b 0x[[#%x, OBJC_STUB:]] <_objc_msgSend$foo>
# ISLANDS-LABEL: Disassembly of section __TEXT,__stubs:
# ISLANDS: [[#STUB]]: adrp x16
# ISLANDS-NEXT: ldr x16
# ISLANDS-NEXT: br x16
# ISLANDS-LABEL: Disassembly of section __TEXT,__objc_stubs:
# ISLANDS: [[#OBJC_STUB]] <_objc_msgSend$foo>:

## Renaming __stubs places it after 128 MiB of __const data. Estimation of branch range span
## must take __const into account and emit branch extension.
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/const-gap.s -o %t/const-gap.o
# RUN: %lld -arch arm64 -dylib -U _external -o %t/const-gap %t/const-gap.o \
# RUN:   -rename_section __TEXT __stubs __TEXT __late_stubs \
# RUN:   -arm64-islands-max-hops 0 --verbose 2> %t/const-gap.log
# RUN: FileCheck %s --check-prefix=GAP-LOG --input-file=%t/const-gap.log
# RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t/const-gap \
# RUN:   | FileCheck %s --check-prefix=GAP --implicit-check-not=.island.

# GAP-LOG: branch extender for __TEXT:
# GAP-LOG-SAME: total extenders = 1

# GAP-LABEL: <_main>:
# GAP-NEXT: bl 0x[[#%x, THUNK:]] <_external.thunk.0>
# GAP-NEXT: ret
# GAP: [[#THUNK]] <_external.thunk.0>:
# GAP-NEXT: adrp x16, 0x[[#%x, STUB_PAGE:]]
# GAP-NEXT: add x16, x16, #[[#%u, STUB_OFFSET:]]
# GAP-NEXT: br x16
# GAP-LABEL: Disassembly of section __TEXT,__late_stubs:
# GAP: [[#%x, STUB_PAGE + STUB_OFFSET]] <__late_stubs>:

## Moving __objc_stubs before __text makes it directly reachable from _main.
## Planning still runs for the distant ordinary stub; the ObjC call must use
## the earlier stub's assigned address without an island or thunk.
# RUN: %lld -arch arm64 -dylib -U _external -U _objc_msgSend -o %t/front %t/islands.o \
# RUN:   -rename_section __TEXT __objc_stubs __TEXT __mach_header \
# RUN:   -arm64-islands-max-hops 2 --verbose 2> %t/front.log
# RUN: FileCheck %s --check-prefix=FRONT-LOG --input-file=%t/front.log
# RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t/front \
# RUN:   | FileCheck %s --check-prefix=FRONT --implicit-check-not=.thunk. \
# RUN:     --implicit-check-not='_objc_msgSend$foo.island.'

# FRONT-LOG: branch extender for __TEXT:
# FRONT-LOG-SAME: total extenders = 2

# FRONT-LABEL: Disassembly of section __TEXT,__mach_header:
# FRONT: [[#%x, OBJC_STUB:]] <_objc_msgSend$foo>:
# FRONT-LABEL: Disassembly of section __TEXT,__text:
# FRONT-LABEL: <_main>:
# FRONT-NEXT: bl 0x{{[0-9a-f]+}} <_external.island.0>
# FRONT-NEXT: bl 0x[[#OBJC_STUB]] <_objc_msgSend$foo>
# FRONT-NEXT: ret

#--- islands.s
.subsections_via_symbols
.text
.globl _main
.p2align 2
_main:
  bl _external
  bl _objc_msgSend$foo
  ret

.space 0x6000000
.globl _f0
_f0:
  ret
.space 0x6000000
.globl _f1
_f1:
  ret
.space 0x6000000
.globl _f2
_f2:
  ret

#--- const-gap.s
.subsections_via_symbols
.text
.globl _main
.p2align 2
_main:
  bl _external
  ret

.section __TEXT,__const
.space 0x8000000
