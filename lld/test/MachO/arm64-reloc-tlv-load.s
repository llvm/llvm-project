# REQUIRES: aarch64

# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/main.s -o %t/main.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/foobar.s -o %t/foobar.o

# RUN: %lld -lSystem -arch arm64 -o %t/static %t/main.o %t/foobar.o
# RUN: llvm-objdump --no-print-imm-hex --macho -d --no-show-raw-insn --syms %t/static | FileCheck %s --check-prefix=STATIC

# RUN: %lld -lSystem -arch arm64 -dylib -o %t/libfoo.dylib %t/foobar.o
# RUN: %lld -lSystem -arch arm64 -o %t/main %t/main.o %t/libfoo.dylib
# RUN: llvm-objdump --no-print-imm-hex --macho -d --no-show-raw-insn --section-headers %t/main | FileCheck %s --check-prefix=DYLIB

# STATIC-LABEL: _main:
# STATIC-NEXT:  adrp x8, [[#]] ; 0x[[#%x,PAGE:]]
# STATIC-NEXT:  add  x8, x8, #[[#%u,FOO_OFF:]]
# STATIC-NEXT:  adrp x8, [[#]] ; 0x[[#PAGE]]
# STATIC-NEXT:  add  x8, x8, #[[#%u,BAR_OFF:]]
# STATIC-NEXT:  ret

# STATIC-LABEL: SYMBOL TABLE:
# STATIC-DAG:   {{0*}}[[#%x,PAGE+FOO_OFF]] g     O __DATA,__thread_vars _foo
# STATIC-DAG:   {{0*}}[[#%x,PAGE+BAR_OFF]] g     O __DATA,__thread_vars _bar

# DYLIB-LABEL: _main:
# DYLIB-NEXT:  adrp x8, [[#]] ; 0x[[#%x,TLV:]]
# DYLIB-NEXT:  ldr  x8, [x8, #8] ; literal pool symbol address: _foo
# DYLIB-NEXT:  adrp x8, [[#]] ; 0x[[#TLV]]
# DYLIB-NEXT:  ldr  x8, [x8] ; literal pool symbol address: _bar
# DYLIB-NEXT:  ret
# DYLIB-EMPTY:
# DYLIB-NEXT:  Sections:
# DYLIB-NEXT:  Idx   Name          Size     VMA              Type
# DYLIB:       [[#]] __thread_ptrs 00000010 {{0*}}[[#TLV]]   DATA

#--- main.s
.globl _main, _foo, _bar
.p2align 2
_main:
  adrp x8, _foo@TLVPPAGE
  ldr  x8, [x8, _foo@TLVPPAGEOFF]
  adrp x8, _bar@TLVPPAGE
  ldr  x8, [x8, _bar@TLVPPAGEOFF]
  ret

#--- foobar.s
.globl _foo, _bar

.section  __DATA,__thread_data,thread_local_regular
_foo$tlv$init:
  .long 123
_bar$tlv$init:
  .long 123

.section  __DATA,__thread_vars,thread_local_variables
_foo:
  .quad __tlv_bootstrap
  .quad 0
  .quad _foo$tlv$init
_bar:
  .quad __tlv_bootstrap
  .quad 0
  .quad _bar$tlv$init
