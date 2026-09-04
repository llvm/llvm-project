# REQUIRES: x86
# RUN: rm -rf %t; split-file %s %t

## A GOT relocation against a thread-local asks for the address of that
## symbol's TLV descriptor, which is what its non-lazy pointer slot holds.
## lld used to reject this valid reference; ld-prime accepts it.

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/deftlv.s -o %t/deftlv.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/libtlv.s -o %t/libtlv.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/got.s -o %t/got.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/both.s -o %t/both.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/tlv.s -o %t/tlv.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/branch.s -o %t/branch.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/direct.s -o %t/direct.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/localtlv.s -o %t/localtlv.o

## A definition in the same image needs no slot: the load relaxes to a direct
## address computation.
# RUN: %lld -dylib -o %t/deftlv.dylib %t/deftlv.o
# RUN: llvm-objdump --macho --section-headers -d --no-show-raw-insn %t/deftlv.dylib | \
# RUN:   FileCheck %s --check-prefix=DEF
# DEF-NOT:   __got
# DEF-NOT:   __thread_ptrs
# DEF-LABEL: _main:
# DEF-NEXT:  leaq _foo(%rip), %rax

# RUN: %lld -dylib -install_name @executable_path/libtlv.dylib -lSystem \
# RUN:   -o %t/libtlv.dylib %t/libtlv.o

## Imported from a dylib, the reference binds through __got, as with ld-prime.
# RUN: %lld -dylib -lSystem -L%t -ltlv -o %t/got.dylib %t/got.o
# RUN: llvm-objdump --macho --bind %t/got.dylib | FileCheck %s --check-prefix=GOT
# GOT: __DATA_CONST __got 0x{{[0-9a-f]+}} pointer 0 libtlv _foo

## Reaching one thread-local both ways must produce a single slot. A second
## one would show up as a second binding.
# RUN: %lld -dylib -lSystem -L%t -ltlv -o %t/both.dylib %t/both.o
# RUN: llvm-objdump --macho --section-headers --bind %t/both.dylib | \
# RUN:   FileCheck %s --check-prefix=BOTH
# BOTH-NOT:   __thread_ptrs
# BOTH:       __got
# BOTH-LABEL: Bind table:
# BOTH:       __DATA_CONST __got 0x{{[0-9a-f]+}} pointer 0 libtlv _foo
# BOTH-NOT:   _foo

## A TLV-only reference also uses __got, matching ld-prime.
# RUN: %lld -dylib -lSystem -L%t -ltlv -o %t/tlv.dylib %t/tlv.o
# RUN: llvm-objdump --macho --section-headers --bind %t/tlv.dylib | \
# RUN:   FileCheck %s --check-prefix=TLV
# TLV-NOT:   __thread_ptrs
# TLV:       __got
# TLV-LABEL: Bind table:
# TLV:       __DATA_CONST __got 0x{{[0-9a-f]+}} pointer 0 libtlv _foo

## A chained-fixup stub and a TLV relocation share the same GOT entry.
# RUN: %lld -dylib -fixup_chains -lSystem -L%t -ltlv \
# RUN:   -o %t/branch.dylib %t/branch.o
# RUN: llvm-objdump --macho --section-headers --chained-fixups \
# RUN:   %t/branch.dylib | FileCheck %s --check-prefix=BRANCH
# BRANCH-NOT: __thread_ptrs
# BRANCH:     __got
# BRANCH:     _foo
# BRANCH-NOT: _foo

## A direct reference can address a same-image descriptor, but an imported
## descriptor has no fixed address. ld-prime accepts the first and rejects the
## second.
# RUN: %lld -dylib -o %t/direct-local.dylib %t/direct.o %t/localtlv.o
# RUN: not %lld -dylib -L%t -ltlv -o /dev/null %t/direct.o 2>&1 | \
# RUN:   FileCheck %s --check-prefix=DIRECT
# DIRECT: SIGNED relocation requires that symbol _foo not be thread-local

#--- deftlv.s
.text
.globl _main
_main:
  movq _foo@GOTPCREL(%rip), %rax
  ret
.section __DATA,__thread_vars,thread_local_variables
_foo:

#--- libtlv.s
.section __DATA,__thread_vars,thread_local_variables
.globl _foo
_foo:

#--- got.s
.text
.globl _main
_main:
  movq _foo@GOTPCREL(%rip), %rax
  ret

#--- both.s
.text
.globl _main
_main:
  movq _foo@GOTPCREL(%rip), %rax
  movq _foo@TLVP(%rip), %rcx
  ret

#--- tlv.s
.text
.globl _main
_main:
  movq _foo@TLVP(%rip), %rax
  ret

#--- branch.s
.text
.globl _main
_main:
  callq _foo
  movq _foo@TLVP(%rip), %rax
  ret

#--- direct.s
.text
.globl _main
_main:
  leaq _foo(%rip), %rax
  ret

#--- localtlv.s
.section __DATA,__thread_vars,thread_local_variables
.globl _foo
_foo:
  .space 24
