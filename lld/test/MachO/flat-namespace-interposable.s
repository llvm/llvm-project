# REQUIRES: x86
# RUN: rm -rf %t; split-file %s %t

## With -flat_namespace, non-weak extern symbols in dylibs become interposable.
## Check that we generate the correct bindings for them. The test also includes
## other symbol types like weak externs to verify we continue to do the same
## (correct) thing even when `-flat_namespace` is enabled, instead of generating
## spurious bindings.

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos -o %t/foo.o %t/foo.s
# RUN: %lld -lSystem -flat_namespace -o %t/foo %t/foo.o
# RUN: %lld -lSystem -dylib -flat_namespace -o %t/foo.dylib %t/foo.o
# RUN: %lld -lSystem -bundle -flat_namespace -o %t/foo.bundle %t/foo.o
# RUN: llvm-objdump --macho --syms --rebase --bind --lazy-bind --weak-bind %t/foo | \
# RUN:   FileCheck %s --check-prefixes=SYMS,EXEC --implicit-check-not=_private_extern
# RUN: llvm-objdump --macho --syms --rebase --bind --lazy-bind --weak-bind %t/foo.dylib | \
# RUN:   FileCheck %s --check-prefixes=SYMS,DYLIB --implicit-check-not=_private_extern
# RUN: llvm-objdump --macho --syms --rebase --bind --lazy-bind --weak-bind %t/foo.bundle | \
# RUN:   FileCheck %s --check-prefixes=SYMS,DYLIB --implicit-check-not=_private_extern

# SYMS: SYMBOL TABLE:
# SYMS-DAG: [[#%x, EXTERN_REF:]] l     O __DATA,__data _extern_ref
# SYMS-DAG: [[#%x, LOCAL_REF:]]  l     O __DATA,__data _local_ref
# SYMS-DAG: [[#%x, WEAK_REF:]]   l     O __DATA,__data _weak_ref
# SYMS-DAG: [[#%x, TLV_REF:]]    g     O __DATA,__thread_vars _tlv

## Executables with -flat_namespace don't have interposable externs.
# EXEC:       Rebase table:
# EXEC-NEXT:  segment  section          address                  type
# EXEC-DAG:   __DATA   __la_symbol_ptr  0x[[#%X, WEAK_LAZY:]]    pointer
# EXEC-DAG:   __DATA   __data           0x[[#%.8X, EXTERN_REF]]  pointer
# EXEC-DAG:   __DATA   __data           0x[[#%.8X, LOCAL_REF]]   pointer
# EXEC-DAG:   __DATA   __data           0x[[#%.8X, WEAK_REF]]    pointer
# EXEC-EMPTY:
# EXEC-NEXT:  Bind table:
# EXEC-NEXT:  segment  section          address  type     addend dylib   symbol
# EXEC-EMPTY:
# EXEC-NEXT:  Lazy bind table:
# EXEC-NEXT:  segment  section          address  dylib    symbol
# EXEC-EMPTY:
# EXEC-NEXT:  Weak bind table:
# EXEC-NEXT:  segment  section          address                type     addend   symbol
# EXEC-NEXT:  __DATA   __la_symbol_ptr  0x[[#%.8X, WEAK_LAZY]] pointer       0   _weak_extern
# EXEC-NEXT:  __DATA   __data           0x[[#%.8X, WEAK_REF]]  pointer       0   _weak_extern
# EXEC-EMPTY:

# DYLIB:       Rebase table:
# DYLIB-NEXT:  segment      section          address                  type
# DYLIB-DAG:   __DATA       __la_symbol_ptr  0x[[#%X, WEAK_LAZY:]]    pointer
# DYLIB-DAG:   __DATA       __la_symbol_ptr  0x[[#%X, EXTERN_LAZY:]]  pointer
# DYLIB-DAG:   __DATA       __data           0x[[#%.8X, EXTERN_REF]]  pointer
# DYLIB-DAG:   __DATA       __data           0x[[#%.8X, LOCAL_REF]]   pointer
# DYLIB-DAG:   __DATA       __data           0x[[#%.8X, WEAK_REF]]    pointer
# DYLIB-DAG:   __DATA       __thread_ptrs    0x[[#%.8X, TLV_REF]]     pointer
# DYLIB-EMPTY:
# DYLIB-NEXT:  Bind table:
# DYLIB-NEXT:  segment      section        address                 type     addend dylib            symbol
# DYLIB-DAG:   __DATA_CONST __got          {{.*}}                  pointer       0 flat-namespace   dyld_stub_binder
# DYLIB-DAG:   __DATA       __data         0x[[#%.8X, EXTERN_REF]] pointer       0 flat-namespace   _extern
# DYLIB-DAG:   __DATA       __thread_ptrs  0x[[#%.8X, TLV_REF]]    pointer       0 flat-namespace   _tlv
# DYLIB-EMPTY:
# DYLIB-NEXT:  Lazy bind table:
# DYLIB-NEXT:  segment  section            address                  dylib           symbol
# DYLIB-NEXT:  __DATA   __la_symbol_ptr    0x[[#%.8X, EXTERN_LAZY]] flat-namespace   _extern
# DYLIB-EMPTY:
# DYLIB-NEXT:  Weak bind table:
# DYLIB-NEXT:  segment  section            address                  type    addend   symbol
# DYLIB-NEXT:  __DATA   __la_symbol_ptr    0x[[#%.8X, WEAK_LAZY]]   pointer      0   _weak_extern
# DYLIB-NEXT:  __DATA   __data             0x[[#%.8X, WEAK_REF]]    pointer      0   _weak_extern
# DYLIB-EMPTY:

#--- foo.s

.globl _main, _extern, _weak_extern, _tlv
.weak_definition _weak_extern
.private_extern _private_extern

_extern:
  retq
_weak_extern:
  retq
_private_extern:
  retq
_local:
  retq

_main:
  callq _extern
  callq _weak_extern
  callq _private_extern
  callq _local
  mov _tlv@TLVP(%rip), %rax
  retq

.data
_extern_ref:
.quad _extern
_weak_ref:
.quad _weak_extern
_local_ref:
.quad _local

.section __DATA,__thread_vars,thread_local_variables
_tlv:

.subsections_via_symbols
