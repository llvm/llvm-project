# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos %s -o %t.o
# RUN: %lld -dylib %t.o -o %t.dylib -lSystem

# RUN: llvm-objdump --macho --bind --weak-bind %t.dylib | FileCheck %s
# CHECK-NOT: __got
# CHECK-NOT: __la_symbol_ptr

# RUN: llvm-objdump --macho --private-header %t.dylib | \
# RUN:     FileCheck --check-prefix=HEADERS %s
# HEADERS-NOT: WEAK_DEFINES
# HEADERS-NOT: BINDS_TO_WEAK

## Check that N_WEAK_DEF isn't set in the symbol table.
## This is different from ld64, which makes private extern weak symbols non-weak
## for binds and relocations, but it still marks them as weak in the symbol table.
## Since `nm -m` doesn't look at N_WEAK_DEF for N_PEXT symbols this is not
## observable via nm, but it feels slightly more correct.
## (It is observable in `llvm-objdump --syms` output.)
# RUN: llvm-readobj --syms %t.dylib | FileCheck --check-prefix=SYMS %s
# SYMS-NOT: WeakDef (0x80)

.globl _use
_use:
  mov _weak_private_extern_gotpcrel@GOTPCREL(%rip), %rax
  callq _weak_private_extern
  retq

.private_extern _weak_private_extern
.globl _weak_private_extern
.weak_definition _weak_private_extern
_weak_private_extern:
  retq

.private_extern _weak_private_extern_gotpcrel
.globl _weak_private_extern_gotpcrel
.weak_definition _weak_private_extern_gotpcrel
_weak_private_extern_gotpcrel:
  .quad 0x1234
