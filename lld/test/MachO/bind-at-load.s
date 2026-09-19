# REQUIRES: x86
# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/dylib.s -o %t/dylib.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/main.s -o %t/main.o
# RUN: %lld -lSystem -dylib %t/dylib.o -o %t/libfoo.dylib
# RUN: %lld -e _main -lSystem %t/main.o %t/libfoo.dylib -o %t/default
# RUN: %lld -e _main -lSystem -bind_at_load %t/main.o %t/libfoo.dylib -o %t/bound

## By default, the stub's pointer is lazily bound: __stub_helper exists and
## _foo appears in the lazy bind opcodes.
# RUN: llvm-objdump --macho --section-headers %t/default | FileCheck %s --check-prefix=DEFAULT
# RUN: obj2yaml %t/default | FileCheck %s --check-prefix=DEFAULT-LAZY

## With -bind_at_load, every bind is eager: no __stub_helper, no
## dyld_stub_binder, no lazy bind opcodes, and _foo is bound in the
## non-lazy bind opcodes at the (formerly lazy) pointer's location.
# RUN: llvm-objdump --macho --section-headers %t/bound | FileCheck %s --check-prefix=BIND-SECTIONS
# RUN: obj2yaml %t/bound | FileCheck %s --check-prefix=BIND-BIND

# DEFAULT:      __stub_helper

# DEFAULT-LAZY:      LazyBindOpcodes:
# DEFAULT-LAZY:      Symbol:          _foo

# BIND-SECTIONS:      __stubs
# BIND-SECTIONS-NOT:  __stub_helper
# BIND-SECTIONS:      __la_symbol_ptr

# BIND-BIND:      BindOpcodes:
# BIND-BIND:      Symbol:          _foo
# BIND-BIND-NOT:  LazyBindOpcodes

#--- dylib.s
.section __TEXT,__text,regular,pure_instructions
.globl _foo
_foo:
  ret

#--- main.s
.section __TEXT,__text,regular,pure_instructions
.globl _main
_main:
  callq _foo
  ret
