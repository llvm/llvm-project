# REQUIRES: x86
# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/libfoo.s -o %t/libfoo.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/strongref.s -o %t/strongref.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/test.s -o %t/test.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/invalid.s -o %t/invalid.o
# RUN: %lld -lSystem -dylib %t/libfoo.o -o %t/libfoo.dylib

# RUN: %lld -lSystem %t/test.o %t/libfoo.dylib -o %t/test
# RUN: llvm-objdump --macho --syms --bind %t/test | FileCheck %s --check-prefixes=SYMS,BIND
## llvm-objdump doesn't print out all the flags info for lazy & weak bindings,
## so we use obj2yaml instead to test them.
# RUN: obj2yaml %t/test | FileCheck %s --check-prefix=YAML

# RUN: %lld -lSystem %t/libfoo.dylib %t/test.o -o %t/test
# RUN: llvm-objdump --macho --syms --bind %t/test | FileCheck %s --check-prefixes=SYMS,BIND
# RUN: obj2yaml %t/test | FileCheck %s --check-prefix=YAML

# SYMS:     SYMBOL TABLE:
# SYMS-DAG: 0000000000000000  w  *UND* _foo
# SYMS-DAG: 0000000000000000  w  *UND* _foo_fn
# SYMS-DAG: 0000000000000000  w  *UND* _foo_tlv
# SYMS-DAG: 0000000000000000  w  *UND* _weak_foo
# SYMS-DAG: 0000000000000000  w  *UND* _weak_foo_fn

# BIND:      Bind table:
# BIND-NEXT: segment       section          address         type     addend dylib   symbol
# BIND-DAG:  __DATA        __data           0x{{[0-9a-f]+}} pointer       0 libfoo  _foo (weak_import)
# BIND-DAG:  __DATA_CONST  __got            0x{{[0-9a-f]+}} pointer       0 libfoo  _foo (weak_import)
# BIND-DAG:  __DATA        __thread_ptrs    0x{{[0-9a-f]+}} pointer       0 libfoo  _foo_tlv (weak_import)
# BIND-DAG:  __DATA        __data           0x{{[0-9a-f]+}} pointer       0 libfoo  _weak_foo (weak_import)
# BIND-DAG:  __DATA        __la_symbol_ptr  0x{{[0-9a-f]+}} pointer       0 libfoo  _weak_foo_fn (weak_import)

# YAML-LABEL: WeakBindOpcodes:
# YAML:        - Opcode:          BIND_OPCODE_SET_SYMBOL_TRAILING_FLAGS_IMM
# YAML-NEXT:     Imm:             0
# YAML-NEXT:     Symbol:          _weak_foo
# YAML:        - Opcode:          BIND_OPCODE_SET_SYMBOL_TRAILING_FLAGS_IMM
# YAML-NEXT:     Imm:             0
# YAML-NEXT:     Symbol:          _weak_foo_fn
# YAML-LABEL: LazyBindOpcodes:
# YAML:        - Opcode:          BIND_OPCODE_SET_SYMBOL_TRAILING_FLAGS_IMM
# YAML-NEXT:     Imm:             1
# YAML-NEXT:     Symbol:          _foo_fn

## Check that if both strong & weak references are present in inputs, the weak
## reference takes priority. NOTE: ld64 actually emits a strong reference if
## the reference is to a function symbol or a TLV. I'm not sure if there's a
## good reason for that, so I'm deviating here for a simpler implementation.
# RUN: %lld -lSystem %t/test.o %t/strongref.o %t/libfoo.dylib -o %t/with-strong
# RUN: llvm-objdump --macho --bind %t/with-strong | FileCheck %s --check-prefix=STRONG-BIND
# RUN: obj2yaml %t/with-strong | FileCheck %s --check-prefix=STRONG-YAML
# RUN: %lld -lSystem %t/strongref.o %t/test.o %t/libfoo.dylib -o %t/with-strong
# RUN: llvm-objdump --macho --bind %t/with-strong | FileCheck %s --check-prefix=STRONG-BIND
# RUN: obj2yaml %t/with-strong | FileCheck %s --check-prefix=STRONG-YAML
# RUN: %lld -lSystem %t/libfoo.dylib %t/strongref.o %t/test.o -o %t/with-strong
# RUN: llvm-objdump --macho --bind %t/with-strong | FileCheck %s --check-prefix=STRONG-BIND
# RUN: obj2yaml %t/with-strong | FileCheck %s --check-prefix=STRONG-YAML
# RUN: %lld -lSystem %t/libfoo.dylib %t/test.o %t/strongref.o -o %t/with-strong
# RUN: llvm-objdump --macho --bind %t/with-strong | FileCheck %s --check-prefix=STRONG-BIND
# RUN: obj2yaml %t/with-strong | FileCheck %s --check-prefix=STRONG-YAML
# RUN: %lld -lSystem %t/test.o %t/libfoo.dylib %t/strongref.o -o %t/with-strong
# RUN: llvm-objdump --macho --bind %t/with-strong | FileCheck %s --check-prefix=STRONG-BIND
# RUN: obj2yaml %t/with-strong | FileCheck %s --check-prefix=STRONG-YAML
# RUN: %lld -lSystem %t/strongref.o %t/libfoo.dylib %t/test.o -o %t/with-strong
# RUN: llvm-objdump --macho --bind %t/with-strong | FileCheck %s --check-prefix=STRONG-BIND
# RUN: obj2yaml %t/with-strong | FileCheck %s --check-prefix=STRONG-YAML

# STRONG-BIND:      Bind table:
# STRONG-BIND-NEXT: segment       section          address         type       addend dylib   symbol
# STRONG-BIND-DAG:  __DATA        __data           0x{{[0-9a-f]+}} pointer         0 libfoo  _foo{{$}}
# STRONG-BIND-DAG:  __DATA        __data           0x{{[0-9a-f]+}} pointer         0 libfoo  _foo{{$}}
# STRONG-BIND-DAG:  __DATA_CONST  __got            0x{{[0-9a-f]+}} pointer         0 libfoo  _foo{{$}}
# STRONG-BIND-DAG:  __DATA        __thread_ptrs    0x{{[0-9a-f]+}} pointer         0 libfoo  _foo_tlv{{$}}
# STRONG-BIND-DAG:  __DATA        __data           0x{{[0-9a-f]+}} pointer         0 libfoo  _weak_foo{{$}}
# STRONG-BIND-DAG:  __DATA        __data           0x{{[0-9a-f]+}} pointer         0 libfoo  _weak_foo{{$}}
# STRONG-BIND-DAG:  __DATA        __la_symbol_ptr  0x{{[0-9a-f]+}} pointer         0 libfoo  _weak_foo_fn{{$}}

# STRONG-YAML-LABEL: WeakBindOpcodes:
# STRONG-YAML:        - Opcode:          BIND_OPCODE_SET_SYMBOL_TRAILING_FLAGS_IMM
# STRONG-YAML-NEXT:     Imm:             0
# STRONG-YAML-NEXT:     Symbol:          _weak_foo
# STRONG-YAML:        - Opcode:          BIND_OPCODE_SET_SYMBOL_TRAILING_FLAGS_IMM
# STRONG-YAML-NEXT:     Imm:             0
# STRONG-YAML-NEXT:     Symbol:          _weak_foo
# STRONG-YAML:        - Opcode:          BIND_OPCODE_SET_SYMBOL_TRAILING_FLAGS_IMM
# STRONG-YAML-NEXT:     Imm:             0
# STRONG-YAML-NEXT:     Symbol:          _weak_foo_fn
# STRONG-YAML-LABEL: LazyBindOpcodes:
# STRONG-YAML:        - Opcode:          BIND_OPCODE_SET_SYMBOL_TRAILING_FLAGS_IMM
# STRONG-YAML-NEXT:     Imm:             0
# STRONG-YAML-NEXT:     Symbol:          _foo_fn

## Weak references must still be satisfied at link time.
# RUN: not %lld -lSystem %t/invalid.o -o /dev/null 2>&1 | FileCheck %s \
# RUN:   --check-prefix=INVALID -DDIR=%t
# INVALID: error: undefined symbol: _missing

#--- libfoo.s
.globl _foo, _foo_fn, _weak_foo, _weak_foo_fn
.weak_definition _weak_foo, _weak_foo_fn
_foo:
_foo_fn:
_weak_foo:
_weak_foo_fn:

.section	__DATA,__thread_vars,thread_local_variables
.globl _foo_tlv
_foo_tlv:

#--- test.s
.globl _main
.weak_reference _foo_fn, _foo, _weak_foo, _weak_foo_fn, _foo_tlv

_main:
  mov _foo@GOTPCREL(%rip), %rax
  mov _foo_tlv@TLVP(%rip), %rax
  callq _foo_fn
  callq _weak_foo_fn
  ret

.data
  .quad _foo
  .quad _weak_foo

#--- strongref.s
.globl _strongref
_strongref:
  mov _foo@GOTPCREL(%rip), %rax
  mov _foo_tlv@TLVP(%rip), %rax
  callq _foo_fn
  callq _weak_foo_fn
  ret

.data
  .quad _foo
  .quad _weak_foo

#--- invalid.s
.globl _main
.weak_reference _missing
_main:
  callq _missing
  ret
