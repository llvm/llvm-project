# REQUIRES: x86
# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/test.s -o %t/test.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/libfoo.s -o %t/libfoo.o
# RUN: %lld -dylib %t/libfoo.o -o %t/libfoo.dylib
# RUN: %lld %t/test.o -L%t -lfoo -o %t/test -lSystem
# RUN: llvm-objdump -d --no-show-raw-insn --bind --lazy-bind --weak-bind --full-contents %t/test | \
# RUN:   FileCheck %s

# CHECK:      Contents of section __DATA_CONST,__got:
## Check that this section contains a nonzero pointer. It should point to
## _weak_external_for_gotpcrel.
# CHECK-NEXT: {{[0-9a-f]+}} {{[0-9a-f ]*[1-9a-f]+[0-9a-f ]*}}

# CHECK:      Contents of section __DATA,__la_symbol_ptr:
## Check that this section contains a nonzero pointer. It should point to
## _weak_external_fn, but we don't have a good way of testing the exact value as
## the bytes here are in little-endian order.
# CHECK-NEXT: {{[0-9a-f]+}} {{[0-9a-f ]*[1-9a-f]+[0-9a-f ]*}}

# CHECK:      <_main>:
# CHECK-NEXT: movq	[[#]](%rip), %rax  # [[#%X,WEAK_DY_GOT_ADDR:]]
# CHECK-NEXT: movq	[[#]](%rip), %rax  # [[#%X,WEAK_EXT_GOT_ADDR:]]
# CHECK-NEXT: leaq	[[#]](%rip), %rax  # [[#%X,WEAK_INT_GOT_ADDR:]]
# CHECK-NEXT: movq	[[#]](%rip), %rax  # [[#%X,WEAK_TLV_ADDR:]]
# CHECK-NEXT: movq	[[#]](%rip), %rax  # [[#%X,WEAK_DY_TLV_ADDR:]]
# CHECK-NEXT: leaq	[[#]](%rip), %rax  # [[#%X,WEAK_INT_TLV_ADDR:]]
# CHECK-NEXT: callq 0x{{[0-9a-f]*}}
# CHECK-NEXT: callq 0x{{[0-9a-f]*}}
# CHECK-NEXT: callq 0x{{[0-9a-f]*}}

# CHECK-LABEL: Bind table:
# CHECK-DAG:   __DATA_CONST  __got           0x[[#WEAK_DY_GOT_ADDR]] pointer 0 libfoo    _weak_dysym_for_gotpcrel
# CHECK-DAG:   __DATA        __la_symbol_ptr 0x[[#%x,WEAK_DY_FN:]]   pointer 0 libfoo    _weak_dysym_fn
# CHECK-DAG:   __DATA        __data          0x[[#%x,WEAK_DY:]]      pointer 0 libfoo    _weak_dysym
# CHECK-DAG:   __DATA        __thread_vars   0x{{[0-9a-f]*}}         pointer 0 libSystem __tlv_bootstrap
# CHECK-DAG:   __DATA        __thread_ptrs   0x[[#WEAK_DY_TLV_ADDR]] pointer 0 libfoo    _weak_dysym_tlv
## Check that we don't have any other bindings
# CHECK-NOT:   pointer

# CHECK-LABEL: Lazy bind table:
## Verify that we have no lazy bindings
# CHECK-NOT:   pointer

# CHECK-LABEL: Weak bind table:
# CHECK-DAG:   __DATA_CONST __got           0x[[#WEAK_DY_GOT_ADDR]]   pointer 0 _weak_dysym_for_gotpcrel
# CHECK-DAG:   __DATA_CONST __got           0x[[#WEAK_EXT_GOT_ADDR]]  pointer 0 _weak_external_for_gotpcrel
# CHECK-DAG:   __DATA       __data          0x[[#WEAK_DY]]            pointer 0 _weak_dysym
# CHECK-DAG:   __DATA       __thread_ptrs   0x[[#WEAK_TLV_ADDR]]      pointer 0 _weak_tlv
# CHECK-DAG:   __DATA       __thread_ptrs   0x[[#WEAK_DY_TLV_ADDR]]   pointer 0 _weak_dysym_tlv
# CHECK-DAG:   __DATA       __data          0x{{[0-9a-f]*}}           pointer 2 _weak_external
# CHECK-DAG:   __DATA       __la_symbol_ptr 0x[[#WEAK_DY_FN]]         pointer 0 _weak_dysym_fn
# CHECK-DAG:   __DATA       __la_symbol_ptr 0x{{[0-9a-f]*}}           pointer 0 _weak_external_fn
## Check that we don't have any other bindings
# CHECK-NOT:   pointer

## Weak internal symbols don't get bindings
# RUN: llvm-objdump --macho --bind --lazy-bind --weak-bind %t/test | FileCheck %s --check-prefix=WEAK-INTERNAL
# WEAK-INTERNAL-NOT: _weak_internal
# WEAK-INTERNAL-NOT: _weak_internal_fn
# WEAK-INTERNAL-NOT: _weak_internal_tlv

#--- libfoo.s

.globl _weak_dysym
.weak_definition _weak_dysym
_weak_dysym:
  .quad 0x1234

.globl _weak_dysym_for_gotpcrel
.weak_definition _weak_dysym_for_gotpcrel
_weak_dysym_for_gotpcrel:
  .quad 0x1234

.globl _weak_dysym_fn
.weak_definition _weak_dysym_fn
_weak_dysym_fn:
  ret

.section __DATA,__thread_vars,thread_local_variables

.globl _weak_dysym_tlv
.weak_definition _weak_dysym_tlv
_weak_dysym_tlv:
  .quad 0x1234

#--- test.s

.globl _main, _weak_external, _weak_external_for_gotpcrel, _weak_external_fn
.weak_definition _weak_external, _weak_external_for_gotpcrel, _weak_external_fn, _weak_internal, _weak_internal_for_gotpcrel, _weak_internal_fn

_main:
  mov _weak_dysym_for_gotpcrel@GOTPCREL(%rip), %rax
  mov _weak_external_for_gotpcrel@GOTPCREL(%rip), %rax
  mov _weak_internal_for_gotpcrel@GOTPCREL(%rip), %rax
  mov _weak_tlv@TLVP(%rip), %rax
  mov _weak_dysym_tlv@TLVP(%rip), %rax
  mov _weak_internal_tlv@TLVP(%rip), %rax
  callq _weak_dysym_fn
  callq _weak_external_fn
  callq _weak_internal_fn
  mov $0, %rax
  ret

_weak_external:
  .quad 0x1234

_weak_external_for_gotpcrel:
  .quad 0x1234

_weak_external_fn:
  ret

_weak_internal:
  .quad 0x1234

_weak_internal_for_gotpcrel:
  .quad 0x1234

_weak_internal_fn:
  ret

.data
  .quad _weak_dysym
  .quad _weak_external + 2
  .quad _weak_internal

.tbss _weak_tlv$tlv$init, 4, 2
.tbss _weak_internal_tlv$tlv$init, 4, 2

.section __DATA,__thread_vars,thread_local_variables
.globl _weak_tlv
.weak_definition  _weak_tlv, _weak_internal_tlv

_weak_tlv:
  .quad __tlv_bootstrap
  .quad 0
  .quad _weak_tlv$tlv$init

_weak_internal_tlv:
  .quad __tlv_bootstrap
  .quad 0
  .quad _weak_internal_tlv$tlv$init
