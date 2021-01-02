# REQUIRES: x86
# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/libfoo.s -o %t/libfoo.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/test.s -o %t/test.o
# RUN: %lld -dylib %t/libfoo.o -o %t/libfoo.dylib -lSystem
# RUN: %lld %t/test.o %t/libfoo.dylib -o %t/test -lSystem
# RUN: llvm-objdump --macho -d --no-show-raw-insn --indirect-symbols %t/test | FileCheck %s

# CHECK:      (__TEXT,__text) section
# CHECK-NEXT: _main:
# CHECK-NEXT: movq	{{.*}}(%rip), %rax ## literal pool symbol address: _foo
# CHECK-NEXT: movq	{{.*}}(%rip), %rax ## literal pool symbol address: _bar
# CHECK-NEXT: movq	{{.*}}(%rip), %rax ## literal pool symbol address: _foo_tlv
# CHECK-NEXT: movq	{{.*}}(%rip), %rax ## literal pool symbol address: _bar_tlv
# CHECK-NEXT: callq	{{.*}} ## symbol stub for: _foo_fn
# CHECK-NEXT: callq	{{.*}} ## symbol stub for: _bar_fn
# CHECK-NEXT: retq

# CHECK:      Indirect symbols for (__TEXT,__stubs) 2 entries
# CHECK-NEXT: address            index name
# CHECK-NEXT: _bar_fn
# CHECK-NEXT: _foo_fn
# CHECK-NEXT: Indirect symbols for (__DATA_CONST,__got) 3 entries
# CHECK-NEXT: address            index name
# CHECK-NEXT: _bar
# CHECK-NEXT: _foo
# CHECK-NEXT: _stub_binder
# CHECK-NEXT: Indirect symbols for (__DATA,__la_symbol_ptr) 2 entries
# CHECK-NEXT: address            index name
# CHECK-NEXT: _bar_fn
# CHECK-NEXT: _foo_fn
# CHECK-NEXT: Indirect symbols for (__DATA,__thread_ptrs) 2 entries
# CHECK-NEXT: address            index name
# CHECK-NEXT: _bar_tlv
# CHECK-NEXT: _foo_tlv

#--- libfoo.s

.globl _foo, _foo_fn, _bar, _bar_fn
_foo:
_foo_fn:
_bar:
_bar_fn:

.section  __DATA,__thread_vars,thread_local_variables
.globl _foo_tlv, _bar_tlv
_foo_tlv:
_bar_tlv:

#--- test.s

.globl _main
_main:
  movq _foo@GOTPCREL(%rip), %rax
  movq _bar@GOTPCREL(%rip), %rax
  mov _foo_tlv@TLVP(%rip), %rax
  mov _bar_tlv@TLVP(%rip), %rax
  callq _foo_fn
  callq _bar_fn
  ret
