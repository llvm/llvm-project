# REQUIRES: x86
# RUN: split-file %s %t

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/libtlv.s -o %t/libtlv.o
# RUN: %lld -dylib -install_name @executable_path/libtlv.dylib \
# RUN:   -lSystem -o %t/libtlv.dylib %t/libtlv.o
# RUN: llvm-objdump --exports-trie -d --no-show-raw-insn %t/libtlv.dylib | FileCheck %s --check-prefix=DYLIB
# DYLIB-DAG: _foo [per-thread]
# DYLIB-DAG: _bar [per-thread]

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/test.s -o %t/test.o
# RUN: %lld -lSystem -L%t -ltlv %t/test.o -o %t/test
# RUN: llvm-objdump --bind -d --no-show-raw-insn %t/test | FileCheck %s

# CHECK:      movq [[#]](%rip), %rax # [[#%x, FOO:]]
# CHECK-NEXT: movq [[#]](%rip), %rax # [[#%x, BAR:]]
# CHECK-NEXT: movq [[#]](%rip), %rax # [[#%x, BAZ:]]

# CHECK-LABEL: Bind table:
# CHECK-DAG: __DATA       __thread_ptrs  0x{{0*}}[[#%x, FOO]] pointer 0   libtlv   _foo
# CHECK-DAG: __DATA       __thread_ptrs  0x{{0*}}[[#%x, BAR]] pointer 0   libtlv   _bar
# CHECK-DAG: __DATA_CONST __got          0x{{0*}}[[#%x, BAZ]] pointer 0   libtlv   _baz

#--- libtlv.s
.section	__DATA,__thread_vars,thread_local_variables
.globl _foo, _bar, _baz
_foo:
_bar:

.text
_baz:

#--- test.s
.globl _main
_main:
  mov _foo@TLVP(%rip), %rax
  mov _bar@TLVP(%rip), %rax
## Add a GOT entry to make sure we don't mix it up with TLVs
  mov _baz@GOTPCREL(%rip), %rax
  ret
