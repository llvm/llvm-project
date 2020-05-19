# REQUIRES: x86, shell
# RUN: mkdir -p %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %p/Inputs/libhello.s \
# RUN:   -o %t/libhello.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %p/Inputs/libgoodbye.s \
# RUN:   -o %t/libgoodbye.o
# RUN: lld -flavor darwinnew -dylib -install_name \
# RUN:   @executable_path/libhello.dylib %t/libhello.o -o %t/libhello.dylib
# RUN: lld -flavor darwinnew -dylib -install_name \
# RUN:   @executable_path/libgoodbye.dylib %t/libgoodbye.o -o %t/libgoodbye.dylib

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t/dylink-lazy.o
# RUN: lld -flavor darwinnew -o %t/dylink-lazy -L%t -lhello -lgoodbye %t/dylink-lazy.o

## When looking at the __stubs section alone, we are unable to easily tell which
## symbol each entry points to. So we call objdump twice in order to get the
## disassembly of __text and the bind tables first, which allow us to check for
## matching entries in __stubs.
# RUN: (llvm-objdump -d --no-show-raw-insn --syms --bind --lazy-bind %t/dylink-lazy; \
# RUN:  llvm-objdump -D --no-show-raw-insn %t/dylink-lazy) | FileCheck %s

# CHECK-LABEL: SYMBOL TABLE:
# CHECK:       {{0*}}[[#%x, IMGLOADER:]] {{.*}} __DATA,__data __dyld_private

# CHECK-LABEL: Disassembly of section __TEXT,__text:
# CHECK:         callq 0x[[#%x, HELLO_STUB:]]
# CHECK-NEXT:    callq 0x[[#%x, GOODBYE_STUB:]]

# CHECK-LABEL: Bind table:
# CHECK:       __DATA_CONST __got 0x[[#%x, BINDER:]] pointer 0 libSystem dyld_stub_binder

# CHECK-LABEL: Lazy bind table:
# CHECK-DAG:   __DATA __la_symbol_ptr 0x{{0*}}[[#%x, HELLO_LAZY_PTR:]] libhello _print_hello
# CHECK-DAG:   __DATA __la_symbol_ptr 0x{{0*}}[[#%x, GOODBYE_LAZY_PTR:]] libgoodbye _print_goodbye

# CHECK-LABEL: Disassembly of section __TEXT,__stubs:
# CHECK-DAG:     [[#%x, HELLO_STUB]]:   jmpq *[[#%u, HELLO_LAZY_PTR - HELLO_STUB - 6]](%rip)
# CHECK-DAG:     [[#%x, GOODBYE_STUB]]: jmpq *[[#%u, GOODBYE_LAZY_PTR - GOODBYE_STUB - 6]](%rip)

# CHECK-LABEL: Disassembly of section __TEXT,__stub_helper:
# CHECK:         {{0*}}[[#%x, STUB_HELPER_ENTRY:]] <__stub_helper>:
# CHECK-NEXT:    leaq [[#%u, IMGLOADER - STUB_HELPER_ENTRY - 7]](%rip), %r11
# CHECK-NEXT:    pushq %r11
# CHECK-NEXT:    jmpq *[[#%u, BINDER_OFF:]](%rip)
# CHECK-NEXT:    [[#%x, BINDER - BINDER_OFF]]: nop
# CHECK-NEXT:    pushq $0
# CHECK-NEXT:    jmp 0x[[#STUB_HELPER_ENTRY]]
# CHECK-NEXT:    pushq $21
# CHECK-NEXT:    jmp 0x[[#STUB_HELPER_ENTRY]]

.text
.globl _main

_main:
  sub $8, %rsp # 16-byte-align the stack; dyld checks for this
  callq _print_hello
  callq _print_goodbye
  add $8, %rsp
  ret
