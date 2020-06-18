# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t

## Default run: sections foo and bar exist in output
# RUN: ld.lld %t -o %tout
# RUN: llvm-objdump -d %tout | FileCheck --check-prefix=DISASM %s

## Check that foo and bar sections are not garbage collected,
## we do not want to reclaim sections if they are referred
## by __start_* and __stop_* symbols.
# RUN: ld.lld %t --gc-sections -o %tout
# RUN: llvm-objdump -d %tout | FileCheck --check-prefix=DISASM %s

# RUN: echo ".global __start_foo; __start_foo:" > %t2.s
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %t2.s -o %t2.o
# RUN: ld.lld -shared %t2.o -o %t2.so
# RUN: ld.lld %t --gc-sections -o %tout %t2.so
# RUN: llvm-objdump -d %tout | FileCheck --check-prefix=DISASM %s

# DISASM:      <_start>:
# DISASM-NEXT:   callq   {{.*}} <__start_foo>
# DISASM-NEXT:   callq   {{.*}} <__stop_bar>
# DISASM-EMPTY:
# DISASM-NEXT: Disassembly of section foo:
# DISASM-EMPTY:
# DISASM-NEXT: <__start_foo>:
# DISASM-NEXT:   nop
# DISASM-EMPTY:
# DISASM-NEXT: Disassembly of section bar:
# DISASM-EMPTY:
# DISASM-NEXT: <bar>:
# DISASM-NEXT:   nop

.global _start
.text
_start:
 callq __start_foo
 callq __stop_bar

.section foo,"ax"
 nop

.section bar,"ax"
 nop
