# REQUIRES: x86

## Don't relax R_X86_64_GOTPCRELX to an absolute symbol.
## In -no-pie mode, it can be relaxed, but it may not worth it.

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: ld.lld %t.o -o %t.so -shared
# RUN: llvm-objdump --no-print-imm-hex -d %t.so | FileCheck %s

# CHECK: movq  4209(%rip), %rax

	movq    bar@GOTPCREL(%rip), %rax
        .data
        .global bar
        .hidden bar
        bar = 42
