# REQUIRES: x86
## Check that group members are retained, if no member has the SHF_ALLOC flag.
## This rule retains .debug_types and .rela.debug_types emitted by clang/gcc.

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: ld.lld --gc-sections %t.o -o %t
# RUN: llvm-readobj -S %t | FileCheck %s

# CHECK: Name: .debug_types

.section .debug_types,"G",@progbits,abcd,comdat
.quad .debug_types
