# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
# RUN: not ld.lld -z notext -shared %t.o -o /dev/null 2>&1 | FileCheck %s
# CHECK: error: relocation R_X86_64_32 cannot be used against symbol 'foo'; recompile with -fPIC

# RUN: ld.lld -z notext %t.o -o %t
# RUN: llvm-readobj -r %t | FileCheck %s --check-prefix=EXE
# EXE:      Relocations [
# EXE-NEXT: ]

.weak foo

_start:
mov $foo,%eax
