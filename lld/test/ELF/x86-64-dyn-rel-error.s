# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %p/Inputs/shared.s -o %t2.o
# RUN: ld.lld %t2.o -shared -o %t2.so --threads=1
# RUN: not ld.lld -pie %t.o %t2.so -o /dev/null --threads=1 2>&1 | FileCheck %s
# RUN: not ld.lld -shared %t.o %t2.so -o /dev/null --threads=1 2>&1 | FileCheck %s

# CHECK:      error: relocation R_X86_64_32 cannot be used against symbol 'zed'; recompile with -fPIC
# CHECK-NEXT: >>> defined in {{.*}}.so
# CHECK-NEXT: >>> referenced by {{.*}}.o:(.data+0x0)
# CHECK-EMPTY:
# CHECK-NEXT: error: relocation R_X86_64_PC32 cannot be used against symbol 'zed'; recompile with -fPIC
# CHECK-NEXT: >>> defined in {{.*}}.so
# CHECK-NEXT: >>> referenced by {{.*}}.o:(.data+0x4)
# CHECK-EMPTY:
# CHECK-NEXT: error: relocation R_X86_64_64 cannot be used against symbol '_start'; recompile with -fPIC
# CHECK:      error: relocation R_X86_64_64 cannot be used against symbol 'main'; recompile with -fPIC
# CHECK:      error: relocation R_X86_64_64 cannot be used against symbol 'data'; recompile with -fPIC
# CHECK-NOT:  error:

# RUN: ld.lld --noinhibit-exec %t.o %t2.so -o /dev/null 2>&1 | FileCheck --check-prefix=WARN %s
# RUN: not ld.lld --export-dynamic --unresolved-symbols=ignore-all %t.o %t2.so -o /dev/null 2>&1 | FileCheck --check-prefix=WARN %s

# WARN: relocation R_X86_64_32 cannot be used against symbol 'zed'; recompile with -fPIC
# WARN: relocation R_X86_64_PC32 cannot be used against symbol 'zed'; recompile with -fPIC

        .global _start, main, data
        .type main, @function
        .type data, @object
_start:
  ret
main:
  ret

.data
data:
.long zed
.long zed - .

.rodata
.quad _start
.quad main
.quad data
