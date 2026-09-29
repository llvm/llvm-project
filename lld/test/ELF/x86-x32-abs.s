# REQUIRES: x86
## Test R_X86_64_32 relocations in x32 (ILP32) mode.
## In x32, R_X86_64_32 is the pointer-sized absolute relocation.

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-gnux32 a.s -o a.o
# RUN: ld.lld -shared a.o -o a.so
# RUN: llvm-readelf -r a.so | FileCheck %s

## Non-resolved non-preemptible R_X86_64_32 get R_X86_64_RELATIVE.
# CHECK:      Relocation section '.rela.dyn' at offset {{.*}} contains 2 entries:
# CHECK:      R_X86_64_RELATIVE
# CHECK-NEXT: {{.*}} R_X86_64_32 {{.*}} und + 0

## R_X86_64_64 is not a supported dynamic relocation in x32 mode.
# RUN: llvm-mc -filetype=obj -triple=x86_64-gnux32 b.s -o b.o
# RUN: not ld.lld -shared b.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR

# ERR:      error: relocation R_X86_64_64 cannot be used against symbol 'und'; recompile with -fPIC
# ERR-NEXT: >>> defined in b.o
# ERR-NEXT: >>> referenced by b.o:(.data+0x0)

#--- a.s
.globl hid
.hidden hid

.data
hid:
  .long und
  .long hid

#--- b.s
.data
  .quad und
