# REQUIRES: x86

## Test canonical PLT can be created for -no-pie, but not for an absolute
## reference in -pie.

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %p/Inputs/canonical-plt-pcrel.s -o %t1.o
# RUN: ld.lld %t1.o -o %t1.so -shared -soname=so

# RUN: ld.lld %t.o %t1.so -o %t
# RUN: llvm-readobj -r %t | FileCheck %s

# RUN: not ld.lld %t.o %t1.so -o /dev/null -pie 2>&1 | FileCheck %s --check-prefix=PIE --implicit-check-not=error:

# CHECK:      Relocations [
# CHECK-NEXT:   .rela.plt {
# CHECK-NEXT:     R_X86_64_JUMP_SLOT func 0x0
# CHECK-NEXT:     R_X86_64_JUMP_SLOT ifunc 0x0
# CHECK-NEXT:   }
# CHECK-NEXT: ]

# PIE:      error: relocation R_X86_64_64 cannot be used against symbol 'func'; recompile with -fPIC
# PIE-NEXT: >>> defined in {{.*}}.so
# PIE-NEXT: >>> referenced by {{.*}}.o:(.rodata+0x0)
# PIE:      error: relocation R_X86_64_64 cannot be used against symbol 'ifunc'; recompile with -fPIC
# PIE-NEXT: >>> defined in {{.*}}.so
# PIE-NEXT: >>> referenced by {{.*}}.o:(.rodata+0x8)

.rodata
  .quad func    # STT_FUNC
  .quad ifunc   # STT_GNU_IFUNC
