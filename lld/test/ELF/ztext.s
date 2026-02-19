# REQUIRES: x86
# RUN: rm -rf %t && mkdir %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o a.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %p/Inputs/ztext.s -o b.o
# RUN: ld.lld b.o -o b.so -shared -soname=so

# RUN: ld.lld -z notext a.o b.so -o out -shared
# RUN: llvm-readobj --dynamic-table -r out | FileCheck %s
# RUN: ld.lld -z notext a.o b.so -o out.pie -pie
# RUN: llvm-readobj --dynamic-table -r out.pie | FileCheck %s
# RUN: ld.lld -z notext a.o b.so -o out.exe
# RUN: llvm-readobj --dynamic-table -r out.exe | FileCheck --check-prefix=STATIC %s

# RUN: not ld.lld a.o b.so -shared 2>&1 | FileCheck --check-prefix=ERR %s --implicit-check-not=error:
# RUN: not ld.lld -z text a.o b.so -shared 2>&1 | FileCheck --check-prefix=ERR %s --implicit-check-not=error:

# ERR:      error: relocation R_X86_64_64 cannot be used against local symbol; recompile with -fPIC
# ERR-NEXT: >>> defined in a.o
# ERR-NEXT: >>> referenced by a.o:(.text+0x0)
# ERR:      error: relocation R_X86_64_64 cannot be used against symbol 'bar'; recompile with -fPIC
# ERR-NEXT: >>> defined in b.so
# ERR-NEXT: >>> referenced by a.o:(.text+0x8)
# ERR:      error: relocation R_X86_64_PC64 cannot be used against symbol 'zed'; recompile with -fPIC
# ERR-NEXT: >>> defined in b.so
# ERR-NEXT: >>> referenced by a.o:(.text+0x10)

## If the preference is to have text relocations, don't create plt of copy relocations.

# CHECK: DynamicSection [
# CHECK:   FLAGS TEXTREL
# CHECK:   TEXTREL 0x0

# CHECK:      Relocations [
# CHECK-NEXT:   Section {{.*}} .rela.dyn {
# CHECK-NEXT:     0x12A0 R_X86_64_RELATIVE - 0x12A0
# CHECK-NEXT:     0x12A8 R_X86_64_64 bar 0x0
# CHECK-NEXT:     0x12B0 R_X86_64_PC64 zed 0x0
# CHECK-NEXT:   }
# CHECK-NEXT: ]

# STATIC: DynamicSection [
# STATIC:   FLAGS TEXTREL
# STATIC:   TEXTREL 0x0

# STATIC:      Relocations [
# STATIC-NEXT:   Section {{.*}} .rela.dyn {
# STATIC-NEXT:     0x201290 R_X86_64_64 bar 0x0
# STATIC-NEXT:     0x201298 R_X86_64_PC64 zed 0x0
# STATIC-NEXT:   }
# STATIC-NEXT: ]

foo:
.quad foo
.quad bar
.quad zed - .
