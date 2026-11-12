# REQUIRES: x86
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 a.s -o a.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 b.s -o b.o
# RUN: ld.lld b.o -o b.so -shared -soname=so

## -z notext allows text relocations for certain relocation types.
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

## R_X86_64_PC64 is not a supported dynamic relocation. It errors even with -z notext.
# RUN: llvm-mc -filetype=obj -triple=x86_64 pc64.s -o pc64.o
# RUN: not ld.lld -z notext pc64.o b.so -shared 2>&1 | FileCheck --check-prefix=ERR-PC64 %s

# ERR-PC64: error: relocation R_X86_64_PC64 cannot be used against symbol 'bar'; recompile with -fPIC
# ERR-PC64-NEXT: >>> defined in b.so
# ERR-PC64-NEXT: >>> referenced by pc64.o:(.text+0x0)

# CHECK: DynamicSection [
# CHECK:   FLAGS TEXTREL
# CHECK:   TEXTREL 0x0

# CHECK:      Relocations [
# CHECK-NEXT:   Section {{.*}} .rela.dyn {
# CHECK-NEXT:     0x1268 R_X86_64_RELATIVE - 0x1268
# CHECK-NEXT:     0x1270 R_X86_64_64 bar 0x0
# CHECK-NEXT:   }
# CHECK-NEXT: ]

# STATIC: DynamicSection [
# STATIC:   FLAGS TEXTREL
# STATIC:   TEXTREL 0x0

# STATIC:      Relocations [
# STATIC-NEXT:   Section {{.*}} .rela.dyn {
# STATIC-NEXT:     0x201258 R_X86_64_64 bar 0x0
# STATIC-NEXT:   }
# STATIC-NEXT: ]

#--- a.s
foo:
.quad foo
.quad bar

#--- b.s
.global bar
.type bar, @object
.size bar, 8
bar:
.quad 0

#--- pc64.s
.quad bar - .
