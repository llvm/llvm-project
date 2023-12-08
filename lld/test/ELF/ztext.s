# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %p/Inputs/ztext.s -o %t2.o
# RUN: ld.lld %t2.o -o %t2.so -shared -soname=so

# RUN: ld.lld -z notext %t.o %t2.so -o %t -shared
# RUN: llvm-readobj  --dynamic-table -r %t | FileCheck %s
# RUN: ld.lld -z notext %t.o %t2.so -o %t2 -pie
# RUN: llvm-readobj  --dynamic-table -r %t2 | FileCheck %s
# RUN: ld.lld -z notext %t.o %t2.so -o %t3
# RUN: llvm-readobj  --dynamic-table -r %t3 | FileCheck --check-prefix=STATIC %s

# RUN: not ld.lld %t.o %t2.so -o /dev/null -shared 2>&1 | FileCheck --check-prefix=ERR %s
# RUN: not ld.lld -z text %t.o %t2.so -o /dev/null -shared 2>&1 | FileCheck --check-prefix=ERR %s
# ERR: error: relocation R_X86_64_64 cannot be used against symbol 'bar'; recompile with -fPIC

# If the preference is to have text relocations, don't create plt of copy relocations.

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
