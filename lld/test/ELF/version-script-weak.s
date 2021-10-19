# REQUIRES: x86

## An weak undefined symbol does not fetch the lazy definition.
## Version scripts do not affect undefined symbols.

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: echo '.globl foo; foo:' | llvm-mc -filetype=obj -triple=x86_64 - -o %t1.o
# RUN: echo "{ local: *; };" > %t.script
# RUN: ld.lld -shared --version-script %t.script %t.o --start-lib %t1.o --end-lib -o %t.so
# RUN: llvm-readobj --dyn-syms -r %t.so | FileCheck %s

# CHECK:      Relocations [
# CHECK-NEXT:   Section ({{.*}}) .rela.plt {
# CHECK-NEXT:     R_X86_64_JUMP_SLOT foo
# CHECK-NEXT:   }
# CHECK-NEXT: ]
# CHECK:      Symbol {
# CHECK:        Name: foo
# CHECK-NEXT:   Value: 0x0
# CHECK-NEXT:   Size: 0
# CHECK-NEXT:   Binding: Weak
# CHECK-NEXT:   Type: None
# CHECK-NEXT:   Other: 0
# CHECK-NEXT:   Section: Undefined
# CHECK-NEXT: }

## The version of an unfetched lazy symbol is VER_NDX_GLOBAL. It is not affected by version scripts.
# RUN: echo "v1 { *; };" > %t2.script
# RUN: ld.lld -shared --version-script %t2.script %t.o --start-lib %t1.o --end-lib -o %t2.so
# RUN: llvm-readelf --dyn-syms %t2.so | FileCheck %s --check-prefix=CHECK2

# CHECK2: NOTYPE WEAK DEFAULT UND foo{{$}}

# RUN: ld.lld -shared --soname=tshared --version-script %t2.script %t1.o -o %tshared.so
# RUN: ld.lld -shared --version-script %t2.script %t.o --start-lib %t1.o --end-lib %tshared.so -o %t3.so
# RUN: llvm-readelf --dyn-syms %t3.so | FileCheck %s --check-prefix=CHECK3

# CHECK3: NOTYPE WEAK DEFAULT UND foo@v1

.text
 callq foo@PLT
.weak foo
