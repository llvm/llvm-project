// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
// RUN: ld.lld %t.o -o %t.so -shared
// RUN: llvm-readelf --symbols -S %t.so | FileCheck %s

        .section        .rodata.cst4,"aM",@progbits,4
        .short 0
foo:
        .short 42

        .short 0
bar:
        .short 42

// CHECK:      Name    Type     Address        {{.*}} ES Flg
// CHECK:      .rodata PROGBITS [[#%x, ADDR:]] {{.*}} 04  AM{{ }}

// CHECK:      Symbol table '.symtab' contains {{.*}} entries:
// CHECK-NEXT:    Num:    Value          {{.*}} Name
// CHECK-DAG:  {{.*}}: {{0*}}[[#ADDR+2]] {{.*}} foo
// CHECK-DAG:  {{.*}}: {{0*}}[[#ADDR+2]] {{.*}} bar
