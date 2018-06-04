// RUN: %clang -x c -ffixed-point -S -emit-llvm -o - %s | FileCheck %s
// RUN: %clang -x c -ffixed-point -S -emit-llvm -o - --target=x86_64-scei-ps4-ubuntu-fast %s | FileCheck %s
// RUN: %clang -x c -ffixed-point -S -emit-llvm -o - --target=ppc64 %s | FileCheck %s
// RUN: %clang -x c -ffixed-point -S -emit-llvm -o - --target=x86_64-scei-ps4-windows10pro-fast %s | FileCheck %s

int size_SsA = sizeof(signed short _Accum);
int size_SA  = sizeof(signed _Accum);
int size_SlA = sizeof(signed long _Accum);
int align_SsA = __alignof(signed short _Accum);
int align_SA  = __alignof(signed _Accum);
int align_SlA = __alignof(signed long _Accum);

int size_UsA = sizeof(unsigned short _Accum);
int size_UA  = sizeof(unsigned _Accum);
int size_UlA = sizeof(unsigned long _Accum);
int align_UsA = __alignof(unsigned short _Accum);
int align_UA  = __alignof(unsigned _Accum);
int align_UlA = __alignof(unsigned long _Accum);

int size_sA = sizeof(short _Accum);
int size_A  = sizeof(_Accum);
int size_lA = sizeof(long _Accum);
int align_sA = __alignof(short _Accum);
int align_A  = __alignof(_Accum);
int align_lA = __alignof(long _Accum);

// CHECK:      @size_SsA  = {{.*}}global i{{[0-9]+}} 2
// CHECK-NEXT: @size_SA   = {{.*}}global i{{[0-9]+}} 4
// CHECK-NEXT: @size_SlA  = {{.*}}global i{{[0-9]+}} 8
// CHECK-NEXT: @align_SsA = {{.*}}global i{{[0-9]+}} 2
// CHECK-NEXT: @align_SA  = {{.*}}global i{{[0-9]+}} 4
// CHECK-NEXT: @align_SlA = {{.*}}global i{{[0-9]+}} 8

// CHECK-NEXT: @size_UsA  = {{.*}}global i{{[0-9]+}} 2
// CHECK-NEXT: @size_UA   = {{.*}}global i{{[0-9]+}} 4
// CHECK-NEXT: @size_UlA  = {{.*}}global i{{[0-9]+}} 8
// CHECK-NEXT: @align_UsA = {{.*}}global i{{[0-9]+}} 2
// CHECK-NEXT: @align_UA  = {{.*}}global i{{[0-9]+}} 4
// CHECK-NEXT: @align_UlA = {{.*}}global i{{[0-9]+}} 8

// CHECK-NEXT: @size_sA   = {{.*}}global i{{[0-9]+}} 2
// CHECK-NEXT: @size_A    = {{.*}}global i{{[0-9]+}} 4
// CHECK-NEXT: @size_lA   = {{.*}}global i{{[0-9]+}} 8
// CHECK-NEXT: @align_sA  = {{.*}}global i{{[0-9]+}} 2
// CHECK-NEXT: @align_A   = {{.*}}global i{{[0-9]+}} 4
// CHECK-NEXT: @align_lA  = {{.*}}global i{{[0-9]+}} 8
