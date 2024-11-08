// RUN: %clang -x c -ffixed-point -S -emit-llvm -o - %s | FileCheck %s
// RUN: %clang -x c -ffixed-point -S -emit-llvm -o - --target=x86_64-scei-ps4 %s | FileCheck %s
// RUN: %clang -x c -ffixed-point -S -emit-llvm -o - --target=ppc64 %s | FileCheck %s
// RUN: %clang -x c -ffixed-point -S -emit-llvm -o - --target=x86_64-scei-ps4 %s | FileCheck %s

/* Primary signed _Accum */

int size_SsA = sizeof(signed short _Accum);
int size_SA  = sizeof(signed _Accum);
int size_SlA = sizeof(signed long _Accum);
int align_SsA = __alignof(signed short _Accum);
int align_SA  = __alignof(signed _Accum);
int align_SlA = __alignof(signed long _Accum);

// CHECK:      @size_SsA  = {{.*}}global i{{[0-9]+}} 2
// CHECK-NEXT: @size_SA   = {{.*}}global i{{[0-9]+}} 4
// CHECK-NEXT: @size_SlA  = {{.*}}global i{{[0-9]+}} 8
// CHECK-NEXT: @align_SsA = {{.*}}global i{{[0-9]+}} 2
// CHECK-NEXT: @align_SA  = {{.*}}global i{{[0-9]+}} 4
// CHECK-NEXT: @align_SlA = {{.*}}global i{{[0-9]+}} 8

/* Primary unsigned _Accum */

int size_UsA = sizeof(unsigned short _Accum);
int size_UA  = sizeof(unsigned _Accum);
int size_UlA = sizeof(unsigned long _Accum);
int align_UsA = __alignof(unsigned short _Accum);
int align_UA  = __alignof(unsigned _Accum);
int align_UlA = __alignof(unsigned long _Accum);

// CHECK-NEXT: @size_UsA  = {{.*}}global i{{[0-9]+}} 2
// CHECK-NEXT: @size_UA   = {{.*}}global i{{[0-9]+}} 4
// CHECK-NEXT: @size_UlA  = {{.*}}global i{{[0-9]+}} 8
// CHECK-NEXT: @align_UsA = {{.*}}global i{{[0-9]+}} 2
// CHECK-NEXT: @align_UA  = {{.*}}global i{{[0-9]+}} 4
// CHECK-NEXT: @align_UlA = {{.*}}global i{{[0-9]+}} 8

/* Primary signed _Fract */

int size_SsF = sizeof(signed short _Fract);
int size_SF  = sizeof(signed _Fract);
int size_SlF = sizeof(signed long _Fract);
int align_SsF = __alignof(signed short _Fract);
int align_SF  = __alignof(signed _Fract);
int align_SlF = __alignof(signed long _Fract);

// CHECK-NEXT: @size_SsF  = {{.*}}global i{{[0-9]+}} 1
// CHECK-NEXT: @size_SF   = {{.*}}global i{{[0-9]+}} 2
// CHECK-NEXT: @size_SlF  = {{.*}}global i{{[0-9]+}} 4
// CHECK-NEXT: @align_SsF = {{.*}}global i{{[0-9]+}} 1
// CHECK-NEXT: @align_SF  = {{.*}}global i{{[0-9]+}} 2
// CHECK-NEXT: @align_SlF = {{.*}}global i{{[0-9]+}} 4

/* Primary unsigned _Fract */

int size_UsF = sizeof(unsigned short _Fract);
int size_UF  = sizeof(unsigned _Fract);
int size_UlF = sizeof(unsigned long _Fract);
int align_UsF = __alignof(unsigned short _Fract);
int align_UF  = __alignof(unsigned _Fract);
int align_UlF = __alignof(unsigned long _Fract);

// CHECK-NEXT: @size_UsF  = {{.*}}global i{{[0-9]+}} 1
// CHECK-NEXT: @size_UF   = {{.*}}global i{{[0-9]+}} 2
// CHECK-NEXT: @size_UlF  = {{.*}}global i{{[0-9]+}} 4
// CHECK-NEXT: @align_UsF = {{.*}}global i{{[0-9]+}} 1
// CHECK-NEXT: @align_UF  = {{.*}}global i{{[0-9]+}} 2
// CHECK-NEXT: @align_UlF = {{.*}}global i{{[0-9]+}} 4

/* Aliased _Accum */

int size_sA = sizeof(short _Accum);
int size_A  = sizeof(_Accum);
int size_lA = sizeof(long _Accum);
int align_sA = __alignof(short _Accum);
int align_A  = __alignof(_Accum);
int align_lA = __alignof(long _Accum);

// CHECK-NEXT: @size_sA   = {{.*}}global i{{[0-9]+}} 2
// CHECK-NEXT: @size_A    = {{.*}}global i{{[0-9]+}} 4
// CHECK-NEXT: @size_lA   = {{.*}}global i{{[0-9]+}} 8
// CHECK-NEXT: @align_sA  = {{.*}}global i{{[0-9]+}} 2
// CHECK-NEXT: @align_A   = {{.*}}global i{{[0-9]+}} 4
// CHECK-NEXT: @align_lA  = {{.*}}global i{{[0-9]+}} 8

/* Aliased _Fract */

int size_sF = sizeof(short _Fract);
int size_F  = sizeof(_Fract);
int size_lF = sizeof(long _Fract);
int align_sF = __alignof(short _Fract);
int align_F  = __alignof(_Fract);
int align_lF = __alignof(long _Fract);

// CHECK-NEXT: @size_sF   = {{.*}}global i{{[0-9]+}} 1
// CHECK-NEXT: @size_F    = {{.*}}global i{{[0-9]+}} 2
// CHECK-NEXT: @size_lF   = {{.*}}global i{{[0-9]+}} 4
// CHECK-NEXT: @align_sF  = {{.*}}global i{{[0-9]+}} 1
// CHECK-NEXT: @align_F   = {{.*}}global i{{[0-9]+}} 2
// CHECK-NEXT: @align_lF  = {{.*}}global i{{[0-9]+}} 4

/* Saturated signed _Accum */

int size_SatSsA = sizeof(_Sat signed short _Accum);
int size_SatSA  = sizeof(_Sat signed _Accum);
int size_SatSlA = sizeof(_Sat signed long _Accum);
int align_SatSsA = __alignof(_Sat signed short _Accum);
int align_SatSA  = __alignof(_Sat signed _Accum);
int align_SatSlA = __alignof(_Sat signed long _Accum);

// CHECK:      @size_SatSsA  = {{.*}}global i{{[0-9]+}} 2
// CHECK-NEXT: @size_SatSA   = {{.*}}global i{{[0-9]+}} 4
// CHECK-NEXT: @size_SatSlA  = {{.*}}global i{{[0-9]+}} 8
// CHECK-NEXT: @align_SatSsA = {{.*}}global i{{[0-9]+}} 2
// CHECK-NEXT: @align_SatSA  = {{.*}}global i{{[0-9]+}} 4
// CHECK-NEXT: @align_SatSlA = {{.*}}global i{{[0-9]+}} 8

/* Saturated unsigned _Accum */

int size_SatUsA = sizeof(_Sat unsigned short _Accum);
int size_SatUA  = sizeof(_Sat unsigned _Accum);
int size_SatUlA = sizeof(_Sat unsigned long _Accum);
int align_SatUsA = __alignof(_Sat unsigned short _Accum);
int align_SatUA  = __alignof(_Sat unsigned _Accum);
int align_SatUlA = __alignof(_Sat unsigned long _Accum);

// CHECK-NEXT: @size_SatUsA  = {{.*}}global i{{[0-9]+}} 2
// CHECK-NEXT: @size_SatUA   = {{.*}}global i{{[0-9]+}} 4
// CHECK-NEXT: @size_SatUlA  = {{.*}}global i{{[0-9]+}} 8
// CHECK-NEXT: @align_SatUsA = {{.*}}global i{{[0-9]+}} 2
// CHECK-NEXT: @align_SatUA  = {{.*}}global i{{[0-9]+}} 4
// CHECK-NEXT: @align_SatUlA = {{.*}}global i{{[0-9]+}} 8

/* Saturated signed _Fract */

int size_SatSsF = sizeof(_Sat signed short _Fract);
int size_SatSF  = sizeof(_Sat signed _Fract);
int size_SatSlF = sizeof(_Sat signed long _Fract);
int align_SatSsF = __alignof(_Sat signed short _Fract);
int align_SatSF  = __alignof(_Sat signed _Fract);
int align_SatSlF = __alignof(_Sat signed long _Fract);

// CHECK-NEXT: @size_SatSsF  = {{.*}}global i{{[0-9]+}} 1
// CHECK-NEXT: @size_SatSF   = {{.*}}global i{{[0-9]+}} 2
// CHECK-NEXT: @size_SatSlF  = {{.*}}global i{{[0-9]+}} 4
// CHECK-NEXT: @align_SatSsF = {{.*}}global i{{[0-9]+}} 1
// CHECK-NEXT: @align_SatSF  = {{.*}}global i{{[0-9]+}} 2
// CHECK-NEXT: @align_SatSlF = {{.*}}global i{{[0-9]+}} 4

/* Saturated unsigned _Fract */

int size_SatUsF = sizeof(_Sat unsigned short _Fract);
int size_SatUF  = sizeof(_Sat unsigned _Fract);
int size_SatUlF = sizeof(_Sat unsigned long _Fract);
int align_SatUsF = __alignof(_Sat unsigned short _Fract);
int align_SatUF  = __alignof(_Sat unsigned _Fract);
int align_SatUlF = __alignof(_Sat unsigned long _Fract);

// CHECK-NEXT: @size_SatUsF  = {{.*}}global i{{[0-9]+}} 1
// CHECK-NEXT: @size_SatUF   = {{.*}}global i{{[0-9]+}} 2
// CHECK-NEXT: @size_SatUlF  = {{.*}}global i{{[0-9]+}} 4
// CHECK-NEXT: @align_SatUsF = {{.*}}global i{{[0-9]+}} 1
// CHECK-NEXT: @align_SatUF  = {{.*}}global i{{[0-9]+}} 2
// CHECK-NEXT: @align_SatUlF = {{.*}}global i{{[0-9]+}} 4

/* Aliased saturated signed _Accum */

int size_SatsA = sizeof(_Sat short _Accum);
int size_SatA  = sizeof(_Sat _Accum);
int size_SatlA = sizeof(_Sat long _Accum);
int align_SatsA = __alignof(_Sat short _Accum);
int align_SatA  = __alignof(_Sat _Accum);
int align_SatlA = __alignof(_Sat long _Accum);

// CHECK-NEXT: @size_SatsA   = {{.*}}global i{{[0-9]+}} 2
// CHECK-NEXT: @size_SatA    = {{.*}}global i{{[0-9]+}} 4
// CHECK-NEXT: @size_SatlA   = {{.*}}global i{{[0-9]+}} 8
// CHECK-NEXT: @align_SatsA  = {{.*}}global i{{[0-9]+}} 2
// CHECK-NEXT: @align_SatA   = {{.*}}global i{{[0-9]+}} 4
// CHECK-NEXT: @align_SatlA  = {{.*}}global i{{[0-9]+}} 8

/* Aliased saturated _Fract */

int size_SatsF = sizeof(_Sat short _Fract);
int size_SatF  = sizeof(_Sat _Fract);
int size_SatlF = sizeof(_Sat long _Fract);
int align_SatsF = __alignof(_Sat short _Fract);
int align_SatF  = __alignof(_Sat _Fract);
int align_SatlF = __alignof(_Sat long _Fract);

// CHECK-NEXT: @size_SatsF   = {{.*}}global i{{[0-9]+}} 1
// CHECK-NEXT: @size_SatF    = {{.*}}global i{{[0-9]+}} 2
// CHECK-NEXT: @size_SatlF   = {{.*}}global i{{[0-9]+}} 4
// CHECK-NEXT: @align_SatsF  = {{.*}}global i{{[0-9]+}} 1
// CHECK-NEXT: @align_SatF   = {{.*}}global i{{[0-9]+}} 2
// CHECK-NEXT: @align_SatlF  = {{.*}}global i{{[0-9]+}} 4
