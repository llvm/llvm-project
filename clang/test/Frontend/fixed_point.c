// RUN: %clang_cc1 -x c -ffixed-point -ast-dump %s | FileCheck %s --strict-whitespace

/*  Various contexts where type _Accum can appear. */

// Primary fixed point types
signed short _Accum s_short_accum;
signed _Accum s_accum;
signed long _Accum s_long_accum;
unsigned short _Accum u_short_accum;
unsigned _Accum u_accum;
unsigned long _Accum u_long_accum;

// Aliased fixed point types
short _Accum short_accum;
_Accum accum;
long _Accum long_accum;

// CHECK:      |-VarDecl {{.*}} s_short_accum 'short _Accum'
// CHECK-NEXT: |-VarDecl {{.*}} s_accum '_Accum'
// CHECK-NEXT: |-VarDecl {{.*}} s_long_accum 'long _Accum'
// CHECK-NEXT: |-VarDecl {{.*}} u_short_accum 'unsigned short _Accum'
// CHECK-NEXT: |-VarDecl {{.*}} u_accum 'unsigned _Accum'
// CHECK-NEXT: |-VarDecl {{.*}} u_long_accum 'unsigned long _Accum'
// CHECK-NEXT: |-VarDecl {{.*}} short_accum 'short _Accum'
// CHECK-NEXT: |-VarDecl {{.*}} accum '_Accum'
// CHECK-NEXT: |-VarDecl {{.*}} long_accum 'long _Accum'

#define MIX_TYPE_SPEC(SPEC, SIGN, SIZE, ID) \
  SPEC SIGN SIZE _Accum ID; \
  SIGN SPEC SIZE _Accum ID ## 2; \
  SIGN SIZE SPEC _Accum ID ## 3; \
  SIGN SIZE _Accum SPEC ID ## 4;

/* Mixing fixed point types with other type specifiers */

#define MIX_VOLATILE(SIGN, SIZE, ID) MIX_TYPE_SPEC(volatile, SIGN, SIZE, ID)
#define MIX_ATOMIC(SIGN, SIZE, ID) MIX_TYPE_SPEC(_Atomic, SIGN, SIZE, ID)
#define MIX_CONST(SIGN, SIZE, ID) MIX_TYPE_SPEC(const, SIGN, SIZE, ID)

MIX_VOLATILE(signed, short, vol_s_short_accum)
MIX_ATOMIC(signed, short, atm_s_short_accum)
MIX_CONST(signed, short, const_s_short_accum)

// CHECK-NEXT: |-VarDecl {{.*}} vol_s_short_accum 'volatile short _Accum'
// CHECK-NEXT: |-VarDecl {{.*}} vol_s_short_accum2 'volatile short _Accum'
// CHECK-NEXT: |-VarDecl {{.*}} vol_s_short_accum3 'volatile short _Accum'
// CHECK-NEXT: |-VarDecl {{.*}} vol_s_short_accum4 'volatile short _Accum'

// CHECK-NEXT: |-VarDecl {{.*}} atm_s_short_accum '_Atomic(short _Accum)'
// CHECK-NEXT: |-VarDecl {{.*}} atm_s_short_accum2 '_Atomic(short _Accum)'
// CHECK-NEXT: |-VarDecl {{.*}} atm_s_short_accum3 '_Atomic(short _Accum)'
// CHECK-NEXT: |-VarDecl {{.*}} atm_s_short_accum4 '_Atomic(short _Accum)'

// CHECK-NEXT: |-VarDecl {{.*}} const_s_short_accum 'const short _Accum'
// CHECK-NEXT: |-VarDecl {{.*}} const_s_short_accum2 'const short _Accum'
// CHECK-NEXT: |-VarDecl {{.*}} const_s_short_accum3 'const short _Accum'
// CHECK-NEXT: `-VarDecl {{.*}} const_s_short_accum4 'const short _Accum'
