// Test without serialization:
// RUN: %clang_cc1 -x c -ffixed-point -ast-dump %s | FileCheck %s --strict-whitespace
//
// Test with serialization:
// RUN: %clang_cc1 -ffixed-point -emit-pch -o %t %s
// RUN: %clang_cc1 -x c -ffixed-point -include-pch %t -ast-dump-all /dev/null \
// RUN: | sed -e "s/ <undeserialized declarations>//" -e "s/ imported//" \
// RUN: | FileCheck %s --strict-whitespace

/*  Various contexts where type _Accum can appear. */

// Primary fixed point types
signed short _Accum s_short_accum;
signed _Accum s_accum;
signed long _Accum s_long_accum;
unsigned short _Accum u_short_accum;
unsigned _Accum u_accum;
unsigned long _Accum u_long_accum;
signed short _Fract s_short_fract;
signed _Fract s_fract;
signed long _Fract s_long_fract;
unsigned short _Fract u_short_fract;
unsigned _Fract u_fract;
unsigned long _Fract u_long_fract;

// Aliased fixed point types
short _Accum short_accum;
_Accum accum;
long _Accum long_accum;
short _Fract short_fract;
_Fract fract;
long _Fract long_fract;

// Saturated fixed point types
_Sat signed short _Accum sat_s_short_accum;
_Sat signed _Accum sat_s_accum;
_Sat signed long _Accum sat_s_long_accum;
_Sat unsigned short _Accum sat_u_short_accum;
_Sat unsigned _Accum sat_u_accum;
_Sat unsigned long _Accum sat_u_long_accum;
_Sat signed short _Fract sat_s_short_fract;
_Sat signed _Fract sat_s_fract;
_Sat signed long _Fract sat_s_long_fract;
_Sat unsigned short _Fract sat_u_short_fract;
_Sat unsigned _Fract sat_u_fract;
_Sat unsigned long _Fract sat_u_long_fract;

// Aliased saturated fixed point types
_Sat short _Accum sat_short_accum;
_Sat _Accum sat_accum;
_Sat long _Accum sat_long_accum;
_Sat short _Fract sat_short_fract;
_Sat _Fract sat_fract;
_Sat long _Fract sat_long_fract;

//CHECK:      |-VarDecl {{.*}} s_short_accum 'short _Accum'
//CHECK-NEXT: |-VarDecl {{.*}} s_accum '_Accum'
//CHECK-NEXT: |-VarDecl {{.*}} s_long_accum 'long _Accum'
//CHECK-NEXT: |-VarDecl {{.*}} u_short_accum 'unsigned short _Accum'
//CHECK-NEXT: |-VarDecl {{.*}} u_accum 'unsigned _Accum'
//CHECK-NEXT: |-VarDecl {{.*}} u_long_accum 'unsigned long _Accum'
//CHECK-NEXT: |-VarDecl {{.*}} s_short_fract 'short _Fract'
//CHECK-NEXT: |-VarDecl {{.*}} s_fract '_Fract'
//CHECK-NEXT: |-VarDecl {{.*}} s_long_fract 'long _Fract'
//CHECK-NEXT: |-VarDecl {{.*}} u_short_fract 'unsigned short _Fract'
//CHECK-NEXT: |-VarDecl {{.*}} u_fract 'unsigned _Fract'
//CHECK-NEXT: |-VarDecl {{.*}} u_long_fract 'unsigned long _Fract'
//CHECK-NEXT: |-VarDecl {{.*}} short_accum 'short _Accum'
//CHECK-NEXT: |-VarDecl {{.*}} accum '_Accum'
//CHECK-NEXT: |-VarDecl {{.*}} long_accum 'long _Accum'
//CHECK-NEXT: |-VarDecl {{.*}} short_fract 'short _Fract'
//CHECK-NEXT: |-VarDecl {{.*}} fract '_Fract'
//CHECK-NEXT: |-VarDecl {{.*}} long_fract 'long _Fract'
//CHECK-NEXT: |-VarDecl {{.*}} sat_s_short_accum '_Sat short _Accum'
//CHECK-NEXT: |-VarDecl {{.*}} sat_s_accum '_Sat _Accum'
//CHECK-NEXT: |-VarDecl {{.*}} sat_s_long_accum '_Sat long _Accum'
//CHECK-NEXT: |-VarDecl {{.*}} sat_u_short_accum '_Sat unsigned short _Accum'
//CHECK-NEXT: |-VarDecl {{.*}} sat_u_accum '_Sat unsigned _Accum'
//CHECK-NEXT: |-VarDecl {{.*}} sat_u_long_accum '_Sat unsigned long _Accum'
//CHECK-NEXT: |-VarDecl {{.*}} sat_s_short_fract '_Sat short _Fract'
//CHECK-NEXT: |-VarDecl {{.*}} sat_s_fract '_Sat _Fract'
//CHECK-NEXT: |-VarDecl {{.*}} sat_s_long_fract '_Sat long _Fract'
//CHECK-NEXT: |-VarDecl {{.*}} sat_u_short_fract '_Sat unsigned short _Fract'
//CHECK-NEXT: |-VarDecl {{.*}} sat_u_fract '_Sat unsigned _Fract'
//CHECK-NEXT: |-VarDecl {{.*}} sat_u_long_fract '_Sat unsigned long _Fract'
//CHECK-NEXT: |-VarDecl {{.*}} sat_short_accum '_Sat short _Accum'
//CHECK-NEXT: |-VarDecl {{.*}} sat_accum '_Sat _Accum'
//CHECK-NEXT: |-VarDecl {{.*}} sat_long_accum '_Sat long _Accum'
//CHECK-NEXT: |-VarDecl {{.*}} sat_short_fract '_Sat short _Fract'
//CHECK-NEXT: |-VarDecl {{.*}} sat_fract '_Sat _Fract'
//CHECK-NEXT: |-VarDecl {{.*}} sat_long_fract '_Sat long _Fract'

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
// CHECK-NEXT: |-VarDecl {{.*}} const_s_short_accum4 'const short _Accum'

/* Typedefs */

// Primary fixed point types
typedef signed short   _Accum SsA_t;
typedef signed         _Accum SA_t;
typedef signed long    _Accum SlA_t;
typedef unsigned short _Accum UsA_t;
typedef unsigned       _Accum UA_t;
typedef unsigned long  _Accum UlA_t;
typedef signed short   _Fract SsF_t;
typedef signed         _Fract SF_t;
typedef signed long    _Fract SlF_t;
typedef unsigned short _Fract UsF_t;
typedef unsigned       _Fract UF_t;
typedef unsigned long  _Fract UlF_t;

// Aliased fixed point types
typedef short _Accum sA_t;
typedef       _Accum A_t;
typedef long  _Accum lA_t;
typedef short _Fract sF_t;
typedef       _Fract F_t;
typedef long  _Fract lF_t;

// Saturated fixed point types
typedef _Sat signed short   _Accum SatSsA_t;
typedef _Sat signed         _Accum SatSA_t;
typedef _Sat signed long    _Accum SatSlA_t;
typedef _Sat unsigned short _Accum SatUsA_t;
typedef _Sat unsigned       _Accum SatUA_t;
typedef _Sat unsigned long  _Accum SatUlA_t;
typedef _Sat signed short   _Fract SatSsF_t;
typedef _Sat signed         _Fract SatSF_t;
typedef _Sat signed long    _Fract SatSlF_t;
typedef _Sat unsigned short _Fract SatUsF_t;
typedef _Sat unsigned       _Fract SatUF_t;
typedef _Sat unsigned long  _Fract SatUlF_t;

// Aliased saturated fixed point types
typedef _Sat short   _Accum SatsA_t;
typedef _Sat         _Accum SatA_t;
typedef _Sat long    _Accum SatlA_t;
typedef _Sat short   _Fract SatsF_t;
typedef _Sat         _Fract SatF_t;
typedef _Sat long    _Fract SatlF_t;

SsA_t     SsA_type;
SA_t      SA_type;
SlA_t     SlA_type;
UsA_t     UsA_type;
UA_t      UA_type;
UlA_t     UlA_type;
SsF_t     SsF_type;
SF_t      SF_type;
SlF_t     SlF_type;
UsF_t     UsF_type;
UF_t      UF_type;
UlF_t     UlF_type;

sA_t      sA_type;
A_t       A_type;
lA_t      lA_type;
sF_t      sF_type;
F_t       F_type;
lF_t      lF_type;

SatSsA_t  SatSsA_type;
SatSA_t   SatSA_type;
SatSlA_t  SatSlA_type;
SatUsA_t  SatUsA_type;
SatUA_t   SatUA_type;
SatUlA_t  SatUlA_type;
SatSsF_t  SatSsF_type;
SatSF_t   SatSF_type;
SatSlF_t  SatSlF_type;
SatUsF_t  SatUsF_type;
SatUF_t   SatUF_type;
SatUlF_t  SatUlF_type;

SatsA_t   SatsA_type;
SatA_t    SatA_type;
SatlA_t   SatlA_type;
SatsF_t   SatsF_type;
SatF_t    SatF_type;
SatlF_t   SatlF_type;

//CHECK:      |-VarDecl {{.*}} SsA_type 'SsA_t':'short _Accum'
//CHECK-NEXT: |-VarDecl {{.*}} SA_type 'SA_t':'_Accum'
//CHECK-NEXT: |-VarDecl {{.*}} SlA_type 'SlA_t':'long _Accum'
//CHECK-NEXT: |-VarDecl {{.*}} UsA_type 'UsA_t':'unsigned short _Accum'
//CHECK-NEXT: |-VarDecl {{.*}} UA_type 'UA_t':'unsigned _Accum'
//CHECK-NEXT: |-VarDecl {{.*}} UlA_type 'UlA_t':'unsigned long _Accum'
//CHECK-NEXT: |-VarDecl {{.*}} SsF_type 'SsF_t':'short _Fract'
//CHECK-NEXT: |-VarDecl {{.*}} SF_type 'SF_t':'_Fract'
//CHECK-NEXT: |-VarDecl {{.*}} SlF_type 'SlF_t':'long _Fract'
//CHECK-NEXT: |-VarDecl {{.*}} UsF_type 'UsF_t':'unsigned short _Fract'
//CHECK-NEXT: |-VarDecl {{.*}} UF_type 'UF_t':'unsigned _Fract'
//CHECK-NEXT: |-VarDecl {{.*}} UlF_type 'UlF_t':'unsigned long _Fract'
//CHECK-NEXT: |-VarDecl {{.*}} sA_type 'sA_t':'short _Accum'
//CHECK-NEXT: |-VarDecl {{.*}} A_type 'A_t':'_Accum'
//CHECK-NEXT: |-VarDecl {{.*}} lA_type 'lA_t':'long _Accum'
//CHECK-NEXT: |-VarDecl {{.*}} sF_type 'sF_t':'short _Fract'
//CHECK-NEXT: |-VarDecl {{.*}} F_type 'F_t':'_Fract'
//CHECK-NEXT: |-VarDecl {{.*}} lF_type 'lF_t':'long _Fract'

//CHECK-NEXT: |-VarDecl {{.*}} SatSsA_type 'SatSsA_t':'_Sat short _Accum'
//CHECK-NEXT: |-VarDecl {{.*}} SatSA_type 'SatSA_t':'_Sat _Accum'
//CHECK-NEXT: |-VarDecl {{.*}} SatSlA_type 'SatSlA_t':'_Sat long _Accum'
//CHECK-NEXT: |-VarDecl {{.*}} SatUsA_type 'SatUsA_t':'_Sat unsigned short _Accum'
//CHECK-NEXT: |-VarDecl {{.*}} SatUA_type 'SatUA_t':'_Sat unsigned _Accum'
//CHECK-NEXT: |-VarDecl {{.*}} SatUlA_type 'SatUlA_t':'_Sat unsigned long _Accum'
//CHECK-NEXT: |-VarDecl {{.*}} SatSsF_type 'SatSsF_t':'_Sat short _Fract'
//CHECK-NEXT: |-VarDecl {{.*}} SatSF_type 'SatSF_t':'_Sat _Fract'
//CHECK-NEXT: |-VarDecl {{.*}} SatSlF_type 'SatSlF_t':'_Sat long _Fract'
//CHECK-NEXT: |-VarDecl {{.*}} SatUsF_type 'SatUsF_t':'_Sat unsigned short _Fract'
//CHECK-NEXT: |-VarDecl {{.*}} SatUF_type 'SatUF_t':'_Sat unsigned _Fract'
//CHECK-NEXT: |-VarDecl {{.*}} SatUlF_type 'SatUlF_t':'_Sat unsigned long _Fract'
//CHECK-NEXT: |-VarDecl {{.*}} SatsA_type 'SatsA_t':'_Sat short _Accum'
//CHECK-NEXT: |-VarDecl {{.*}} SatA_type 'SatA_t':'_Sat _Accum'
//CHECK-NEXT: |-VarDecl {{.*}} SatlA_type 'SatlA_t':'_Sat long _Accum'
//CHECK-NEXT: |-VarDecl {{.*}} SatsF_type 'SatsF_t':'_Sat short _Fract'
//CHECK-NEXT: |-VarDecl {{.*}} SatF_type 'SatF_t':'_Sat _Fract'
//CHECK-NEXT: |-VarDecl {{.*}} SatlF_type 'SatlF_t':'_Sat long _Fract'

// Fixed point literal exponent syntax
_Accum decexp1 = 1.575e1k;
_Accum decexp2 = 1.575E1k;
_Accum decexp3 = 1575e-2k;
_Accum decexp4 = 1575E-2k;

_Accum hexexp1 = 0x0.3p10k;
_Accum hexexp2 = 0x0.3P10k;
_Accum hexexp3 = 0x30000p-10k;
_Accum hexexp4 = 0x30000P-10k;

_Accum zeroexp1 = 1e0k;
_Accum zeroexp2 = 1e-0k;

//CHECK-NEXT: |-VarDecl {{.*}} decexp1 '_Accum' cinit
//CHECK-NEXT:   `-FixedPointLiteral {{.*}} '_Accum' 15.75
//CHECK-NEXT: |-VarDecl {{.*}} decexp2 '_Accum' cinit
//CHECK-NEXT:   `-FixedPointLiteral {{.*}} '_Accum' 15.75
//CHECK-NEXT: |-VarDecl {{.*}} decexp3 '_Accum' cinit
//CHECK-NEXT:   `-FixedPointLiteral {{.*}} '_Accum' 15.75
//CHECK-NEXT: |-VarDecl {{.*}} decexp4 '_Accum' cinit
//CHECK-NEXT:   `-FixedPointLiteral {{.*}} '_Accum' 15.75

//CHECK-NEXT: |-VarDecl {{.*}} hexexp1 '_Accum' cinit
//CHECK-NEXT:   `-FixedPointLiteral {{.*}} '_Accum' 192.0
//CHECK-NEXT: |-VarDecl {{.*}} hexexp2 '_Accum' cinit
//CHECK-NEXT:   `-FixedPointLiteral {{.*}} '_Accum' 192.0
//CHECK-NEXT: |-VarDecl {{.*}} hexexp3 '_Accum' cinit
//CHECK-NEXT:   `-FixedPointLiteral {{.*}} '_Accum' 192.0
//CHECK-NEXT: |-VarDecl {{.*}} hexexp4 '_Accum' cinit
//CHECK-NEXT:   `-FixedPointLiteral {{.*}} '_Accum' 192.0

//CHECK-NEXT: |-VarDecl {{.*}} zeroexp1 '_Accum' cinit
//CHECK-NEXT:   `-FixedPointLiteral {{.*}} '_Accum' 1.0
//CHECK-NEXT: |-VarDecl {{.*}} zeroexp2 '_Accum' cinit
//CHECK-NEXT:   `-FixedPointLiteral {{.*}} '_Accum' 1.0

// Fixed point literal values
_Accum literal1 = 2.5k;       // Precise decimal
_Accum literal2 = 0.0k;       // Zero
_Accum literal3 = 1.1k;       // Imprecise decimal
_Accum literal4 = 1.11k;
_Accum literal5 = 1.111k;
_Accum literal6 = 1.1111k;
_Accum literal7 = 1.11111k;   // After some point after the radix, adding any more
                              // digits to the literal will not result in any
                              // further precision since the nth digit added may
                              // be less than the precision that can be
                              // represented by the fractional bits of the type.
                              // This results in the same value being stored for
                              // the type.

//CHECK-NEXT: |-VarDecl {{.*}} literal1 '_Accum' cinit
//CHECK-NEXT:   `-FixedPointLiteral {{.*}} '_Accum' 2.5
//CHECK-NEXT: |-VarDecl {{.*}} literal2 '_Accum' cinit
//CHECK-NEXT:   `-FixedPointLiteral {{.*}} '_Accum' 0.0
//CHECK-NEXT: |-VarDecl {{.*}} literal3 '_Accum' cinit
//CHECK-NEXT:   `-FixedPointLiteral {{.*}} '_Accum' 1.0999755859375
//CHECK-NEXT: |-VarDecl {{.*}} literal4 '_Accum' cinit
//CHECK-NEXT:   `-FixedPointLiteral {{.*}} '_Accum' 1.1099853515625
//CHECK-NEXT: |-VarDecl {{.*}} literal5 '_Accum' cinit
//CHECK-NEXT:   `-FixedPointLiteral {{.*}} '_Accum' 1.110992431640625
//CHECK-NEXT: |-VarDecl {{.*}} literal6 '_Accum' cinit
//CHECK-NEXT:   `-FixedPointLiteral {{.*}} '_Accum' 1.111083984375
//CHECK-NEXT: |-VarDecl {{.*}} literal7 '_Accum' cinit
//CHECK-NEXT:   `-FixedPointLiteral {{.*}} '_Accum' 1.111083984375

long _Accum longaccumliteral     = 0.99999999lk;
long _Accum longaccumliteral2    = 0.999999999lk;
long _Accum verylongaccumliteral = 0.99999999999999999999999999lk;
long _Fract longfractliteral     = 0.99999999lr;
long _Fract longfractliteral2    = 0.999999999lr;
long _Fract verylongfractliteral = 0.99999999999999999999999999lr;

//CHECK-NEXT: |-VarDecl {{.*}} longaccumliteral 'long _Accum' cinit
//CHECK-NEXT:   `-FixedPointLiteral {{.*}} 'long _Accum' 0.999999989755451679229736328125
//CHECK-NEXT: |-VarDecl {{.*}} longaccumliteral2 'long _Accum' cinit
//CHECK-NEXT:   `-FixedPointLiteral {{.*}} 'long _Accum' 0.9999999986030161380767822265625
//CHECK-NEXT: |-VarDecl {{.*}} verylongaccumliteral 'long _Accum' cinit
//CHECK-NEXT:   `-FixedPointLiteral {{.*}} 'long _Accum' 0.9999999995343387126922607421875
//CHECK-NEXT: |-VarDecl {{.*}} longfractliteral 'long _Fract' cinit
//CHECK-NEXT:   `-FixedPointLiteral {{.*}} 'long _Fract' 0.999999989755451679229736328125
//CHECK-NEXT: |-VarDecl {{.*}} longfractliteral2 'long _Fract' cinit
//CHECK-NEXT:   `-FixedPointLiteral {{.*}} 'long _Fract' 0.9999999986030161380767822265625
//CHECK-NEXT: |-VarDecl {{.*}} verylongfractliteral 'long _Fract' cinit
//CHECK-NEXT:   `-FixedPointLiteral {{.*}} 'long _Fract' 0.9999999995343387126922607421875

unsigned _Accum uliteral1 = 2.5uk;    // Unsigned
_Accum literal8 = -2.5k;              // Negative

//CHECK-NEXT: |-VarDecl {{.*}} uliteral1 'unsigned _Accum' cinit
//CHECK-NEXT:   `-FixedPointLiteral {{.*}} 'unsigned _Accum' 2.5
//CHECK-NEXT: |-VarDecl {{.*}} literal8 '_Accum' cinit
//CHECK-NEXT:   `-UnaryOperator {{.*}} '_Accum' prefix '-'
//CHECK-NEXT:     `-FixedPointLiteral {{.*}} '_Accum' 2.5

short _Accum  literalexact1 = 0.9921875hk;  // Exact value
_Accum        literalexact2 = 0.999969482421875k;

//CHECK-NEXT: |-VarDecl {{.*}} literalexact1 'short _Accum' cinit
//CHECK-NEXT:   `-FixedPointLiteral {{.*}} 'short _Accum' 0.9921875
//CHECK-NEXT: |-VarDecl {{.*}} literalexact2 '_Accum' cinit
//CHECK-NEXT:   `-FixedPointLiteral {{.*}} '_Accum' 0.999969482421875

// Unfortunately we do not have enough space to store the exact decimal value of
// 0.9999999995343387126922607421875 ((1 << 31) - 1), but we can still use a
// large number of 9s to get the max fractional value.
long _Accum   long_accum_max = 0.999999999999999999999999999lk;

//CHECK-NEXT: |-VarDecl {{.*}} long_accum_max 'long _Accum' cinit
//CHECK-NEXT:   `-FixedPointLiteral {{.*}} 'long _Accum' 0.9999999995343387126922607421875

// Epsilon
short _Accum  short_accum_eps   = 0.0078125hk;
short _Accum  short_accum_eps2  = 0.0078124hk;  // Less than epsilon floors to zero
_Accum        accum_eps         = 0.000030517578125k;
_Accum        accum_eps2        = 0.000030517578124k;
long _Accum   long_accum_eps    = 0x1p-31lk;
long _Accum   long_accum_eps2   = 0x0.99999999p-31lk;

//CHECK-NEXT: |-VarDecl {{.*}} short_accum_eps 'short _Accum' cinit
//CHECK-NEXT:   `-FixedPointLiteral {{.*}} 'short _Accum' 0.0078125
//CHECK-NEXT: |-VarDecl {{.*}} short_accum_eps2 'short _Accum' cinit
//CHECK-NEXT:   `-FixedPointLiteral {{.*}} 'short _Accum' 0.0
//CHECK-NEXT: |-VarDecl {{.*}} accum_eps '_Accum' cinit
//CHECK-NEXT:   `-FixedPointLiteral {{.*}} '_Accum' 0.000030517578125
//CHECK-NEXT: |-VarDecl {{.*}} accum_eps2 '_Accum' cinit
//CHECK-NEXT:   `-FixedPointLiteral {{.*}} '_Accum' 0.0
//CHECK-NEXT: |-VarDecl {{.*}} long_accum_eps 'long _Accum' cinit
//CHECK-NEXT:   `-FixedPointLiteral {{.*}} 'long _Accum' 0.0000000004656612873077392578125
//CHECK-NEXT: |-VarDecl {{.*}} long_accum_eps2 'long _Accum' cinit
//CHECK-NEXT:   `-FixedPointLiteral {{.*}} 'long _Accum' 0.0

// Fract literals can be one but evaluate to the respective Fract max
short _Fract           short_fract_one   = 1.0hr;
_Fract                 fract_one         = 1.0r;
long _Fract            long_fract_one    = 1.0lr;
unsigned short _Fract  u_short_fract_one = 1.0uhr;
unsigned _Fract        u_fract_one       = 1.0ur;
unsigned long _Fract   u_long_fract_one  = 1.0ulr;

//CHECK-NEXT: |-VarDecl {{.*}} short_fract_one 'short _Fract' cinit
//CHECK-NEXT:   `-FixedPointLiteral {{.*}} 'short _Fract' 0.9921875
//CHECK-NEXT: |-VarDecl {{.*}} fract_one '_Fract' cinit
//CHECK-NEXT:   `-FixedPointLiteral {{.*}} '_Fract' 0.999969482421875
//CHECK-NEXT: |-VarDecl {{.*}} long_fract_one 'long _Fract' cinit
//CHECK-NEXT:   `-FixedPointLiteral {{.*}} 'long _Fract' 0.9999999995343387126922607421875

//CHECK-NEXT: |-VarDecl {{.*}} u_short_fract_one 'unsigned short _Fract' cinit
//CHECK-NEXT:   `-FixedPointLiteral {{.*}} 'unsigned short _Fract' 0.99609375
//CHECK-NEXT: |-VarDecl {{.*}} u_fract_one 'unsigned _Fract' cinit
//CHECK-NEXT:   `-FixedPointLiteral {{.*}} 'unsigned _Fract' 0.9999847412109375
//CHECK-NEXT: |-VarDecl {{.*}} u_long_fract_one 'unsigned long _Fract' cinit
//CHECK-NEXT:   `-FixedPointLiteral {{.*}} 'unsigned long _Fract' 0.99999999976716935634613037109375

_Accum literallast = 1.0k;    // One

//CHECK-NEXT: `-VarDecl {{.*}} literallast '_Accum' cinit
//CHECK-NEXT:   `-FixedPointLiteral {{.*}} '_Accum' 1.0
