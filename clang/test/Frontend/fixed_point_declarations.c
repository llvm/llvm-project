// RUN: %clang -ffixed-point -S -emit-llvm %s -o - --target=x86_64-linux | FileCheck %s

// Primary fixed point types
signed short _Accum   s_short_accum;  // CHECK-DAG: @s_short_accum  = {{.*}}global i16 0, align 2
signed _Accum         s_accum;        // CHECK-DAG: @s_accum        = {{.*}}global i32 0, align 4
signed long _Accum    s_long_accum;   // CHECK-DAG: @s_long_accum   = {{.*}}global i64 0, align 8
unsigned short _Accum u_short_accum;  // CHECK-DAG: @u_short_accum  = {{.*}}global i16 0, align 2
unsigned _Accum       u_accum;        // CHECK-DAG: @u_accum        = {{.*}}global i32 0, align 4
unsigned long _Accum  u_long_accum;   // CHECK-DAG: @u_long_accum   = {{.*}}global i64 0, align 8
signed short _Fract   s_short_fract;  // CHECK-DAG: @s_short_fract  = {{.*}}global i8  0, align 1
signed _Fract         s_fract;        // CHECK-DAG: @s_fract        = {{.*}}global i16 0, align 2
signed long _Fract    s_long_fract;   // CHECK-DAG: @s_long_fract   = {{.*}}global i32 0, align 4
unsigned short _Fract u_short_fract;  // CHECK-DAG: @u_short_fract  = {{.*}}global i8  0, align 1
unsigned _Fract       u_fract;        // CHECK-DAG: @u_fract        = {{.*}}global i16 0, align 2
unsigned long _Fract  u_long_fract;   // CHECK-DAG: @u_long_fract   = {{.*}}global i32 0, align 4

// Aliased
short _Accum  short_accum;            // CHECK-DAG: @short_accum    = {{.*}}global i16 0, align 2
_Accum        accum;                  // CHECK-DAG: @accum          = {{.*}}global i32 0, align 4
long _Accum   long_accum;             // CHECK-DAG: @long_accum     = {{.*}}global i64 0, align 8
short _Fract  short_fract;            // CHECK-DAG: @short_fract    = {{.*}}global i8  0, align 1
_Fract        fract;                  // CHECK-DAG: @fract          = {{.*}}global i16 0, align 2
long _Fract   long_fract;             // CHECK-DAG: @long_fract     = {{.*}}global i32 0, align 4

// Primary saturated
_Sat signed short _Accum   sat_s_short_accum;  // CHECK-DAG: @sat_s_short_accum  = {{.*}}global i16 0, align 2
_Sat signed _Accum         sat_s_accum;        // CHECK-DAG: @sat_s_accum        = {{.*}}global i32 0, align 4
_Sat signed long _Accum    sat_s_long_accum;   // CHECK-DAG: @sat_s_long_accum   = {{.*}}global i64 0, align 8
_Sat unsigned short _Accum sat_u_short_accum;  // CHECK-DAG: @sat_u_short_accum  = {{.*}}global i16 0, align 2
_Sat unsigned _Accum       sat_u_accum;        // CHECK-DAG: @sat_u_accum        = {{.*}}global i32 0, align 4
_Sat unsigned long _Accum  sat_u_long_accum;   // CHECK-DAG: @sat_u_long_accum   = {{.*}}global i64 0, align 8
_Sat signed short _Fract   sat_s_short_fract;  // CHECK-DAG: @sat_s_short_fract  = {{.*}}global i8  0, align 1
_Sat signed _Fract         sat_s_fract;        // CHECK-DAG: @sat_s_fract        = {{.*}}global i16 0, align 2
_Sat signed long _Fract    sat_s_long_fract;   // CHECK-DAG: @sat_s_long_fract   = {{.*}}global i32 0, align 4
_Sat unsigned short _Fract sat_u_short_fract;  // CHECK-DAG: @sat_u_short_fract  = {{.*}}global i8  0, align 1
_Sat unsigned _Fract       sat_u_fract;        // CHECK-DAG: @sat_u_fract        = {{.*}}global i16 0, align 2
_Sat unsigned long _Fract  sat_u_long_fract;   // CHECK-DAG: @sat_u_long_fract   = {{.*}}global i32 0, align 4

// Aliased saturated
_Sat short _Accum  sat_short_accum;            // CHECK-DAG: @sat_short_accum    = {{.*}}global i16 0, align 2
_Sat _Accum        sat_accum;                  // CHECK-DAG: @sat_accum          = {{.*}}global i32 0, align 4
_Sat long _Accum   sat_long_accum;             // CHECK-DAG: @sat_long_accum     = {{.*}}global i64 0, align 8
_Sat short _Fract  sat_short_fract;            // CHECK-DAG: @sat_short_fract    = {{.*}}global i8  0, align 1
_Sat _Fract        sat_fract;                  // CHECK-DAG: @sat_fract          = {{.*}}global i16 0, align 2
_Sat long _Fract   sat_long_fract;             // CHECK-DAG: @sat_long_fract     = {{.*}}global i32 0, align 4

/* Fixed point literals */
short _Accum  short_accum_literal = 2.5hk;    // CHECK-DAG: @short_accum_literal  = {{.*}}global i16 320, align 2
_Accum        accum_literal       = 2.5k;     // CHECK-DAG: @accum_literal        = {{.*}}global i32 81920, align 4
long _Accum   long_accum_literal  = 2.5lk;    // CHECK-DAG: @long_accum_literal   = {{.*}}global i64 5368709120, align 8
short _Fract  short_fract_literal = 0.5hr;    // CHECK-DAG: @short_fract_literal  = {{.*}}global i8  64, align 1
_Fract        fract_literal       = 0.5r;     // CHECK-DAG: @fract_literal        = {{.*}}global i16 16384, align 2
long _Fract   long_fract_literal  = 0.5lr;    // CHECK-DAG: @long_fract_literal   = {{.*}}global i32 1073741824, align 4

unsigned short _Accum  u_short_accum_literal = 2.5uhk;    // CHECK-DAG: @u_short_accum_literal  = {{.*}}global i16 640, align 2
unsigned _Accum        u_accum_literal       = 2.5uk;     // CHECK-DAG: @u_accum_literal        = {{.*}}global i32 163840, align 4
unsigned long _Accum   u_long_accum_literal  = 2.5ulk;    // CHECK-DAG: @u_long_accum_literal   = {{.*}}global i64 10737418240, align 8
unsigned short _Fract  u_short_fract_literal = 0.5uhr;    // CHECK-DAG: @u_short_fract_literal  = {{.*}}global i8  -128, align 1
unsigned _Fract        u_fract_literal       = 0.5ur;     // CHECK-DAG: @u_fract_literal        = {{.*}}global i16 -32768, align 2
unsigned long _Fract   u_long_fract_literal  = 0.5ulr;    // CHECK-DAG: @u_long_fract_literal   = {{.*}}global i32 -2147483648, align 4

// Max literal values
short _Accum          short_accum_max   = 255.9999999999999999hk;         // CHECK-DAG: @short_accum_max   = {{.*}}global i16 32767, align 2
_Accum                accum_max         = 65535.9999999999999999k;        // CHECK-DAG: @accum_max         = {{.*}}global i32 2147483647, align 4
long _Accum           long_accum_max    = 4294967295.9999999999999999lk;  // CHECK-DAG: @long_accum_max    = {{.*}}global i64 9223372036854775807, align 8
unsigned short _Accum u_short_accum_max = 255.9999999999999999uhk;        // CHECK-DAG: @u_short_accum_max = {{.*}}global i16 -1, align 2
unsigned _Accum       u_accum_max       = 65535.9999999999999999uk;       // CHECK-DAG: @u_accum_max       = {{.*}}global i32 -1, align 4
unsigned long _Accum  u_long_accum_max  = 4294967295.9999999999999999ulk; // CHECK-DAG: @u_long_accum_max  = {{.*}}global i64 -1, align 8

short _Fract          short_fract_max   = 0.9999999999999999hr;           // CHECK-DAG: @short_fract_max   = {{.*}}global i8  127, align 1
_Fract                fract_max         = 0.9999999999999999r;            // CHECK-DAG: @fract_max         = {{.*}}global i16 32767, align 2
long _Fract           long_fract_max    = 0.9999999999999999lr;           // CHECK-DAG: @long_fract_max    = {{.*}}global i32 2147483647, align 4
unsigned short _Fract u_short_fract_max = 0.9999999999999999uhr;          // CHECK-DAG: @u_short_fract_max = {{.*}}global i8  -1, align 1
unsigned _Fract       u_fract_max       = 0.9999999999999999ur;           // CHECK-DAG: @u_fract_max       = {{.*}}global i16 -1, align 2
unsigned long _Fract  u_long_fract_max  = 0.9999999999999999ulr;          // CHECK-DAG: @u_long_fract_max  = {{.*}}global i32 -1, align 4

// Fracts may be exactly one but evaluate to the Fract max
short _Fract          short_fract_one   = 1.0hr;    // CHECK-DAG: @short_fract_one    = {{.*}}global i8  127, align 1
_Fract                fract_one         = 1.0r;     // CHECK-DAG: @fract_one          = {{.*}}global i16 32767, align 2
long _Fract           long_fract_one    = 1.0lr;    // CHECK-DAG: @long_fract_one     = {{.*}}global i32 2147483647, align 4
unsigned short _Fract u_short_fract_one = 1.0uhr;   // CHECK-DAG: @u_short_fract_one  = {{.*}}global i8  -1, align 1
unsigned _Fract       u_fract_one       = 1.0ur;    // CHECK-DAG: @u_fract_one        = {{.*}}global i16 -1, align 2
unsigned long _Fract  u_long_fract_one  = 1.0ulr;   // CHECK-DAG: @u_long_fract_one   = {{.*}}global i32 -1, align 4

short _Fract          short_fract_exp_one   = 0.1e1hr;    // CHECK-DAG: @short_fract_exp_one    = {{.*}}global i8  127, align 1
_Fract                fract_exp_one         = 0.1e1r;     // CHECK-DAG: @fract_exp_one          = {{.*}}global i16 32767, align 2
long _Fract           long_fract_exp_one    = 0.1e1lr;    // CHECK-DAG: @long_fract_exp_one     = {{.*}}global i32 2147483647, align 4
unsigned short _Fract u_short_fract_exp_one = 0.1e1uhr;   // CHECK-DAG: @u_short_fract_exp_one  = {{.*}}global i8  -1, align 1
unsigned _Fract       u_fract_exp_one       = 0.1e1ur;    // CHECK-DAG: @u_fract_exp_one        = {{.*}}global i16 -1, align 2
unsigned long _Fract  u_long_fract_exp_one  = 0.1e1ulr;   // CHECK-DAG: @u_long_fract_exp_one   = {{.*}}global i32 -1, align 4

short _Fract          short_fract_hex_exp_one   = 0x0.8p1hr;    // CHECK-DAG: @short_fract_hex_exp_one    = {{.*}}global i8  127, align 1
_Fract                fract_hex_exp_one         = 0x0.8p1r;     // CHECK-DAG: @fract_hex_exp_one          = {{.*}}global i16 32767, align 2
long _Fract           long_fract_hex_exp_one    = 0x0.8p1lr;    // CHECK-DAG: @long_fract_hex_exp_one     = {{.*}}global i32 2147483647, align 4
unsigned short _Fract u_short_fract_hex_exp_one = 0x0.8p1uhr;   // CHECK-DAG: @u_short_fract_hex_exp_one  = {{.*}}global i8  -1, align 1
unsigned _Fract       u_fract_hex_exp_one       = 0x0.8p1ur;    // CHECK-DAG: @u_fract_hex_exp_one        = {{.*}}global i16 -1, align 2
unsigned long _Fract  u_long_fract_hex_exp_one  = 0x0.8p1ulr;   // CHECK-DAG: @u_long_fract_hex_exp_one   = {{.*}}global i32 -1, align 4

// Expsilon values
short _Accum          short_accum_eps   = 0x1p-7hk;         // CHECK-DAG: @short_accum_eps   = {{.*}}global i16 1, align 2
_Accum                accum_eps         = 0x1p-15k;         // CHECK-DAG: @accum_eps         = {{.*}}global i32 1, align 4
long _Accum           long_accum_eps    = 0x1p-31lk;        // CHECK-DAG: @long_accum_eps    = {{.*}}global i64 1, align 8
unsigned short _Accum u_short_accum_eps = 0x1p-8uhk;        // CHECK-DAG: @u_short_accum_eps = {{.*}}global i16 1, align 2
unsigned _Accum       u_accum_eps       = 0x1p-16uk;        // CHECK-DAG: @u_accum_eps       = {{.*}}global i32 1, align 4
unsigned long _Accum  u_long_accum_eps  = 0x1p-32ulk;       // CHECK-DAG: @u_long_accum_eps  = {{.*}}global i64 1, align 8

short _Fract          short_fract_eps   = 0x1p-7hr;         // CHECK-DAG: @short_fract_eps   = {{.*}}global i8  1, align 1
_Fract                fract_eps         = 0x1p-15r;         // CHECK-DAG: @fract_eps         = {{.*}}global i16 1, align 2
long _Fract           long_fract_eps    = 0x1p-31lr;        // CHECK-DAG: @long_fract_eps    = {{.*}}global i32 1, align 4
unsigned short _Fract u_short_fract_eps = 0x1p-8uhr;        // CHECK-DAG: @u_short_fract_eps = {{.*}}global i8  1, align 1
unsigned _Fract       u_fract_eps       = 0x1p-16ur;        // CHECK-DAG: @u_fract_eps       = {{.*}}global i16 1, align 2
unsigned long _Fract  u_long_fract_eps  = 0x1p-32ulr;       // CHECK-DAG: @u_long_fract_eps  = {{.*}}global i32 1, align 4

// Zero
short _Accum          short_accum_zero    = 0.0hk;    // CHECK-DAG: @short_accum_zero     = {{.*}}global i16 0, align 2
 _Accum               accum_zero          = 0.0k;     // CHECK-DAG: @accum_zero           = {{.*}}global i32 0, align 4
long _Accum           long_accum_zero     = 0.0lk;    // CHECK-DAG: @long_accum_zero      = {{.*}}global i64 0, align 8
unsigned short _Accum u_short_accum_zero  = 0.0uhk;   // CHECK-DAG: @u_short_accum_zero   = {{.*}}global i16 0, align 2
unsigned  _Accum      u_accum_zero        = 0.0uk;    // CHECK-DAG: @u_accum_zero         = {{.*}}global i32 0, align 4
unsigned long _Accum  u_long_accum_zero   = 0.0ulk;   // CHECK-DAG: @u_long_accum_zero    = {{.*}}global i64 0, align 8

short _Fract          short_fract_zero    = 0.0hr;    // CHECK-DAG: @short_fract_zero     = {{.*}}global i8  0, align 1
 _Fract               fract_zero          = 0.0r;     // CHECK-DAG: @fract_zero           = {{.*}}global i16 0, align 2
long _Fract           long_fract_zero     = 0.0lr;    // CHECK-DAG: @long_fract_zero      = {{.*}}global i32 0, align 4
unsigned short _Fract u_short_fract_zero  = 0.0uhr;   // CHECK-DAG: @u_short_fract_zero   = {{.*}}global i8  0, align 1
unsigned  _Fract      u_fract_zero        = 0.0ur;    // CHECK-DAG: @u_fract_zero         = {{.*}}global i16 0, align 2
unsigned long _Fract  u_long_fract_zero   = 0.0ulr;   // CHECK-DAG: @u_long_fract_zero    = {{.*}}global i32 0, align 4
