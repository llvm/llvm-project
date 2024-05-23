// RUN: %clang_cc1 -E -triple i386-linux -Wno-unknown-pragmas %s -o - | FileCheck %s

v1 = __ROUNDING_MODE__;
// CHECK: v1 = ;

#pragma STDC FENV_ROUND FE_TONEAREST
v2 = __ROUNDING_MODE__;
// CHECK: v2 = _rte;

#pragma STDC FENV_ROUND FE_TOWARDZERO
v3 = __ROUNDING_MODE__;
// CHECK: v3 = _rtz;

#pragma STDC FENV_ROUND FE_DOWNWARD
v4 = __ROUNDING_MODE__;
// CHECK: v4 = _rtn;

#pragma STDC FENV_ROUND FE_UPWARD
v5 = __ROUNDING_MODE__;
// CHECK: v5 = _rtp;

#pragma STDC FENV_ROUND FE_TONEARESTFROMZERO
v6 = __ROUNDING_MODE__;
// CHECK: v6 = _rta;

#pragma STDC FENV_ROUND FE_DYNAMIC
v7 = __ROUNDING_MODE__;
// CHECK: v7 = ;


#pragma STDC FENV_ROUND FE_TOWARDZERO
#pragma STDC FENV_ROUND FE_UPWARD
v10 = __ROUNDING_MODE__;
// CHECK: v10 = _rtp;

#pragma STDC FENV_ROUND FE_DYNAMIC
{
  #pragma STDC FENV_ROUND FE_TOWARDZERO
  {
    #pragma STDC FENV_ROUND FE_TONEAREST
    {
      #pragma STDC FENV_ROUND FE_DOWNWARD
      {
        #pragma STDC FENV_ROUND FE_UPWARD
        v11 = __ROUNDING_MODE__;
        // CHECK: v11 = _rtp;
      }
      v12 = __ROUNDING_MODE__;
      // CHECK: v12 = _rtn;
    }
    v13 = __ROUNDING_MODE__;
    // CHECK: v13 = _rte;
  }
  v14 = __ROUNDING_MODE__;
  // CHECK: v14 = _rtz;
}
v15 = __ROUNDING_MODE__;
// CHECK: v15 = ;


#define CONCAT(a, b) CONCAT_(a, b)
#define CONCAT_(a, b) a##b
#define ADD_ROUNDING_MODE_SUFFIX(func) CONCAT(func, __ROUNDING_MODE__)

#define sin(x) ADD_ROUNDING_MODE_SUFFIX(sin)(x)

sin(x);
// CHECK: sin(x);

#pragma STDC FENV_ROUND FE_TOWARDZERO
sin(x);
// CHECK: sin_rtz(x);

#pragma STDC FENV_ROUND FE_TONEAREST
sin(x);
// CHECK: sin_rte(x);

#pragma STDC FENV_ROUND FE_DOWNWARD
sin(x);
// CHECK: sin_rtn(x);

#pragma STDC FENV_ROUND FE_UPWARD
sin(x);
// CHECK: sin_rtp(x);

#pragma STDC FENV_ROUND FE_TONEARESTFROMZERO
sin(x);
// CHECK: sin_rta(x);
