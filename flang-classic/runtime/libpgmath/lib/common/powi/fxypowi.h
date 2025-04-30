/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

extern void __pgi_abort(int, char*);

static const vrs4_t Csp1_4={1.0, 1.0, 1.0, 1.0};
static const vrd2_t Cdp1_2={1.0, 1.0};
static const vrs8_t Csp1_8={1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
static const vrd4_t Cdp1_4={1.0, 1.0, 1.0, 1.0};
static const vrs16_t Csp1_16={1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                             1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
static const vrd8_t Cdp1_8={1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

#define	powk1_body(_xyz,_prec,_vl)                                        \
  r = C##_prec##p1_##_vl;                                                 \
                                                                          \
  i = iy >= 0 ? iy : -iy;                                                 \
  if (iy > 0 && iy <= 10) {                                               \
    vr##_prec##_vl##_t t2;                                                \
    t2 = t * t;                                                           \
    switch(iy) {                                                          \
      case 1:                                                             \
        r = t;                                                            \
        break;                                                            \
                                                                          \
      case 2:                                                             \
        r = t2;                                                           \
        break;                                                            \
                                                                          \
      case 3:                                                             \
        r = t2 * t;                                                       \
        break;                                                            \
                                                                          \
      case 4:                                                             \
        r = t2 * t2;                                                      \
        break;                                                            \
                                                                          \
      case 5:                                                             \
        r = (t2 * t2) * t;                                                \
        break;                                                            \
                                                                          \
      case 6:                                                             \
        r = (t2 * t2) * t2;                                               \
        break;                                                            \
                                                                          \
      case 7:                                                             \
        r = (t2 * t2) * (t2 * t);                                         \
        break;                                                            \
                                                                          \
      case 8:                                                             \
        r = (t2 * t2) * (t2 * t2);                                        \
        break;                                                            \
                                                                          \
      case 9:                                                             \
        r = (t2 * t2) * (t2 * t2) * t;                                    \
        break;                                                            \
                                                                          \
      case 10:                                                            \
        r = (t2 * t2) * (t2 * t2) * t2;                                   \
        break;                                                            \
    }                                                                     \
  } else if (i != 0) {                                                    \
    while (true) {                                                        \
      if (i & 0x1)  r *= t;                                               \
       i >>= 1;                                                           \
       if (i == 0) break;                                                 \
       t *= t;                                                            \
    }                                                                     \
  }

// Yes, prolog should be spelled prologue
#define	powk1m_prolog(_xyz,_prec,_vl) \
  t = (vr##_prec##_vl##_t)((vi##_prec##_vl##_t)x & m);
#define	powk1m_epilog(_xyz,_prec,_vl) \
    t = (vr##_prec##_vl##_t)((vi##_prec##_vl##_t)C##_prec##p1_##_vl & ~m); \
    t = (vr##_prec##_vl##_t)((vi##_prec##_vl##_t)t | (vi##_prec##_vl##_t)r);

#define powk1m(_xyz,_prec,_vl) \
vr##_prec##_vl##_t __f##_xyz##_powk1_##_vl##m(vr##_prec##_vl##_t x, int64_t iy, vi##_prec##_vl##_t m) \
{ \
  vr##_prec##_vl##_t r; \
  vr##_prec##_vl##_t t; \
  int64_t i; \
\
  t = x; \
  powk1m_prolog(_xyz,_prec,_vl) \
  powk1_body(_xyz,_prec,_vl) \
  if (iy < 0) { \
    powk1m_epilog(_xyz,_prec,_vl) \
    r = C##_prec##p1_##_vl / t; \
  } \
  return r; \
} \
 \
vr##_prec##_vl##_t __f##_xyz##_powi1_##_vl##m(vr##_prec##_vl##_t x, int32_t iy, vi##_prec##_vl##_t m) \
{ \
  return __f##_xyz##_powk1_##_vl##m(x, iy, m); \
}

#define powk1(_xyz,_prec,_vl) \
vr##_prec##_vl##_t __f##_xyz##_powk1_##_vl(vr##_prec##_vl##_t x, int64_t iy) \
{ \
  vr##_prec##_vl##_t r; \
  vr##_prec##_vl##_t t; \
  int64_t i; \
\
  t = x; \
  powk1_body(_xyz,_prec,_vl) \
  if (iy < 0) { \
    r = C##_prec##p1_##_vl / r; \
  } \
  return r; \
} \
 \
vr##_prec##_vl##_t __f##_xyz##_powi1_##_vl(vr##_prec##_vl##_t x, int32_t iy) \
{ \
  return __f##_xyz##_powk1_##_vl(x, iy); \
}
