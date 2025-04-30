/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* red.h -- header for intrinsic reduction functions */
/* FIXME: still used */

#include "fort_vars.h"

/* intrinsic reduction function enumeration */

typedef enum {
  __ALL,     /*  0 logical and */
  __ANY,     /*  1 logical or */
  __COUNT,   /*  2 logical count */
  __IALL,    /*  3 bitwise and */
  __IANY,    /*  4 bitwise or */
  __IPARITY, /*  5 bitwise xor */
  __MAXLOC,  /*  6 location of maximum */
  __MAXVAL,  /*  7 maximum value */
  __MINLOC,  /*  8 location of minimum */
  __MINVAL,  /*  9 minimum value */
  __PARITY,  /* 10 logical xor */
  __PRODUCT, /* 11 product */
  __SUM,     /* 12 sum */
  __FINDLOC, /* 13 location of value */
  __NREDS    /* 14 number of reduction functions */
} red_enum;

/* parameter struct for intrinsic reductions */

typedef struct {
  local_reduc_fn      l_fn;   /* local reduction function */
  local_reduc_back_fn l_fn_b; /* local reduction function with "back" arg */
  global_reduc_fn     g_fn;   /* global reduction function */
  char *rb, *ab; /* result, array base addresses */
  void *zb;      /* null value */
  __LOG_T *mb;   /* mask base address */
  __INT_T *xb;   /* location base address (max/minloc) */
  DECL_HDR_PTRS(rs);
  DECL_HDR_PTRS(as);
  DECL_HDR_PTRS(ms); /* result, array, mask descriptors */
  int dim;           /* dim argument (when present) */
  dtype kind;        /* result (max/minloc temp) kind & length */
  int len;
  __LOG_T back;        /* back argument (when present) */
  __INT_T mi[MAXDIMS]; /* mask index */
  int mask_present;    /* mask is non-scalar */
  int mask_stored_alike;
  int lk_shift; /* mask logical kind, where kind value is
                 * computed as 1<<lk_shift, where,
                 *     lk_shift = 0, 1, 2, 3, ...
                 */
} red_parm;

#define INIT_RED_PARM(z) memset(&z, '\0', sizeof(red_parm))

/* prototypes */

void __fort_red_unimplemented();

void __fort_red_abort(const char *msg);

void I8(__fort_red_scalar)(red_parm *z, char *rb, char *ab, char *mb,
                          F90_Desc *rs, F90_Desc *as, F90_Desc *ms, __INT_T *xb,
                          red_enum op);

void I8(__fort_red_scalarlk)(red_parm *z, char *rb, char *ab, char *mb,
                            F90_Desc *rs, F90_Desc *as, F90_Desc *ms,
                            __INT_T *xb, red_enum op);

void I8(__fort_kred_scalarlk)(red_parm *z, char *rb, char *ab, char *mb,
                             F90_Desc *rs, F90_Desc *as, F90_Desc *ms,
                             __INT8_T *xb, red_enum op);

void I8(__fort_red_array)(red_parm *z, char *rb0, char *ab, char *mb, char *db,
                         F90_Desc *rs0, F90_Desc *as, F90_Desc *ms,
                         F90_Desc *ds, red_enum op);

void I8(__fort_red_arraylk)(red_parm *z, char *rb0, char *ab, char *mb, char *db,
                           F90_Desc *rs0, F90_Desc *as, F90_Desc *ms,
                           F90_Desc *ds, red_enum op);

void I8(__fort_kred_arraylk)(red_parm *z, char *rb0, char *ab, char *mb,
                            char *db, F90_Desc *rs0, F90_Desc *as, F90_Desc *ms,
                            F90_Desc *ds, red_enum op);

void I8(__fort_global_reduce)(char *rb, char *hb, int dims, F90_Desc *rd,
                             F90_Desc *hd, const char *what, void (*fn[__NTYPES])());

/* prototype local reduction function (name beginning with l_):

   void l_NAME(void *r, __INT_T n, void *v, __INT_T vs, __LOG_T *m,
               __INT_T ms, __INT_T *loc, __INT_T li, __INT_T ls, __INT_T len);
   where
      r   = result address (scalar)
      n   = vector length
      v   = vector base address
      vs  = vector stride
      m   = mask vector address
      ms  = mask vector stride
      loc = maxloc/minloc element location
      li  = initial location
      ls  = location stride
      len = use for length of string

   prototype global parallel reduction function (name beginning with g_):

   void g_NAME(__INT_T n, RTYP *rl, RTYP *rr, void *vl, void *vr, __INT_T len);
   where
      n   = vector length
      lr  = local result vector
      rr  = remote result vector
      lv  = local min/max value vector
      rv  = remote min/max value vector
      len = use for length of string
*/

/* arithmetic reduction functions
   RTYP = result & vector type
   ATYP = accumulator type
*/

#define ARITHFNL(OP, NAME, RTYP, ATYP)                                          \
  static void l_##NAME(RTYP *r, __INT_T n, RTYP *v, __INT_T vs, __LOG_T *m,    \
                       __INT_T ms, __INT_T *loc, __INT_T li, __INT_T ls,       \
                       __INT_T len)                                            \
  {                                                                            \
    __INT_T i, j;                                                              \
    ATYP x = *r;                                                               \
    __LOG_T mask_log;                                                          \
    if (ms == 0)                                                               \
      for (i = 0; n > 0; n--, i += vs) {                                       \
        x = x OP v[i];                                                         \
      }                                                                        \
    else {                                                                     \
      mask_log = GET_DIST_MASK_LOG;                                           \
      for (i = j = 0; n > 0; n--, i += vs, j += ms) {                          \
        if (m[j] & mask_log)                                                   \
          x = x OP v[i];                                                       \
      }                                                                        \
    }                                                                          \
    *r = x;                                                                    \
  }

#define ARITHFNG(OP, NAME, RTYP, ATYP)                                         \
  static void g_##NAME(__INT_T n, RTYP *lr, RTYP *rr, void *lv, void *rv,      \
                       __INT_T len)                                            \
  {                                                                            \
    __INT_T i;                                                                 \
    for (i = 0; i < n; i++) {                                                  \
      lr[i] = lr[i] OP rr[i];                                                  \
    }                                                                          \
  }

#define ARITHFNLKN(OP, NAME, RTYP, ATYP, N)                                    \
  static void l_##NAME##l##N(RTYP *r, __INT_T n, RTYP *v, __INT_T vs,          \
                             __LOG##N##_T *m, __INT_T ms, __INT_T *loc,        \
                             __INT_T li, __INT_T ls, __INT_T len)              \
  {                                                                            \
    __INT_T i, j;                                                              \
    ATYP x = *r;                                                               \
    __LOG##N##_T mask_log;                                                     \
    if (ms == 0)                                                               \
      for (i = 0; n > 0; n--, i += vs) {                                       \
        x = x OP v[i];                                                         \
      }                                                                        \
    else {                                                                     \
      mask_log = GET_DIST_MASK_LOG##N;                                        \
      for (i = j = 0; n > 0; n--, i += vs, j += ms) {                          \
        if (m[j] & mask_log)                                                   \
          x = x OP v[i];                                                       \
      }                                                                        \
    }                                                                          \
    *r = x;                                                                    \
  }

/* note: all, any, parity, and count do not have mask arguments */

#define LOGFNL(OP, NAME, RTYP)                                                  \
  static void l_##NAME(RTYP *r, __INT_T n, RTYP *v, __INT_T vs, __LOG_T *m,    \
                       __INT_T ms, __INT_T *loc, __INT_T li, __INT_T ls,       \
                       __INT_T len)                                            \
  {                                                                            \
    int x;                                                                     \
    __INT_T i;                                                                 \
    __LOG_T mask_log = GET_DIST_MASK_LOG;                                     \
    x = ((*r & mask_log) != 0);                                                \
    for (i = 0; n > 0; n--, i += vs) {                                         \
      x = x OP((v[i] & mask_log) != 0);                                        \
    }                                                                          \
    *r = (RTYP)(x ? GET_DIST_TRUE_LOG : 0);                                   \
  }

#define LOGFNG(OP, NAME, RTYP)                                                \
  static void g_##NAME(__INT_T n, RTYP *lr, RTYP *rr, void *lv, void *rv,      \
                       __INT_T len)                                            \
  {                                                                            \
    __INT_T i;                                                                 \
    for (i = 0; i < n; i++) {                                                  \
      lr[i] = lr[i] OP rr[i];                                                  \
    }                                                                          \
  }

#define LOGFNLKN(OP, NAME, RTYP, N)                                            \
  static void l_##NAME##l##N(RTYP *r, __INT_T n, RTYP *v, __INT_T vs,          \
                             __LOG##N##_T *m, __INT_T ms, __INT_T *loc,        \
                             __INT_T li, __INT_T ls, __INT_T len)              \
  {                                                                            \
    int x;                                                                     \
    __INT_T i;                                                                 \
    __LOG##N##_T mask_log = GET_DIST_MASK_LOG##N;                             \
    x = ((*r & mask_log) != 0);                                                \
    for (i = 0; n > 0; n--, i += vs) {                                         \
      x = x OP((v[i] & mask_log) != 0);                                        \
    }                                                                          \
    *r = (RTYP)(x ? GET_DIST_TRUE_LOG : 0);                                   \
  }

#define CONDFN(COND, NAME, RTYP)                                               \
  static void l_##NAME(RTYP *r, __INT_T n, RTYP *v, __INT_T vs, __LOG_T *m,    \
                       __INT_T ms, __INT_T *loc, __INT_T li, __INT_T ls,       \
                       __INT_T len)                                            \
  {                                                                            \
    __INT_T i, j;                                                              \
    RTYP x = *r;                                                               \
    __LOG_T mask_log;                                                          \
    if (ms == 0)                                                               \
      for (i = 0; n > 0; n--, i += vs) {                                       \
        if (v[i] COND x)                                                       \
          x = v[i];                                                            \
      }                                                                        \
    else {                                                                     \
      mask_log = GET_DIST_MASK_LOG;                                           \
      for (i = j = 0; n > 0; n--, i += vs, j += ms) {                          \
        if (m[j] & mask_log && v[i] COND x)                                    \
          x = v[i];                                                            \
      }                                                                        \
    }                                                                          \
    *r = x;                                                                    \
  }                                                                            \
  static void g_##NAME(__INT_T n, RTYP *lr, RTYP *rr, void *lv, void *rv,      \
                       __INT_T len)                                            \
  {                                                                            \
    __INT_T i;                                                                 \
    for (i = 0; i < n; i++) {                                                  \
      if (rr[i] COND lr[i])                                                    \
        lr[i] = rr[i];                                                         \
    }                                                                          \
  }

#define CONDFNG(COND, NAME, RTYP)                                              \
  static void g_##NAME(__INT_T n, RTYP *lr, RTYP *rr, void *lv, void *rv,      \
                       __INT_T len)                                            \
  {                                                                            \
    __INT_T i;                                                                 \
    for (i = 0; i < n; i++) {                                                  \
      if (rr[i] COND lr[i])                                                    \
        lr[i] = rr[i];                                                         \
    }                                                                          \
  }

#define CONDSTRFNG(COND, NAME, RTYP)                                           \
  static void g_##NAME(__INT_T n, RTYP *lr, RTYP *rr, void *lv, void *rv,      \
                       __INT_T len)                                            \
  {                                                                            \
    __INT_T i;                                                                 \
    for (i = 0; i < n; i++, lr += len, rr += len) {                            \
      if (strncmp(rr, lr, len) COND 0)                                         \
        strncpy(lr, rr, len);                                                  \
    }                                                                          \
  }

#define CONDFNLKN(COND, NAME, RTYP, N)                                         \
  static void l_##NAME##l##N(RTYP *r, __INT_T n, RTYP *v, __INT_T vs,          \
                             __LOG##N##_T *m, __INT_T ms, __INT_T *loc,        \
                             __INT_T li, __INT_T ls, __INT_T len)              \
  {                                                                            \
    __INT_T i, j;                                                              \
    RTYP x = *r;                                                               \
    __LOG##N##_T mask_log;                                                     \
    if (ms == 0)                                                               \
      for (i = 0; n > 0; n--, i += vs) {                                       \
        if (v[i] COND x)                                                       \
          x = v[i];                                                            \
      }                                                                        \
    else {                                                                     \
      mask_log = GET_DIST_MASK_LOG##N;                                        \
      for (i = j = 0; n > 0; n--, i += vs, j += ms) {                          \
        if (m[j] & mask_log && v[i] COND x)                                    \
          x = v[i];                                                            \
      }                                                                        \
    }                                                                          \
    *r = x;                                                                    \
  }

#define CONDSTRFNLKN(COND, NAME, RTYP, N)                                      \
  static void l_##NAME##l##N(RTYP *r, __INT_T n, RTYP *v, __INT_T vs,          \
                             __LOG##N##_T *m, __INT_T ms, __INT_T *loc,        \
                             __INT_T li, __INT_T ls, __INT_T len)              \
  {                                                                            \
    __INT_T i, j, ahop;                                                        \
    RTYP *x = r;                                                               \
    __LOG##N##_T mask_log;                                                     \
    ahop = len * vs;                                                           \
    if (ms == 0)                                                               \
      for (i = 0; n > 0; n--, i += vs, v += (ahop)) {                          \
        if (strncmp(v, x, len) COND 0)                                         \
          x = v;                                                               \
      }                                                                        \
    else {                                                                     \
      mask_log = GET_DIST_MASK_LOG##N;                                        \
      for (i = j = 0; n > 0; n--, i += vs, j += ms, v += (ahop)) {             \
        if (m[j] & mask_log && strncmp(v, x, len) COND 0)                      \
          x = v;                                                               \
      }                                                                        \
    }                                                                          \
    strncpy(r, x, len);                                                        \
  }

#define MLOCFN(COND, NAME, RTYP)                                               \
  static void l_##NAME(RTYP *r, __INT_T n, RTYP *v, __INT_T vs, __LOG_T *m,    \
                       __INT_T ms, __INT4_T *loc, __INT_T li, __INT_T ls,      \
                       __INT_T len)                                            \
  {                                                                            \
    __INT4_T i, j;                                                             \
    __INT4_T t_loc = 0;                                                        \
    RTYP val = *r;                                                             \
    __LOG_T mask_log;                                                          \
    if (ms == 0) {                                                             \
      for (i = 0; n > 0; n--, i += vs, li += ls) {                             \
        if (v[i] COND val) {                                                   \
          t_loc = li;                                                          \
          val = v[i];                                                          \
        } else if (v[i] == val && t_loc == 0 && *loc == 0) {                   \
          t_loc = li;                                                          \
        }                                                                      \
      }                                                                        \
    } else {                                                                   \
      mask_log = GET_DIST_MASK_LOG;                                           \
      for (i = j = 0; n > 0; n--, i += vs, j += ms, li += ls) {                \
        if ((m[j] & mask_log)) {                                               \
          if (v[i] COND val) {                                                 \
            t_loc = li;                                                        \
            val = v[i];                                                        \
          } else if (v[i] == val && t_loc == 0 && *loc == 0) {                 \
            t_loc = li;                                                        \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    }                                                                          \
    *r = val;                                                                  \
    if (t_loc != 0)                                                            \
      *loc = t_loc;                                                            \
  }                                                                            \
  static void g_##NAME(__INT_T n, RTYP *lval, RTYP *rval, __INT4_T *lloc,      \
                       __INT_T *rloc, __INT_T len)                             \
  {                                                                            \
    __INT4_T i;                                                                \
    for (i = 0; i < n; i++) {                                                  \
      if (rval[i] COND lval[i]) {                                              \
        lloc[i] = rloc[i];                                                     \
        lval[i] = rval[i];                                                     \
      } else if (rval[i] == lval[i] && rloc[i] < lloc[i]) {                    \
        lloc[i] = rloc[i];                                                     \
      }                                                                        \
    }                                                                          \
  }

#define MLOCSTRFN(COND, NAME, RTYP)                                            \
  static void l_##NAME(RTYP *r, __INT_T n, RTYP *v, __INT_T vs, __LOG_T *m,    \
                       __INT_T ms, __INT4_T *loc, __INT_T li, __INT_T ls,      \
                       __INT_T len)                                            \
  {                                                                            \
    __INT4_T i, j, ahop;                                                       \
    __INT4_T t_loc = 0;                                                        \
    RTYP *val = r;                                                             \
    __LOG_T mask_log;                                                          \
    ahop = len * vs;                                                           \
    if (ms == 0) {                                                             \
      for (i = 0; n > 0; n--, i += vs, li += ls, v += (ahop)) {                \
        if (strncmp(v, val, len) COND 0) {                                     \
          t_loc = li;                                                          \
          val = v;                                                             \
        } else if (strncmp(v, val, len) == 0 && t_loc == 0 && *loc == 0) {     \
          t_loc = li;                                                          \
        }                                                                      \
      }                                                                        \
    } else {                                                                   \
      mask_log = GET_DIST_MASK_LOG;                                           \
      for (i = j = 0; n > 0; n--, i += vs, j += ms, li += ls, v += (ahop)) {   \
        if ((m[j] & mask_log)) {                                               \
          if (strncmp(v, val, len) COND 0) {                                   \
            t_loc = li;                                                        \
            val = v;                                                           \
          } else if (strncmp(v, val, len) == 0 && t_loc == 0 && *loc == 0) {   \
            t_loc = li;                                                        \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    }                                                                          \
    strncpy(r, val, len);                                                      \
    if (t_loc != 0)                                                            \
      *loc = t_loc;                                                            \
  }                                                                            \
  static void g_##NAME(__INT_T n, RTYP *lval, RTYP *rval, __INT4_T *lloc,      \
                       __INT_T *rloc, __INT_T len)                             \
  {                                                                            \
    __INT4_T i;                                                                \
    for (i = 0; i < n; i++, rval += len, lval += len) {                        \
      if (strncmp(rval, lval, len) COND 0) {                                   \
        lloc[i] = rloc[i];                                                     \
        strncpy(lval, rval, len);                                              \
      } else if (strncmp(rval, lval, len) == 0 && rloc[i] < lloc[i]) {         \
        lloc[i] = rloc[i];                                                     \
      }                                                                        \
    }                                                                          \
  }

#define MLOCFNG(COND, NAME, RTYP)                                              \
  static void g_##NAME(__INT_T n, RTYP *lval, RTYP *rval, __INT4_T *lloc,      \
                       __INT_T *rloc, __INT_T len)                             \
  {                                                                            \
    __INT4_T i;                                                                \
    for (i = 0; i < n; i++) {                                                  \
      if (rval[i] COND lval[i]) {                                              \
        lloc[i] = rloc[i];                                                     \
        lval[i] = rval[i];                                                     \
      } else if (rval[i] == lval[i] && rloc[i] < lloc[i]) {                    \
        lloc[i] = rloc[i];                                                     \
      }                                                                        \
    }                                                                          \
  }

#define MLOCSTRFNG(COND, NAME, RTYP)                                           \
  static void g_##NAME(__INT_T n, RTYP *lval, RTYP *rval, __INT4_T *lloc,      \
                       __INT_T *rloc, __INT_T len)                             \
  {                                                                            \
    __INT4_T i;                                                                \
    for (i = 0; i < n; i++, rval += len, lval += len) {                        \
      if (strncmp(rval, lval, len) COND 0) {                                   \
        lloc[i] = rloc[i];                                                     \
        strncpy(lval, rval, len);                                              \
      } else if (strncmp(rval, lval, len) == 0 && rloc[i] < lloc[i]) {         \
        lloc[i] = rloc[i];                                                     \
      }                                                                        \
    }                                                                          \
  }

#define MLOCFNLKN(COND, NAME, RTYP, N)                                         \
  static void l_##NAME##l##N(RTYP *r, __INT_T n, RTYP *v, __INT_T vs,          \
                             __LOG##N##_T *m, __INT_T ms, __INT4_T *loc,       \
                             __INT_T li, __INT_T ls, __INT_T len, __LOG_T back)\
  {                                                                            \
    __INT4_T i, j, t_loc = 0;                                                  \
    RTYP val = *r;                                                             \
    __LOG##N##_T mask_log;                                                     \
    if (ms == 0) {                                                             \
      for (i = 0; n > 0; n--, i += vs, li += ls) {                             \
        if (v[i] COND val) {                                                   \
          t_loc = li;                                                          \
          val = v[i];                                                          \
        } else if (v[i] == val && (back || (t_loc == 0 && *loc == 0))) {       \
          t_loc = li;                                                          \
        }                                                                      \
      }                                                                        \
    } else {                                                                   \
      mask_log = GET_DIST_MASK_LOG##N;                                         \
      for (i = j = 0; n > 0; n--, i += vs, j += ms, li += ls) {                \
        if ((m[j] & mask_log)) {                                               \
          if (v[i] COND val) {                                                 \
            t_loc = li;                                                        \
            val = v[i];                                                        \
          } else if (v[i] == val && (back || (t_loc == 0 && *loc == 0))) {     \
            t_loc = li;                                                        \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    }                                                                          \
    *r = val;                                                                  \
    if (t_loc != 0)                                                            \
      *loc = t_loc;                                                            \
  }

#define MLOCSTRFNLKN(COND, NAME, RTYP, N)                                      \
  static void l_##NAME##l##N(RTYP *r, __INT_T n, RTYP *v, __INT_T vs,          \
                             __LOG##N##_T *m, __INT_T ms, __INT4_T *loc,       \
                             __INT_T li, __INT_T ls, __INT_T len, __LOG_T back)\
  {                                                                            \
    __INT4_T i, j, ahop, t_loc = 0;                                            \
    RTYP *val = r;                                                             \
    __LOG##N##_T mask_log;                                                     \
    ahop = len * vs;                                                           \
    if (ms == 0) {                                                             \
      for (i = 0; n > 0; n--, i += vs, li += ls, v += (ahop)) {                \
        if (strncmp(v, val, len) COND 0) {                                     \
          t_loc = li;                                                          \
          val = v;                                                             \
        } else if (strncmp(v, val, len) == 0                                   \
                  && (back || (t_loc == 0 && *loc == 0))) {                    \
          t_loc = li;                                                          \
        }                                                                      \
      }                                                                        \
    } else {                                                                   \
      mask_log = GET_DIST_MASK_LOG##N;                                         \
      for (i = j = 0; n > 0; n--, i += vs, j += ms, li += ls, v += (ahop)) {   \
        if ((m[j] & mask_log)) {                                               \
          if (strncmp(v, val, len) COND 0) {                                   \
            t_loc = li;                                                        \
            val = v;                                                           \
          } else if (strncmp(v, val, len) == 0                                 \
                    && (back || (t_loc == 0 && *loc == 0))) {                  \
            t_loc = li;                                                        \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    }                                                                          \
    strncpy(r, val, len);                                                      \
    if (t_loc != 0)                                                            \
      *loc = t_loc;                                                            \
  }

#define KMLOCFNG(COND, NAME, RTYP)                                             \
  static void g_##NAME(__INT_T n, RTYP *lval, RTYP *rval, __INT8_T *lloc,      \
                       __INT8_T *rloc, __INT_T len)                            \
  {                                                                            \
    __INT_T i;                                                                 \
    for (i = 0; i < n; i++) {                                                  \
      if (rval[i] COND lval[i]) {                                              \
        lloc[i] = rloc[i];                                                     \
        lval[i] = rval[i];                                                     \
      } else if (rval[i] == lval[i] && rloc[i] < lloc[i]) {                    \
        lloc[i] = rloc[i];                                                     \
      }                                                                        \
    }                                                                          \
  }

#define KMLOCSTRFNG(COND, NAME, RTYP)                                          \
  static void g_##NAME(__INT_T n, RTYP *lval, RTYP *rval, __INT8_T *lloc,      \
                       __INT8_T *rloc, __INT_T len)                            \
  {                                                                            \
    __INT_T i;                                                                 \
    for (i = 0; i < n; i++, lval += len, rval += len) {                        \
      if (strncmp(rval, lval, len) COND 0) {                                   \
        lloc[i] = rloc[i];                                                     \
        strncpy(lval, rval, len);                                              \
      } else if (strncmp(rval, lval, len) == 0 && rloc[i] < lloc[i]) {         \
        lloc[i] = rloc[i];                                                     \
      }                                                                        \
    }                                                                          \
  }

#define KMLOCFNLKN(COND, NAME, RTYP, N)                                        \
  static void l_##NAME##l##N(RTYP *r, __INT_T n, RTYP *v, __INT_T vs,          \
                             __LOG##N##_T *m, __INT_T ms, __INT8_T *loc,       \
                             __INT_T li, __INT_T ls, __INT_T len, __LOG_T back)\
  {                                                                            \
    __INT_T i, j, t_loc = 0;                                                   \
    RTYP val = *r;                                                             \
    __LOG##N##_T mask_log;                                                     \
    if (ms == 0) {                                                             \
      for (i = 0; n > 0; n--, i += vs, li += ls) {                             \
        if (v[i] COND val) {                                                   \
          t_loc = li;                                                          \
          val = v[i];                                                          \
        } else if (v[i] == val && (back || (t_loc == 0 && *loc == 0))) {       \
          t_loc = li;                                                          \
        }                                                                      \
      }                                                                        \
    } else {                                                                   \
      mask_log = GET_DIST_MASK_LOG##N;                                        \
      for (i = j = 0; n > 0; n--, i += vs, j += ms, li += ls) {                \
        if ((m[j] & mask_log)) {                                               \
          if (v[i] COND val) {                                                 \
            t_loc = li;                                                        \
            val = v[i];                                                        \
          } else if (v[i] == val && (back || (t_loc == 0 && *loc == 0))) {     \
            t_loc = li;                                                        \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    }                                                                          \
    *r = val;                                                                  \
    if (t_loc != 0)                                                            \
      *loc = t_loc;                                                            \
  }

#define KMLOCSTRFNLKN(COND, NAME, RTYP, N)                                     \
  static void l_##NAME##l##N(RTYP *r, __INT_T n, RTYP *v, __INT_T vs,          \
                             __LOG##N##_T *m, __INT_T ms, __INT8_T *loc,       \
                             __INT_T li, __INT_T ls, __INT_T len, __INT_T back)\
  {                                                                            \
    __INT_T i, j, ahop, t_loc = 0;                                             \
    RTYP *val = r;                                                             \
    __LOG##N##_T mask_log;                                                     \
    ahop = len * vs;                                                           \
    if (ms == 0) {                                                             \
      for (i = 0; n > 0; n--, i += vs, li += ls, v += (ahop)) {                \
        if (strncmp(v, val, len) COND 0) {                                     \
          t_loc = li;                                                          \
          val = v;                                                             \
        } else if (strncmp(v, val, len) == 0                                   \
                  && (back || (t_loc == 0 && *loc == 0))) {                    \
          t_loc = li;                                                          \
        }                                                                      \
      }                                                                        \
    } else {                                                                   \
      mask_log = GET_DIST_MASK_LOG##N;                                         \
      for (i = j = 0; n > 0; n--, i += vs, j += ms, li += ls, v += (ahop)) {   \
        if ((m[j] & mask_log)) {                                               \
          if (strncmp(v, val, len) COND 0) {                                   \
            t_loc = li;                                                        \
            val = v;                                                           \
          } else if (strncmp(v, val, len) == 0                                 \
                    && (back || (t_loc == 0 && *loc == 0))) {                  \
            t_loc = li;                                                        \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    }                                                                          \
    strncpy(r, val, len);                                                      \
    if (t_loc != 0)                                                            \
      *loc = t_loc;                                                            \
  }

#define FLOCFN(COND, NAME, RTYP)                                               \
  static void l_##NAME(RTYP *r, __INT_T n, RTYP *v, __INT_T vs, __LOG_T *m,    \
                       __INT_T ms, __INT4_T *loc, __INT_T li, __INT_T ls,      \
                       __INT_T len, __LOG_T back)                              \
  {                                                                            \
    __INT4_T i, j;                                                             \
    __INT4_T t_loc = 0;                                                        \
    RTYP val = *r;                                                             \
    __LOG_T mask_log;                                                          \
    if (!back && *loc != 0)                                                    \
      return;                                                                  \
    if (ms == 0) {                                                             \
      for (i = 0; n > 0; n--, i += vs, li += ls) {                             \
        if (v[i] COND val) {                                                   \
          t_loc = li;                                                          \
          if (!back)                                                           \
            break;                                                             \
        }                                                                      \
      }                                                                        \
    } else {                                                                   \
      mask_log = GET_DIST_MASK_LOG;                                           \
      for (i = j = 0; n > 0; n--, i += vs, j += ms, li += ls) {                \
        if ((m[j] & mask_log)) {                                               \
          if (v[i] COND val) {                                                 \
            t_loc = li;                                                        \
            if (!back)                                                         \
              break;                                                           \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    }                                                                          \
    if (t_loc != 0)                                                            \
      *loc = t_loc;                                                            \
  }                                                                            \
  static void g_##NAME(__INT_T n, RTYP *lval, RTYP *rval, __INT4_T *lloc,      \
                       __INT_T *rloc, __INT_T len, __LOG_T back)               \
  {                                                                            \
    __INT4_T i;                                                                \
    for (i = 0; i < n; i++) {                                                  \
      if (rval[i] COND lval[i]) {                                              \
        lloc[i] = rloc[i];                                                     \
      } else if (rval[i] == lval[i] && rloc[i] < lloc[i]) {                    \
        lloc[i] = rloc[i];                                                     \
      }                                                                        \
    }                                                                          \
  }

#define FLOCSTRFN(COND, NAME, RTYP)                                                                 \
static void l_ ## NAME(RTYP *r, __INT_T n, RTYP *v, __INT_T vs, \
                       __LOG_T *m, __INT_T ms, __INT4_T *loc, __INT_T li, __INT_T ls, __INT_T len), \
                       __LOG_T back )                                                               \
  {                                                                                                 \
    __INT4_T i, j, ahop;                                                                            \
    __INT4_T t_loc = 0;                                                                             \
    RTYP *val = v;                                                                                  \
    __LOG_T mask_log;                                                                               \
    if (!back && *loc != 0)                                                                         \
      return;                                                                                       \
    ahop = len * vs;                                                                                \
    if (ms == 0) {                                                                                  \
      for (i = 0; n > 0; n--, i += vs, li += ls, v += (ahop)) {                                     \
        if (strncmp(r, v, len) COND 0) {                                                            \
          t_loc = li;                                                                               \
          if (!back)                                                                                \
            break;                                                                                  \
        }                                                                                           \
      }                                                                                             \
    } else {                                                                                        \
      mask_log = GET_DIST_MASK_LOG;                                                                \
      for (i = j = 0; n > 0; n--, i += vs, j += ms, li += ls, v += (ahop)) {                        \
        if ((m[j] & mask_log)) {                                                                    \
          if (strncmp(r, v, len) COND 0) {                                                          \
            t_loc = li;                                                                             \
            if (!back)                                                                              \
              break;                                                                                \
          }                                                                                         \
        }                                                                                           \
      }                                                                                             \
    }                                                                                               \
    if (t_loc != 0)                                                                                 \
      *loc = t_loc;                                                                                 \
  }                                                                                                 \
  static void g_##NAME(__INT_T n, RTYP *lval, RTYP *rval, __INT4_T *lloc,                           \
                       __INT_T *rloc, __INT_T len, __LOG_T back)                                    \
  {                                                                                                 \
    __INT4_T i;                                                                                     \
    for (i = 0; i < n; i++, rval += len, lval += len) {                                             \
      if (strncmp(rval, lval, len) COND 0) {                                                        \
        lloc[i] = rloc[i];                                                                          \
        if (!back)                                                                                  \
          break;                                                                                    \
      } else if (strncmp(rval, lval, len) == 0 && rloc[i] < lloc[i]) {                              \
        lloc[i] = rloc[i];                                                                          \
        if (!back)                                                                                  \
          break;                                                                                    \
      }                                                                                             \
    }                                                                                               \
  }

#define FLOCFNG(COND, NAME, RTYP)                                              \
  static void g_##NAME(__INT_T n, RTYP *lval, RTYP *rval, __INT4_T *lloc,      \
                       __INT_T *rloc, __INT_T len, __LOG_T back)               \
  {                                                                            \
    __INT4_T i;                                                                \
    for (i = 0; i < n; i++) {                                                  \
      if (rval[i] COND lval[i]) {                                              \
        lloc[i] = rloc[i];                                                     \
      } else if (rval[i] == lval[i] && rloc[i] < lloc[i]) {                    \
        lloc[i] = rloc[i];                                                     \
      }                                                                        \
    }                                                                          \
  }

#define FLOCSTRFNG(COND, NAME, RTYP)                                           \
  static void g_##NAME(__INT_T n, RTYP *lval, RTYP *rval, __INT4_T *lloc,      \
                       __INT_T *rloc, __INT_T len, __LOG_T back)               \
  {                                                                            \
    __INT4_T i;                                                                \
    for (i = 0; i < n; i++, rval += len, lval += len) {                        \
      if (strncmp(rval, lval, len) COND 0) {                                   \
        lloc[i] = rloc[i];                                                     \
      }                                                                        \
    }                                                                          \
  }

#define FLOCFNLKN(COND, NAME, RTYP, N)                                         \
  static void l_##NAME##l##N(                                                  \
      RTYP *r, __INT_T n, RTYP *v, __INT_T vs, __LOG##N##_T *m, __INT_T ms,    \
      __INT4_T *loc, __INT_T li, __INT_T ls, __INT_T len, __LOG_T back)        \
  {                                                                            \
    __INT4_T i, j, t_loc = 0;                                                  \
    RTYP val = *r;                                                             \
    __LOG##N##_T mask_log;                                                     \
    if (!back && *loc != 0)                                                    \
      return;                                                                  \
    if (ms == 0) {                                                             \
      for (i = 0; n > 0; n--, i += vs, li += ls) {                             \
        if (v[i] COND val) {                                                   \
          t_loc = li;                                                          \
          if (!back)                                                           \
            break;                                                             \
        }                                                                      \
      }                                                                        \
    } else {                                                                   \
      mask_log = GET_DIST_MASK_LOG##N;                                        \
      for (i = j = 0; n > 0; n--, i += vs, j += ms, li += ls) {                \
        if ((m[j] & mask_log)) {                                               \
          if (v[i] COND val) {                                                 \
            t_loc = li;                                                        \
            if (!back)                                                         \
              break;                                                           \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    }                                                                          \
    if (t_loc != 0)                                                            \
      *loc = t_loc;                                                            \
  }

#define FLOCSTRFNLKN(COND, NAME, RTYP, N)                                      \
  static void l_##NAME##l##N(                                                  \
      RTYP *r, __INT_T n, RTYP *v, __INT_T vs, __LOG##N##_T *m, __INT_T ms,    \
      __INT4_T *loc, __INT_T li, __INT_T ls, __INT_T len, __LOG_T back)        \
  {                                                                            \
    __INT4_T i, j, ahop, t_loc = 0;                                            \
    __LOG##N##_T mask_log;                                                     \
    if (!back && *loc != 0)                                                    \
      return;                                                                  \
    ahop = len * vs;                                                           \
    if (ms == 0) {                                                             \
      for (i = 0; n > 0; n--, i += vs, li += ls, v += (ahop)) {                \
        if (strncmp(r, v, len) COND 0) {                                       \
          t_loc = li;                                                          \
          if (!back)                                                           \
            break;                                                             \
        }                                                                      \
      }                                                                        \
    } else {                                                                   \
      mask_log = GET_DIST_MASK_LOG##N;                                        \
      for (i = j = 0; n > 0; n--, i += vs, j += ms, li += ls, v += (ahop)) {   \
        if ((m[j] & mask_log)) {                                               \
          if (strncmp(r, v, len) COND 0) {                                     \
            t_loc = li;                                                        \
            if (!back)                                                         \
              break;                                                           \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    }                                                                          \
    if (t_loc != 0)                                                            \
      *loc = t_loc;                                                            \
  }

#define KFLOCFNG(COND, NAME, RTYP)                                             \
  static void g_##NAME(__INT_T n, RTYP *lval, RTYP *rval, __INT8_T *lloc,      \
                       __INT8_T *rloc, __INT_T len, __LOG_T back)              \
  {                                                                            \
    __INT_T i;                                                                 \
    for (i = 0; i < n; i++) {                                                  \
      if (rval[i] COND lval[i]) {                                              \
        lloc[i] = rloc[i];                                                     \
        if (!back)                                                             \
          break;                                                               \
      } else if (rval[i] == lval[i] && rloc[i] < lloc[i]) {                    \
        lloc[i] = rloc[i];                                                     \
        if (!back)                                                             \
          break;                                                               \
      }                                                                        \
    }                                                                          \
  }

#define KFLOCSTRFNG(COND, NAME, RTYP)                                          \
  static void g_##NAME(__INT_T n, RTYP *lval, RTYP *rval, __INT8_T *lloc,      \
                       __INT8_T *rloc, __INT_T len, __LOG_T back)              \
  {                                                                            \
    __INT_T i;                                                                 \
    for (i = 0; i < n; i++, lval += len, rval += len) {                        \
      if (strncmp(rval, lval, len) COND 0) {                                   \
        lloc[i] = rloc[i];                                                     \
        if (!back)                                                             \
          break;                                                               \
      } else if (strncmp(rval, lval, len) == 0 && rloc[i] < lloc[i]) {         \
        lloc[i] = rloc[i];                                                     \
        if (!back)                                                             \
          break;                                                               \
      }                                                                        \
    }                                                                          \
  }

#define KFLOCFNLKN(COND, NAME, RTYP, N)                                        \
  static void l_##NAME##l##N(                                                  \
      RTYP *r, __INT_T n, RTYP *v, __INT_T vs, __LOG##N##_T *m, __INT_T ms,    \
      __INT8_T *loc, __INT_T li, __INT_T ls, __INT_T len, __LOG_T back)        \
  {                                                                            \
    __INT_T i, j, t_loc = 0;                                                   \
    RTYP val = *r;                                                             \
    __LOG##N##_T mask_log;                                                     \
    if (!back && *loc != 0)                                                    \
      return;                                                                  \
    if (ms == 0) {                                                             \
      for (i = 0; n > 0; n--, i += vs, li += ls) {                             \
        if (v[i] COND val) {                                                   \
          t_loc = li;                                                          \
          if (!back)                                                           \
            break;                                                             \
        }                                                                      \
      }                                                                        \
    } else {                                                                   \
      mask_log = GET_DIST_MASK_LOG##N;                                        \
      for (i = j = 0; n > 0; n--, i += vs, j += ms, li += ls) {                \
        if ((m[j] & mask_log)) {                                               \
          if (v[i] COND val) {                                                 \
            t_loc = li;                                                        \
            if (!back)                                                         \
              break;                                                           \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    }                                                                          \
    if (t_loc != 0)                                                            \
      *loc = t_loc;                                                            \
  }

#define KFLOCSTRFNLKN(COND, NAME, RTYP, N)                                     \
  static void l_##NAME##l##N(                                                  \
      RTYP *r, __INT_T n, RTYP *v, __INT_T vs, __LOG##N##_T *m, __INT_T ms,    \
      __INT8_T *loc, __INT_T li, __INT_T ls, __INT_T len, __LOG_T back)        \
  {                                                                            \
    __INT_T i, j, ahop, t_loc = 0;                                             \
    __LOG##N##_T mask_log;                                                     \
    if (!back && *loc != 0)                                                    \
      return;                                                                  \
    ahop = len * vs;                                                           \
    if (ms == 0) {                                                             \
      for (i = 0; n > 0; n--, i += vs, li += ls, v += (ahop)) {                \
        if (strncmp(r, v, len) COND 0) {                                       \
          t_loc = li;                                                          \
          if (!back)                                                           \
            break;                                                             \
        }                                                                      \
      }                                                                        \
    } else {                                                                   \
      mask_log = GET_DIST_MASK_LOG##N;                                        \
      for (i = j = 0; n > 0; n--, i += vs, j += ms, li += ls, v += (ahop)) {   \
        if ((m[j] & mask_log)) {                                               \
          if (strncmp(r, v, len) COND 0) {                                     \
            t_loc = li;                                                        \
            if (!back)                                                         \
              break;                                                           \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    }                                                                          \
    if (t_loc != 0)                                                            \
      *loc = t_loc;                                                            \
  }

/* type list 1 -- sum, product */

#define TYPELIST1(NAME)                                                        \
  {                                                                            \
    __fort_red_unimplemented,     /*  0 __NONE       no type */                 \
        __fort_red_unimplemented, /*  1 __SHORT      short */                   \
        __fort_red_unimplemented, /*  2 __USHORT     unsigned short */          \
        __fort_red_unimplemented, /*  3 __CINT       int */                     \
        __fort_red_unimplemented, /*  4 __UINT       unsigned int */            \
        __fort_red_unimplemented, /*  5 __LONG       long */                    \
        __fort_red_unimplemented, /*  6 __ULONG      unsigned long */           \
        __fort_red_unimplemented, /*  7 __FLOAT      float */                   \
        __fort_red_unimplemented, /*  8 __DOUBLE     double */                  \
        NAME##cplx8,             /*  9 __CPLX8      float complex */           \
        NAME##cplx16,            /* 10 __CPLX16     double complex */          \
        __fort_red_unimplemented, /* 11 __CHAR       char */                    \
        __fort_red_unimplemented, /* 12 __UCHAR      unsigned char */           \
        __fort_red_unimplemented, /* 13 __LONGDOUBLE long double */             \
        __fort_red_unimplemented, /* 14 __STR        string */                  \
        __fort_red_unimplemented, /* 15 __LONGLONG   long long */               \
        __fort_red_unimplemented, /* 16 __ULONGLONG  unsigned long long */      \
        __fort_red_unimplemented, /* 17 __LOG1       logical*1 */               \
        __fort_red_unimplemented, /* 18 __LOG2       logical*2 */               \
        __fort_red_unimplemented, /* 19 __LOG4       logical*4 */               \
        __fort_red_unimplemented, /* 20 __LOG8       logical*8 */               \
        __fort_red_unimplemented, /* 21 __WORD4      typeless */                \
        __fort_red_unimplemented, /* 22 __WORD8      double typeless */         \
        __fort_red_unimplemented, /* 23 __NCHAR      ncharacter - kanji */      \
        NAME##int2,              /* 24 __INT2       integer*2 */               \
        NAME##int4,              /* 25 __INT4       integer*4 */               \
        NAME##int8,              /* 26 __INT8       integer*8 */               \
        NAME##real4,             /* 27 __REAL4      real*4 */                  \
        NAME##real8,             /* 28 __REAL8      real*8 */                  \
        NAME##real16,            /* 29 __REAL16     real*16 */                 \
        NAME##cplx32,            /* 30 __CPLX32     complex*32 */              \
        __fort_red_unimplemented, /* 31 __WORD16     quad typeless */           \
        NAME##int1,              /* 32 __INT1       integer*1 */               \
        __fort_red_unimplemented  /* 33 __DERIVED    derived type */            \
  }

/* type list 1 with logical kind -- sum, product */

#define TYPELIST1LKN(NAME, N)                                                  \
  {                                                                            \
    __fort_red_unimplemented,     /*  0 __NONE       no type */                 \
        __fort_red_unimplemented, /*  1 __SHORT      short */                   \
        __fort_red_unimplemented, /*  2 __USHORT     unsigned short */          \
        __fort_red_unimplemented, /*  3 __CINT       int */                     \
        __fort_red_unimplemented, /*  4 __UINT       unsigned int */            \
        __fort_red_unimplemented, /*  5 __LONG       long */                    \
        __fort_red_unimplemented, /*  6 __ULONG      unsigned long */           \
        __fort_red_unimplemented, /*  7 __FLOAT      float */                   \
        __fort_red_unimplemented, /*  8 __DOUBLE     double */                  \
        NAME##cplx8##l##N,       /*  9 __CPLX8      float complex */           \
        NAME##cplx16##l##N,      /* 10 __CPLX16     double complex */          \
        __fort_red_unimplemented, /* 11 __CHAR       char */                    \
        __fort_red_unimplemented, /* 12 __UCHAR      unsigned char */           \
        __fort_red_unimplemented, /* 13 __LONGDOUBLE long double */             \
        __fort_red_unimplemented, /* 14 __STR        string */                  \
        __fort_red_unimplemented, /* 15 __LONGLONG   long long */               \
        __fort_red_unimplemented, /* 16 __ULONGLONG  unsigned long long */      \
        __fort_red_unimplemented, /* 17 __LOG1       logical*1 */               \
        __fort_red_unimplemented, /* 18 __LOG2       logical*2 */               \
        __fort_red_unimplemented, /* 19 __LOG4       logical*4 */               \
        __fort_red_unimplemented, /* 20 __LOG8       logical*8 */               \
        __fort_red_unimplemented, /* 21 __WORD4      typeless */                \
        __fort_red_unimplemented, /* 22 __WORD8      double typeless */         \
        __fort_red_unimplemented, /* 23 __NCHAR      ncharacter - kanji */      \
        NAME##int2##l##N,        /* 24 __INT2       integer*2 */               \
        NAME##int4##l##N,        /* 25 __INT4       integer*4 */               \
        NAME##int8##l##N,        /* 26 __INT8       integer*8 */               \
        NAME##real4##l##N,       /* 27 __REAL4      real*4 */                  \
        NAME##real8##l##N,       /* 28 __REAL8      real*8 */                  \
        NAME##real16##l##N,      /* 29 __REAL16     real*16 */                 \
        NAME##cplx32##l##N,      /* 30 __CPLX32     complex*32 */              \
        __fort_red_unimplemented, /* 31 __WORD16     quad typeless */           \
        NAME##int1##l##N,        /* 32 __INT1       integer*1 */               \
        __fort_red_unimplemented  /* 33 __DERIVED    derived type */            \
  }

/* type list 1 for all logical kind -- for sum, product */

#define TYPELIST1LK(NAME)                                                                                        \
  {                                                                                                              \
    TYPELIST1LKN(NAME, 1), \
    TYPELIST1LKN(NAME, 2), \
    TYPELIST1LKN(NAME, 4), \
    TYPELIST1LKN(NAME, 8) \
  }

/* type list 2 -- iall, iany, iparity, all, any, parity, count */

#define TYPELIST2(NAME)                                                        \
  {                                                                            \
    __fort_red_unimplemented,     /*  0 __NONE       no type */                 \
        __fort_red_unimplemented, /*  1 __SHORT      short */                   \
        __fort_red_unimplemented, /*  2 __USHORT     unsigned short */          \
        __fort_red_unimplemented, /*  3 __CINT       int */                     \
        __fort_red_unimplemented, /*  4 __UINT       unsigned int */            \
        __fort_red_unimplemented, /*  5 __LONG       long */                    \
        __fort_red_unimplemented, /*  6 __ULONG      unsigned long */           \
        __fort_red_unimplemented, /*  7 __FLOAT      float */                   \
        __fort_red_unimplemented, /*  8 __DOUBLE     double */                  \
        __fort_red_unimplemented, /*  9 __CPLX8      float complex */           \
        __fort_red_unimplemented, /* 10 __CPLX16     double complex */          \
        __fort_red_unimplemented, /* 11 __CHAR       char */                    \
        __fort_red_unimplemented, /* 12 __UCHAR      unsigned char */           \
        __fort_red_unimplemented, /* 13 __LONGDOUBLE long double */             \
        __fort_red_unimplemented, /* 14 __STR        string */                  \
        __fort_red_unimplemented, /* 15 __LONGLONG   long long */               \
        __fort_red_unimplemented, /* 16 __ULONGLONG  unsigned long long */      \
        NAME##log1,              /* 17 __LOG1       logical*1 */               \
        NAME##log2,              /* 18 __LOG2       logical*2 */               \
        NAME##log4,              /* 19 __LOG4       logical*4 */               \
        NAME##log8,              /* 20 __LOG8       logical*8 */               \
        __fort_red_unimplemented, /* 21 __WORD4      typeless */                \
        __fort_red_unimplemented, /* 22 __WORD8      double typeless */         \
        __fort_red_unimplemented, /* 23 __NCHAR      ncharacter - kanji */      \
        NAME##int2,              /* 24 __INT2       integer*2 */               \
        NAME##int4,              /* 25 __INT4       integer*4 */               \
        NAME##int8,              /* 26 __INT8       integer*8 */               \
        __fort_red_unimplemented, /* 27 __REAL4      real*4 */                  \
        __fort_red_unimplemented, /* 28 __REAL8      real*8 */                  \
        __fort_red_unimplemented, /* 29 __REAL16     real*16 */                 \
        __fort_red_unimplemented, /* 30 __CPLX32     complex*32 */              \
        __fort_red_unimplemented, /* 31 __WORD16     quad typeless */           \
        NAME##int1,              /* 32 __INT1       integer*1 */               \
        __fort_red_unimplemented  /* 33 __DERIVED    derived type */            \
  }

/* type list 2 with logical kind mask -- iall, iany, iparity, all, any, parity,
 * count */

#define TYPELIST2LKN(NAME, N)                                                  \
  {                                                                            \
    __fort_red_unimplemented,     /*  0 __NONE       no type */                 \
        __fort_red_unimplemented, /*  1 __SHORT      short */                   \
        __fort_red_unimplemented, /*  2 __USHORT     unsigned short */          \
        __fort_red_unimplemented, /*  3 __CINT       int */                     \
        __fort_red_unimplemented, /*  4 __UINT       unsigned int */            \
        __fort_red_unimplemented, /*  5 __LONG       long */                    \
        __fort_red_unimplemented, /*  6 __ULONG      unsigned long */           \
        __fort_red_unimplemented, /*  7 __FLOAT      float */                   \
        __fort_red_unimplemented, /*  8 __DOUBLE     double */                  \
        __fort_red_unimplemented, /*  9 __CPLX8      float complex */           \
        __fort_red_unimplemented, /* 10 __CPLX16     double complex */          \
        __fort_red_unimplemented, /* 11 __CHAR       char */                    \
        __fort_red_unimplemented, /* 12 __UCHAR      unsigned char */           \
        __fort_red_unimplemented, /* 13 __LONGDOUBLE long double */             \
        __fort_red_unimplemented, /* 14 __STR        string */                  \
        __fort_red_unimplemented, /* 15 __LONGLONG   long long */               \
        __fort_red_unimplemented, /* 16 __ULONGLONG  unsigned long long */      \
        NAME##log1##l##N,        /* 17 __LOG1       logical*1 */               \
        NAME##log2##l##N,        /* 18 __LOG2       logical*2 */               \
        NAME##log4##l##N,        /* 19 __LOG4       logical*4 */               \
        NAME##log8##l##N,        /* 20 __LOG8       logical*8 */               \
        __fort_red_unimplemented, /* 21 __WORD4      typeless */                \
        __fort_red_unimplemented, /* 22 __WORD8      double typeless */         \
        __fort_red_unimplemented, /* 23 __NCHAR      ncharacter - kanji */      \
        NAME##int2##l##N,        /* 24 __INT2       integer*2 */               \
        NAME##int4##l##N,        /* 25 __INT4       integer*4 */               \
        NAME##int8##l##N,        /* 26 __INT8       integer*8 */               \
        __fort_red_unimplemented, /* 27 __REAL4      real*4 */                  \
        __fort_red_unimplemented, /* 28 __REAL8      real*8 */                  \
        __fort_red_unimplemented, /* 29 __REAL16     real*16 */                 \
        __fort_red_unimplemented, /* 30 __CPLX32     complex*32 */              \
        __fort_red_unimplemented, /* 31 __WORD16     quad typeless */           \
        NAME##int1##l##N,        /* 32 __INT1       integer*1 */               \
        __fort_red_unimplemented  /* 33 __DERIVED    derived type */            \
  }

/* type list 2 for all logical kind -- for sum, product */

#define TYPELIST2LK(NAME)                                                                                        \
  {                                                                                                              \
    TYPELIST2LKN(NAME, 1), \
    TYPELIST2LKN(NAME, 2), \
    TYPELIST2LKN(NAME, 4), \
    TYPELIST2LKN(NAME, 8) \
  }

/* type list 3 -- for maxval, minval, maxloc, minloc */

#define TYPELIST3(NAME)                                                        \
  {                                                                            \
    __fort_red_unimplemented,     /*  0 __NONE       no type */                 \
        __fort_red_unimplemented, /*  1 __SHORT      short */                   \
        __fort_red_unimplemented, /*  2 __USHORT     unsigned short */          \
        __fort_red_unimplemented, /*  3 __CINT       int */                     \
        __fort_red_unimplemented, /*  4 __UINT       unsigned int */            \
        __fort_red_unimplemented, /*  5 __LONG       long */                    \
        __fort_red_unimplemented, /*  6 __ULONG      unsigned long */           \
        __fort_red_unimplemented, /*  7 __FLOAT      float */                   \
        __fort_red_unimplemented, /*  8 __DOUBLE     double */                  \
        __fort_red_unimplemented, /*  9 __CPLX8      float complex */           \
        __fort_red_unimplemented, /* 10 __CPLX16     double complex */          \
        __fort_red_unimplemented, /* 11 __CHAR       char */                    \
        __fort_red_unimplemented, /* 12 __UCHAR      unsigned char */           \
        __fort_red_unimplemented, /* 13 __LONGDOUBLE long double */             \
        NAME##str,               /* 14 __STR        string */                  \
        __fort_red_unimplemented, /* 15 __LONGLONG   long long */               \
        __fort_red_unimplemented, /* 16 __ULONGLONG  unsigned long long */      \
        __fort_red_unimplemented, /* 17 __LOG1       logical*1 */               \
        __fort_red_unimplemented, /* 18 __LOG2       logical*2 */               \
        __fort_red_unimplemented, /* 19 __LOG4       logical*4 */               \
        __fort_red_unimplemented, /* 20 __LOG8       logical*8 */               \
        __fort_red_unimplemented, /* 21 __WORD4      typeless */                \
        __fort_red_unimplemented, /* 22 __WORD8      double typeless */         \
        __fort_red_unimplemented, /* 23 __NCHAR      ncharacter - kanji */      \
        NAME##int2,              /* 24 __INT2       integer*2 */               \
        NAME##int4,              /* 25 __INT4       integer*4 */               \
        NAME##int8,              /* 26 __INT8       integer*8 */               \
        NAME##real4,             /* 27 __REAL4      real*4 */                  \
        NAME##real8,             /* 28 __REAL8      real*8 */                  \
        NAME##real16,            /* 29 __REAL16     real*16 */                 \
        __fort_red_unimplemented, /* 30 __CPLX32     complex*32 */              \
        __fort_red_unimplemented, /* 31 __WORD16     quad typeless */           \
        NAME##int1,              /* 32 __INT1       integer*1 */               \
        __fort_red_unimplemented  /* 33 __DERIVED    derived type */            \
  }

/* type list 3 with logical kind -- for maxval, minval, maxloc, minloc */

#define TYPELIST3LKN(NAME, N)                                                  \
  {                                                                            \
    __fort_red_unimplemented,     /*  0 __NONE       no type */                 \
        __fort_red_unimplemented, /*  1 __SHORT      short */                   \
        __fort_red_unimplemented, /*  2 __USHORT     unsigned short */          \
        __fort_red_unimplemented, /*  3 __CINT       int */                     \
        __fort_red_unimplemented, /*  4 __UINT       unsigned int */            \
        __fort_red_unimplemented, /*  5 __LONG       long */                    \
        __fort_red_unimplemented, /*  6 __ULONG      unsigned long */           \
        __fort_red_unimplemented, /*  7 __FLOAT      float */                   \
        __fort_red_unimplemented, /*  8 __DOUBLE     double */                  \
        __fort_red_unimplemented, /*  9 __CPLX8      float complex */           \
        __fort_red_unimplemented, /* 10 __CPLX16     double complex */          \
        __fort_red_unimplemented, /* 11 __CHAR       char */                    \
        __fort_red_unimplemented, /* 12 __UCHAR      unsigned char */           \
        __fort_red_unimplemented, /* 13 __LONGDOUBLE long double */             \
        NAME##str##l##N,         /* 14 __STR        string */                  \
        __fort_red_unimplemented, /* 15 __LONGLONG   long long */               \
        __fort_red_unimplemented, /* 16 __ULONGLONG  unsigned long long */      \
        __fort_red_unimplemented, /* 17 __LOG1       logical*1 */               \
        __fort_red_unimplemented, /* 18 __LOG2       logical*2 */               \
        __fort_red_unimplemented, /* 19 __LOG4       logical*4 */               \
        __fort_red_unimplemented, /* 20 __LOG8       logical*8 */               \
        __fort_red_unimplemented, /* 21 __WORD4      typeless */                \
        __fort_red_unimplemented, /* 22 __WORD8      double typeless */         \
        __fort_red_unimplemented, /* 23 __NCHAR      ncharacter - kanji */      \
        NAME##int2##l##N,        /* 24 __INT2       integer*2 */               \
        NAME##int4##l##N,        /* 25 __INT4       integer*4 */               \
        NAME##int8##l##N,        /* 26 __INT8       integer*8 */               \
        NAME##real4##l##N,       /* 27 __REAL4      real*4 */                  \
        NAME##real8##l##N,       /* 28 __REAL8      real*8 */                  \
        NAME##real16##l##N,      /* 29 __REAL16     real*16 */                 \
        __fort_red_unimplemented, /* 30 __CPLX32     complex*32 */              \
        __fort_red_unimplemented, /* 31 __WORD16     quad typeless */           \
        NAME##int1##l##N,        /* 32 __INT1       integer*1 */               \
        __fort_red_unimplemented  /* 33 __DERIVED    derived type */            \
  }

/* type list 3 for all logical kind -- for maxval, minval, maxloc, minloc */

#define TYPELIST3LK(NAME)                                                                                        \
  {                                                                                                              \
    TYPELIST3LKN(NAME, 1), \
    TYPELIST3LKN(NAME, 2), \
    TYPELIST3LKN(NAME, 4), \
    TYPELIST3LKN(NAME, 8) \
  }
