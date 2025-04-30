/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */
/** \file
 * \brief Legacy constant folding API.
 *
 *  This header comprises declarations and definitions from the
 *  original scutil/hammer/linux86-64/include/scutil.h header file
 *  that pertain to the representation of constants as arrays of
 *  32-bit integers, scanning and formatting of numeric literals,
 *  and compile-time evaluation of operations.
 */

#ifndef LEGACY_FOLDING_API_H_
#define LEGACY_FOLDING_API_H_
#ifdef __cplusplus
extern "C" {
#endif

#include "legacy-ints.h"

/*
 *  Constants that are larger than 32 bits are expected to have been
 *  broken into big-endian arrays of 32-bit int chunks, regardless of
 *  the endianness of the host and target.
 */

typedef int32_t IEEE32;		/* IEEE single precision float number */
typedef int32_t IEEE64[2];	/* IEEE double precision float number */
typedef int32_t IEEE80[3];	/* x87 80-bit extended precision float number */
typedef IEEE64 IEEE6464[2];	/* double-double float number (OpenPOWER) */
typedef int32_t IEEE128[4];	/* IEEE quad precision float number */
typedef int32_t INT128[4];
typedef int32_t UINT128[4];

/* Synonyms */
typedef IEEE32  SNGL;
typedef IEEE64  DBLE;
typedef IEEE128 QUAD;

/*
 *  fperror() is called by the library in exceptional conditions.
 *  There's a simple default implementation of fperror() in the library
 *  that is typically overridden by compilers.
 */
#define FPE_NOERR	0		/* Everything OK */
#define FPE_FPOVF	(-2)		/* floating point overflow */
#define FPE_FPUNF	(-3)		/* floating point underflow */
#define FPE_IOVF	(-2)		/* integer overflow (fix/dfix only) */
#define FPE_INVOP	(-1)		/* invalid operand */
#define FPE_DIVZ	(-2)		/* reciprocal of zero */
void fperror(int errcode);

#define ftomf(f,mf) ((mf)= *((float *)&(f)))
#define mftof(mf,f) (*((float *)&(f))=(mf))
void xdtomd(IEEE64 d, double *md);
void xmdtod(double md, IEEE64 d);
void xmqtoq(long double mq, IEEE128 q);

int cmp64(DBLINT64 arg1, DBLINT64 arg2);
int ucmp64(DBLUINT64 arg1, DBLUINT64 arg2);
void add64(DBLINT64 arg1, DBLINT64 arg2, DBLINT64 result);
void div64(DBLINT64 arg1, DBLINT64 arg2, DBLINT64 result);
void exp64(DBLINT64 base, int exp, DBLINT64 result);
void mul64(DBLINT64 arg1, DBLINT64 arg2, DBLINT64 result);
void mul64_10(DBLINT64 arg1, DBLINT64 result);
void neg64(DBLINT64 arg, DBLINT64 result);
void shf64(DBLINT64 arg, int count, DBLINT64 result);
void sub64(DBLINT64 arg1, DBLINT64 arg2, DBLINT64 result);
void uadd64(DBLUINT64 arg1, DBLUINT64 arg2, DBLUINT64 result);
void uneg64(DBLUINT64 arg, DBLUINT64 result);
void ushf64(DBLUINT64 arg, int count, DBLUINT64 result);
void usub64(DBLUINT64 arg1, DBLUINT64 arg2, DBLUINT64 result);
void udiv64(DBLUINT64 arg1, DBLUINT64 arg2, DBLUINT64 result);
void umul64(DBLUINT64 arg1, DBLUINT64 arg2, DBLUINT64 result);
void umul64_10(DBLUINT64 arg1, DBLUINT64 result);
void and64(DBLINT64 arg1, DBLINT64 arg2, DBLINT64 result);
void not64(DBLINT64 arg, DBLINT64 result);
void or64(DBLINT64 arg1, DBLINT64 arg2, DBLINT64 result);
void xor64(DBLINT64 arg1, DBLINT64 arg2, DBLINT64 result);
void mul128(DBLINT64 arg1, DBLINT64 arg2, INT result[4]);
void shf128(INT arg[4], int count, INT result[4]);
void shf128_1(INT arg[4], INT result[4]);
void shf128_m1(INT arg[4], INT result[4]);
void neg128(INT arg[4], INT result[4]);

void add128(INT128, INT128, INT128);
void sub128(INT128, INT128, INT128);
int cmp128(INT128, INT128);
int ucmp128(UINT128, UINT128);
void div128(INT128, INT128, INT128);
void mul256(INT128, INT128, INT []);
void mul128l(INT128, INT128, INT128);
void negN(INT [], INT [], INT);
void shfN(INT [], INT, INT [], INT);
void shfN_1(INT [], INT [], INT);
void shfN_m1(INT [], INT [], INT);

int atoxi(const char *s, INT *i, int n, int base);
int atosi32(const char *s, INT *i, int n, int base);
int atoxf(const char *s, IEEE32 *f, int n);
int atoxd(const char *s, IEEE64 d, int n);
int atoxe(const char *s, IEEE80 e, int n);
int atoxdd(const char *s, IEEE6464 dd, int n);
int atoxq(const char *s, IEEE128 q, int n);
int hxatoxf(const char *s, IEEE32 *f, int n);
int hxatoxd(const char *s, IEEE64 d, int n);
int hxatoxe(const char *s, IEEE80 e, int n);
int hxatoxdd(const char *s, IEEE6464 dd, int n);
int hxatoxq(const char *s, IEEE128 q, int n);
void cprintf(char *s, const char *format, INT *val);
int atoxi64(const char *s, DBLINT64 ir, int n, int radix);
int atosi64(const char *, DBLINT64, int, int);
void ui64toax(DBLINT64 from, char *to, int count, int sign, int radix);

void xqfix64(IEEE128 q, DBLINT64 i);
void xqfixu64(IEEE128 q, DBLUINT64 i);
void xefix64(IEEE80 e, DBLINT64 i);
void xefixu64(IEEE80 e, DBLUINT64 i);
void xddfix64(IEEE6464 dd, DBLINT64 i);
void xddfixu64(IEEE6464 dd, DBLUINT64 i);
void xdfix64(IEEE64 d, DBLINT64 i);
void xdfixu64(IEEE64 d, DBLUINT64 i);
void xfix64(IEEE32 f, DBLINT64 i);
void xfixu64(IEEE32 f, DBLUINT64 i);
void xfixu(IEEE32 f, UINT *r);

void xqflt64(DBLINT64 i, IEEE128 q);
void xqfltu64(DBLUINT64 i, IEEE128 q);
void xeflt64(DBLINT64 i, IEEE80 e);
void xefltu64(DBLUINT64 i, IEEE80 e);
void xddflt64(DBLINT64 i, IEEE6464 dd);
void xddfltu64(DBLUINT64 i, IEEE6464 dd);
void xdflt64(DBLINT64 i, IEEE64 d);
void xdfltu64(DBLUINT64 i, IEEE64 d);
void xflt64(DBLINT64 i, IEEE32 *f);
void xfltu64(DBLUINT64 i, IEEE32 *f);

void xfadd(IEEE32 f1, IEEE32 f2, IEEE32 *r);
void xdadd(IEEE64 d1, IEEE64 d2, IEEE64 r);
void xeadd(IEEE80 e1, IEEE80 e2, IEEE80 r);
void xddadd(IEEE6464 dd1, IEEE6464 dd2, IEEE6464 r);
void xqadd(IEEE128 q1, IEEE128 q2, IEEE128 r);
void xfsub(IEEE32 f1, IEEE32 f2, IEEE32 *r);
void xdsub(IEEE64 d1, IEEE64 d2, IEEE64 r);
void xesub(IEEE80 e1, IEEE80 e2, IEEE80 r);
void xddsub(IEEE6464 dd1, IEEE6464 dd2, IEEE6464 r);
void xqsub(IEEE128 q1, IEEE128 q2, IEEE128 r);
int xdisint(IEEE64 d, int *r);
int xqisint(IEEE128 q, int *r);
void xfneg(IEEE32 f1, IEEE32 *r);
void xdneg(IEEE64 d1, IEEE64 r);
void xeneg(IEEE80 e1, IEEE80 r);
void xddneg(IEEE6464 dd1, IEEE6464 r);
void xqneg(IEEE128 q1, IEEE128 r);
void xfmul(IEEE32 f1, IEEE32 f2, IEEE32 *r);
void xdmul(IEEE64 d1, IEEE64 d2, IEEE64 r);
void xemul(IEEE80 e1, IEEE80 e2, IEEE80 r);
void xddmul(IEEE6464 dd1, IEEE6464 dd2, IEEE6464 r);
void xqmul(IEEE128 q1, IEEE128 q2, IEEE128 r);
void xfdiv(IEEE32 f1, IEEE32 f2, IEEE32 *r);
void xddiv(IEEE64 d1, IEEE64 d2, IEEE64 r);
void xediv(IEEE80 e1, IEEE80 e2, IEEE80 r);
void xdddiv(IEEE6464 dd1, IEEE6464 dd2, IEEE6464 r);
void xqdiv(IEEE128 q1, IEEE128 q2, IEEE128 r);
void xdrcp(IEEE64 d, IEEE64 recip);
void xfrcp(IEEE32 f, IEEE32 *recip);
void xfabsv(IEEE32 f, IEEE32 *r);
int xfisint(IEEE32 f, int *r);
void xfsqrt(IEEE32 f, IEEE32 *r);
void xdabsv(IEEE64 f, IEEE64 r);
void xeabsv(IEEE80 e, IEEE80 r);
void xddabsv(IEEE6464 dd, IEEE6464 r);
void xqabsv(IEEE128 f, IEEE128 r);
void xdsqrt(IEEE64 f, IEEE64 r);
void xqsqrt(IEEE128 f, IEEE128 r);
void xfpow(IEEE32 f1, IEEE32 f2, IEEE32 *r);
void xdpow(IEEE64 d1, IEEE64 d2, IEEE64 r);
void xepow(IEEE80 e1, IEEE80 e2, IEEE80 r);
void xddpow(IEEE6464 dd1, IEEE6464 dd2, IEEE6464 r);
void xqpow(IEEE128 q1, IEEE128 q2, IEEE128 r);
void xfsin(IEEE32 , IEEE32 *);
void xdsin(IEEE64 , IEEE64  );
void xesin(IEEE80 , IEEE80  );
void xddsin(IEEE6464, IEEE6464);
void xqsin(IEEE128, IEEE128 );
void xfcos(IEEE32 , IEEE32 *);
void xdcos(IEEE64 , IEEE64  );
void xecos(IEEE80 , IEEE80  );
void xddcos(IEEE6464, IEEE6464);
void xqcos(IEEE128, IEEE128 );
void xftan(IEEE32 , IEEE32 *);
void xdtan(IEEE64 , IEEE64  );
void xetan(IEEE80 , IEEE80  );
void xddtan(IEEE6464, IEEE6464);
void xqtan(IEEE128, IEEE128 );
void xfasin (IEEE32 , IEEE32 *);
void xdasin (IEEE64 , IEEE64  );
void xeasin (IEEE80 , IEEE80  );
void xddasin(IEEE6464, IEEE6464);
void xqasin (IEEE128, IEEE128 );
void xfacos (IEEE32 , IEEE32 *);
void xdacos (IEEE64 , IEEE64  );
void xeacos (IEEE80 , IEEE80  );
void xddacos(IEEE6464, IEEE6464);
void xqacos (IEEE128, IEEE128 );
void xfatan (IEEE32 , IEEE32 *);
void xdatan (IEEE64 , IEEE64  );
void xeatan (IEEE80 , IEEE80  );
void xddatan(IEEE6464, IEEE6464);
void xqatan (IEEE128, IEEE128 );
void xfatan2(IEEE32 , IEEE32 , IEEE32  *);
void xdatan2(IEEE64 , IEEE64 , IEEE64   );
void xeatan2(IEEE80 , IEEE80 , IEEE80   );
void xddatan2(IEEE6464, IEEE6464, IEEE6464);
void xqatan2(IEEE128, IEEE128, IEEE128  );
void xfexp   (IEEE32 , IEEE32 *);
void xdexp   (IEEE64 , IEEE64  );
void xeexp   (IEEE80 , IEEE80  );
void xddexp(IEEE6464, IEEE6464);
void xqexp   (IEEE128, IEEE128 );
void xflog   (IEEE32 , IEEE32 *);
void xdlog   (IEEE64 , IEEE64  );
void xelog   (IEEE80 , IEEE80  );
void xddlog(IEEE6464, IEEE6464);
void xqlog   (IEEE128, IEEE128 );
void xflog10 (IEEE32 , IEEE32 *);
void xdlog10 (IEEE64 , IEEE64  );
void xelog10 (IEEE80 , IEEE80  );
void xddlog10(IEEE6464, IEEE6464);
void xqlog10 (IEEE128, IEEE128 );

void xffloat(INT i, IEEE32 *f);
void xdfloat(INT i, IEEE64 d);
void xefloat(INT i, IEEE80 e);
void xddfloat(INT i, IEEE6464 dd);
void xqfloat(INT i, IEEE128 q);
void xfix(IEEE32 f, INT *i);
void xdfix(IEEE64 d, INT *i);
void xefix(IEEE80 e, INT *i);
void xddfix(IEEE6464 dd, INT *i);
void xqfix(IEEE128 q, INT *i);

void xdble(IEEE32 f, IEEE64 r);
void xsngl(IEEE64 d, IEEE32 *r);
void xdtoe(IEEE64 d, IEEE80 r);
void xdtodd(IEEE64 d, IEEE6464 r);
void xdtoq(IEEE64 d, IEEE128 r);
void xftoe(IEEE32 d, IEEE80 r);
void xftodd(IEEE32 d, IEEE6464 r);
void xftoq(IEEE32 d, IEEE128 r);
void xetof(IEEE80 e, IEEE32 *r);
void xetod(IEEE80 e, IEEE64 r);
void xetoq(IEEE80 e, IEEE128 r);
void xddtof(IEEE6464 dd, IEEE32 *r);
void xddtod(IEEE6464 dd, IEEE64 r);
void xddtoq(IEEE6464 dd, IEEE128 r);
void xqtodd(IEEE128 q, IEEE6464 r);
void xqtoe(IEEE128 q, IEEE80 r);
void xqtod(IEEE128 q, IEEE64 r);
void xqtof(IEEE128 q, IEEE32 *);

int xudiv(UINT n, UINT d, UINT *r);
int xumod(UINT n, UINT d, UINT *r);

void xqfixu(IEEE128 q, UINT *r);
void xqfloatu(UINT i, IEEE128 r);
void xddfixu(IEEE6464 dd, UINT *r);
void xddfloatu(UINT i, IEEE6464 r);
void xefixu(IEEE80 e, UINT *r);
void xefloatu(UINT i, IEEE80 r);
void xdfixu(IEEE64 d, UINT *r);
void xdfloatu(UINT i, IEEE64 r);
void xffixu(IEEE32 f, UINT *r);
void xffloatu(UINT i, IEEE32 *r);

int xqcmp(IEEE128 q1, IEEE128 q2);
int xddcmp(IEEE6464 dd1, IEEE6464 dd2);
int xecmp(IEEE80 e1, IEEE80 e2);
int xdcmp(IEEE64 d1, IEEE64 d2);
int xfcmp(IEEE32 f1, IEEE32 f2);
int xucmp(INT a, INT b);

void xcfpow(IEEE32 r1, IEEE32 i1, IEEE32 r2, IEEE32 i2, IEEE32 *rr, IEEE32 *ir);
void xcdpow(IEEE64 r1, IEEE64 i1, IEEE64 r2, IEEE64 i2, IEEE64 rr, IEEE64 ir);
void xcqpow(IEEE128 r1, IEEE128 i1, IEEE128 r2, IEEE128 i2, IEEE128 rr, IEEE128 ir);

#ifdef __cplusplus
}
#endif
#endif /* LEGACY_FOLDING_API_H_ */
