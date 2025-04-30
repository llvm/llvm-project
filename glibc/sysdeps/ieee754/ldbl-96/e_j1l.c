/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */

/* Long double expansions are
  Copyright (C) 2001 Stephen L. Moshier <moshier@na-net.ornl.gov>
  and are incorporated herein by permission of the author.  The author
  reserves the right to distribute this material elsewhere under different
  copying permissions.  These modifications are distributed here under
  the following terms:

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2.1 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, see
    <https://www.gnu.org/licenses/>.  */

/* __ieee754_j1(x), __ieee754_y1(x)
 * Bessel function of the first and second kinds of order zero.
 * Method -- j1(x):
 *	1. For tiny x, we use j1(x) = x/2 - x^3/16 + x^5/384 - ...
 *	2. Reduce x to |x| since j1(x)=-j1(-x),  and
 *	   for x in (0,2)
 *		j1(x) = x/2 + x*z*R0/S0,  where z = x*x;
 *	   for x in (2,inf)
 *		j1(x) = sqrt(2/(pi*x))*(p1(x)*cos(x1)-q1(x)*sin(x1))
 *		y1(x) = sqrt(2/(pi*x))*(p1(x)*sin(x1)+q1(x)*cos(x1))
 *	   where x1 = x-3*pi/4. It is better to compute sin(x1),cos(x1)
 *	   as follow:
 *		cos(x1) =  cos(x)cos(3pi/4)+sin(x)sin(3pi/4)
 *			=  1/sqrt(2) * (sin(x) - cos(x))
 *		sin(x1) =  sin(x)cos(3pi/4)-cos(x)sin(3pi/4)
 *			= -1/sqrt(2) * (sin(x) + cos(x))
 *	   (To avoid cancellation, use
 *		sin(x) +- cos(x) = -cos(2x)/(sin(x) -+ cos(x))
 *	    to compute the worse one.)
 *
 *	3 Special cases
 *		j1(nan)= nan
 *		j1(0) = 0
 *		j1(inf) = 0
 *
 * Method -- y1(x):
 *	1. screen out x<=0 cases: y1(0)=-inf, y1(x<0)=NaN
 *	2. For x<2.
 *	   Since
 *		y1(x) = 2/pi*(j1(x)*(ln(x/2)+Euler)-1/x-x/2+5/64*x^3-...)
 *	   therefore y1(x)-2/pi*j1(x)*ln(x)-1/x is an odd function.
 *	   We use the following function to approximate y1,
 *		y1(x) = x*U(z)/V(z) + (2/pi)*(j1(x)*ln(x)-1/x), z= x^2
 *	   Note: For tiny x, 1/x dominate y1 and hence
 *		y1(tiny) = -2/pi/tiny
 *	3. For x>=2.
 *		y1(x) = sqrt(2/(pi*x))*(p1(x)*sin(x1)+q1(x)*cos(x1))
 *	   where x1 = x-3*pi/4. It is better to compute sin(x1),cos(x1)
 *	   by method mentioned above.
 */

#include <errno.h>
#include <float.h>
#include <math.h>
#include <math_private.h>
#include <math-underflow.h>
#include <libm-alias-finite.h>

static long double pone (long double), qone (long double);

static const long double
  huge = 1e4930L,
 one = 1.0L,
 invsqrtpi = 5.6418958354775628694807945156077258584405e-1L,
  tpi =  6.3661977236758134307553505349005744813784e-1L,

  /* J1(x) = .5 x + x x^2 R(x^2) / S(x^2)
     0 <= x <= 2
     Peak relative error 4.5e-21 */
R[5] = {
    -9.647406112428107954753770469290757756814E7L,
    2.686288565865230690166454005558203955564E6L,
    -3.689682683905671185891885948692283776081E4L,
    2.195031194229176602851429567792676658146E2L,
    -5.124499848728030297902028238597308971319E-1L,
},

  S[4] =
{
  1.543584977988497274437410333029029035089E9L,
  2.133542369567701244002565983150952549520E7L,
  1.394077011298227346483732156167414670520E5L,
  5.252401789085732428842871556112108446506E2L,
  /*  1.000000000000000000000000000000000000000E0L, */
};

static const long double zero = 0.0;


long double
__ieee754_j1l (long double x)
{
  long double z, c, r, s, ss, cc, u, v, y;
  int32_t ix;
  uint32_t se;

  GET_LDOUBLE_EXP (se, x);
  ix = se & 0x7fff;
  if (__glibc_unlikely (ix >= 0x7fff))
    return one / x;
  y = fabsl (x);
  if (ix >= 0x4000)
    {				/* |x| >= 2.0 */
      __sincosl (y, &s, &c);
      ss = -s - c;
      cc = s - c;
      if (ix < 0x7ffe)
	{			/* make sure y+y not overflow */
	  z = __cosl (y + y);
	  if ((s * c) > zero)
	    cc = z / ss;
	  else
	    ss = z / cc;
	}
      /*
       * j1(x) = 1/sqrt(pi) * (P(1,x)*cc - Q(1,x)*ss) / sqrt(x)
       * y1(x) = 1/sqrt(pi) * (P(1,x)*ss + Q(1,x)*cc) / sqrt(x)
       */
      if (__glibc_unlikely (ix > 0x408e))
	z = (invsqrtpi * cc) / sqrtl (y);
      else
	{
	  u = pone (y);
	  v = qone (y);
	  z = invsqrtpi * (u * cc - v * ss) / sqrtl (y);
	}
      if (se & 0x8000)
	return -z;
      else
	return z;
    }
  if (__glibc_unlikely (ix < 0x3fde))       /* |x| < 2^-33 */
    {
      if (huge + x > one)		/* inexact if x!=0 necessary */
	{
	  long double ret = 0.5 * x;
	  math_check_force_underflow (ret);
	  if (ret == 0 && x != 0)
	    __set_errno (ERANGE);
	  return ret;
	}
    }
  z = x * x;
  r = z * (R[0] + z * (R[1]+ z * (R[2] + z * (R[3] + z * R[4]))));
  s = S[0] + z * (S[1] + z * (S[2] + z * (S[3] + z)));
  r *= x;
  return (x * 0.5 + r / s);
}
libm_alias_finite (__ieee754_j1l, __j1l)


/* Y1(x) = 2/pi * (log(x) * j1(x) - 1/x) + x R(x^2)
   0 <= x <= 2
   Peak relative error 2.3e-23 */
static const long double U0[6] = {
  -5.908077186259914699178903164682444848615E10L,
  1.546219327181478013495975514375773435962E10L,
  -6.438303331169223128870035584107053228235E8L,
  9.708540045657182600665968063824819371216E6L,
  -6.138043997084355564619377183564196265471E4L,
  1.418503228220927321096904291501161800215E2L,
};
static const long double V0[5] = {
  3.013447341682896694781964795373783679861E11L,
  4.669546565705981649470005402243136124523E9L,
  3.595056091631351184676890179233695857260E7L,
  1.761554028569108722903944659933744317994E5L,
  5.668480419646516568875555062047234534863E2L,
  /*  1.000000000000000000000000000000000000000E0L, */
};


long double
__ieee754_y1l (long double x)
{
  long double z, s, c, ss, cc, u, v;
  int32_t ix;
  uint32_t se, i0, i1;

  GET_LDOUBLE_WORDS (se, i0, i1, x);
  ix = se & 0x7fff;
  /* if Y1(NaN) is NaN, Y1(-inf) is NaN, Y1(inf) is 0 */
  if (__glibc_unlikely (se & 0x8000))
    return zero / (zero * x);
  if (__glibc_unlikely (ix >= 0x7fff))
    return one / (x + x * x);
  if (__glibc_unlikely ((i0 | i1) == 0))
    return -HUGE_VALL + x;  /* -inf and overflow exception.  */
  if (ix >= 0x4000)
    {				/* |x| >= 2.0 */
      __sincosl (x, &s, &c);
      ss = -s - c;
      cc = s - c;
      if (ix < 0x7ffe)
	{			/* make sure x+x not overflow */
	  z = __cosl (x + x);
	  if ((s * c) > zero)
	    cc = z / ss;
	  else
	    ss = z / cc;
	}
      /* y1(x) = sqrt(2/(pi*x))*(p1(x)*sin(x0)+q1(x)*cos(x0))
       * where x0 = x-3pi/4
       *      Better formula:
       *              cos(x0) = cos(x)cos(3pi/4)+sin(x)sin(3pi/4)
       *                      =  1/sqrt(2) * (sin(x) - cos(x))
       *              sin(x0) = sin(x)cos(3pi/4)-cos(x)sin(3pi/4)
       *                      = -1/sqrt(2) * (cos(x) + sin(x))
       * To avoid cancellation, use
       *              sin(x) +- cos(x) = -cos(2x)/(sin(x) -+ cos(x))
       * to compute the worse one.
       */
      if (__glibc_unlikely (ix > 0x408e))
	z = (invsqrtpi * ss) / sqrtl (x);
      else
	{
	  u = pone (x);
	  v = qone (x);
	  z = invsqrtpi * (u * ss + v * cc) / sqrtl (x);
	}
      return z;
    }
  if (__glibc_unlikely (ix <= 0x3fbe))
    {				/* x < 2**-65 */
      z = -tpi / x;
      if (isinf (z))
	__set_errno (ERANGE);
      return z;
    }
  z = x * x;
 u = U0[0] + z * (U0[1] + z * (U0[2] + z * (U0[3] + z * (U0[4] + z * U0[5]))));
 v = V0[0] + z * (V0[1] + z * (V0[2] + z * (V0[3] + z * (V0[4] + z))));
  return (x * (u / v) +
	  tpi * (__ieee754_j1l (x) * __ieee754_logl (x) - one / x));
}
libm_alias_finite (__ieee754_y1l, __y1l)


/* For x >= 8, the asymptotic expansions of pone is
 *	1 + 15/128 s^2 - 4725/2^15 s^4 - ...,	where s = 1/x.
 * We approximate pone by
 *	pone(x) = 1 + (R/S)
 */

/* J1(x) cosX + Y1(x) sinX  =  sqrt( 2/(pi x)) P1(x)
   P1(x) = 1 + z^2 R(z^2), z=1/x
   8 <= x <= inf  (0 <= z <= 0.125)
   Peak relative error 5.2e-22  */

static const long double pr8[7] = {
  8.402048819032978959298664869941375143163E-9L,
  1.813743245316438056192649247507255996036E-6L,
  1.260704554112906152344932388588243836276E-4L,
  3.439294839869103014614229832700986965110E-3L,
  3.576910849712074184504430254290179501209E-2L,
  1.131111483254318243139953003461511308672E-1L,
  4.480715825681029711521286449131671880953E-2L,
};
static const long double ps8[6] = {
  7.169748325574809484893888315707824924354E-8L,
  1.556549720596672576431813934184403614817E-5L,
  1.094540125521337139209062035774174565882E-3L,
  3.060978962596642798560894375281428805840E-2L,
  3.374146536087205506032643098619414507024E-1L,
  1.253830208588979001991901126393231302559E0L,
  /* 1.000000000000000000000000000000000000000E0L, */
};

/* J1(x) cosX + Y1(x) sinX  =  sqrt( 2/(pi x)) P1(x)
   P1(x) = 1 + z^2 R(z^2), z=1/x
   4.54541015625 <= x <= 8
   Peak relative error 7.7e-22  */
static const long double pr5[7] = {
  4.318486887948814529950980396300969247900E-7L,
  4.715341880798817230333360497524173929315E-5L,
  1.642719430496086618401091544113220340094E-3L,
  2.228688005300803935928733750456396149104E-2L,
  1.142773760804150921573259605730018327162E-1L,
  1.755576530055079253910829652698703791957E-1L,
  3.218803858282095929559165965353784980613E-2L,
};
static const long double ps5[6] = {
  3.685108812227721334719884358034713967557E-6L,
  4.069102509511177498808856515005792027639E-4L,
  1.449728676496155025507893322405597039816E-2L,
  2.058869213229520086582695850441194363103E-1L,
  1.164890985918737148968424972072751066553E0L,
  2.274776933457009446573027260373361586841E0L,
  /*  1.000000000000000000000000000000000000000E0L,*/
};

/* J1(x) cosX + Y1(x) sinX  =  sqrt( 2/(pi x)) P1(x)
   P1(x) = 1 + z^2 R(z^2), z=1/x
   2.85711669921875 <= x <= 4.54541015625
   Peak relative error 6.5e-21  */
static const long double pr3[7] = {
  1.265251153957366716825382654273326407972E-5L,
  8.031057269201324914127680782288352574567E-4L,
  1.581648121115028333661412169396282881035E-2L,
  1.179534658087796321928362981518645033967E-1L,
  3.227936912780465219246440724502790727866E-1L,
  2.559223765418386621748404398017602935764E-1L,
  2.277136933287817911091370397134882441046E-2L,
};
static const long double ps3[6] = {
  1.079681071833391818661952793568345057548E-4L,
  6.986017817100477138417481463810841529026E-3L,
  1.429403701146942509913198539100230540503E-1L,
  1.148392024337075609460312658938700765074E0L,
  3.643663015091248720208251490291968840882E0L,
  3.990702269032018282145100741746633960737E0L,
  /* 1.000000000000000000000000000000000000000E0L, */
};

/* J1(x) cosX + Y1(x) sinX  =  sqrt( 2/(pi x)) P1(x)
   P1(x) = 1 + z^2 R(z^2), z=1/x
   2 <= x <= 2.85711669921875
   Peak relative error 3.5e-21  */
static const long double pr2[7] = {
  2.795623248568412225239401141338714516445E-4L,
  1.092578168441856711925254839815430061135E-2L,
  1.278024620468953761154963591853679640560E-1L,
  5.469680473691500673112904286228351988583E-1L,
  8.313769490922351300461498619045639016059E-1L,
  3.544176317308370086415403567097130611468E-1L,
  1.604142674802373041247957048801599740644E-2L,
};
static const long double ps2[6] = {
  2.385605161555183386205027000675875235980E-3L,
  9.616778294482695283928617708206967248579E-2L,
  1.195215570959693572089824415393951258510E0L,
  5.718412857897054829999458736064922974662E0L,
  1.065626298505499086386584642761602177568E1L,
  6.809140730053382188468983548092322151791E0L,
 /* 1.000000000000000000000000000000000000000E0L, */
};


static long double
pone (long double x)
{
  const long double *p, *q;
  long double z, r, s;
  int32_t ix;
  uint32_t se, i0, i1;

  GET_LDOUBLE_WORDS (se, i0, i1, x);
  ix = se & 0x7fff;
  /* ix >= 0x4000 for all calls to this function.  */
  if (ix >= 0x4002) /* x >= 8 */
    {
      p = pr8;
      q = ps8;
    }
  else
    {
      i1 = (ix << 16) | (i0 >> 16);
      if (i1 >= 0x40019174)	/* x >= 4.54541015625 */
	{
	  p = pr5;
	  q = ps5;
	}
      else if (i1 >= 0x4000b6db)	/* x >= 2.85711669921875 */
	{
	  p = pr3;
	  q = ps3;
	}
      else	/* x >= 2 */
	{
	  p = pr2;
	  q = ps2;
	}
    }
  z = one / (x * x);
 r = p[0] + z * (p[1] +
		 z * (p[2] + z * (p[3] + z * (p[4] + z * (p[5] + z * p[6])))));
 s = q[0] + z * (q[1] + z * (q[2] + z * (q[3] + z * (q[4] + z * (q[5] + z)))));
  return one + z * r / s;
}


/* For x >= 8, the asymptotic expansions of qone is
 *	3/8 s - 105/1024 s^3 - ..., where s = 1/x.
 * We approximate pone by
 *	qone(x) = s*(0.375 + (R/S))
 */

/* Y1(x)cosX - J1(x)sinX = sqrt( 2/(pi x)) Q1(x),
   Q1(x) = z(.375 + z^2 R(z^2)), z=1/x
   8 <= x <= inf
   Peak relative error 8.3e-22 */

static const long double qr8[7] = {
  -5.691925079044209246015366919809404457380E-10L,
  -1.632587664706999307871963065396218379137E-7L,
  -1.577424682764651970003637263552027114600E-5L,
  -6.377627959241053914770158336842725291713E-4L,
  -1.087408516779972735197277149494929568768E-2L,
  -6.854943629378084419631926076882330494217E-2L,
  -1.055448290469180032312893377152490183203E-1L,
};
static const long double qs8[7] = {
  5.550982172325019811119223916998393907513E-9L,
  1.607188366646736068460131091130644192244E-6L,
  1.580792530091386496626494138334505893599E-4L,
  6.617859900815747303032860443855006056595E-3L,
  1.212840547336984859952597488863037659161E-1L,
  9.017885953937234900458186716154005541075E-1L,
  2.201114489712243262000939120146436167178E0L,
  /* 1.000000000000000000000000000000000000000E0L, */
};

/* Y1(x)cosX - J1(x)sinX = sqrt( 2/(pi x)) Q1(x),
   Q1(x) = z(.375 + z^2 R(z^2)), z=1/x
   4.54541015625 <= x <= 8
   Peak relative error 4.1e-22 */
static const long double qr5[7] = {
  -6.719134139179190546324213696633564965983E-8L,
  -9.467871458774950479909851595678622044140E-6L,
  -4.429341875348286176950914275723051452838E-4L,
  -8.539898021757342531563866270278505014487E-3L,
  -6.818691805848737010422337101409276287170E-2L,
  -1.964432669771684034858848142418228214855E-1L,
  -1.333896496989238600119596538299938520726E-1L,
};
static const long double qs5[7] = {
  6.552755584474634766937589285426911075101E-7L,
  9.410814032118155978663509073200494000589E-5L,
  4.561677087286518359461609153655021253238E-3L,
  9.397742096177905170800336715661091535805E-2L,
  8.518538116671013902180962914473967738771E-1L,
  3.177729183645800174212539541058292579009E0L,
  4.006745668510308096259753538973038902990E0L,
  /* 1.000000000000000000000000000000000000000E0L, */
};

/* Y1(x)cosX - J1(x)sinX = sqrt( 2/(pi x)) Q1(x),
   Q1(x) = z(.375 + z^2 R(z^2)), z=1/x
   2.85711669921875 <= x <= 4.54541015625
   Peak relative error 2.2e-21 */
static const long double qr3[7] = {
  -3.618746299358445926506719188614570588404E-6L,
  -2.951146018465419674063882650970344502798E-4L,
  -7.728518171262562194043409753656506795258E-3L,
  -8.058010968753999435006488158237984014883E-2L,
  -3.356232856677966691703904770937143483472E-1L,
  -4.858192581793118040782557808823460276452E-1L,
  -1.592399251246473643510898335746432479373E-1L,
};
static const long double qs3[7] = {
  3.529139957987837084554591421329876744262E-5L,
  2.973602667215766676998703687065066180115E-3L,
  8.273534546240864308494062287908662592100E-2L,
  9.613359842126507198241321110649974032726E-1L,
  4.853923697093974370118387947065402707519E0L,
  1.002671608961669247462020977417828796933E1L,
  7.028927383922483728931327850683151410267E0L,
  /* 1.000000000000000000000000000000000000000E0L, */
};

/* Y1(x)cosX - J1(x)sinX = sqrt( 2/(pi x)) Q1(x),
   Q1(x) = z(.375 + z^2 R(z^2)), z=1/x
   2 <= x <= 2.85711669921875
   Peak relative error 6.9e-22 */
static const long double qr2[7] = {
  -1.372751603025230017220666013816502528318E-4L,
  -6.879190253347766576229143006767218972834E-3L,
  -1.061253572090925414598304855316280077828E-1L,
  -6.262164224345471241219408329354943337214E-1L,
  -1.423149636514768476376254324731437473915E0L,
  -1.087955310491078933531734062917489870754E0L,
  -1.826821119773182847861406108689273719137E-1L,
};
static const long double qs2[7] = {
  1.338768933634451601814048220627185324007E-3L,
  7.071099998918497559736318523932241901810E-2L,
  1.200511429784048632105295629933382142221E0L,
  8.327301713640367079030141077172031825276E0L,
  2.468479301872299311658145549931764426840E1L,
  2.961179686096262083509383820557051621644E1L,
  1.201402313144305153005639494661767354977E1L,
 /* 1.000000000000000000000000000000000000000E0L, */
};


static long double
qone (long double x)
{
  const long double *p, *q;
  long double s, r, z;
  int32_t ix;
  uint32_t se, i0, i1;

  GET_LDOUBLE_WORDS (se, i0, i1, x);
  ix = se & 0x7fff;
  /* ix >= 0x4000 for all calls to this function.  */
  if (ix >= 0x4002)		/* x >= 8 */
    {
      p = qr8;
      q = qs8;
    }
  else
    {
      i1 = (ix << 16) | (i0 >> 16);
      if (i1 >= 0x40019174)	/* x >= 4.54541015625 */
	{
	  p = qr5;
	  q = qs5;
	}
      else if (i1 >= 0x4000b6db)	/* x >= 2.85711669921875 */
	{
	  p = qr3;
	  q = qs3;
	}
      else	/* x >= 2 */
	{
	  p = qr2;
	  q = qs2;
	}
    }
  z = one / (x * x);
  r =
    p[0] + z * (p[1] +
		z * (p[2] + z * (p[3] + z * (p[4] + z * (p[5] + z * p[6])))));
  s =
    q[0] + z * (q[1] +
		z * (q[2] +
		     z * (q[3] + z * (q[4] + z * (q[5] + z * (q[6] + z))))));
  return (.375 + z * r / s) / x;
}
