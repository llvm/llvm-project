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

/* __ieee754_j0(x), __ieee754_y0(x)
 * Bessel function of the first and second kinds of order zero.
 * Method -- j0(x):
 *	1. For tiny x, we use j0(x) = 1 - x^2/4 + x^4/64 - ...
 *	2. Reduce x to |x| since j0(x)=j0(-x),  and
 *	   for x in (0,2)
 *		j0(x) = 1 - z/4 + z^2*R0/S0,  where z = x*x;
 *	   for x in (2,inf)
 *		j0(x) = sqrt(2/(pi*x))*(p0(x)*cos(x0)-q0(x)*sin(x0))
 *	   where x0 = x-pi/4. It is better to compute sin(x0),cos(x0)
 *	   as follow:
 *		cos(x0) = cos(x)cos(pi/4)+sin(x)sin(pi/4)
 *			= 1/sqrt(2) * (cos(x) + sin(x))
 *		sin(x0) = sin(x)cos(pi/4)-cos(x)sin(pi/4)
 *			= 1/sqrt(2) * (sin(x) - cos(x))
 *	   (To avoid cancellation, use
 *		sin(x) +- cos(x) = -cos(2x)/(sin(x) -+ cos(x))
 *	    to compute the worse one.)
 *
 *	3 Special cases
 *		j0(nan)= nan
 *		j0(0) = 1
 *		j0(inf) = 0
 *
 * Method -- y0(x):
 *	1. For x<2.
 *	   Since
 *		y0(x) = 2/pi*(j0(x)*(ln(x/2)+Euler) + x^2/4 - ...)
 *	   therefore y0(x)-2/pi*j0(x)*ln(x) is an even function.
 *	   We use the following function to approximate y0,
 *		y0(x) = U(z)/V(z) + (2/pi)*(j0(x)*ln(x)), z= x^2
 *
 *	   Note: For tiny x, U/V = u0 and j0(x)~1, hence
 *		y0(tiny) = u0 + (2/pi)*ln(tiny), (choose tiny<2**-27)
 *	2. For x>=2.
 *		y0(x) = sqrt(2/(pi*x))*(p0(x)*cos(x0)+q0(x)*sin(x0))
 *	   where x0 = x-pi/4. It is better to compute sin(x0),cos(x0)
 *	   by the method mentioned above.
 *	3. Special cases: y0(0)=-inf, y0(x<0)=NaN, y0(inf)=0.
 */

#include <math.h>
#include <math-barriers.h>
#include <math_private.h>
#include <libm-alias-finite.h>

static long double pzero (long double), qzero (long double);

static const long double
  huge = 1e4930L,
  one = 1.0L,
  invsqrtpi = 5.6418958354775628694807945156077258584405e-1L,
  tpi = 6.3661977236758134307553505349005744813784e-1L,

  /* J0(x) = 1 - x^2 / 4 + x^4 R0(x^2) / S0(x^2)
     0 <= x <= 2
     peak relative error 1.41e-22 */
  R[5] = {
  4.287176872744686992880841716723478740566E7L,
  -6.652058897474241627570911531740907185772E5L,
  7.011848381719789863458364584613651091175E3L,
  -3.168040850193372408702135490809516253693E1L,
  6.030778552661102450545394348845599300939E-2L,
},

 S[4] = {
   2.743793198556599677955266341699130654342E9L,
   3.364330079384816249840086842058954076201E7L,
   1.924119649412510777584684927494642526573E5L,
   6.239282256012734914211715620088714856494E2L,
   /*   1.000000000000000000000000000000000000000E0L,*/
};

static const long double zero = 0.0;

long double
__ieee754_j0l (long double x)
{
  long double z, s, c, ss, cc, r, u, v;
  int32_t ix;
  uint32_t se;

  GET_LDOUBLE_EXP (se, x);
  ix = se & 0x7fff;
  if (__glibc_unlikely (ix >= 0x7fff))
    return one / (x * x);
  x = fabsl (x);
  if (ix >= 0x4000)		/* |x| >= 2.0 */
    {
      __sincosl (x, &s, &c);
      ss = s - c;
      cc = s + c;
      if (ix < 0x7ffe)
	{			/* make sure x+x not overflow */
	  z = -__cosl (x + x);
	  if ((s * c) < zero)
	    cc = z / ss;
	  else
	    ss = z / cc;
	}
      /*
       * j0(x) = 1/sqrt(pi) * (P(0,x)*cc - Q(0,x)*ss) / sqrt(x)
       * y0(x) = 1/sqrt(pi) * (P(0,x)*ss + Q(0,x)*cc) / sqrt(x)
       */
      if (__glibc_unlikely (ix > 0x408e))      	/* 2^143 */
	z = (invsqrtpi * cc) / sqrtl (x);
      else
	{
	  u = pzero (x);
	  v = qzero (x);
	  z = invsqrtpi * (u * cc - v * ss) / sqrtl (x);
	}
      return z;
    }
  if (__glibc_unlikely (ix < 0x3fef))       /* |x| < 2**-16 */
    {
      /* raise inexact if x != 0 */
      math_force_eval (huge + x);
      if (ix < 0x3fde) /* |x| < 2^-33 */
	return one;
      else
	return one - 0.25 * x * x;
    }
  z = x * x;
  r = z * (R[0] + z * (R[1] + z * (R[2] + z * (R[3] + z * R[4]))));
  s = S[0] + z * (S[1] + z * (S[2] + z * (S[3] + z)));
  if (ix < 0x3fff)
    {				/* |x| < 1.00 */
      return (one - 0.25 * z + z * (r / s));
    }
  else
    {
      u = 0.5 * x;
      return ((one + u) * (one - u) + z * (r / s));
    }
}
libm_alias_finite (__ieee754_j0l, __j0l)


/* y0(x) = 2/pi ln(x) J0(x) + U(x^2)/V(x^2)
   0 < x <= 2
   peak relative error 1.7e-21 */
static const long double
U[6] = {
  -1.054912306975785573710813351985351350861E10L,
  2.520192609749295139432773849576523636127E10L,
  -1.856426071075602001239955451329519093395E9L,
  4.079209129698891442683267466276785956784E7L,
  -3.440684087134286610316661166492641011539E5L,
  1.005524356159130626192144663414848383774E3L,
};
static const long double
V[5] = {
  1.429337283720789610137291929228082613676E11L,
  2.492593075325119157558811370165695013002E9L,
  2.186077620785925464237324417623665138376E7L,
  1.238407896366385175196515057064384929222E5L,
  4.693924035211032457494368947123233101664E2L,
  /*  1.000000000000000000000000000000000000000E0L */
};

long double
__ieee754_y0l (long double x)
{
  long double z, s, c, ss, cc, u, v;
  int32_t ix;
  uint32_t se, i0, i1;

  GET_LDOUBLE_WORDS (se, i0, i1, x);
  ix = se & 0x7fff;
  /* Y0(NaN) is NaN, y0(-inf) is Nan, y0(inf) is 0  */
  if (__glibc_unlikely (se & 0x8000))
    return zero / (zero * x);
  if (__glibc_unlikely (ix >= 0x7fff))
    return one / (x + x * x);
  if (__glibc_unlikely ((i0 | i1) == 0))
    return -HUGE_VALL + x;  /* -inf and overflow exception.  */
  if (ix >= 0x4000)
    {				/* |x| >= 2.0 */

      /* y0(x) = sqrt(2/(pi*x))*(p0(x)*sin(x0)+q0(x)*cos(x0))
       * where x0 = x-pi/4
       *      Better formula:
       *              cos(x0) = cos(x)cos(pi/4)+sin(x)sin(pi/4)
       *                      =  1/sqrt(2) * (sin(x) + cos(x))
       *              sin(x0) = sin(x)cos(3pi/4)-cos(x)sin(3pi/4)
       *                      =  1/sqrt(2) * (sin(x) - cos(x))
       * To avoid cancellation, use
       *              sin(x) +- cos(x) = -cos(2x)/(sin(x) -+ cos(x))
       * to compute the worse one.
       */
      __sincosl (x, &s, &c);
      ss = s - c;
      cc = s + c;
      /*
       * j0(x) = 1/sqrt(pi) * (P(0,x)*cc - Q(0,x)*ss) / sqrt(x)
       * y0(x) = 1/sqrt(pi) * (P(0,x)*ss + Q(0,x)*cc) / sqrt(x)
       */
      if (ix < 0x7ffe)
	{			/* make sure x+x not overflow */
	  z = -__cosl (x + x);
	  if ((s * c) < zero)
	    cc = z / ss;
	  else
	    ss = z / cc;
	}
      if (__glibc_unlikely (ix > 0x408e))      	/* 2^143 */
	z = (invsqrtpi * ss) / sqrtl (x);
      else
	{
	  u = pzero (x);
	  v = qzero (x);
	  z = invsqrtpi * (u * ss + v * cc) / sqrtl (x);
	}
      return z;
    }
  if (__glibc_unlikely (ix <= 0x3fde))       /* x < 2^-33 */
    {
      z = -7.380429510868722527629822444004602747322E-2L
	+ tpi * __ieee754_logl (x);
      return z;
    }
  z = x * x;
  u = U[0] + z * (U[1] + z * (U[2] + z * (U[3] + z * (U[4] + z * U[5]))));
  v = V[0] + z * (V[1] + z * (V[2] + z * (V[3] + z * (V[4] + z))));
  return (u / v + tpi * (__ieee754_j0l (x) * __ieee754_logl (x)));
}
libm_alias_finite (__ieee754_y0l, __y0l)

/* The asymptotic expansions of pzero is
 *	1 - 9/128 s^2 + 11025/98304 s^4 - ...,	where s = 1/x.
 * For x >= 2, We approximate pzero by
 *	pzero(x) = 1 + s^2 R(s^2) / S(s^2)
 */
static const long double pR8[7] = {
  /* 8 <= x <= inf
     Peak relative error 4.62 */
  -4.094398895124198016684337960227780260127E-9L,
  -8.929643669432412640061946338524096893089E-7L,
  -6.281267456906136703868258380673108109256E-5L,
  -1.736902783620362966354814353559382399665E-3L,
  -1.831506216290984960532230842266070146847E-2L,
  -5.827178869301452892963280214772398135283E-2L,
  -2.087563267939546435460286895807046616992E-2L,
};
static const long double pS8[6] = {
  5.823145095287749230197031108839653988393E-8L,
  1.279281986035060320477759999428992730280E-5L,
  9.132668954726626677174825517150228961304E-4L,
  2.606019379433060585351880541545146252534E-2L,
  2.956262215119520464228467583516287175244E-1L,
  1.149498145388256448535563278632697465675E0L,
  /* 1.000000000000000000000000000000000000000E0L, */
};

static const long double pR5[7] = {
  /* 4.54541015625 <= x <= 8
     Peak relative error 6.51E-22 */
  -2.041226787870240954326915847282179737987E-7L,
  -2.255373879859413325570636768224534428156E-5L,
  -7.957485746440825353553537274569102059990E-4L,
  -1.093205102486816696940149222095559439425E-2L,
  -5.657957849316537477657603125260701114646E-2L,
  -8.641175552716402616180994954177818461588E-2L,
  -1.354654710097134007437166939230619726157E-2L,
};
static const long double pS5[6] = {
  2.903078099681108697057258628212823545290E-6L,
  3.253948449946735405975737677123673867321E-4L,
  1.181269751723085006534147920481582279979E-2L,
  1.719212057790143888884745200257619469363E-1L,
  1.006306498779212467670654535430694221924E0L,
  2.069568808688074324555596301126375951502E0L,
  /* 1.000000000000000000000000000000000000000E0L, */
};

static const long double pR3[7] = {
  /* 2.85711669921875 <= x <= 4.54541015625
     peak relative error 5.25e-21 */
  -5.755732156848468345557663552240816066802E-6L,
  -3.703675625855715998827966962258113034767E-4L,
  -7.390893350679637611641350096842846433236E-3L,
  -5.571922144490038765024591058478043873253E-2L,
  -1.531290690378157869291151002472627396088E-1L,
  -1.193350853469302941921647487062620011042E-1L,
  -8.567802507331578894302991505331963782905E-3L,
};
static const long double pS3[6] = {
  8.185931139070086158103309281525036712419E-5L,
  5.398016943778891093520574483111255476787E-3L,
  1.130589193590489566669164765853409621081E-1L,
  9.358652328786413274673192987670237145071E-1L,
  3.091711512598349056276917907005098085273E0L,
  3.594602474737921977972586821673124231111E0L,
  /* 1.000000000000000000000000000000000000000E0L, */
};

static const long double pR2[7] = {
  /* 2 <= x <= 2.85711669921875
     peak relative error 2.64e-21 */
  -1.219525235804532014243621104365384992623E-4L,
  -4.838597135805578919601088680065298763049E-3L,
  -5.732223181683569266223306197751407418301E-2L,
  -2.472947430526425064982909699406646503758E-1L,
  -3.753373645974077960207588073975976327695E-1L,
  -1.556241316844728872406672349347137975495E-1L,
  -5.355423239526452209595316733635519506958E-3L,
};
static const long double pS2[6] = {
  1.734442793664291412489066256138894953823E-3L,
  7.158111826468626405416300895617986926008E-2L,
  9.153839713992138340197264669867993552641E-1L,
  4.539209519433011393525841956702487797582E0L,
  8.868932430625331650266067101752626253644E0L,
  6.067161890196324146320763844772857713502E0L,
  /* 1.000000000000000000000000000000000000000E0L, */
};

static long double
pzero (long double x)
{
  const long double *p, *q;
  long double z, r, s;
  int32_t ix;
  uint32_t se, i0, i1;

  GET_LDOUBLE_WORDS (se, i0, i1, x);
  ix = se & 0x7fff;
  /* ix >= 0x4000 for all calls to this function.  */
  if (ix >= 0x4002)
    {
      p = pR8;
      q = pS8;
    }				/* x >= 8 */
  else
    {
      i1 = (ix << 16) | (i0 >> 16);
      if (i1 >= 0x40019174)	/* x >= 4.54541015625 */
	{
	  p = pR5;
	  q = pS5;
	}
      else if (i1 >= 0x4000b6db)	/* x >= 2.85711669921875 */
	{
	  p = pR3;
	  q = pS3;
	}
      else	/* x >= 2 */
	{
	  p = pR2;
	  q = pS2;
	}
    }
  z = one / (x * x);
  r =
    p[0] + z * (p[1] +
		z * (p[2] + z * (p[3] + z * (p[4] + z * (p[5] + z * p[6])))));
  s =
    q[0] + z * (q[1] + z * (q[2] + z * (q[3] + z * (q[4] + z * (q[5] + z)))));
  return (one + z * r / s);
}


/* For x >= 8, the asymptotic expansions of qzero is
 *	-1/8 s + 75/1024 s^3 - ..., where s = 1/x.
 * We approximate qzero by
 *	qzero(x) = s*(-.125 + R(s^2) / S(s^2))
 */
static const long double qR8[7] = {
  /* 8 <= x <= inf
     peak relative error 2.23e-21 */
  3.001267180483191397885272640777189348008E-10L,
  8.693186311430836495238494289942413810121E-8L,
  8.496875536711266039522937037850596580686E-6L,
  3.482702869915288984296602449543513958409E-4L,
  6.036378380706107692863811938221290851352E-3L,
  3.881970028476167836382607922840452192636E-2L,
  6.132191514516237371140841765561219149638E-2L,
};
static const long double qS8[7] = {
  4.097730123753051126914971174076227600212E-9L,
  1.199615869122646109596153392152131139306E-6L,
  1.196337580514532207793107149088168946451E-4L,
  5.099074440112045094341500497767181211104E-3L,
  9.577420799632372483249761659674764460583E-2L,
  7.385243015344292267061953461563695918646E-1L,
  1.917266424391428937962682301561699055943E0L,
  /* 1.000000000000000000000000000000000000000E0L, */
};

static const long double qR5[7] = {
  /* 4.54541015625 <= x <= 8
     peak relative error 1.03e-21 */
  3.406256556438974327309660241748106352137E-8L,
  4.855492710552705436943630087976121021980E-6L,
  2.301011739663737780613356017352912281980E-4L,
  4.500470249273129953870234803596619899226E-3L,
  3.651376459725695502726921248173637054828E-2L,
  1.071578819056574524416060138514508609805E-1L,
  7.458950172851611673015774675225656063757E-2L,
};
static const long double qS5[7] = {
  4.650675622764245276538207123618745150785E-7L,
  6.773573292521412265840260065635377164455E-5L,
  3.340711249876192721980146877577806687714E-3L,
  7.036218046856839214741678375536970613501E-2L,
  6.569599559163872573895171876511377891143E-1L,
  2.557525022583599204591036677199171155186E0L,
  3.457237396120935674982927714210361269133E0L,
  /* 1.000000000000000000000000000000000000000E0L,*/
};

static const long double qR3[7] = {
  /* 2.85711669921875 <= x <= 4.54541015625
     peak relative error 5.24e-21 */
  1.749459596550816915639829017724249805242E-6L,
  1.446252487543383683621692672078376929437E-4L,
  3.842084087362410664036704812125005761859E-3L,
  4.066369994699462547896426554180954233581E-2L,
  1.721093619117980251295234795188992722447E-1L,
  2.538595333972857367655146949093055405072E-1L,
  8.560591367256769038905328596020118877936E-2L,
};
static const long double qS3[7] = {
  2.388596091707517488372313710647510488042E-5L,
  2.048679968058758616370095132104333998147E-3L,
  5.824663198201417760864458765259945181513E-2L,
  6.953906394693328750931617748038994763958E-1L,
  3.638186936390881159685868764832961092476E0L,
  7.900169524705757837298990558459547842607E0L,
  5.992718532451026507552820701127504582907E0L,
  /* 1.000000000000000000000000000000000000000E0L, */
};

static const long double qR2[7] = {
  /* 2 <= x <= 2.85711669921875
     peak relative error 1.58e-21  */
  6.306524405520048545426928892276696949540E-5L,
  3.209606155709930950935893996591576624054E-3L,
  5.027828775702022732912321378866797059604E-2L,
  3.012705561838718956481911477587757845163E-1L,
  6.960544893905752937420734884995688523815E-1L,
  5.431871999743531634887107835372232030655E-1L,
  9.447736151202905471899259026430157211949E-2L,
};
static const long double qS2[7] = {
  8.610579901936193494609755345106129102676E-4L,
  4.649054352710496997203474853066665869047E-2L,
  8.104282924459837407218042945106320388339E-1L,
  5.807730930825886427048038146088828206852E0L,
  1.795310145936848873627710102199881642939E1L,
  2.281313316875375733663657188888110605044E1L,
  1.011242067883822301487154844458322200143E1L,
  /* 1.000000000000000000000000000000000000000E0L, */
};

static long double
qzero (long double x)
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
      p = qR8;
      q = qS8;
    }
  else
    {
      i1 = (ix << 16) | (i0 >> 16);
      if (i1 >= 0x40019174)	/* x >= 4.54541015625 */
	{
	  p = qR5;
	  q = qS5;
	}
      else if (i1 >= 0x4000b6db)	/* x >= 2.85711669921875 */
	{
	  p = qR3;
	  q = qS3;
	}
      else	/* x >= 2 */
	{
	  p = qR2;
	  q = qS2;
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
  return (-.125 + z * r / s) / x;
}
