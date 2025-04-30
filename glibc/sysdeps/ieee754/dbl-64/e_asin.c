/*
 * IBM Accurate Mathematical Library
 * written by International Business Machines Corp.
 * Copyright (C) 2001-2021 Free Software Foundation, Inc.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, see <https://www.gnu.org/licenses/>.
 */
/******************************************************************/
/*     MODULE_NAME:uasncs.c                                       */
/*                                                                */
/*     FUNCTIONS: uasin                                           */
/*                uacos                                           */
/* FILES NEEDED: dla.h endian.h mydefs.h  usncs.h                 */
/*               sincos.tbl  asincos.tbl  powtwo.tbl root.tbl     */
/*                                                                */
/******************************************************************/
#include "endian.h"
#include "mydefs.h"
#include "asincos.tbl"
#include "root.tbl"
#include "powtwo.tbl"
#include "uasncs.h"
#include <float.h>
#include <math.h>
#include <math_private.h>
#include <math-underflow.h>
#include <libm-alias-finite.h>

#ifndef SECTION
# define SECTION
#endif

/* asin with max ULP of ~0.516 based on random sampling.  */
double
SECTION
__ieee754_asin(double x){
  double x2,xx,res1,p,t,res,r,cor,cc,y,c,z;
  mynumber u,v;
  int4 k,m,n;

  u.x = x;
  m = u.i[HIGH_HALF];
  k = 0x7fffffff&m;              /* no sign */

  if (k < 0x3e500000)
    {
      math_check_force_underflow (x);
      return x;  /* for x->0 => sin(x)=x */
    }
  /*----------------------2^-26 <= |x| < 2^ -3    -----------------*/
  else
  if (k < 0x3fc00000) {
    x2 = x*x;
    t = (((((f6*x2 + f5)*x2 + f4)*x2 + f3)*x2 + f2)*x2 + f1)*(x2*x);
    res = x+t;         /*  res=arcsin(x) according to Taylor series  */
    /* Max ULP is 0.513.  */
    return res;
  }
  /*---------------------0.125 <= |x| < 0.5 -----------------------------*/
  else if (k < 0x3fe00000) {
    if (k<0x3fd00000) n = 11*((k&0x000fffff)>>15);
    else n = 11*((k&0x000fffff)>>14)+352;
    if (m>0) xx = x - asncs.x[n];
    else xx = -x - asncs.x[n];
    t = asncs.x[n+1]*xx;
    p=xx*xx*(asncs.x[n+2]+xx*(asncs.x[n+3]+xx*(asncs.x[n+4]+xx*(asncs.x[n+5]
     +xx*asncs.x[n+6]))))+asncs.x[n+7];
    t+=p;
    res =asncs.x[n+8] +t;
    /* Max ULP is 0.524.  */
    return (m>0)?res:-res;
  }    /*   else  if (k < 0x3fe00000)    */
  /*-------------------- 0.5 <= |x| < 0.75 -----------------------------*/
  else
  if (k < 0x3fe80000) {
    n = 1056+((k&0x000fe000)>>11)*3;
    if (m>0) xx = x - asncs.x[n];
    else xx = -x - asncs.x[n];
    t = asncs.x[n+1]*xx;
    p=xx*xx*(asncs.x[n+2]+xx*(asncs.x[n+3]+xx*(asncs.x[n+4]+xx*(asncs.x[n+5]
	   +xx*(asncs.x[n+6]+xx*asncs.x[n+7])))))+asncs.x[n+8];
    t+=p;
    res =asncs.x[n+9] +t;
    /* Max ULP is 0.505.  */
    return (m>0)?res:-res;
  }    /*   else  if (k < 0x3fe80000)    */
  /*--------------------- 0.75 <= |x|< 0.921875 ----------------------*/
  else
  if (k < 0x3fed8000) {
    n = 992+((k&0x000fe000)>>13)*13;
    if (m>0) xx = x - asncs.x[n];
    else xx = -x - asncs.x[n];
    t = asncs.x[n+1]*xx;
    p=xx*xx*(asncs.x[n+2]+xx*(asncs.x[n+3]+xx*(asncs.x[n+4]+xx*(asncs.x[n+5]
     +xx*(asncs.x[n+6]+xx*(asncs.x[n+7]+xx*asncs.x[n+8]))))))+asncs.x[n+9];
    t+=p;
    res =asncs.x[n+10] +t;
    /* Max ULP is 0.505.  */
    return (m>0)?res:-res;
  }    /*   else  if (k < 0x3fed8000)    */
  /*-------------------0.921875 <= |x| < 0.953125 ------------------------*/
  else
  if (k < 0x3fee8000) {
    n = 884+((k&0x000fe000)>>13)*14;
    if (m>0) xx = x - asncs.x[n];
    else xx = -x - asncs.x[n];
    t = asncs.x[n+1]*xx;
    p=xx*xx*(asncs.x[n+2]+xx*(asncs.x[n+3]+xx*(asncs.x[n+4]+
		      xx*(asncs.x[n+5]+xx*(asncs.x[n+6]
		      +xx*(asncs.x[n+7]+xx*(asncs.x[n+8]+
		      xx*asncs.x[n+9])))))))+asncs.x[n+10];
    t+=p;
    res =asncs.x[n+11] +t;
    /* Max ULP is 0.505.  */
    return (m>0)?res:-res;
  }    /*   else  if (k < 0x3fee8000)    */

  /*--------------------0.953125 <= |x| < 0.96875 ------------------------*/
  else
  if (k < 0x3fef0000) {
    n = 768+((k&0x000fe000)>>13)*15;
    if (m>0) xx = x - asncs.x[n];
    else xx = -x - asncs.x[n];
    t = asncs.x[n+1]*xx;
    p=xx*xx*(asncs.x[n+2]+xx*(asncs.x[n+3]+xx*(asncs.x[n+4]+
			 xx*(asncs.x[n+5]+xx*(asncs.x[n+6]
			 +xx*(asncs.x[n+7]+xx*(asncs.x[n+8]+
		    xx*(asncs.x[n+9]+xx*asncs.x[n+10]))))))))+asncs.x[n+11];
    t+=p;
    res =asncs.x[n+12] +t;
    /* Max ULP is 0.505.  */
    return (m>0)?res:-res;
  }    /*   else  if (k < 0x3fef0000)    */
  /*--------------------0.96875 <= |x| < 1 --------------------------------*/
  else
  if (k<0x3ff00000)  {
    z = 0.5*((m>0)?(1.0-x):(1.0+x));
    v.x=z;
    k=v.i[HIGH_HALF];
    t=inroot[(k&0x001fffff)>>14]*powtwo[511-(k>>21)];
    r=1.0-t*t*z;
    t = t*(rt0+r*(rt1+r*(rt2+r*rt3)));
    c=t*z;
    t=c*(1.5-0.5*t*c);
    y=(c+t24)-t24;
    cc = (z-y*y)/(t+y);
    p=(((((f6*z+f5)*z+f4)*z+f3)*z+f2)*z+f1)*z;
    cor = (hp1.x - 2.0*cc)-2.0*(y+cc)*p;
    res1 = hp0.x - 2.0*y;
    res =res1 + cor;
    /* Max ULP is 0.5015.  */
    return (m>0)?res:-res;
  }    /*   else  if (k < 0x3ff00000)    */
  /*---------------------------- |x|>=1 -------------------------------*/
  else if (k==0x3ff00000 && u.i[LOW_HALF]==0) return (m>0)?hp0.x:-hp0.x;
  else
  if (k>0x7ff00000 || (k == 0x7ff00000 && u.i[LOW_HALF] != 0)) return x + x;
  else {
    u.i[HIGH_HALF]=0x7ff00000;
    v.i[HIGH_HALF]=0x7ff00000;
    u.i[LOW_HALF]=0;
    v.i[LOW_HALF]=0;
    return u.x/v.x;  /* NaN */
 }
}
#ifndef __ieee754_asin
libm_alias_finite (__ieee754_asin, __asin)
#endif

/*******************************************************************/
/*                                                                 */
/*         End of arcsine,  below is arccosine                     */
/*                                                                 */
/*******************************************************************/

/* acos with max ULP of ~0.523 based on random sampling.  */
double
SECTION
__ieee754_acos(double x)
{
  double x2,xx,res1,p,t,res,r,cor,cc,y,c,z;
  mynumber u,v;
  int4 k,m,n;
  u.x = x;
  m = u.i[HIGH_HALF];
  k = 0x7fffffff&m;
  /*-------------------  |x|<2.77556*10^-17 ----------------------*/
  if (k < 0x3c880000) return hp0.x;

  /*-----------------  2.77556*10^-17 <= |x| < 2^-3 --------------*/
  else
  if (k < 0x3fc00000) {
    x2 = x*x;
    t = (((((f6*x2 + f5)*x2 + f4)*x2 + f3)*x2 + f2)*x2 + f1)*(x2*x);
    r=hp0.x-x;
    cor=(((hp0.x-r)-x)+hp1.x)-t;
    res = r+cor;
    /* Max ULP is 0.502.  */
    return res;
  }    /*   else  if (k < 0x3fc00000)    */
  /*----------------------  0.125 <= |x| < 0.5 --------------------*/
  else
  if (k < 0x3fe00000) {
    if (k<0x3fd00000) n = 11*((k&0x000fffff)>>15);
    else n = 11*((k&0x000fffff)>>14)+352;
    if (m>0) xx = x - asncs.x[n];
    else xx = -x - asncs.x[n];
    t = asncs.x[n+1]*xx;
    p=xx*xx*(asncs.x[n+2]+xx*(asncs.x[n+3]+xx*(asncs.x[n+4]+
		   xx*(asncs.x[n+5]+xx*asncs.x[n+6]))))+asncs.x[n+7];
    t+=p;
    y = (m>0)?(hp0.x-asncs.x[n+8]):(hp0.x+asncs.x[n+8]);
    t = (m>0)?(hp1.x-t):(hp1.x+t);
    res = y+t;
   /* Max ULP is 0.51.  */
    return res;
  }    /*   else  if (k < 0x3fe00000)    */

  /*--------------------------- 0.5 <= |x| < 0.75 ---------------------*/
  else
  if (k < 0x3fe80000) {
    n = 1056+((k&0x000fe000)>>11)*3;
    if (m>0) {xx = x - asncs.x[n]; }
    else {xx = -x - asncs.x[n]; }
    t = asncs.x[n+1]*xx;
    p=xx*xx*(asncs.x[n+2]+xx*(asncs.x[n+3]+xx*(asncs.x[n+4]+
		   xx*(asncs.x[n+5]+xx*(asncs.x[n+6]+
		   xx*asncs.x[n+7])))))+asncs.x[n+8];
    t+=p;
   y = (m>0)?(hp0.x-asncs.x[n+9]):(hp0.x+asncs.x[n+9]);
   t = (m>0)?(hp1.x-t):(hp1.x+t);
   res = y+t;
   /* Max ULP is 0.523 based on random sampling.  */
   return res;
  }    /*   else  if (k < 0x3fe80000)    */

/*------------------------- 0.75 <= |x| < 0.921875 -------------*/
  else
  if (k < 0x3fed8000) {
    n = 992+((k&0x000fe000)>>13)*13;
    if (m>0) {xx = x - asncs.x[n]; }
    else {xx = -x - asncs.x[n]; }
    t = asncs.x[n+1]*xx;
    p=xx*xx*(asncs.x[n+2]+xx*(asncs.x[n+3]+xx*(asncs.x[n+4]+
		      xx*(asncs.x[n+5]+xx*(asncs.x[n+6]+xx*(asncs.x[n+7]+
		      xx*asncs.x[n+8]))))))+asncs.x[n+9];
    t+=p;
    y = (m>0)?(hp0.x-asncs.x[n+10]):(hp0.x+asncs.x[n+10]);
    t = (m>0)?(hp1.x-t):(hp1.x+t);
    res = y+t;
   /* Max ULP is 0.523 based on random sampling.  */
    return res;
  }    /*   else  if (k < 0x3fed8000)    */

/*-------------------0.921875 <= |x| < 0.953125 ------------------*/
  else
  if (k < 0x3fee8000) {
    n = 884+((k&0x000fe000)>>13)*14;
    if (m>0) {xx = x - asncs.x[n]; }
    else {xx = -x - asncs.x[n]; }
    t = asncs.x[n+1]*xx;
    p=xx*xx*(asncs.x[n+2]+xx*(asncs.x[n+3]+xx*(asncs.x[n+4]+
		   xx*(asncs.x[n+5]+xx*(asncs.x[n+6]
		   +xx*(asncs.x[n+7]+xx*(asncs.x[n+8]+
		   xx*asncs.x[n+9])))))))+asncs.x[n+10];
    t+=p;
    y = (m>0)?(hp0.x-asncs.x[n+11]):(hp0.x+asncs.x[n+11]);
    t = (m>0)?(hp1.x-t):(hp1.x+t);
    res = y+t;
   /* Max ULP is 0.523 based on random sampling.  */
    return res;
  }    /*   else  if (k < 0x3fee8000)    */

  /*--------------------0.953125 <= |x| < 0.96875 ----------------*/
  else
  if (k < 0x3fef0000) {
    n = 768+((k&0x000fe000)>>13)*15;
    if (m>0) {xx = x - asncs.x[n]; }
    else {xx = -x - asncs.x[n]; }
    t = asncs.x[n+1]*xx;
    p=xx*xx*(asncs.x[n+2]+xx*(asncs.x[n+3]+xx*(asncs.x[n+4]+
	    xx*(asncs.x[n+5]+xx*(asncs.x[n+6]
	    +xx*(asncs.x[n+7]+xx*(asncs.x[n+8]+xx*(asncs.x[n+9]+
	    xx*asncs.x[n+10]))))))))+asncs.x[n+11];
    t+=p;
    y = (m>0)?(hp0.x-asncs.x[n+12]):(hp0.x+asncs.x[n+12]);
   t = (m>0)?(hp1.x-t):(hp1.x+t);
   res = y+t;
   /* Max ULP is 0.523 based on random sampling.  */
   return res;
  }    /*   else  if (k < 0x3fef0000)    */
  /*-----------------0.96875 <= |x| < 1 ---------------------------*/

  else
  if (k<0x3ff00000)  {
    z = 0.5*((m>0)?(1.0-x):(1.0+x));
    v.x=z;
    k=v.i[HIGH_HALF];
    t=inroot[(k&0x001fffff)>>14]*powtwo[511-(k>>21)];
    r=1.0-t*t*z;
    t = t*(rt0+r*(rt1+r*(rt2+r*rt3)));
    c=t*z;
    t=c*(1.5-0.5*t*c);
    y = (t27*c+c)-t27*c;
    cc = (z-y*y)/(t+y);
    p=(((((f6*z+f5)*z+f4)*z+f3)*z+f2)*z+f1)*z;
    if (m<0) {
      cor = (hp1.x - cc)-(y+cc)*p;
      res1 = hp0.x - y;
      res =res1 + cor;
      /* Max ULP is 0.501.  */
      return (res+res);
    }
    else {
      cor = cc+p*(y+cc);
      res = y + cor;
      /* Max ULP is 0.515.  */
      return (res+res);
    }
  }    /*   else  if (k < 0x3ff00000)    */

  /*---------------------------- |x|>=1 -----------------------*/
  else
  if (k==0x3ff00000 && u.i[LOW_HALF]==0) return (m>0)?0:2.0*hp0.x;
  else
  if (k>0x7ff00000 || (k == 0x7ff00000 && u.i[LOW_HALF] != 0)) return x + x;
  else {
    u.i[HIGH_HALF]=0x7ff00000;
    v.i[HIGH_HALF]=0x7ff00000;
    u.i[LOW_HALF]=0;
    v.i[LOW_HALF]=0;
    return u.x/v.x;
  }
}
#ifndef __ieee754_acos
libm_alias_finite (__ieee754_acos, __acos)
#endif
