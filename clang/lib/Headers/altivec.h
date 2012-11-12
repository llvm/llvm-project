/*===---- altivec.h - Standard header for type generic math ---------------===*\
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
\*===----------------------------------------------------------------------===*/

#ifndef __ALTIVEC_H
#define __ALTIVEC_H

#ifndef __ALTIVEC__
#error "AltiVec support not enabled"
#endif

/* constants for mapping CR6 bits to predicate result. */

#define __CR6_EQ     0
#define __CR6_EQ_REV 1
#define __CR6_LT     2
#define __CR6_LT_REV 3

#define __ATTRS_o_ai __attribute__((__overloadable__, __always_inline__))

static vector signed char __ATTRS_o_ai
vec_perm(vector signed char a, vector signed char b, vector unsigned char c);

static vector unsigned char __ATTRS_o_ai
vec_perm(vector unsigned char a,
         vector unsigned char b, 
         vector unsigned char c);

static vector bool char __ATTRS_o_ai
vec_perm(vector bool char a, vector bool char b, vector unsigned char c);

static vector short __ATTRS_o_ai
vec_perm(vector short a, vector short b, vector unsigned char c);

static vector unsigned short __ATTRS_o_ai
vec_perm(vector unsigned short a,
         vector unsigned short b, 
         vector unsigned char c);

static vector bool short __ATTRS_o_ai
vec_perm(vector bool short a, vector bool short b, vector unsigned char c);

static vector pixel __ATTRS_o_ai
vec_perm(vector pixel a, vector pixel b, vector unsigned char c);

static vector int __ATTRS_o_ai
vec_perm(vector int a, vector int b, vector unsigned char c);

static vector unsigned int __ATTRS_o_ai
vec_perm(vector unsigned int a, vector unsigned int b, vector unsigned char c);

static vector bool int __ATTRS_o_ai
vec_perm(vector bool int a, vector bool int b, vector unsigned char c);

static vector float __ATTRS_o_ai
vec_perm(vector float a, vector float b, vector unsigned char c);

/* vec_abs */

#define __builtin_altivec_abs_v16qi vec_abs
#define __builtin_altivec_abs_v8hi  vec_abs
#define __builtin_altivec_abs_v4si  vec_abs

static vector signed char __ATTRS_o_ai
vec_abs(vector signed char a)
{
  return __builtin_altivec_vmaxsb(a, -a);
}

static vector signed short __ATTRS_o_ai
vec_abs(vector signed short a)
{
  return __builtin_altivec_vmaxsh(a, -a);
}

static vector signed int __ATTRS_o_ai
vec_abs(vector signed int a)
{
  return __builtin_altivec_vmaxsw(a, -a);
}

static vector float __ATTRS_o_ai
vec_abs(vector float a)
{
  vector unsigned int res = (vector unsigned int)a 
                            & (vector unsigned int)(0x7FFFFFFF);
  return (vector float)res;
}

/* vec_abss */

#define __builtin_altivec_abss_v16qi vec_abss
#define __builtin_altivec_abss_v8hi  vec_abss
#define __builtin_altivec_abss_v4si  vec_abss

static vector signed char __ATTRS_o_ai
vec_abss(vector signed char a)
{
  return __builtin_altivec_vmaxsb
           (a, __builtin_altivec_vsubsbs((vector signed char)(0), a));
}

static vector signed short __ATTRS_o_ai
vec_abss(vector signed short a)
{
  return __builtin_altivec_vmaxsh
           (a, __builtin_altivec_vsubshs((vector signed short)(0), a));
}

static vector signed int __ATTRS_o_ai
vec_abss(vector signed int a)
{
  return __builtin_altivec_vmaxsw
           (a, __builtin_altivec_vsubsws((vector signed int)(0), a));
}

/* vec_add */

static vector signed char __ATTRS_o_ai
vec_add(vector signed char a, vector signed char b)
{
  return a + b;
}

static vector signed char __ATTRS_o_ai
vec_add(vector bool char a, vector signed char b)
{
  return (vector signed char)a + b;
}

static vector signed char __ATTRS_o_ai
vec_add(vector signed char a, vector bool char b)
{
  return a + (vector signed char)b;
}

static vector unsigned char __ATTRS_o_ai
vec_add(vector unsigned char a, vector unsigned char b)
{
  return a + b;
}

static vector unsigned char __ATTRS_o_ai
vec_add(vector bool char a, vector unsigned char b)
{
  return (vector unsigned char)a + b;
}

static vector unsigned char __ATTRS_o_ai
vec_add(vector unsigned char a, vector bool char b)
{
  return a + (vector unsigned char)b;
}

static vector short __ATTRS_o_ai
vec_add(vector short a, vector short b)
{
  return a + b;
}

static vector short __ATTRS_o_ai
vec_add(vector bool short a, vector short b)
{
  return (vector short)a + b;
}

static vector short __ATTRS_o_ai
vec_add(vector short a, vector bool short b)
{
  return a + (vector short)b;
}

static vector unsigned short __ATTRS_o_ai
vec_add(vector unsigned short a, vector unsigned short b)
{
  return a + b;
}

static vector unsigned short __ATTRS_o_ai
vec_add(vector bool short a, vector unsigned short b)
{
  return (vector unsigned short)a + b;
}

static vector unsigned short __ATTRS_o_ai
vec_add(vector unsigned short a, vector bool short b)
{
  return a + (vector unsigned short)b;
}

static vector int __ATTRS_o_ai
vec_add(vector int a, vector int b)
{
  return a + b;
}

static vector int __ATTRS_o_ai
vec_add(vector bool int a, vector int b)
{
  return (vector int)a + b;
}

static vector int __ATTRS_o_ai
vec_add(vector int a, vector bool int b)
{
  return a + (vector int)b;
}

static vector unsigned int __ATTRS_o_ai
vec_add(vector unsigned int a, vector unsigned int b)
{
  return a + b;
}

static vector unsigned int __ATTRS_o_ai
vec_add(vector bool int a, vector unsigned int b)
{
  return (vector unsigned int)a + b;
}

static vector unsigned int __ATTRS_o_ai
vec_add(vector unsigned int a, vector bool int b)
{
  return a + (vector unsigned int)b;
}

static vector float __ATTRS_o_ai
vec_add(vector float a, vector float b)
{
  return a + b;
}

/* vec_vaddubm */

#define __builtin_altivec_vaddubm vec_vaddubm

static vector signed char __ATTRS_o_ai
vec_vaddubm(vector signed char a, vector signed char b)
{
  return a + b;
}

static vector signed char __ATTRS_o_ai
vec_vaddubm(vector bool char a, vector signed char b)
{
  return (vector signed char)a + b;
}

static vector signed char __ATTRS_o_ai
vec_vaddubm(vector signed char a, vector bool char b)
{
  return a + (vector signed char)b;
}

static vector unsigned char __ATTRS_o_ai
vec_vaddubm(vector unsigned char a, vector unsigned char b)
{
  return a + b;
}

static vector unsigned char __ATTRS_o_ai
vec_vaddubm(vector bool char a, vector unsigned char b)
{
  return (vector unsigned char)a + b;
}

static vector unsigned char __ATTRS_o_ai
vec_vaddubm(vector unsigned char a, vector bool char b)
{
  return a + (vector unsigned char)b;
}

/* vec_vadduhm */

#define __builtin_altivec_vadduhm vec_vadduhm

static vector short __ATTRS_o_ai
vec_vadduhm(vector short a, vector short b)
{
  return a + b;
}

static vector short __ATTRS_o_ai
vec_vadduhm(vector bool short a, vector short b)
{
  return (vector short)a + b;
}

static vector short __ATTRS_o_ai
vec_vadduhm(vector short a, vector bool short b)
{
  return a + (vector short)b;
}

static vector unsigned short __ATTRS_o_ai
vec_vadduhm(vector unsigned short a, vector unsigned short b)
{
  return a + b;
}

static vector unsigned short __ATTRS_o_ai
vec_vadduhm(vector bool short a, vector unsigned short b)
{
  return (vector unsigned short)a + b;
}

static vector unsigned short __ATTRS_o_ai
vec_vadduhm(vector unsigned short a, vector bool short b)
{
  return a + (vector unsigned short)b;
}

/* vec_vadduwm */

#define __builtin_altivec_vadduwm vec_vadduwm

static vector int __ATTRS_o_ai
vec_vadduwm(vector int a, vector int b)
{
  return a + b;
}

static vector int __ATTRS_o_ai
vec_vadduwm(vector bool int a, vector int b)
{
  return (vector int)a + b;
}

static vector int __ATTRS_o_ai
vec_vadduwm(vector int a, vector bool int b)
{
  return a + (vector int)b;
}

static vector unsigned int __ATTRS_o_ai
vec_vadduwm(vector unsigned int a, vector unsigned int b)
{
  return a + b;
}

static vector unsigned int __ATTRS_o_ai
vec_vadduwm(vector bool int a, vector unsigned int b)
{
  return (vector unsigned int)a + b;
}

static vector unsigned int __ATTRS_o_ai
vec_vadduwm(vector unsigned int a, vector bool int b)
{
  return a + (vector unsigned int)b;
}

/* vec_vaddfp */

#define __builtin_altivec_vaddfp  vec_vaddfp

static vector float __attribute__((__always_inline__))
vec_vaddfp(vector float a, vector float b)
{
  return a + b;
}

/* vec_addc */

static vector unsigned int __attribute__((__always_inline__))
vec_addc(vector unsigned int a, vector unsigned int b)
{
  return __builtin_altivec_vaddcuw(a, b);
}

/* vec_vaddcuw */

static vector unsigned int __attribute__((__always_inline__))
vec_vaddcuw(vector unsigned int a, vector unsigned int b)
{
  return __builtin_altivec_vaddcuw(a, b);
}

/* vec_adds */

static vector signed char __ATTRS_o_ai
vec_adds(vector signed char a, vector signed char b)
{
  return __builtin_altivec_vaddsbs(a, b);
}

static vector signed char __ATTRS_o_ai
vec_adds(vector bool char a, vector signed char b)
{
  return __builtin_altivec_vaddsbs((vector signed char)a, b);
}

static vector signed char __ATTRS_o_ai
vec_adds(vector signed char a, vector bool char b)
{
  return __builtin_altivec_vaddsbs(a, (vector signed char)b);
}

static vector unsigned char __ATTRS_o_ai
vec_adds(vector unsigned char a, vector unsigned char b)
{
  return __builtin_altivec_vaddubs(a, b);
}

static vector unsigned char __ATTRS_o_ai
vec_adds(vector bool char a, vector unsigned char b)
{
  return __builtin_altivec_vaddubs((vector unsigned char)a, b);
}

static vector unsigned char __ATTRS_o_ai
vec_adds(vector unsigned char a, vector bool char b)
{
  return __builtin_altivec_vaddubs(a, (vector unsigned char)b);
}

static vector short __ATTRS_o_ai
vec_adds(vector short a, vector short b)
{
  return __builtin_altivec_vaddshs(a, b);
}

static vector short __ATTRS_o_ai
vec_adds(vector bool short a, vector short b)
{
  return __builtin_altivec_vaddshs((vector short)a, b);
}

static vector short __ATTRS_o_ai
vec_adds(vector short a, vector bool short b)
{
  return __builtin_altivec_vaddshs(a, (vector short)b);
}

static vector unsigned short __ATTRS_o_ai
vec_adds(vector unsigned short a, vector unsigned short b)
{
  return __builtin_altivec_vadduhs(a, b);
}

static vector unsigned short __ATTRS_o_ai
vec_adds(vector bool short a, vector unsigned short b)
{
  return __builtin_altivec_vadduhs((vector unsigned short)a, b);
}

static vector unsigned short __ATTRS_o_ai
vec_adds(vector unsigned short a, vector bool short b)
{
  return __builtin_altivec_vadduhs(a, (vector unsigned short)b);
}

static vector int __ATTRS_o_ai
vec_adds(vector int a, vector int b)
{
  return __builtin_altivec_vaddsws(a, b);
}

static vector int __ATTRS_o_ai
vec_adds(vector bool int a, vector int b)
{
  return __builtin_altivec_vaddsws((vector int)a, b);
}

static vector int __ATTRS_o_ai
vec_adds(vector int a, vector bool int b)
{
  return __builtin_altivec_vaddsws(a, (vector int)b);
}

static vector unsigned int __ATTRS_o_ai
vec_adds(vector unsigned int a, vector unsigned int b)
{
  return __builtin_altivec_vadduws(a, b);
}

static vector unsigned int __ATTRS_o_ai
vec_adds(vector bool int a, vector unsigned int b)
{
  return __builtin_altivec_vadduws((vector unsigned int)a, b);
}

static vector unsigned int __ATTRS_o_ai
vec_adds(vector unsigned int a, vector bool int b)
{
  return __builtin_altivec_vadduws(a, (vector unsigned int)b);
}

/* vec_vaddsbs */

static vector signed char __ATTRS_o_ai
vec_vaddsbs(vector signed char a, vector signed char b)
{
  return __builtin_altivec_vaddsbs(a, b);
}

static vector signed char __ATTRS_o_ai
vec_vaddsbs(vector bool char a, vector signed char b)
{
  return __builtin_altivec_vaddsbs((vector signed char)a, b);
}

static vector signed char __ATTRS_o_ai
vec_vaddsbs(vector signed char a, vector bool char b)
{
  return __builtin_altivec_vaddsbs(a, (vector signed char)b);
}

/* vec_vaddubs */

static vector unsigned char __ATTRS_o_ai
vec_vaddubs(vector unsigned char a, vector unsigned char b)
{
  return __builtin_altivec_vaddubs(a, b);
}

static vector unsigned char __ATTRS_o_ai
vec_vaddubs(vector bool char a, vector unsigned char b)
{
  return __builtin_altivec_vaddubs((vector unsigned char)a, b);
}

static vector unsigned char __ATTRS_o_ai
vec_vaddubs(vector unsigned char a, vector bool char b)
{
  return __builtin_altivec_vaddubs(a, (vector unsigned char)b);
}

/* vec_vaddshs */

static vector short __ATTRS_o_ai
vec_vaddshs(vector short a, vector short b)
{
  return __builtin_altivec_vaddshs(a, b);
}

static vector short __ATTRS_o_ai
vec_vaddshs(vector bool short a, vector short b)
{
  return __builtin_altivec_vaddshs((vector short)a, b);
}

static vector short __ATTRS_o_ai
vec_vaddshs(vector short a, vector bool short b)
{
  return __builtin_altivec_vaddshs(a, (vector short)b);
}

/* vec_vadduhs */

static vector unsigned short __ATTRS_o_ai
vec_vadduhs(vector unsigned short a, vector unsigned short b)
{
  return __builtin_altivec_vadduhs(a, b);
}

static vector unsigned short __ATTRS_o_ai
vec_vadduhs(vector bool short a, vector unsigned short b)
{
  return __builtin_altivec_vadduhs((vector unsigned short)a, b);
}

static vector unsigned short __ATTRS_o_ai
vec_vadduhs(vector unsigned short a, vector bool short b)
{
  return __builtin_altivec_vadduhs(a, (vector unsigned short)b);
}

/* vec_vaddsws */

static vector int __ATTRS_o_ai
vec_vaddsws(vector int a, vector int b)
{
  return __builtin_altivec_vaddsws(a, b);
}

static vector int __ATTRS_o_ai
vec_vaddsws(vector bool int a, vector int b)
{
  return __builtin_altivec_vaddsws((vector int)a, b);
}

static vector int __ATTRS_o_ai
vec_vaddsws(vector int a, vector bool int b)
{
  return __builtin_altivec_vaddsws(a, (vector int)b);
}

/* vec_vadduws */

static vector unsigned int __ATTRS_o_ai
vec_vadduws(vector unsigned int a, vector unsigned int b)
{
  return __builtin_altivec_vadduws(a, b);
}

static vector unsigned int __ATTRS_o_ai
vec_vadduws(vector bool int a, vector unsigned int b)
{
  return __builtin_altivec_vadduws((vector unsigned int)a, b);
}

static vector unsigned int __ATTRS_o_ai
vec_vadduws(vector unsigned int a, vector bool int b)
{
  return __builtin_altivec_vadduws(a, (vector unsigned int)b);
}

/* vec_and */

#define __builtin_altivec_vand vec_and

static vector signed char __ATTRS_o_ai
vec_and(vector signed char a, vector signed char b)
{
  return a & b;
}

static vector signed char __ATTRS_o_ai
vec_and(vector bool char a, vector signed char b)
{
  return (vector signed char)a & b;
}

static vector signed char __ATTRS_o_ai
vec_and(vector signed char a, vector bool char b)
{
  return a & (vector signed char)b;
}

static vector unsigned char __ATTRS_o_ai
vec_and(vector unsigned char a, vector unsigned char b)
{
  return a & b;
}

static vector unsigned char __ATTRS_o_ai
vec_and(vector bool char a, vector unsigned char b)
{
  return (vector unsigned char)a & b;
}

static vector unsigned char __ATTRS_o_ai
vec_and(vector unsigned char a, vector bool char b)
{
  return a & (vector unsigned char)b;
}

static vector bool char __ATTRS_o_ai
vec_and(vector bool char a, vector bool char b)
{
  return a & b;
}

static vector short __ATTRS_o_ai
vec_and(vector short a, vector short b)
{
  return a & b;
}

static vector short __ATTRS_o_ai
vec_and(vector bool short a, vector short b)
{
  return (vector short)a & b;
}

static vector short __ATTRS_o_ai
vec_and(vector short a, vector bool short b)
{
  return a & (vector short)b;
}

static vector unsigned short __ATTRS_o_ai
vec_and(vector unsigned short a, vector unsigned short b)
{
  return a & b;
}

static vector unsigned short __ATTRS_o_ai
vec_and(vector bool short a, vector unsigned short b)
{
  return (vector unsigned short)a & b;
}

static vector unsigned short __ATTRS_o_ai
vec_and(vector unsigned short a, vector bool short b)
{
  return a & (vector unsigned short)b;
}

static vector bool short __ATTRS_o_ai
vec_and(vector bool short a, vector bool short b)
{
  return a & b;
}

static vector int __ATTRS_o_ai
vec_and(vector int a, vector int b)
{
  return a & b;
}

static vector int __ATTRS_o_ai
vec_and(vector bool int a, vector int b)
{
  return (vector int)a & b;
}

static vector int __ATTRS_o_ai
vec_and(vector int a, vector bool int b)
{
  return a & (vector int)b;
}

static vector unsigned int __ATTRS_o_ai
vec_and(vector unsigned int a, vector unsigned int b)
{
  return a & b;
}

static vector unsigned int __ATTRS_o_ai
vec_and(vector bool int a, vector unsigned int b)
{
  return (vector unsigned int)a & b;
}

static vector unsigned int __ATTRS_o_ai
vec_and(vector unsigned int a, vector bool int b)
{
  return a & (vector unsigned int)b;
}

static vector bool int __ATTRS_o_ai
vec_and(vector bool int a, vector bool int b)
{
  return a & b;
}

static vector float __ATTRS_o_ai
vec_and(vector float a, vector float b)
{
  vector unsigned int res = (vector unsigned int)a & (vector unsigned int)b;
  return (vector float)res;
}

static vector float __ATTRS_o_ai
vec_and(vector bool int a, vector float b)
{
  vector unsigned int res = (vector unsigned int)a & (vector unsigned int)b;
  return (vector float)res;
}

static vector float __ATTRS_o_ai
vec_and(vector float a, vector bool int b)
{
  vector unsigned int res = (vector unsigned int)a & (vector unsigned int)b;
  return (vector float)res;
}

/* vec_vand */

static vector signed char __ATTRS_o_ai
vec_vand(vector signed char a, vector signed char b)
{
  return a & b;
}

static vector signed char __ATTRS_o_ai
vec_vand(vector bool char a, vector signed char b)
{
  return (vector signed char)a & b;
}

static vector signed char __ATTRS_o_ai
vec_vand(vector signed char a, vector bool char b)
{
  return a & (vector signed char)b;
}

static vector unsigned char __ATTRS_o_ai
vec_vand(vector unsigned char a, vector unsigned char b)
{
  return a & b;
}

static vector unsigned char __ATTRS_o_ai
vec_vand(vector bool char a, vector unsigned char b)
{
  return (vector unsigned char)a & b;
}

static vector unsigned char __ATTRS_o_ai
vec_vand(vector unsigned char a, vector bool char b)
{
  return a & (vector unsigned char)b;
}

static vector bool char __ATTRS_o_ai
vec_vand(vector bool char a, vector bool char b)
{
  return a & b;
}

static vector short __ATTRS_o_ai
vec_vand(vector short a, vector short b)
{
  return a & b;
}

static vector short __ATTRS_o_ai
vec_vand(vector bool short a, vector short b)
{
  return (vector short)a & b;
}

static vector short __ATTRS_o_ai
vec_vand(vector short a, vector bool short b)
{
  return a & (vector short)b;
}

static vector unsigned short __ATTRS_o_ai
vec_vand(vector unsigned short a, vector unsigned short b)
{
  return a & b;
}

static vector unsigned short __ATTRS_o_ai
vec_vand(vector bool short a, vector unsigned short b)
{
  return (vector unsigned short)a & b;
}

static vector unsigned short __ATTRS_o_ai
vec_vand(vector unsigned short a, vector bool short b)
{
  return a & (vector unsigned short)b;
}

static vector bool short __ATTRS_o_ai
vec_vand(vector bool short a, vector bool short b)
{
  return a & b;
}

static vector int __ATTRS_o_ai
vec_vand(vector int a, vector int b)
{
  return a & b;
}

static vector int __ATTRS_o_ai
vec_vand(vector bool int a, vector int b)
{
  return (vector int)a & b;
}

static vector int __ATTRS_o_ai
vec_vand(vector int a, vector bool int b)
{
  return a & (vector int)b;
}

static vector unsigned int __ATTRS_o_ai
vec_vand(vector unsigned int a, vector unsigned int b)
{
  return a & b;
}

static vector unsigned int __ATTRS_o_ai
vec_vand(vector bool int a, vector unsigned int b)
{
  return (vector unsigned int)a & b;
}

static vector unsigned int __ATTRS_o_ai
vec_vand(vector unsigned int a, vector bool int b)
{
  return a & (vector unsigned int)b;
}

static vector bool int __ATTRS_o_ai
vec_vand(vector bool int a, vector bool int b)
{
  return a & b;
}

static vector float __ATTRS_o_ai
vec_vand(vector float a, vector float b)
{
  vector unsigned int res = (vector unsigned int)a & (vector unsigned int)b;
  return (vector float)res;
}

static vector float __ATTRS_o_ai
vec_vand(vector bool int a, vector float b)
{
  vector unsigned int res = (vector unsigned int)a & (vector unsigned int)b;
  return (vector float)res;
}

static vector float __ATTRS_o_ai
vec_vand(vector float a, vector bool int b)
{
  vector unsigned int res = (vector unsigned int)a & (vector unsigned int)b;
  return (vector float)res;
}

/* vec_andc */

#define __builtin_altivec_vandc vec_andc

static vector signed char __ATTRS_o_ai
vec_andc(vector signed char a, vector signed char b)
{
  return a & ~b;
}

static vector signed char __ATTRS_o_ai
vec_andc(vector bool char a, vector signed char b)
{
  return (vector signed char)a & ~b;
}

static vector signed char __ATTRS_o_ai
vec_andc(vector signed char a, vector bool char b)
{
  return a & ~(vector signed char)b;
}

static vector unsigned char __ATTRS_o_ai
vec_andc(vector unsigned char a, vector unsigned char b)
{
  return a & ~b;
}

static vector unsigned char __ATTRS_o_ai
vec_andc(vector bool char a, vector unsigned char b)
{
  return (vector unsigned char)a & ~b;
}

static vector unsigned char __ATTRS_o_ai
vec_andc(vector unsigned char a, vector bool char b)
{
  return a & ~(vector unsigned char)b;
}

static vector bool char __ATTRS_o_ai
vec_andc(vector bool char a, vector bool char b)
{
  return a & ~b;
}

static vector short __ATTRS_o_ai
vec_andc(vector short a, vector short b)
{
  return a & ~b;
}

static vector short __ATTRS_o_ai
vec_andc(vector bool short a, vector short b)
{
  return (vector short)a & ~b;
}

static vector short __ATTRS_o_ai
vec_andc(vector short a, vector bool short b)
{
  return a & ~(vector short)b;
}

static vector unsigned short __ATTRS_o_ai
vec_andc(vector unsigned short a, vector unsigned short b)
{
  return a & ~b;
}

static vector unsigned short __ATTRS_o_ai
vec_andc(vector bool short a, vector unsigned short b)
{
  return (vector unsigned short)a & ~b;
}

static vector unsigned short __ATTRS_o_ai
vec_andc(vector unsigned short a, vector bool short b)
{
  return a & ~(vector unsigned short)b;
}

static vector bool short __ATTRS_o_ai
vec_andc(vector bool short a, vector bool short b)
{
  return a & ~b;
}

static vector int __ATTRS_o_ai
vec_andc(vector int a, vector int b)
{
  return a & ~b;
}

static vector int __ATTRS_o_ai
vec_andc(vector bool int a, vector int b)
{
  return (vector int)a & ~b;
}

static vector int __ATTRS_o_ai
vec_andc(vector int a, vector bool int b)
{
  return a & ~(vector int)b;
}

static vector unsigned int __ATTRS_o_ai
vec_andc(vector unsigned int a, vector unsigned int b)
{
  return a & ~b;
}

static vector unsigned int __ATTRS_o_ai
vec_andc(vector bool int a, vector unsigned int b)
{
  return (vector unsigned int)a & ~b;
}

static vector unsigned int __ATTRS_o_ai
vec_andc(vector unsigned int a, vector bool int b)
{
  return a & ~(vector unsigned int)b;
}

static vector bool int __ATTRS_o_ai
vec_andc(vector bool int a, vector bool int b)
{
  return a & ~b;
}

static vector float __ATTRS_o_ai
vec_andc(vector float a, vector float b)
{
  vector unsigned int res = (vector unsigned int)a & ~(vector unsigned int)b;
  return (vector float)res;
}

static vector float __ATTRS_o_ai
vec_andc(vector bool int a, vector float b)
{
  vector unsigned int res = (vector unsigned int)a & ~(vector unsigned int)b;
  return (vector float)res;
}

static vector float __ATTRS_o_ai
vec_andc(vector float a, vector bool int b)
{
  vector unsigned int res = (vector unsigned int)a & ~(vector unsigned int)b;
  return (vector float)res;
}

/* vec_vandc */

static vector signed char __ATTRS_o_ai
vec_vandc(vector signed char a, vector signed char b)
{
  return a & ~b;
}

static vector signed char __ATTRS_o_ai
vec_vandc(vector bool char a, vector signed char b)
{
  return (vector signed char)a & ~b;
}

static vector signed char __ATTRS_o_ai
vec_vandc(vector signed char a, vector bool char b)
{
  return a & ~(vector signed char)b;
}

static vector unsigned char __ATTRS_o_ai
vec_vandc(vector unsigned char a, vector unsigned char b)
{
  return a & ~b;
}

static vector unsigned char __ATTRS_o_ai
vec_vandc(vector bool char a, vector unsigned char b)
{
  return (vector unsigned char)a & ~b;
}

static vector unsigned char __ATTRS_o_ai
vec_vandc(vector unsigned char a, vector bool char b)
{
  return a & ~(vector unsigned char)b;
}

static vector bool char __ATTRS_o_ai
vec_vandc(vector bool char a, vector bool char b)
{
  return a & ~b;
}

static vector short __ATTRS_o_ai
vec_vandc(vector short a, vector short b)
{
  return a & ~b;
}

static vector short __ATTRS_o_ai
vec_vandc(vector bool short a, vector short b)
{
  return (vector short)a & ~b;
}

static vector short __ATTRS_o_ai
vec_vandc(vector short a, vector bool short b)
{
  return a & ~(vector short)b;
}

static vector unsigned short __ATTRS_o_ai
vec_vandc(vector unsigned short a, vector unsigned short b)
{
  return a & ~b;
}

static vector unsigned short __ATTRS_o_ai
vec_vandc(vector bool short a, vector unsigned short b)
{
  return (vector unsigned short)a & ~b;
}

static vector unsigned short __ATTRS_o_ai
vec_vandc(vector unsigned short a, vector bool short b)
{
  return a & ~(vector unsigned short)b;
}

static vector bool short __ATTRS_o_ai
vec_vandc(vector bool short a, vector bool short b)
{
  return a & ~b;
}

static vector int __ATTRS_o_ai
vec_vandc(vector int a, vector int b)
{
  return a & ~b;
}

static vector int __ATTRS_o_ai
vec_vandc(vector bool int a, vector int b)
{
  return (vector int)a & ~b;
}

static vector int __ATTRS_o_ai
vec_vandc(vector int a, vector bool int b)
{
  return a & ~(vector int)b;
}

static vector unsigned int __ATTRS_o_ai
vec_vandc(vector unsigned int a, vector unsigned int b)
{
  return a & ~b;
}

static vector unsigned int __ATTRS_o_ai
vec_vandc(vector bool int a, vector unsigned int b)
{
  return (vector unsigned int)a & ~b;
}

static vector unsigned int __ATTRS_o_ai
vec_vandc(vector unsigned int a, vector bool int b)
{
  return a & ~(vector unsigned int)b;
}

static vector bool int __ATTRS_o_ai
vec_vandc(vector bool int a, vector bool int b)
{
  return a & ~b;
}

static vector float __ATTRS_o_ai
vec_vandc(vector float a, vector float b)
{
  vector unsigned int res = (vector unsigned int)a & ~(vector unsigned int)b;
  return (vector float)res;
}

static vector float __ATTRS_o_ai
vec_vandc(vector bool int a, vector float b)
{
  vector unsigned int res = (vector unsigned int)a & ~(vector unsigned int)b;
  return (vector float)res;
}

static vector float __ATTRS_o_ai
vec_vandc(vector float a, vector bool int b)
{
  vector unsigned int res = (vector unsigned int)a & ~(vector unsigned int)b;
  return (vector float)res;
}

/* vec_avg */

static vector signed char __ATTRS_o_ai
vec_avg(vector signed char a, vector signed char b)
{
  return __builtin_altivec_vavgsb(a, b);
}

static vector unsigned char __ATTRS_o_ai
vec_avg(vector unsigned char a, vector unsigned char b)
{
  return __builtin_altivec_vavgub(a, b);
}

static vector short __ATTRS_o_ai
vec_avg(vector short a, vector short b)
{
  return __builtin_altivec_vavgsh(a, b);
}

static vector unsigned short __ATTRS_o_ai
vec_avg(vector unsigned short a, vector unsigned short b)
{
  return __builtin_altivec_vavguh(a, b);
}

static vector int __ATTRS_o_ai
vec_avg(vector int a, vector int b)
{
  return __builtin_altivec_vavgsw(a, b);
}

static vector unsigned int __ATTRS_o_ai
vec_avg(vector unsigned int a, vector unsigned int b)
{
  return __builtin_altivec_vavguw(a, b);
}

/* vec_vavgsb */

static vector signed char __attribute__((__always_inline__))
vec_vavgsb(vector signed char a, vector signed char b)
{
  return __builtin_altivec_vavgsb(a, b);
}

/* vec_vavgub */

static vector unsigned char __attribute__((__always_inline__))
vec_vavgub(vector unsigned char a, vector unsigned char b)
{
  return __builtin_altivec_vavgub(a, b);
}

/* vec_vavgsh */

static vector short __attribute__((__always_inline__))
vec_vavgsh(vector short a, vector short b)
{
  return __builtin_altivec_vavgsh(a, b);
}

/* vec_vavguh */

static vector unsigned short __attribute__((__always_inline__))
vec_vavguh(vector unsigned short a, vector unsigned short b)
{
  return __builtin_altivec_vavguh(a, b);
}

/* vec_vavgsw */

static vector int __attribute__((__always_inline__))
vec_vavgsw(vector int a, vector int b)
{
  return __builtin_altivec_vavgsw(a, b);
}

/* vec_vavguw */

static vector unsigned int __attribute__((__always_inline__))
vec_vavguw(vector unsigned int a, vector unsigned int b)
{
  return __builtin_altivec_vavguw(a, b);
}

/* vec_ceil */

static vector float __attribute__((__always_inline__))
vec_ceil(vector float a)
{
  return __builtin_altivec_vrfip(a);
}

/* vec_vrfip */

static vector float __attribute__((__always_inline__))
vec_vrfip(vector float a)
{
  return __builtin_altivec_vrfip(a);
}

/* vec_cmpb */

static vector int __attribute__((__always_inline__))
vec_cmpb(vector float a, vector float b)
{
  return __builtin_altivec_vcmpbfp(a, b);
}

/* vec_vcmpbfp */

static vector int __attribute__((__always_inline__))
vec_vcmpbfp(vector float a, vector float b)
{
  return __builtin_altivec_vcmpbfp(a, b);
}

/* vec_cmpeq */

static vector bool char __ATTRS_o_ai
vec_cmpeq(vector signed char a, vector signed char b)
{
  return (vector bool char)
    __builtin_altivec_vcmpequb((vector char)a, (vector char)b);
}

static vector bool char __ATTRS_o_ai
vec_cmpeq(vector unsigned char a, vector unsigned char b)
{
  return (vector bool char)
    __builtin_altivec_vcmpequb((vector char)a, (vector char)b);
}

static vector bool short __ATTRS_o_ai
vec_cmpeq(vector short a, vector short b)
{
  return (vector bool short)__builtin_altivec_vcmpequh(a, b);
}

static vector bool short __ATTRS_o_ai
vec_cmpeq(vector unsigned short a, vector unsigned short b)
{
  return (vector bool short)
    __builtin_altivec_vcmpequh((vector short)a, (vector short)b);
}

static vector bool int __ATTRS_o_ai
vec_cmpeq(vector int a, vector int b)
{
  return (vector bool int)__builtin_altivec_vcmpequw(a, b);
}

static vector bool int __ATTRS_o_ai
vec_cmpeq(vector unsigned int a, vector unsigned int b)
{
  return (vector bool int)
    __builtin_altivec_vcmpequw((vector int)a, (vector int)b);
}

static vector bool int __ATTRS_o_ai
vec_cmpeq(vector float a, vector float b)
{
  return (vector bool int)__builtin_altivec_vcmpeqfp(a, b);
}

/* vec_cmpge */

static vector bool int __attribute__((__always_inline__))
vec_cmpge(vector float a, vector float b)
{
  return (vector bool int)__builtin_altivec_vcmpgefp(a, b);
}

/* vec_vcmpgefp */

static vector bool int __attribute__((__always_inline__))
vec_vcmpgefp(vector float a, vector float b)
{
  return (vector bool int)__builtin_altivec_vcmpgefp(a, b);
}

/* vec_cmpgt */

static vector bool char __ATTRS_o_ai
vec_cmpgt(vector signed char a, vector signed char b)
{
  return (vector bool char)__builtin_altivec_vcmpgtsb(a, b);
}

static vector bool char __ATTRS_o_ai
vec_cmpgt(vector unsigned char a, vector unsigned char b)
{
  return (vector bool char)__builtin_altivec_vcmpgtub(a, b);
}

static vector bool short __ATTRS_o_ai
vec_cmpgt(vector short a, vector short b)
{
  return (vector bool short)__builtin_altivec_vcmpgtsh(a, b);
}

static vector bool short __ATTRS_o_ai
vec_cmpgt(vector unsigned short a, vector unsigned short b)
{
  return (vector bool short)__builtin_altivec_vcmpgtuh(a, b);
}

static vector bool int __ATTRS_o_ai
vec_cmpgt(vector int a, vector int b)
{
  return (vector bool int)__builtin_altivec_vcmpgtsw(a, b);
}

static vector bool int __ATTRS_o_ai
vec_cmpgt(vector unsigned int a, vector unsigned int b)
{
  return (vector bool int)__builtin_altivec_vcmpgtuw(a, b);
}

static vector bool int __ATTRS_o_ai
vec_cmpgt(vector float a, vector float b)
{
  return (vector bool int)__builtin_altivec_vcmpgtfp(a, b);
}

/* vec_vcmpgtsb */

static vector bool char __attribute__((__always_inline__))
vec_vcmpgtsb(vector signed char a, vector signed char b)
{
  return (vector bool char)__builtin_altivec_vcmpgtsb(a, b);
}

/* vec_vcmpgtub */

static vector bool char __attribute__((__always_inline__))
vec_vcmpgtub(vector unsigned char a, vector unsigned char b)
{
  return (vector bool char)__builtin_altivec_vcmpgtub(a, b);
}

/* vec_vcmpgtsh */

static vector bool short __attribute__((__always_inline__))
vec_vcmpgtsh(vector short a, vector short b)
{
  return (vector bool short)__builtin_altivec_vcmpgtsh(a, b);
}

/* vec_vcmpgtuh */

static vector bool short __attribute__((__always_inline__))
vec_vcmpgtuh(vector unsigned short a, vector unsigned short b)
{
  return (vector bool short)__builtin_altivec_vcmpgtuh(a, b);
}

/* vec_vcmpgtsw */

static vector bool int __attribute__((__always_inline__))
vec_vcmpgtsw(vector int a, vector int b)
{
  return (vector bool int)__builtin_altivec_vcmpgtsw(a, b);
}

/* vec_vcmpgtuw */

static vector bool int __attribute__((__always_inline__))
vec_vcmpgtuw(vector unsigned int a, vector unsigned int b)
{
  return (vector bool int)__builtin_altivec_vcmpgtuw(a, b);
}

/* vec_vcmpgtfp */

static vector bool int __attribute__((__always_inline__))
vec_vcmpgtfp(vector float a, vector float b)
{
  return (vector bool int)__builtin_altivec_vcmpgtfp(a, b);
}

/* vec_cmple */

static vector bool int __attribute__((__always_inline__))
vec_cmple(vector float a, vector float b)
{
  return (vector bool int)__builtin_altivec_vcmpgefp(b, a);
}

/* vec_cmplt */

static vector bool char __ATTRS_o_ai
vec_cmplt(vector signed char a, vector signed char b)
{
  return (vector bool char)__builtin_altivec_vcmpgtsb(b, a);
}

static vector bool char __ATTRS_o_ai
vec_cmplt(vector unsigned char a, vector unsigned char b)
{
  return (vector bool char)__builtin_altivec_vcmpgtub(b, a);
}

static vector bool short __ATTRS_o_ai
vec_cmplt(vector short a, vector short b)
{
  return (vector bool short)__builtin_altivec_vcmpgtsh(b, a);
}

static vector bool short __ATTRS_o_ai
vec_cmplt(vector unsigned short a, vector unsigned short b)
{
  return (vector bool short)__builtin_altivec_vcmpgtuh(b, a);
}

static vector bool int __ATTRS_o_ai
vec_cmplt(vector int a, vector int b)
{
  return (vector bool int)__builtin_altivec_vcmpgtsw(b, a);
}

static vector bool int __ATTRS_o_ai
vec_cmplt(vector unsigned int a, vector unsigned int b)
{
  return (vector bool int)__builtin_altivec_vcmpgtuw(b, a);
}

static vector bool int __ATTRS_o_ai
vec_cmplt(vector float a, vector float b)
{
  return (vector bool int)__builtin_altivec_vcmpgtfp(b, a);
}

/* vec_ctf */

static vector float __ATTRS_o_ai
vec_ctf(vector int a, int b)
{
  return __builtin_altivec_vcfsx(a, b);
}

static vector float __ATTRS_o_ai
vec_ctf(vector unsigned int a, int b)
{
  return __builtin_altivec_vcfux((vector int)a, b);
}

/* vec_vcfsx */

static vector float __attribute__((__always_inline__))
vec_vcfsx(vector int a, int b)
{
  return __builtin_altivec_vcfsx(a, b);
}

/* vec_vcfux */

static vector float __attribute__((__always_inline__))
vec_vcfux(vector unsigned int a, int b)
{
  return __builtin_altivec_vcfux((vector int)a, b);
}

/* vec_cts */

static vector int __attribute__((__always_inline__))
vec_cts(vector float a, int b)
{
  return __builtin_altivec_vctsxs(a, b);
}

/* vec_vctsxs */

static vector int __attribute__((__always_inline__))
vec_vctsxs(vector float a, int b)
{
  return __builtin_altivec_vctsxs(a, b);
}

/* vec_ctu */

static vector unsigned int __attribute__((__always_inline__))
vec_ctu(vector float a, int b)
{
  return __builtin_altivec_vctuxs(a, b);
}

/* vec_vctuxs */

static vector unsigned int __attribute__((__always_inline__))
vec_vctuxs(vector float a, int b)
{
  return __builtin_altivec_vctuxs(a, b);
}

/* vec_dss */

static void __attribute__((__always_inline__))
vec_dss(int a)
{
  __builtin_altivec_dss(a);
}

/* vec_dssall */

static void __attribute__((__always_inline__))
vec_dssall(void)
{
  __builtin_altivec_dssall();
}

/* vec_dst */

static void __attribute__((__always_inline__))
vec_dst(const void *a, int b, int c)
{
  __builtin_altivec_dst(a, b, c);
}

/* vec_dstst */

static void __attribute__((__always_inline__))
vec_dstst(const void *a, int b, int c)
{
  __builtin_altivec_dstst(a, b, c);
}

/* vec_dststt */

static void __attribute__((__always_inline__))
vec_dststt(const void *a, int b, int c)
{
  __builtin_altivec_dststt(a, b, c);
}

/* vec_dstt */

static void __attribute__((__always_inline__))
vec_dstt(const void *a, int b, int c)
{
  __builtin_altivec_dstt(a, b, c);
}

/* vec_expte */

static vector float __attribute__((__always_inline__))
vec_expte(vector float a)
{
  return __builtin_altivec_vexptefp(a);
}

/* vec_vexptefp */

static vector float __attribute__((__always_inline__))
vec_vexptefp(vector float a)
{
  return __builtin_altivec_vexptefp(a);
}

/* vec_floor */

static vector float __attribute__((__always_inline__))
vec_floor(vector float a)
{
  return __builtin_altivec_vrfim(a);
}

/* vec_vrfim */

static vector float __attribute__((__always_inline__))
vec_vrfim(vector float a)
{
  return __builtin_altivec_vrfim(a);
}

/* vec_ld */

static vector signed char __ATTRS_o_ai
vec_ld(int a, const vector signed char *b)
{
  return (vector signed char)__builtin_altivec_lvx(a, b);
}

static vector signed char __ATTRS_o_ai
vec_ld(int a, const signed char *b)
{
  return (vector signed char)__builtin_altivec_lvx(a, b);
}

static vector unsigned char __ATTRS_o_ai
vec_ld(int a, const vector unsigned char *b)
{
  return (vector unsigned char)__builtin_altivec_lvx(a, b);
}

static vector unsigned char __ATTRS_o_ai
vec_ld(int a, const unsigned char *b)
{
  return (vector unsigned char)__builtin_altivec_lvx(a, b);
}

static vector bool char __ATTRS_o_ai
vec_ld(int a, const vector bool char *b)
{
  return (vector bool char)__builtin_altivec_lvx(a, b);
}

static vector short __ATTRS_o_ai
vec_ld(int a, const vector short *b)
{
  return (vector short)__builtin_altivec_lvx(a, b);
}

static vector short __ATTRS_o_ai
vec_ld(int a, const short *b)
{
  return (vector short)__builtin_altivec_lvx(a, b);
}

static vector unsigned short __ATTRS_o_ai
vec_ld(int a, const vector unsigned short *b)
{
  return (vector unsigned short)__builtin_altivec_lvx(a, b);
}

static vector unsigned short __ATTRS_o_ai
vec_ld(int a, const unsigned short *b)
{
  return (vector unsigned short)__builtin_altivec_lvx(a, b);
}

static vector bool short __ATTRS_o_ai
vec_ld(int a, const vector bool short *b)
{
  return (vector bool short)__builtin_altivec_lvx(a, b);
}

static vector pixel __ATTRS_o_ai
vec_ld(int a, const vector pixel *b)
{
  return (vector pixel)__builtin_altivec_lvx(a, b);
}

static vector int __ATTRS_o_ai
vec_ld(int a, const vector int *b)
{
  return (vector int)__builtin_altivec_lvx(a, b);
}

static vector int __ATTRS_o_ai
vec_ld(int a, const int *b)
{
  return (vector int)__builtin_altivec_lvx(a, b);
}

static vector unsigned int __ATTRS_o_ai
vec_ld(int a, const vector unsigned int *b)
{
  return (vector unsigned int)__builtin_altivec_lvx(a, b);
}

static vector unsigned int __ATTRS_o_ai
vec_ld(int a, const unsigned int *b)
{
  return (vector unsigned int)__builtin_altivec_lvx(a, b);
}

static vector bool int __ATTRS_o_ai
vec_ld(int a, const vector bool int *b)
{
  return (vector bool int)__builtin_altivec_lvx(a, b);
}

static vector float __ATTRS_o_ai
vec_ld(int a, const vector float *b)
{
  return (vector float)__builtin_altivec_lvx(a, b);
}

static vector float __ATTRS_o_ai
vec_ld(int a, const float *b)
{
  return (vector float)__builtin_altivec_lvx(a, b);
}

/* vec_lvx */

static vector signed char __ATTRS_o_ai
vec_lvx(int a, const vector signed char *b)
{
  return (vector signed char)__builtin_altivec_lvx(a, b);
}

static vector signed char __ATTRS_o_ai
vec_lvx(int a, const signed char *b)
{
  return (vector signed char)__builtin_altivec_lvx(a, b);
}

static vector unsigned char __ATTRS_o_ai
vec_lvx(int a, const vector unsigned char *b)
{
  return (vector unsigned char)__builtin_altivec_lvx(a, b);
}

static vector unsigned char __ATTRS_o_ai
vec_lvx(int a, const unsigned char *b)
{
  return (vector unsigned char)__builtin_altivec_lvx(a, b);
}

static vector bool char __ATTRS_o_ai
vec_lvx(int a, const vector bool char *b)
{
  return (vector bool char)__builtin_altivec_lvx(a, b);
}

static vector short __ATTRS_o_ai
vec_lvx(int a, const vector short *b)
{
  return (vector short)__builtin_altivec_lvx(a, b);
}

static vector short __ATTRS_o_ai
vec_lvx(int a, const short *b)
{
  return (vector short)__builtin_altivec_lvx(a, b);
}

static vector unsigned short __ATTRS_o_ai
vec_lvx(int a, const vector unsigned short *b)
{
  return (vector unsigned short)__builtin_altivec_lvx(a, b);
}

static vector unsigned short __ATTRS_o_ai
vec_lvx(int a, const unsigned short *b)
{
  return (vector unsigned short)__builtin_altivec_lvx(a, b);
}

static vector bool short __ATTRS_o_ai
vec_lvx(int a, const vector bool short *b)
{
  return (vector bool short)__builtin_altivec_lvx(a, b);
}

static vector pixel __ATTRS_o_ai
vec_lvx(int a, const vector pixel *b)
{
  return (vector pixel)__builtin_altivec_lvx(a, b);
}

static vector int __ATTRS_o_ai
vec_lvx(int a, const vector int *b)
{
  return (vector int)__builtin_altivec_lvx(a, b);
}

static vector int __ATTRS_o_ai
vec_lvx(int a, const int *b)
{
  return (vector int)__builtin_altivec_lvx(a, b);
}

static vector unsigned int __ATTRS_o_ai
vec_lvx(int a, const vector unsigned int *b)
{
  return (vector unsigned int)__builtin_altivec_lvx(a, b);
}

static vector unsigned int __ATTRS_o_ai
vec_lvx(int a, const unsigned int *b)
{
  return (vector unsigned int)__builtin_altivec_lvx(a, b);
}

static vector bool int __ATTRS_o_ai
vec_lvx(int a, const vector bool int *b)
{
  return (vector bool int)__builtin_altivec_lvx(a, b);
}

static vector float __ATTRS_o_ai
vec_lvx(int a, const vector float *b)
{
  return (vector float)__builtin_altivec_lvx(a, b);
}

static vector float __ATTRS_o_ai
vec_lvx(int a, const float *b)
{
  return (vector float)__builtin_altivec_lvx(a, b);
}

/* vec_lde */

static vector signed char __ATTRS_o_ai
vec_lde(int a, const vector signed char *b)
{
  return (vector signed char)__builtin_altivec_lvebx(a, b);
}

static vector unsigned char __ATTRS_o_ai
vec_lde(int a, const vector unsigned char *b)
{
  return (vector unsigned char)__builtin_altivec_lvebx(a, b);
}

static vector short __ATTRS_o_ai
vec_lde(int a, const vector short *b)
{
  return (vector short)__builtin_altivec_lvehx(a, b);
}

static vector unsigned short __ATTRS_o_ai
vec_lde(int a, const vector unsigned short *b)
{
  return (vector unsigned short)__builtin_altivec_lvehx(a, b);
}

static vector int __ATTRS_o_ai
vec_lde(int a, const vector int *b)
{
  return (vector int)__builtin_altivec_lvewx(a, b);
}

static vector unsigned int __ATTRS_o_ai
vec_lde(int a, const vector unsigned int *b)
{
  return (vector unsigned int)__builtin_altivec_lvewx(a, b);
}

static vector float __ATTRS_o_ai
vec_lde(int a, const vector float *b)
{
  return (vector float)__builtin_altivec_lvewx(a, b);
}

/* vec_lvebx */

static vector signed char __ATTRS_o_ai
vec_lvebx(int a, const vector signed char *b)
{
  return (vector signed char)__builtin_altivec_lvebx(a, b);
}

static vector unsigned char __ATTRS_o_ai
vec_lvebx(int a, const vector unsigned char *b)
{
  return (vector unsigned char)__builtin_altivec_lvebx(a, b);
}

/* vec_lvehx */

static vector short __ATTRS_o_ai
vec_lvehx(int a, const vector short *b)
{
  return (vector short)__builtin_altivec_lvehx(a, b);
}

static vector unsigned short __ATTRS_o_ai
vec_lvehx(int a, const vector unsigned short *b)
{
  return (vector unsigned short)__builtin_altivec_lvehx(a, b);
}

/* vec_lvewx */

static vector int __ATTRS_o_ai
vec_lvewx(int a, const vector int *b)
{
  return (vector int)__builtin_altivec_lvewx(a, b);
}

static vector unsigned int __ATTRS_o_ai
vec_lvewx(int a, const vector unsigned int *b)
{
  return (vector unsigned int)__builtin_altivec_lvewx(a, b);
}

static vector float __ATTRS_o_ai
vec_lvewx(int a, const vector float *b)
{
  return (vector float)__builtin_altivec_lvewx(a, b);
}

/* vec_ldl */

static vector signed char __ATTRS_o_ai
vec_ldl(int a, const vector signed char *b)
{
  return (vector signed char)__builtin_altivec_lvxl(a, b);
}

static vector signed char __ATTRS_o_ai
vec_ldl(int a, const signed char *b)
{
  return (vector signed char)__builtin_altivec_lvxl(a, b);
}

static vector unsigned char __ATTRS_o_ai
vec_ldl(int a, const vector unsigned char *b)
{
  return (vector unsigned char)__builtin_altivec_lvxl(a, b);
}

static vector unsigned char __ATTRS_o_ai
vec_ldl(int a, const unsigned char *b)
{
  return (vector unsigned char)__builtin_altivec_lvxl(a, b);
}

static vector bool char __ATTRS_o_ai
vec_ldl(int a, const vector bool char *b)
{
  return (vector bool char)__builtin_altivec_lvxl(a, b);
}

static vector short __ATTRS_o_ai
vec_ldl(int a, const vector short *b)
{
  return (vector short)__builtin_altivec_lvxl(a, b);
}

static vector short __ATTRS_o_ai
vec_ldl(int a, const short *b)
{
  return (vector short)__builtin_altivec_lvxl(a, b);
}

static vector unsigned short __ATTRS_o_ai
vec_ldl(int a, const vector unsigned short *b)
{
  return (vector unsigned short)__builtin_altivec_lvxl(a, b);
}

static vector unsigned short __ATTRS_o_ai
vec_ldl(int a, const unsigned short *b)
{
  return (vector unsigned short)__builtin_altivec_lvxl(a, b);
}

static vector bool short __ATTRS_o_ai
vec_ldl(int a, const vector bool short *b)
{
  return (vector bool short)__builtin_altivec_lvxl(a, b);
}

static vector pixel __ATTRS_o_ai
vec_ldl(int a, const vector pixel *b)
{
  return (vector pixel short)__builtin_altivec_lvxl(a, b);
}

static vector int __ATTRS_o_ai
vec_ldl(int a, const vector int *b)
{
  return (vector int)__builtin_altivec_lvxl(a, b);
}

static vector int __ATTRS_o_ai
vec_ldl(int a, const int *b)
{
  return (vector int)__builtin_altivec_lvxl(a, b);
}

static vector unsigned int __ATTRS_o_ai
vec_ldl(int a, const vector unsigned int *b)
{
  return (vector unsigned int)__builtin_altivec_lvxl(a, b);
}

static vector unsigned int __ATTRS_o_ai
vec_ldl(int a, const unsigned int *b)
{
  return (vector unsigned int)__builtin_altivec_lvxl(a, b);
}

static vector bool int __ATTRS_o_ai
vec_ldl(int a, const vector bool int *b)
{
  return (vector bool int)__builtin_altivec_lvxl(a, b);
}

static vector float __ATTRS_o_ai
vec_ldl(int a, const vector float *b)
{
  return (vector float)__builtin_altivec_lvxl(a, b);
}

static vector float __ATTRS_o_ai
vec_ldl(int a, const float *b)
{
  return (vector float)__builtin_altivec_lvxl(a, b);
}

/* vec_lvxl */

static vector signed char __ATTRS_o_ai
vec_lvxl(int a, const vector signed char *b)
{
  return (vector signed char)__builtin_altivec_lvxl(a, b);
}

static vector signed char __ATTRS_o_ai
vec_lvxl(int a, const signed char *b)
{
  return (vector signed char)__builtin_altivec_lvxl(a, b);
}

static vector unsigned char __ATTRS_o_ai
vec_lvxl(int a, const vector unsigned char *b)
{
  return (vector unsigned char)__builtin_altivec_lvxl(a, b);
}

static vector unsigned char __ATTRS_o_ai
vec_lvxl(int a, const unsigned char *b)
{
  return (vector unsigned char)__builtin_altivec_lvxl(a, b);
}

static vector bool char __ATTRS_o_ai
vec_lvxl(int a, const vector bool char *b)
{
  return (vector bool char)__builtin_altivec_lvxl(a, b);
}

static vector short __ATTRS_o_ai
vec_lvxl(int a, const vector short *b)
{
  return (vector short)__builtin_altivec_lvxl(a, b);
}

static vector short __ATTRS_o_ai
vec_lvxl(int a, const short *b)
{
  return (vector short)__builtin_altivec_lvxl(a, b);
}

static vector unsigned short __ATTRS_o_ai
vec_lvxl(int a, const vector unsigned short *b)
{
  return (vector unsigned short)__builtin_altivec_lvxl(a, b);
}

static vector unsigned short __ATTRS_o_ai
vec_lvxl(int a, const unsigned short *b)
{
  return (vector unsigned short)__builtin_altivec_lvxl(a, b);
}

static vector bool short __ATTRS_o_ai
vec_lvxl(int a, const vector bool short *b)
{
  return (vector bool short)__builtin_altivec_lvxl(a, b);
}

static vector pixel __ATTRS_o_ai
vec_lvxl(int a, const vector pixel *b)
{
  return (vector pixel)__builtin_altivec_lvxl(a, b);
}

static vector int __ATTRS_o_ai
vec_lvxl(int a, const vector int *b)
{
  return (vector int)__builtin_altivec_lvxl(a, b);
}

static vector int __ATTRS_o_ai
vec_lvxl(int a, const int *b)
{
  return (vector int)__builtin_altivec_lvxl(a, b);
}

static vector unsigned int __ATTRS_o_ai
vec_lvxl(int a, const vector unsigned int *b)
{
  return (vector unsigned int)__builtin_altivec_lvxl(a, b);
}

static vector unsigned int __ATTRS_o_ai
vec_lvxl(int a, const unsigned int *b)
{
  return (vector unsigned int)__builtin_altivec_lvxl(a, b);
}

static vector bool int __ATTRS_o_ai
vec_lvxl(int a, const vector bool int *b)
{
  return (vector bool int)__builtin_altivec_lvxl(a, b);
}

static vector float __ATTRS_o_ai
vec_lvxl(int a, const vector float *b)
{
  return (vector float)__builtin_altivec_lvxl(a, b);
}

static vector float __ATTRS_o_ai
vec_lvxl(int a, const float *b)
{
  return (vector float)__builtin_altivec_lvxl(a, b);
}

/* vec_loge */

static vector float __attribute__((__always_inline__))
vec_loge(vector float a)
{
  return __builtin_altivec_vlogefp(a);
}

/* vec_vlogefp */

static vector float __attribute__((__always_inline__))
vec_vlogefp(vector float a)
{
  return __builtin_altivec_vlogefp(a);
}

/* vec_lvsl */

static vector unsigned char __ATTRS_o_ai
vec_lvsl(int a, const signed char *b)
{
  return (vector unsigned char)__builtin_altivec_lvsl(a, b);
}

static vector unsigned char __ATTRS_o_ai
vec_lvsl(int a, const unsigned char *b)
{
  return (vector unsigned char)__builtin_altivec_lvsl(a, b);
}

static vector unsigned char __ATTRS_o_ai
vec_lvsl(int a, const short *b)
{
  return (vector unsigned char)__builtin_altivec_lvsl(a, b);
}

static vector unsigned char __ATTRS_o_ai
vec_lvsl(int a, const unsigned short *b)
{
  return (vector unsigned char)__builtin_altivec_lvsl(a, b);
}

static vector unsigned char __ATTRS_o_ai
vec_lvsl(int a, const int *b)
{
  return (vector unsigned char)__builtin_altivec_lvsl(a, b);
}

static vector unsigned char __ATTRS_o_ai
vec_lvsl(int a, const unsigned int *b)
{
  return (vector unsigned char)__builtin_altivec_lvsl(a, b);
}

static vector unsigned char __ATTRS_o_ai
vec_lvsl(int a, const float *b)
{
  return (vector unsigned char)__builtin_altivec_lvsl(a, b);
}

/* vec_lvsr */

static vector unsigned char __ATTRS_o_ai
vec_lvsr(int a, const signed char *b)
{
  return (vector unsigned char)__builtin_altivec_lvsr(a, b);
}

static vector unsigned char __ATTRS_o_ai
vec_lvsr(int a, const unsigned char *b)
{
  return (vector unsigned char)__builtin_altivec_lvsr(a, b);
}

static vector unsigned char __ATTRS_o_ai
vec_lvsr(int a, const short *b)
{
  return (vector unsigned char)__builtin_altivec_lvsr(a, b);
}

static vector unsigned char __ATTRS_o_ai
vec_lvsr(int a, const unsigned short *b)
{
  return (vector unsigned char)__builtin_altivec_lvsr(a, b);
}

static vector unsigned char __ATTRS_o_ai
vec_lvsr(int a, const int *b)
{
  return (vector unsigned char)__builtin_altivec_lvsr(a, b);
}

static vector unsigned char __ATTRS_o_ai
vec_lvsr(int a, const unsigned int *b)
{
  return (vector unsigned char)__builtin_altivec_lvsr(a, b);
}

static vector unsigned char __ATTRS_o_ai
vec_lvsr(int a, const float *b)
{
  return (vector unsigned char)__builtin_altivec_lvsr(a, b);
}

/* vec_madd */

static vector float __attribute__((__always_inline__))
vec_madd(vector float a, vector float b, vector float c)
{
  return __builtin_altivec_vmaddfp(a, b, c);
}

/* vec_vmaddfp */

static vector float __attribute__((__always_inline__))
vec_vmaddfp(vector float a, vector float b, vector float c)
{
  return __builtin_altivec_vmaddfp(a, b, c);
}

/* vec_madds */

static vector signed short __attribute__((__always_inline__))
vec_madds(vector signed short a, vector signed short b, vector signed short c)
{
  return __builtin_altivec_vmhaddshs(a, b, c);
}

/* vec_vmhaddshs */
static vector signed short __attribute__((__always_inline__))
vec_vmhaddshs(vector signed short a,
              vector signed short b, 
              vector signed short c)
{
  return __builtin_altivec_vmhaddshs(a, b, c);
}

/* vec_max */

static vector signed char __ATTRS_o_ai
vec_max(vector signed char a, vector signed char b)
{
  return __builtin_altivec_vmaxsb(a, b);
}

static vector signed char __ATTRS_o_ai
vec_max(vector bool char a, vector signed char b)
{
  return __builtin_altivec_vmaxsb((vector signed char)a, b);
}

static vector signed char __ATTRS_o_ai
vec_max(vector signed char a, vector bool char b)
{
  return __builtin_altivec_vmaxsb(a, (vector signed char)b);
}

static vector unsigned char __ATTRS_o_ai
vec_max(vector unsigned char a, vector unsigned char b)
{
  return __builtin_altivec_vmaxub(a, b);
}

static vector unsigned char __ATTRS_o_ai
vec_max(vector bool char a, vector unsigned char b)
{
  return __builtin_altivec_vmaxub((vector unsigned char)a, b);
}

static vector unsigned char __ATTRS_o_ai
vec_max(vector unsigned char a, vector bool char b)
{
  return __builtin_altivec_vmaxub(a, (vector unsigned char)b);
}

static vector short __ATTRS_o_ai
vec_max(vector short a, vector short b)
{
  return __builtin_altivec_vmaxsh(a, b);
}

static vector short __ATTRS_o_ai
vec_max(vector bool short a, vector short b)
{
  return __builtin_altivec_vmaxsh((vector short)a, b);
}

static vector short __ATTRS_o_ai
vec_max(vector short a, vector bool short b)
{
  return __builtin_altivec_vmaxsh(a, (vector short)b);
}

static vector unsigned short __ATTRS_o_ai
vec_max(vector unsigned short a, vector unsigned short b)
{
  return __builtin_altivec_vmaxuh(a, b);
}

static vector unsigned short __ATTRS_o_ai
vec_max(vector bool short a, vector unsigned short b)
{
  return __builtin_altivec_vmaxuh((vector unsigned short)a, b);
}

static vector unsigned short __ATTRS_o_ai
vec_max(vector unsigned short a, vector bool short b)
{
  return __builtin_altivec_vmaxuh(a, (vector unsigned short)b);
}

static vector int __ATTRS_o_ai
vec_max(vector int a, vector int b)
{
  return __builtin_altivec_vmaxsw(a, b);
}

static vector int __ATTRS_o_ai
vec_max(vector bool int a, vector int b)
{
  return __builtin_altivec_vmaxsw((vector int)a, b);
}

static vector int __ATTRS_o_ai
vec_max(vector int a, vector bool int b)
{
  return __builtin_altivec_vmaxsw(a, (vector int)b);
}

static vector unsigned int __ATTRS_o_ai
vec_max(vector unsigned int a, vector unsigned int b)
{
  return __builtin_altivec_vmaxuw(a, b);
}

static vector unsigned int __ATTRS_o_ai
vec_max(vector bool int a, vector unsigned int b)
{
  return __builtin_altivec_vmaxuw((vector unsigned int)a, b);
}

static vector unsigned int __ATTRS_o_ai
vec_max(vector unsigned int a, vector bool int b)
{
  return __builtin_altivec_vmaxuw(a, (vector unsigned int)b);
}

static vector float __ATTRS_o_ai
vec_max(vector float a, vector float b)
{
  return __builtin_altivec_vmaxfp(a, b);
}

/* vec_vmaxsb */

static vector signed char __ATTRS_o_ai
vec_vmaxsb(vector signed char a, vector signed char b)
{
  return __builtin_altivec_vmaxsb(a, b);
}

static vector signed char __ATTRS_o_ai
vec_vmaxsb(vector bool char a, vector signed char b)
{
  return __builtin_altivec_vmaxsb((vector signed char)a, b);
}

static vector signed char __ATTRS_o_ai
vec_vmaxsb(vector signed char a, vector bool char b)
{
  return __builtin_altivec_vmaxsb(a, (vector signed char)b);
}

/* vec_vmaxub */

static vector unsigned char __ATTRS_o_ai
vec_vmaxub(vector unsigned char a, vector unsigned char b)
{
  return __builtin_altivec_vmaxub(a, b);
}

static vector unsigned char __ATTRS_o_ai
vec_vmaxub(vector bool char a, vector unsigned char b)
{
  return __builtin_altivec_vmaxub((vector unsigned char)a, b);
}

static vector unsigned char __ATTRS_o_ai
vec_vmaxub(vector unsigned char a, vector bool char b)
{
  return __builtin_altivec_vmaxub(a, (vector unsigned char)b);
}

/* vec_vmaxsh */

static vector short __ATTRS_o_ai
vec_vmaxsh(vector short a, vector short b)
{
  return __builtin_altivec_vmaxsh(a, b);
}

static vector short __ATTRS_o_ai
vec_vmaxsh(vector bool short a, vector short b)
{
  return __builtin_altivec_vmaxsh((vector short)a, b);
}

static vector short __ATTRS_o_ai
vec_vmaxsh(vector short a, vector bool short b)
{
  return __builtin_altivec_vmaxsh(a, (vector short)b);
}

/* vec_vmaxuh */

static vector unsigned short __ATTRS_o_ai
vec_vmaxuh(vector unsigned short a, vector unsigned short b)
{
  return __builtin_altivec_vmaxuh(a, b);
}

static vector unsigned short __ATTRS_o_ai
vec_vmaxuh(vector bool short a, vector unsigned short b)
{
  return __builtin_altivec_vmaxuh((vector unsigned short)a, b);
}

static vector unsigned short __ATTRS_o_ai
vec_vmaxuh(vector unsigned short a, vector bool short b)
{
  return __builtin_altivec_vmaxuh(a, (vector unsigned short)b);
}

/* vec_vmaxsw */

static vector int __ATTRS_o_ai
vec_vmaxsw(vector int a, vector int b)
{
  return __builtin_altivec_vmaxsw(a, b);
}

static vector int __ATTRS_o_ai
vec_vmaxsw(vector bool int a, vector int b)
{
  return __builtin_altivec_vmaxsw((vector int)a, b);
}

static vector int __ATTRS_o_ai
vec_vmaxsw(vector int a, vector bool int b)
{
  return __builtin_altivec_vmaxsw(a, (vector int)b);
}

/* vec_vmaxuw */

static vector unsigned int __ATTRS_o_ai
vec_vmaxuw(vector unsigned int a, vector unsigned int b)
{
  return __builtin_altivec_vmaxuw(a, b);
}

static vector unsigned int __ATTRS_o_ai
vec_vmaxuw(vector bool int a, vector unsigned int b)
{
  return __builtin_altivec_vmaxuw((vector unsigned int)a, b);
}

static vector unsigned int __ATTRS_o_ai
vec_vmaxuw(vector unsigned int a, vector bool int b)
{
  return __builtin_altivec_vmaxuw(a, (vector unsigned int)b);
}

/* vec_vmaxfp */

static vector float __attribute__((__always_inline__))
vec_vmaxfp(vector float a, vector float b)
{
  return __builtin_altivec_vmaxfp(a, b);
}

/* vec_mergeh */

static vector signed char __ATTRS_o_ai
vec_mergeh(vector signed char a, vector signed char b)
{
  return vec_perm(a, b, (vector unsigned char)
    (0x00, 0x10, 0x01, 0x11, 0x02, 0x12, 0x03, 0x13, 
     0x04, 0x14, 0x05, 0x15, 0x06, 0x16, 0x07, 0x17));
}

static vector unsigned char __ATTRS_o_ai
vec_mergeh(vector unsigned char a, vector unsigned char b)
{
  return vec_perm(a, b, (vector unsigned char)
    (0x00, 0x10, 0x01, 0x11, 0x02, 0x12, 0x03, 0x13, 
     0x04, 0x14, 0x05, 0x15, 0x06, 0x16, 0x07, 0x17));
}

static vector bool char __ATTRS_o_ai
vec_mergeh(vector bool char a, vector bool char b)
{
  return vec_perm(a, b, (vector unsigned char)
    (0x00, 0x10, 0x01, 0x11, 0x02, 0x12, 0x03, 0x13, 
     0x04, 0x14, 0x05, 0x15, 0x06, 0x16, 0x07, 0x17));
}

static vector short __ATTRS_o_ai
vec_mergeh(vector short a, vector short b)
{
  return vec_perm(a, b, (vector unsigned char)
    (0x00, 0x01, 0x10, 0x11, 0x02, 0x03, 0x12, 0x13,
     0x04, 0x05, 0x14, 0x15, 0x06, 0x07, 0x16, 0x17));
}

static vector unsigned short __ATTRS_o_ai
vec_mergeh(vector unsigned short a, vector unsigned short b)
{
  return vec_perm(a, b, (vector unsigned char)
    (0x00, 0x01, 0x10, 0x11, 0x02, 0x03, 0x12, 0x13,
     0x04, 0x05, 0x14, 0x15, 0x06, 0x07, 0x16, 0x17));
}

static vector bool short __ATTRS_o_ai
vec_mergeh(vector bool short a, vector bool short b)
{
  return vec_perm(a, b, (vector unsigned char)
    (0x00, 0x01, 0x10, 0x11, 0x02, 0x03, 0x12, 0x13,
     0x04, 0x05, 0x14, 0x15, 0x06, 0x07, 0x16, 0x17));
}

static vector pixel __ATTRS_o_ai
vec_mergeh(vector pixel a, vector pixel b)
{
  return vec_perm(a, b, (vector unsigned char)
    (0x00, 0x01, 0x10, 0x11, 0x02, 0x03, 0x12, 0x13,
     0x04, 0x05, 0x14, 0x15, 0x06, 0x07, 0x16, 0x17));
}

static vector int __ATTRS_o_ai
vec_mergeh(vector int a, vector int b)
{
  return vec_perm(a, b, (vector unsigned char)
    (0x00, 0x01, 0x02, 0x03, 0x10, 0x11, 0x12, 0x13,
     0x04, 0x05, 0x06, 0x07, 0x14, 0x15, 0x16, 0x17));
}

static vector unsigned int __ATTRS_o_ai
vec_mergeh(vector unsigned int a, vector unsigned int b)
{
  return vec_perm(a, b, (vector unsigned char)
    (0x00, 0x01, 0x02, 0x03, 0x10, 0x11, 0x12, 0x13,
     0x04, 0x05, 0x06, 0x07, 0x14, 0x15, 0x16, 0x17));
}

static vector bool int __ATTRS_o_ai
vec_mergeh(vector bool int a, vector bool int b)
{
  return vec_perm(a, b, (vector unsigned char)
    (0x00, 0x01, 0x02, 0x03, 0x10, 0x11, 0x12, 0x13,
     0x04, 0x05, 0x06, 0x07, 0x14, 0x15, 0x16, 0x17));
}

static vector float __ATTRS_o_ai
vec_mergeh(vector float a, vector float b)
{
  return vec_perm(a, b, (vector unsigned char)
    (0x00, 0x01, 0x02, 0x03, 0x10, 0x11, 0x12, 0x13,
     0x04, 0x05, 0x06, 0x07, 0x14, 0x15, 0x16, 0x17));
}

/* vec_vmrghb */

#define __builtin_altivec_vmrghb vec_vmrghb

static vector signed char __ATTRS_o_ai
vec_vmrghb(vector signed char a, vector signed char b)
{
  return vec_perm(a, b, (vector unsigned char)
    (0x00, 0x10, 0x01, 0x11, 0x02, 0x12, 0x03, 0x13, 
     0x04, 0x14, 0x05, 0x15, 0x06, 0x16, 0x07, 0x17));
}

static vector unsigned char __ATTRS_o_ai
vec_vmrghb(vector unsigned char a, vector unsigned char b)
{
  return vec_perm(a, b, (vector unsigned char)
    (0x00, 0x10, 0x01, 0x11, 0x02, 0x12, 0x03, 0x13, 
     0x04, 0x14, 0x05, 0x15, 0x06, 0x16, 0x07, 0x17));
}

static vector bool char __ATTRS_o_ai
vec_vmrghb(vector bool char a, vector bool char b)
{
  return vec_perm(a, b, (vector unsigned char)
    (0x00, 0x10, 0x01, 0x11, 0x02, 0x12, 0x03, 0x13, 
     0x04, 0x14, 0x05, 0x15, 0x06, 0x16, 0x07, 0x17));
}

/* vec_vmrghh */

#define __builtin_altivec_vmrghh vec_vmrghh

static vector short __ATTRS_o_ai
vec_vmrghh(vector short a, vector short b)
{
  return vec_perm(a, b, (vector unsigned char)
    (0x00, 0x01, 0x10, 0x11, 0x02, 0x03, 0x12, 0x13,
     0x04, 0x05, 0x14, 0x15, 0x06, 0x07, 0x16, 0x17));
}

static vector unsigned short __ATTRS_o_ai
vec_vmrghh(vector unsigned short a, vector unsigned short b)
{
  return vec_perm(a, b, (vector unsigned char)
    (0x00, 0x01, 0x10, 0x11, 0x02, 0x03, 0x12, 0x13,
     0x04, 0x05, 0x14, 0x15, 0x06, 0x07, 0x16, 0x17));
}

static vector bool short __ATTRS_o_ai
vec_vmrghh(vector bool short a, vector bool short b)
{
  return vec_perm(a, b, (vector unsigned char)
    (0x00, 0x01, 0x10, 0x11, 0x02, 0x03, 0x12, 0x13,
     0x04, 0x05, 0x14, 0x15, 0x06, 0x07, 0x16, 0x17));
}

static vector pixel __ATTRS_o_ai
vec_vmrghh(vector pixel a, vector pixel b)
{
  return vec_perm(a, b, (vector unsigned char)
    (0x00, 0x01, 0x10, 0x11, 0x02, 0x03, 0x12, 0x13,
     0x04, 0x05, 0x14, 0x15, 0x06, 0x07, 0x16, 0x17));
}

/* vec_vmrghw */

#define __builtin_altivec_vmrghw vec_vmrghw

static vector int __ATTRS_o_ai
vec_vmrghw(vector int a, vector int b)
{
  return vec_perm(a, b, (vector unsigned char)
    (0x00, 0x01, 0x02, 0x03, 0x10, 0x11, 0x12, 0x13,
     0x04, 0x05, 0x06, 0x07, 0x14, 0x15, 0x16, 0x17));
}

static vector unsigned int __ATTRS_o_ai
vec_vmrghw(vector unsigned int a, vector unsigned int b)
{
  return vec_perm(a, b, (vector unsigned char)
    (0x00, 0x01, 0x02, 0x03, 0x10, 0x11, 0x12, 0x13,
     0x04, 0x05, 0x06, 0x07, 0x14, 0x15, 0x16, 0x17));
}

static vector bool int __ATTRS_o_ai
vec_vmrghw(vector bool int a, vector bool int b)
{
  return vec_perm(a, b, (vector unsigned char)
    (0x00, 0x01, 0x02, 0x03, 0x10, 0x11, 0x12, 0x13,
     0x04, 0x05, 0x06, 0x07, 0x14, 0x15, 0x16, 0x17));
}

static vector float __ATTRS_o_ai
vec_vmrghw(vector float a, vector float b)
{
  return vec_perm(a, b, (vector unsigned char)
    (0x00, 0x01, 0x02, 0x03, 0x10, 0x11, 0x12, 0x13,
     0x04, 0x05, 0x06, 0x07, 0x14, 0x15, 0x16, 0x17));
}

/* vec_mergel */

static vector signed char __ATTRS_o_ai
vec_mergel(vector signed char a, vector signed char b)
{
  return vec_perm(a, b, (vector unsigned char)
    (0x08, 0x18, 0x09, 0x19, 0x0A, 0x1A, 0x0B, 0x1B, 
     0x0C, 0x1C, 0x0D, 0x1D, 0x0E, 0x1E, 0x0F, 0x1F));
}

static vector unsigned char __ATTRS_o_ai
vec_mergel(vector unsigned char a, vector unsigned char b)
{
  return vec_perm(a, b, (vector unsigned char)
    (0x08, 0x18, 0x09, 0x19, 0x0A, 0x1A, 0x0B, 0x1B, 
     0x0C, 0x1C, 0x0D, 0x1D, 0x0E, 0x1E, 0x0F, 0x1F));
}

static vector bool char __ATTRS_o_ai
vec_mergel(vector bool char a, vector bool char b)
{
  return vec_perm(a, b, (vector unsigned char)
    (0x08, 0x18, 0x09, 0x19, 0x0A, 0x1A, 0x0B, 0x1B, 
     0x0C, 0x1C, 0x0D, 0x1D, 0x0E, 0x1E, 0x0F, 0x1F));
}

static vector short __ATTRS_o_ai
vec_mergel(vector short a, vector short b)
{
  return vec_perm(a, b, (vector unsigned char)
    (0x08, 0x09, 0x18, 0x19, 0x0A, 0x0B, 0x1A, 0x1B,
     0x0C, 0x0D, 0x1C, 0x1D, 0x0E, 0x0F, 0x1E, 0x1F));
}

static vector unsigned short __ATTRS_o_ai
vec_mergel(vector unsigned short a, vector unsigned short b)
{
  return vec_perm(a, b, (vector unsigned char)
    (0x08, 0x09, 0x18, 0x19, 0x0A, 0x0B, 0x1A, 0x1B,
     0x0C, 0x0D, 0x1C, 0x1D, 0x0E, 0x0F, 0x1E, 0x1F));
}

static vector bool short __ATTRS_o_ai
vec_mergel(vector bool short a, vector bool short b)
{
  return vec_perm(a, b, (vector unsigned char)
    (0x08, 0x09, 0x18, 0x19, 0x0A, 0x0B, 0x1A, 0x1B,
     0x0C, 0x0D, 0x1C, 0x1D, 0x0E, 0x0F, 0x1E, 0x1F));
}

static vector pixel __ATTRS_o_ai
vec_mergel(vector pixel a, vector pixel b)
{
  return vec_perm(a, b, (vector unsigned char)
    (0x08, 0x09, 0x18, 0x19, 0x0A, 0x0B, 0x1A, 0x1B,
     0x0C, 0x0D, 0x1C, 0x1D, 0x0E, 0x0F, 0x1E, 0x1F));
}

static vector int __ATTRS_o_ai
vec_mergel(vector int a, vector int b)
{
  return vec_perm(a, b, (vector unsigned char)
    (0x08, 0x09, 0x0A, 0x0B, 0x18, 0x19, 0x1A, 0x1B,
     0x0C, 0x0D, 0x0E, 0x0F, 0x1C, 0x1D, 0x1E, 0x1F));
}

static vector unsigned int __ATTRS_o_ai
vec_mergel(vector unsigned int a, vector unsigned int b)
{
  return vec_perm(a, b, (vector unsigned char)
    (0x08, 0x09, 0x0A, 0x0B, 0x18, 0x19, 0x1A, 0x1B,
     0x0C, 0x0D, 0x0E, 0x0F, 0x1C, 0x1D, 0x1E, 0x1F));
}

static vector bool int __ATTRS_o_ai
vec_mergel(vector bool int a, vector bool int b)
{
  return vec_perm(a, b, (vector unsigned char)
    (0x08, 0x09, 0x0A, 0x0B, 0x18, 0x19, 0x1A, 0x1B,
     0x0C, 0x0D, 0x0E, 0x0F, 0x1C, 0x1D, 0x1E, 0x1F));
}

static vector float __ATTRS_o_ai
vec_mergel(vector float a, vector float b)
{
  return vec_perm(a, b, (vector unsigned char)
    (0x08, 0x09, 0x0A, 0x0B, 0x18, 0x19, 0x1A, 0x1B,
     0x0C, 0x0D, 0x0E, 0x0F, 0x1C, 0x1D, 0x1E, 0x1F));
}

/* vec_vmrglb */

#define __builtin_altivec_vmrglb vec_vmrglb

static vector signed char __ATTRS_o_ai
vec_vmrglb(vector signed char a, vector signed char b)
{
  return vec_perm(a, b, (vector unsigned char)
    (0x08, 0x18, 0x09, 0x19, 0x0A, 0x1A, 0x0B, 0x1B, 
     0x0C, 0x1C, 0x0D, 0x1D, 0x0E, 0x1E, 0x0F, 0x1F));
}

static vector unsigned char __ATTRS_o_ai
vec_vmrglb(vector unsigned char a, vector unsigned char b)
{
  return vec_perm(a, b, (vector unsigned char)
    (0x08, 0x18, 0x09, 0x19, 0x0A, 0x1A, 0x0B, 0x1B, 
     0x0C, 0x1C, 0x0D, 0x1D, 0x0E, 0x1E, 0x0F, 0x1F));
}

static vector bool char __ATTRS_o_ai
vec_vmrglb(vector bool char a, vector bool char b)
{
  return vec_perm(a, b, (vector unsigned char)
    (0x08, 0x18, 0x09, 0x19, 0x0A, 0x1A, 0x0B, 0x1B, 
     0x0C, 0x1C, 0x0D, 0x1D, 0x0E, 0x1E, 0x0F, 0x1F));
}

/* vec_vmrglh */

#define __builtin_altivec_vmrglh vec_vmrglh

static vector short __ATTRS_o_ai
vec_vmrglh(vector short a, vector short b)
{
  return vec_perm(a, b, (vector unsigned char)
    (0x08, 0x09, 0x18, 0x19, 0x0A, 0x0B, 0x1A, 0x1B,
     0x0C, 0x0D, 0x1C, 0x1D, 0x0E, 0x0F, 0x1E, 0x1F));
}

static vector unsigned short __ATTRS_o_ai
vec_vmrglh(vector unsigned short a, vector unsigned short b)
{
  return vec_perm(a, b, (vector unsigned char)
    (0x08, 0x09, 0x18, 0x19, 0x0A, 0x0B, 0x1A, 0x1B,
     0x0C, 0x0D, 0x1C, 0x1D, 0x0E, 0x0F, 0x1E, 0x1F));
}

static vector bool short __ATTRS_o_ai
vec_vmrglh(vector bool short a, vector bool short b)
{
  return vec_perm(a, b, (vector unsigned char)
    (0x08, 0x09, 0x18, 0x19, 0x0A, 0x0B, 0x1A, 0x1B,
     0x0C, 0x0D, 0x1C, 0x1D, 0x0E, 0x0F, 0x1E, 0x1F));
}

static vector pixel __ATTRS_o_ai
vec_vmrglh(vector pixel a, vector pixel b)
{
  return vec_perm(a, b, (vector unsigned char)
    (0x08, 0x09, 0x18, 0x19, 0x0A, 0x0B, 0x1A, 0x1B,
     0x0C, 0x0D, 0x1C, 0x1D, 0x0E, 0x0F, 0x1E, 0x1F));
}

/* vec_vmrglw */

#define __builtin_altivec_vmrglw vec_vmrglw

static vector int __ATTRS_o_ai
vec_vmrglw(vector int a, vector int b)
{
  return vec_perm(a, b, (vector unsigned char)
    (0x08, 0x09, 0x0A, 0x0B, 0x18, 0x19, 0x1A, 0x1B,
     0x0C, 0x0D, 0x0E, 0x0F, 0x1C, 0x1D, 0x1E, 0x1F));
}

static vector unsigned int __ATTRS_o_ai
vec_vmrglw(vector unsigned int a, vector unsigned int b)
{
  return vec_perm(a, b, (vector unsigned char)
    (0x08, 0x09, 0x0A, 0x0B, 0x18, 0x19, 0x1A, 0x1B,
     0x0C, 0x0D, 0x0E, 0x0F, 0x1C, 0x1D, 0x1E, 0x1F));
}

static vector bool int __ATTRS_o_ai
vec_vmrglw(vector bool int a, vector bool int b)
{
  return vec_perm(a, b, (vector unsigned char)
    (0x08, 0x09, 0x0A, 0x0B, 0x18, 0x19, 0x1A, 0x1B,
     0x0C, 0x0D, 0x0E, 0x0F, 0x1C, 0x1D, 0x1E, 0x1F));
}

static vector float __ATTRS_o_ai
vec_vmrglw(vector float a, vector float b)
{
  return vec_perm(a, b, (vector unsigned char)
    (0x08, 0x09, 0x0A, 0x0B, 0x18, 0x19, 0x1A, 0x1B,
     0x0C, 0x0D, 0x0E, 0x0F, 0x1C, 0x1D, 0x1E, 0x1F));
}

/* vec_mfvscr */

static vector unsigned short __attribute__((__always_inline__))
vec_mfvscr(void)
{
  return __builtin_altivec_mfvscr();
}

/* vec_min */

static vector signed char __ATTRS_o_ai
vec_min(vector signed char a, vector signed char b)
{
  return __builtin_altivec_vminsb(a, b);
}

static vector signed char __ATTRS_o_ai
vec_min(vector bool char a, vector signed char b)
{
  return __builtin_altivec_vminsb((vector signed char)a, b);
}

static vector signed char __ATTRS_o_ai
vec_min(vector signed char a, vector bool char b)
{
  return __builtin_altivec_vminsb(a, (vector signed char)b);
}

static vector unsigned char __ATTRS_o_ai
vec_min(vector unsigned char a, vector unsigned char b)
{
  return __builtin_altivec_vminub(a, b);
}

static vector unsigned char __ATTRS_o_ai
vec_min(vector bool char a, vector unsigned char b)
{
  return __builtin_altivec_vminub((vector unsigned char)a, b);
}

static vector unsigned char __ATTRS_o_ai
vec_min(vector unsigned char a, vector bool char b)
{
  return __builtin_altivec_vminub(a, (vector unsigned char)b);
}

static vector short __ATTRS_o_ai
vec_min(vector short a, vector short b)
{
  return __builtin_altivec_vminsh(a, b);
}

static vector short __ATTRS_o_ai
vec_min(vector bool short a, vector short b)
{
  return __builtin_altivec_vminsh((vector short)a, b);
}

static vector short __ATTRS_o_ai
vec_min(vector short a, vector bool short b)
{
  return __builtin_altivec_vminsh(a, (vector short)b);
}

static vector unsigned short __ATTRS_o_ai
vec_min(vector unsigned short a, vector unsigned short b)
{
  return __builtin_altivec_vminuh(a, b);
}

static vector unsigned short __ATTRS_o_ai
vec_min(vector bool short a, vector unsigned short b)
{
  return __builtin_altivec_vminuh((vector unsigned short)a, b);
}

static vector unsigned short __ATTRS_o_ai
vec_min(vector unsigned short a, vector bool short b)
{
  return __builtin_altivec_vminuh(a, (vector unsigned short)b);
}

static vector int __ATTRS_o_ai
vec_min(vector int a, vector int b)
{
  return __builtin_altivec_vminsw(a, b);
}

static vector int __ATTRS_o_ai
vec_min(vector bool int a, vector int b)
{
  return __builtin_altivec_vminsw((vector int)a, b);
}

static vector int __ATTRS_o_ai
vec_min(vector int a, vector bool int b)
{
  return __builtin_altivec_vminsw(a, (vector int)b);
}

static vector unsigned int __ATTRS_o_ai
vec_min(vector unsigned int a, vector unsigned int b)
{
  return __builtin_altivec_vminuw(a, b);
}

static vector unsigned int __ATTRS_o_ai
vec_min(vector bool int a, vector unsigned int b)
{
  return __builtin_altivec_vminuw((vector unsigned int)a, b);
}

static vector unsigned int __ATTRS_o_ai
vec_min(vector unsigned int a, vector bool int b)
{
  return __builtin_altivec_vminuw(a, (vector unsigned int)b);
}

static vector float __ATTRS_o_ai
vec_min(vector float a, vector float b)
{
  return __builtin_altivec_vminfp(a, b);
}

/* vec_vminsb */

static vector signed char __ATTRS_o_ai
vec_vminsb(vector signed char a, vector signed char b)
{
  return __builtin_altivec_vminsb(a, b);
}

static vector signed char __ATTRS_o_ai
vec_vminsb(vector bool char a, vector signed char b)
{
  return __builtin_altivec_vminsb((vector signed char)a, b);
}

static vector signed char __ATTRS_o_ai
vec_vminsb(vector signed char a, vector bool char b)
{
  return __builtin_altivec_vminsb(a, (vector signed char)b);
}

/* vec_vminub */

static vector unsigned char __ATTRS_o_ai
vec_vminub(vector unsigned char a, vector unsigned char b)
{
  return __builtin_altivec_vminub(a, b);
}

static vector unsigned char __ATTRS_o_ai
vec_vminub(vector bool char a, vector unsigned char b)
{
  return __builtin_altivec_vminub((vector unsigned char)a, b);
}

static vector unsigned char __ATTRS_o_ai
vec_vminub(vector unsigned char a, vector bool char b)
{
  return __builtin_altivec_vminub(a, (vector unsigned char)b);
}

/* vec_vminsh */

static vector short __ATTRS_o_ai
vec_vminsh(vector short a, vector short b)
{
  return __builtin_altivec_vminsh(a, b);
}

static vector short __ATTRS_o_ai
vec_vminsh(vector bool short a, vector short b)
{
  return __builtin_altivec_vminsh((vector short)a, b);
}

static vector short __ATTRS_o_ai
vec_vminsh(vector short a, vector bool short b)
{
  return __builtin_altivec_vminsh(a, (vector short)b);
}

/* vec_vminuh */

static vector unsigned short __ATTRS_o_ai
vec_vminuh(vector unsigned short a, vector unsigned short b)
{
  return __builtin_altivec_vminuh(a, b);
}

static vector unsigned short __ATTRS_o_ai
vec_vminuh(vector bool short a, vector unsigned short b)
{
  return __builtin_altivec_vminuh((vector unsigned short)a, b);
}

static vector unsigned short __ATTRS_o_ai
vec_vminuh(vector unsigned short a, vector bool short b)
{
  return __builtin_altivec_vminuh(a, (vector unsigned short)b);
}

/* vec_vminsw */

static vector int __ATTRS_o_ai
vec_vminsw(vector int a, vector int b)
{
  return __builtin_altivec_vminsw(a, b);
}

static vector int __ATTRS_o_ai
vec_vminsw(vector bool int a, vector int b)
{
  return __builtin_altivec_vminsw((vector int)a, b);
}

static vector int __ATTRS_o_ai
vec_vminsw(vector int a, vector bool int b)
{
  return __builtin_altivec_vminsw(a, (vector int)b);
}

/* vec_vminuw */

static vector unsigned int __ATTRS_o_ai
vec_vminuw(vector unsigned int a, vector unsigned int b)
{
  return __builtin_altivec_vminuw(a, b);
}

static vector unsigned int __ATTRS_o_ai
vec_vminuw(vector bool int a, vector unsigned int b)
{
  return __builtin_altivec_vminuw((vector unsigned int)a, b);
}

static vector unsigned int __ATTRS_o_ai
vec_vminuw(vector unsigned int a, vector bool int b)
{
  return __builtin_altivec_vminuw(a, (vector unsigned int)b);
}

/* vec_vminfp */

static vector float __attribute__((__always_inline__))
vec_vminfp(vector float a, vector float b)
{
  return __builtin_altivec_vminfp(a, b);
}

/* vec_mladd */

#define __builtin_altivec_vmladduhm vec_mladd

static vector short __ATTRS_o_ai
vec_mladd(vector short a, vector short b, vector short c)
{
  return a * b + c;
}

static vector short __ATTRS_o_ai
vec_mladd(vector short a, vector unsigned short b, vector unsigned short c)
{
  return a * (vector short)b + (vector short)c;
}

static vector short __ATTRS_o_ai
vec_mladd(vector unsigned short a, vector short b, vector short c)
{
  return (vector short)a * b + c;
}

static vector unsigned short __ATTRS_o_ai
vec_mladd(vector unsigned short a,
          vector unsigned short b, 
          vector unsigned short c)
{
  return a * b + c;
}

/* vec_vmladduhm */

static vector short __ATTRS_o_ai
vec_vmladduhm(vector short a, vector short b, vector short c)
{
  return a * b + c;
}

static vector short __ATTRS_o_ai
vec_vmladduhm(vector short a, vector unsigned short b, vector unsigned short c)
{
  return a * (vector short)b + (vector short)c;
}

static vector short __ATTRS_o_ai
vec_vmladduhm(vector unsigned short a, vector short b, vector short c)
{
  return (vector short)a * b + c;
}

static vector unsigned short __ATTRS_o_ai
vec_vmladduhm(vector unsigned short a,
              vector unsigned short b,
              vector unsigned short c)
{
  return a * b + c;
}

/* vec_mradds */

static vector short __attribute__((__always_inline__))
vec_mradds(vector short a, vector short b, vector short c)
{
  return __builtin_altivec_vmhraddshs(a, b, c);
}

/* vec_vmhraddshs */

static vector short __attribute__((__always_inline__))
vec_vmhraddshs(vector short a, vector short b, vector short c)
{
  return __builtin_altivec_vmhraddshs(a, b, c);
}

/* vec_msum */

static vector int __ATTRS_o_ai
vec_msum(vector signed char a, vector unsigned char b, vector int c)
{
  return __builtin_altivec_vmsummbm(a, b, c);
}

static vector unsigned int __ATTRS_o_ai
vec_msum(vector unsigned char a, vector unsigned char b, vector unsigned int c)
{
  return __builtin_altivec_vmsumubm(a, b, c);
}

static vector int __ATTRS_o_ai
vec_msum(vector short a, vector short b, vector int c)
{
  return __builtin_altivec_vmsumshm(a, b, c);
}

static vector unsigned int __ATTRS_o_ai
vec_msum(vector unsigned short a,
         vector unsigned short b,
         vector unsigned int c)
{
  return __builtin_altivec_vmsumuhm(a, b, c);
}

/* vec_vmsummbm */

static vector int __attribute__((__always_inline__))
vec_vmsummbm(vector signed char a, vector unsigned char b, vector int c)
{
  return __builtin_altivec_vmsummbm(a, b, c);
}

/* vec_vmsumubm */

static vector unsigned int __attribute__((__always_inline__))
vec_vmsumubm(vector unsigned char a,
             vector unsigned char b,
             vector unsigned int c)
{
  return __builtin_altivec_vmsumubm(a, b, c);
}

/* vec_vmsumshm */

static vector int __attribute__((__always_inline__))
vec_vmsumshm(vector short a, vector short b, vector int c)
{
  return __builtin_altivec_vmsumshm(a, b, c);
}

/* vec_vmsumuhm */

static vector unsigned int __attribute__((__always_inline__))
vec_vmsumuhm(vector unsigned short a,
             vector unsigned short b,
             vector unsigned int c)
{
  return __builtin_altivec_vmsumuhm(a, b, c);
}

/* vec_msums */

static vector int __ATTRS_o_ai
vec_msums(vector short a, vector short b, vector int c)
{
  return __builtin_altivec_vmsumshs(a, b, c);
}

static vector unsigned int __ATTRS_o_ai
vec_msums(vector unsigned short a,
          vector unsigned short b,
          vector unsigned int c)
{
  return __builtin_altivec_vmsumuhs(a, b, c);
}

/* vec_vmsumshs */

static vector int __attribute__((__always_inline__))
vec_vmsumshs(vector short a, vector short b, vector int c)
{
  return __builtin_altivec_vmsumshs(a, b, c);
}

/* vec_vmsumuhs */

static vector unsigned int __attribute__((__always_inline__))
vec_vmsumuhs(vector unsigned short a,
             vector unsigned short b,
             vector unsigned int c)
{
  return __builtin_altivec_vmsumuhs(a, b, c);
}

/* vec_mtvscr */

static void __ATTRS_o_ai
vec_mtvscr(vector signed char a)
{
  __builtin_altivec_mtvscr((vector int)a);
}

static void __ATTRS_o_ai
vec_mtvscr(vector unsigned char a)
{
  __builtin_altivec_mtvscr((vector int)a);
}

static void __ATTRS_o_ai
vec_mtvscr(vector bool char a)
{
  __builtin_altivec_mtvscr((vector int)a);
}

static void __ATTRS_o_ai
vec_mtvscr(vector short a)
{
  __builtin_altivec_mtvscr((vector int)a);
}

static void __ATTRS_o_ai
vec_mtvscr(vector unsigned short a)
{
  __builtin_altivec_mtvscr((vector int)a);
}

static void __ATTRS_o_ai
vec_mtvscr(vector bool short a)
{
  __builtin_altivec_mtvscr((vector int)a);
}

static void __ATTRS_o_ai
vec_mtvscr(vector pixel a)
{
  __builtin_altivec_mtvscr((vector int)a);
}

static void __ATTRS_o_ai
vec_mtvscr(vector int a)
{
  __builtin_altivec_mtvscr((vector int)a);
}

static void __ATTRS_o_ai
vec_mtvscr(vector unsigned int a)
{
  __builtin_altivec_mtvscr((vector int)a);
}

static void __ATTRS_o_ai
vec_mtvscr(vector bool int a)
{
  __builtin_altivec_mtvscr((vector int)a);
}

static void __ATTRS_o_ai
vec_mtvscr(vector float a)
{
  __builtin_altivec_mtvscr((vector int)a);
}

/* vec_mule */

static vector short __ATTRS_o_ai
vec_mule(vector signed char a, vector signed char b)
{
  return __builtin_altivec_vmulesb(a, b);
}

static vector unsigned short __ATTRS_o_ai
vec_mule(vector unsigned char a, vector unsigned char b)
{
  return __builtin_altivec_vmuleub(a, b);
}

static vector int __ATTRS_o_ai
vec_mule(vector short a, vector short b)
{
  return __builtin_altivec_vmulesh(a, b);
}

static vector unsigned int __ATTRS_o_ai
vec_mule(vector unsigned short a, vector unsigned short b)
{
  return __builtin_altivec_vmuleuh(a, b);
}

/* vec_vmulesb */

static vector short __attribute__((__always_inline__))
vec_vmulesb(vector signed char a, vector signed char b)
{
  return __builtin_altivec_vmulesb(a, b);
}

/* vec_vmuleub */

static vector unsigned short __attribute__((__always_inline__))
vec_vmuleub(vector unsigned char a, vector unsigned char b)
{
  return __builtin_altivec_vmuleub(a, b);
}

/* vec_vmulesh */

static vector int __attribute__((__always_inline__))
vec_vmulesh(vector short a, vector short b)
{
  return __builtin_altivec_vmulesh(a, b);
}

/* vec_vmuleuh */

static vector unsigned int __attribute__((__always_inline__))
vec_vmuleuh(vector unsigned short a, vector unsigned short b)
{
  return __builtin_altivec_vmuleuh(a, b);
}

/* vec_mulo */

static vector short __ATTRS_o_ai
vec_mulo(vector signed char a, vector signed char b)
{
  return __builtin_altivec_vmulosb(a, b);
}

static vector unsigned short __ATTRS_o_ai
vec_mulo(vector unsigned char a, vector unsigned char b)
{
  return __builtin_altivec_vmuloub(a, b);
}

static vector int __ATTRS_o_ai
vec_mulo(vector short a, vector short b)
{
  return __builtin_altivec_vmulosh(a, b);
}

static vector unsigned int __ATTRS_o_ai
vec_mulo(vector unsigned short a, vector unsigned short b)
{
  return __builtin_altivec_vmulouh(a, b);
}

/* vec_vmulosb */

static vector short __attribute__((__always_inline__))
vec_vmulosb(vector signed char a, vector signed char b)
{
  return __builtin_altivec_vmulosb(a, b);
}

/* vec_vmuloub */

static vector unsigned short __attribute__((__always_inline__))
vec_vmuloub(vector unsigned char a, vector unsigned char b)
{
  return __builtin_altivec_vmuloub(a, b);
}

/* vec_vmulosh */

static vector int __attribute__((__always_inline__))
vec_vmulosh(vector short a, vector short b)
{
  return __builtin_altivec_vmulosh(a, b);
}

/* vec_vmulouh */

static vector unsigned int __attribute__((__always_inline__))
vec_vmulouh(vector unsigned short a, vector unsigned short b)
{
  return __builtin_altivec_vmulouh(a, b);
}

/* vec_nmsub */

static vector float __attribute__((__always_inline__))
vec_nmsub(vector float a, vector float b, vector float c)
{
  return __builtin_altivec_vnmsubfp(a, b, c);
}

/* vec_vnmsubfp */

static vector float __attribute__((__always_inline__))
vec_vnmsubfp(vector float a, vector float b, vector float c)
{
  return __builtin_altivec_vnmsubfp(a, b, c);
}

/* vec_nor */

#define __builtin_altivec_vnor vec_nor

static vector signed char __ATTRS_o_ai
vec_nor(vector signed char a, vector signed char b)
{
  return ~(a | b);
}

static vector unsigned char __ATTRS_o_ai
vec_nor(vector unsigned char a, vector unsigned char b)
{
  return ~(a | b);
}

static vector bool char __ATTRS_o_ai
vec_nor(vector bool char a, vector bool char b)
{
  return ~(a | b);
}

static vector short __ATTRS_o_ai
vec_nor(vector short a, vector short b)
{
  return ~(a | b);
}

static vector unsigned short __ATTRS_o_ai
vec_nor(vector unsigned short a, vector unsigned short b)
{
  return ~(a | b);
}

static vector bool short __ATTRS_o_ai
vec_nor(vector bool short a, vector bool short b)
{
  return ~(a | b);
}

static vector int __ATTRS_o_ai
vec_nor(vector int a, vector int b)
{
  return ~(a | b);
}

static vector unsigned int __ATTRS_o_ai
vec_nor(vector unsigned int a, vector unsigned int b)
{
  return ~(a | b);
}

static vector bool int __ATTRS_o_ai
vec_nor(vector bool int a, vector bool int b)
{
  return ~(a | b);
}

static vector float __ATTRS_o_ai
vec_nor(vector float a, vector float b)
{
  vector unsigned int res = ~((vector unsigned int)a | (vector unsigned int)b);
  return (vector float)res;
}

/* vec_vnor */

static vector signed char __ATTRS_o_ai
vec_vnor(vector signed char a, vector signed char b)
{
  return ~(a | b);
}

static vector unsigned char __ATTRS_o_ai
vec_vnor(vector unsigned char a, vector unsigned char b)
{
  return ~(a | b);
}

static vector bool char __ATTRS_o_ai
vec_vnor(vector bool char a, vector bool char b)
{
  return ~(a | b);
}

static vector short __ATTRS_o_ai
vec_vnor(vector short a, vector short b)
{
  return ~(a | b);
}

static vector unsigned short __ATTRS_o_ai
vec_vnor(vector unsigned short a, vector unsigned short b)
{
  return ~(a | b);
}

static vector bool short __ATTRS_o_ai
vec_vnor(vector bool short a, vector bool short b)
{
  return ~(a | b);
}

static vector int __ATTRS_o_ai
vec_vnor(vector int a, vector int b)
{
  return ~(a | b);
}

static vector unsigned int __ATTRS_o_ai
vec_vnor(vector unsigned int a, vector unsigned int b)
{
  return ~(a | b);
}

static vector bool int __ATTRS_o_ai
vec_vnor(vector bool int a, vector bool int b)
{
  return ~(a | b);
}

static vector float __ATTRS_o_ai
vec_vnor(vector float a, vector float b)
{
  vector unsigned int res = ~((vector unsigned int)a | (vector unsigned int)b);
  return (vector float)res;
}

/* vec_or */

#define __builtin_altivec_vor vec_or

static vector signed char __ATTRS_o_ai
vec_or(vector signed char a, vector signed char b)
{
  return a | b;
}

static vector signed char __ATTRS_o_ai
vec_or(vector bool char a, vector signed char b)
{
  return (vector signed char)a | b;
}

static vector signed char __ATTRS_o_ai
vec_or(vector signed char a, vector bool char b)
{
  return a | (vector signed char)b;
}

static vector unsigned char __ATTRS_o_ai
vec_or(vector unsigned char a, vector unsigned char b)
{
  return a | b;
}

static vector unsigned char __ATTRS_o_ai
vec_or(vector bool char a, vector unsigned char b)
{
  return (vector unsigned char)a | b;
}

static vector unsigned char __ATTRS_o_ai
vec_or(vector unsigned char a, vector bool char b)
{
  return a | (vector unsigned char)b;
}

static vector bool char __ATTRS_o_ai
vec_or(vector bool char a, vector bool char b)
{
  return a | b;
}

static vector short __ATTRS_o_ai
vec_or(vector short a, vector short b)
{
  return a | b;
}

static vector short __ATTRS_o_ai
vec_or(vector bool short a, vector short b)
{
  return (vector short)a | b;
}

static vector short __ATTRS_o_ai
vec_or(vector short a, vector bool short b)
{
  return a | (vector short)b;
}

static vector unsigned short __ATTRS_o_ai
vec_or(vector unsigned short a, vector unsigned short b)
{
  return a | b;
}

static vector unsigned short __ATTRS_o_ai
vec_or(vector bool short a, vector unsigned short b)
{
  return (vector unsigned short)a | b;
}

static vector unsigned short __ATTRS_o_ai
vec_or(vector unsigned short a, vector bool short b)
{
  return a | (vector unsigned short)b;
}

static vector bool short __ATTRS_o_ai
vec_or(vector bool short a, vector bool short b)
{
  return a | b;
}

static vector int __ATTRS_o_ai
vec_or(vector int a, vector int b)
{
  return a | b;
}

static vector int __ATTRS_o_ai
vec_or(vector bool int a, vector int b)
{
  return (vector int)a | b;
}

static vector int __ATTRS_o_ai
vec_or(vector int a, vector bool int b)
{
  return a | (vector int)b;
}

static vector unsigned int __ATTRS_o_ai
vec_or(vector unsigned int a, vector unsigned int b)
{
  return a | b;
}

static vector unsigned int __ATTRS_o_ai
vec_or(vector bool int a, vector unsigned int b)
{
  return (vector unsigned int)a | b;
}

static vector unsigned int __ATTRS_o_ai
vec_or(vector unsigned int a, vector bool int b)
{
  return a | (vector unsigned int)b;
}

static vector bool int __ATTRS_o_ai
vec_or(vector bool int a, vector bool int b)
{
  return a | b;
}

static vector float __ATTRS_o_ai
vec_or(vector float a, vector float b)
{
  vector unsigned int res = (vector unsigned int)a | (vector unsigned int)b;
  return (vector float)res;
}

static vector float __ATTRS_o_ai
vec_or(vector bool int a, vector float b)
{
  vector unsigned int res = (vector unsigned int)a | (vector unsigned int)b;
  return (vector float)res;
}

static vector float __ATTRS_o_ai
vec_or(vector float a, vector bool int b)
{
  vector unsigned int res = (vector unsigned int)a | (vector unsigned int)b;
  return (vector float)res;
}

/* vec_vor */

static vector signed char __ATTRS_o_ai
vec_vor(vector signed char a, vector signed char b)
{
  return a | b;
}

static vector signed char __ATTRS_o_ai
vec_vor(vector bool char a, vector signed char b)
{
  return (vector signed char)a | b;
}

static vector signed char __ATTRS_o_ai
vec_vor(vector signed char a, vector bool char b)
{
  return a | (vector signed char)b;
}

static vector unsigned char __ATTRS_o_ai
vec_vor(vector unsigned char a, vector unsigned char b)
{
  return a | b;
}

static vector unsigned char __ATTRS_o_ai
vec_vor(vector bool char a, vector unsigned char b)
{
  return (vector unsigned char)a | b;
}

static vector unsigned char __ATTRS_o_ai
vec_vor(vector unsigned char a, vector bool char b)
{
  return a | (vector unsigned char)b;
}

static vector bool char __ATTRS_o_ai
vec_vor(vector bool char a, vector bool char b)
{
  return a | b;
}

static vector short __ATTRS_o_ai
vec_vor(vector short a, vector short b)
{
  return a | b;
}

static vector short __ATTRS_o_ai
vec_vor(vector bool short a, vector short b)
{
  return (vector short)a | b;
}

static vector short __ATTRS_o_ai
vec_vor(vector short a, vector bool short b)
{
  return a | (vector short)b;
}

static vector unsigned short __ATTRS_o_ai
vec_vor(vector unsigned short a, vector unsigned short b)
{
  return a | b;
}

static vector unsigned short __ATTRS_o_ai
vec_vor(vector bool short a, vector unsigned short b)
{
  return (vector unsigned short)a | b;
}

static vector unsigned short __ATTRS_o_ai
vec_vor(vector unsigned short a, vector bool short b)
{
  return a | (vector unsigned short)b;
}

static vector bool short __ATTRS_o_ai
vec_vor(vector bool short a, vector bool short b)
{
  return a | b;
}

static vector int __ATTRS_o_ai
vec_vor(vector int a, vector int b)
{
  return a | b;
}

static vector int __ATTRS_o_ai
vec_vor(vector bool int a, vector int b)
{
  return (vector int)a | b;
}

static vector int __ATTRS_o_ai
vec_vor(vector int a, vector bool int b)
{
  return a | (vector int)b;
}

static vector unsigned int __ATTRS_o_ai
vec_vor(vector unsigned int a, vector unsigned int b)
{
  return a | b;
}

static vector unsigned int __ATTRS_o_ai
vec_vor(vector bool int a, vector unsigned int b)
{
  return (vector unsigned int)a | b;
}

static vector unsigned int __ATTRS_o_ai
vec_vor(vector unsigned int a, vector bool int b)
{
  return a | (vector unsigned int)b;
}

static vector bool int __ATTRS_o_ai
vec_vor(vector bool int a, vector bool int b)
{
  return a | b;
}

static vector float __ATTRS_o_ai
vec_vor(vector float a, vector float b)
{
  vector unsigned int res = (vector unsigned int)a | (vector unsigned int)b;
  return (vector float)res;
}

static vector float __ATTRS_o_ai
vec_vor(vector bool int a, vector float b)
{
  vector unsigned int res = (vector unsigned int)a | (vector unsigned int)b;
  return (vector float)res;
}

static vector float __ATTRS_o_ai
vec_vor(vector float a, vector bool int b)
{
  vector unsigned int res = (vector unsigned int)a | (vector unsigned int)b;
  return (vector float)res;
}

/* vec_pack */

static vector signed char __ATTRS_o_ai
vec_pack(vector signed short a, vector signed short b)
{
  return (vector signed char)vec_perm(a, b, (vector unsigned char)
    (0x01, 0x03, 0x05, 0x07, 0x09, 0x0B, 0x0D, 0x0F,
     0x11, 0x13, 0x15, 0x17, 0x19, 0x1B, 0x1D, 0x1F));
}

static vector unsigned char __ATTRS_o_ai
vec_pack(vector unsigned short a, vector unsigned short b)
{
  return (vector unsigned char)vec_perm(a, b, (vector unsigned char)
    (0x01, 0x03, 0x05, 0x07, 0x09, 0x0B, 0x0D, 0x0F,
     0x11, 0x13, 0x15, 0x17, 0x19, 0x1B, 0x1D, 0x1F));
}

static vector bool char __ATTRS_o_ai
vec_pack(vector bool short a, vector bool short b)
{
  return (vector bool char)vec_perm(a, b, (vector unsigned char)
    (0x01, 0x03, 0x05, 0x07, 0x09, 0x0B, 0x0D, 0x0F,
     0x11, 0x13, 0x15, 0x17, 0x19, 0x1B, 0x1D, 0x1F));
}

static vector short __ATTRS_o_ai
vec_pack(vector int a, vector int b)
{
  return (vector short)vec_perm(a, b, (vector unsigned char)
    (0x02, 0x03, 0x06, 0x07, 0x0A, 0x0B, 0x0E, 0x0F,
     0x12, 0x13, 0x16, 0x17, 0x1A, 0x1B, 0x1E, 0x1F));
}

static vector unsigned short __ATTRS_o_ai
vec_pack(vector unsigned int a, vector unsigned int b)
{
  return (vector unsigned short)vec_perm(a, b, (vector unsigned char)
    (0x02, 0x03, 0x06, 0x07, 0x0A, 0x0B, 0x0E, 0x0F,
     0x12, 0x13, 0x16, 0x17, 0x1A, 0x1B, 0x1E, 0x1F));
}

static vector bool short __ATTRS_o_ai
vec_pack(vector bool int a, vector bool int b)
{
  return (vector bool short)vec_perm(a, b, (vector unsigned char)
    (0x02, 0x03, 0x06, 0x07, 0x0A, 0x0B, 0x0E, 0x0F,
     0x12, 0x13, 0x16, 0x17, 0x1A, 0x1B, 0x1E, 0x1F));
}

/* vec_vpkuhum */

#define __builtin_altivec_vpkuhum vec_vpkuhum

static vector signed char __ATTRS_o_ai
vec_vpkuhum(vector signed short a, vector signed short b)
{
  return (vector signed char)vec_perm(a, b, (vector unsigned char)
    (0x01, 0x03, 0x05, 0x07, 0x09, 0x0B, 0x0D, 0x0F,
     0x11, 0x13, 0x15, 0x17, 0x19, 0x1B, 0x1D, 0x1F));
}

static vector unsigned char __ATTRS_o_ai
vec_vpkuhum(vector unsigned short a, vector unsigned short b)
{
  return (vector unsigned char)vec_perm(a, b, (vector unsigned char)
    (0x01, 0x03, 0x05, 0x07, 0x09, 0x0B, 0x0D, 0x0F,
     0x11, 0x13, 0x15, 0x17, 0x19, 0x1B, 0x1D, 0x1F));
}

static vector bool char __ATTRS_o_ai
vec_vpkuhum(vector bool short a, vector bool short b)
{
  return (vector bool char)vec_perm(a, b, (vector unsigned char)
    (0x01, 0x03, 0x05, 0x07, 0x09, 0x0B, 0x0D, 0x0F,
     0x11, 0x13, 0x15, 0x17, 0x19, 0x1B, 0x1D, 0x1F));
}

/* vec_vpkuwum */

#define __builtin_altivec_vpkuwum vec_vpkuwum

static vector short __ATTRS_o_ai
vec_vpkuwum(vector int a, vector int b)
{
  return (vector short)vec_perm(a, b, (vector unsigned char)
    (0x02, 0x03, 0x06, 0x07, 0x0A, 0x0B, 0x0E, 0x0F,
     0x12, 0x13, 0x16, 0x17, 0x1A, 0x1B, 0x1E, 0x1F));
}

static vector unsigned short __ATTRS_o_ai
vec_vpkuwum(vector unsigned int a, vector unsigned int b)
{
  return (vector unsigned short)vec_perm(a, b, (vector unsigned char)
    (0x02, 0x03, 0x06, 0x07, 0x0A, 0x0B, 0x0E, 0x0F,
     0x12, 0x13, 0x16, 0x17, 0x1A, 0x1B, 0x1E, 0x1F));
}

static vector bool short __ATTRS_o_ai
vec_vpkuwum(vector bool int a, vector bool int b)
{
  return (vector bool short)vec_perm(a, b, (vector unsigned char)
    (0x02, 0x03, 0x06, 0x07, 0x0A, 0x0B, 0x0E, 0x0F,
     0x12, 0x13, 0x16, 0x17, 0x1A, 0x1B, 0x1E, 0x1F));
}

/* vec_packpx */

static vector pixel __attribute__((__always_inline__))
vec_packpx(vector unsigned int a, vector unsigned int b)
{
  return (vector pixel)__builtin_altivec_vpkpx(a, b);
}

/* vec_vpkpx */

static vector pixel __attribute__((__always_inline__))
vec_vpkpx(vector unsigned int a, vector unsigned int b)
{
  return (vector pixel)__builtin_altivec_vpkpx(a, b);
}

/* vec_packs */

static vector signed char __ATTRS_o_ai
vec_packs(vector short a, vector short b)
{
  return __builtin_altivec_vpkshss(a, b);
}

static vector unsigned char __ATTRS_o_ai
vec_packs(vector unsigned short a, vector unsigned short b)
{
  return __builtin_altivec_vpkuhus(a, b);
}

static vector signed short __ATTRS_o_ai
vec_packs(vector int a, vector int b)
{
  return __builtin_altivec_vpkswss(a, b);
}

static vector unsigned short __ATTRS_o_ai
vec_packs(vector unsigned int a, vector unsigned int b)
{
  return __builtin_altivec_vpkuwus(a, b);
}

/* vec_vpkshss */

static vector signed char __attribute__((__always_inline__))
vec_vpkshss(vector short a, vector short b)
{
  return __builtin_altivec_vpkshss(a, b);
}

/* vec_vpkuhus */

static vector unsigned char __attribute__((__always_inline__))
vec_vpkuhus(vector unsigned short a, vector unsigned short b)
{
  return __builtin_altivec_vpkuhus(a, b);
}

/* vec_vpkswss */

static vector signed short __attribute__((__always_inline__))
vec_vpkswss(vector int a, vector int b)
{
  return __builtin_altivec_vpkswss(a, b);
}

/* vec_vpkuwus */

static vector unsigned short __attribute__((__always_inline__))
vec_vpkuwus(vector unsigned int a, vector unsigned int b)
{
  return __builtin_altivec_vpkuwus(a, b);
}

/* vec_packsu */

static vector unsigned char __ATTRS_o_ai
vec_packsu(vector short a, vector short b)
{
  return __builtin_altivec_vpkshus(a, b);
}

static vector unsigned char __ATTRS_o_ai
vec_packsu(vector unsigned short a, vector unsigned short b)
{
  return __builtin_altivec_vpkuhus(a, b);
}

static vector unsigned short __ATTRS_o_ai
vec_packsu(vector int a, vector int b)
{
  return __builtin_altivec_vpkswus(a, b);
}

static vector unsigned short __ATTRS_o_ai
vec_packsu(vector unsigned int a, vector unsigned int b)
{
  return __builtin_altivec_vpkuwus(a, b);
}

/* vec_vpkshus */

static vector unsigned char __ATTRS_o_ai
vec_vpkshus(vector short a, vector short b)
{
  return __builtin_altivec_vpkshus(a, b);
}

static vector unsigned char __ATTRS_o_ai
vec_vpkshus(vector unsigned short a, vector unsigned short b)
{
  return __builtin_altivec_vpkuhus(a, b);
}

/* vec_vpkswus */

static vector unsigned short __ATTRS_o_ai
vec_vpkswus(vector int a, vector int b)
{
  return __builtin_altivec_vpkswus(a, b);
}

static vector unsigned short __ATTRS_o_ai
vec_vpkswus(vector unsigned int a, vector unsigned int b)
{
  return __builtin_altivec_vpkuwus(a, b);
}

/* vec_perm */

vector signed char __ATTRS_o_ai
vec_perm(vector signed char a, vector signed char b, vector unsigned char c)
{
  return (vector signed char)
           __builtin_altivec_vperm_4si((vector int)a, (vector int)b, c);
}

vector unsigned char __ATTRS_o_ai
vec_perm(vector unsigned char a,
         vector unsigned char b,
         vector unsigned char c)
{
  return (vector unsigned char)
           __builtin_altivec_vperm_4si((vector int)a, (vector int)b, c);
}

vector bool char __ATTRS_o_ai
vec_perm(vector bool char a, vector bool char b, vector unsigned char c)
{
  return (vector bool char)
           __builtin_altivec_vperm_4si((vector int)a, (vector int)b, c);
}

vector short __ATTRS_o_ai
vec_perm(vector short a, vector short b, vector unsigned char c)
{
  return (vector short)
           __builtin_altivec_vperm_4si((vector int)a, (vector int)b, c);
}

vector unsigned short __ATTRS_o_ai
vec_perm(vector unsigned short a,
         vector unsigned short b,
         vector unsigned char c)
{
  return (vector unsigned short)
           __builtin_altivec_vperm_4si((vector int)a, (vector int)b, c);
}

vector bool short __ATTRS_o_ai
vec_perm(vector bool short a, vector bool short b, vector unsigned char c)
{
  return (vector bool short)
           __builtin_altivec_vperm_4si((vector int)a, (vector int)b, c);
}

vector pixel __ATTRS_o_ai
vec_perm(vector pixel a, vector pixel b, vector unsigned char c)
{
  return (vector pixel)
           __builtin_altivec_vperm_4si((vector int)a, (vector int)b, c);
}

vector int __ATTRS_o_ai
vec_perm(vector int a, vector int b, vector unsigned char c)
{
  return (vector int)__builtin_altivec_vperm_4si(a, b, c);
}

vector unsigned int __ATTRS_o_ai
vec_perm(vector unsigned int a, vector unsigned int b, vector unsigned char c)
{
  return (vector unsigned int)
           __builtin_altivec_vperm_4si((vector int)a, (vector int)b, c);
}

vector bool int __ATTRS_o_ai
vec_perm(vector bool int a, vector bool int b, vector unsigned char c)
{
  return (vector bool int)
           __builtin_altivec_vperm_4si((vector int)a, (vector int)b, c);
}

vector float __ATTRS_o_ai
vec_perm(vector float a, vector float b, vector unsigned char c)
{
  return (vector float)
           __builtin_altivec_vperm_4si((vector int)a, (vector int)b, c);
}

/* vec_vperm */

static vector signed char __ATTRS_o_ai
vec_vperm(vector signed char a, vector signed char b, vector unsigned char c)
{
  return (vector signed char)
           __builtin_altivec_vperm_4si((vector int)a, (vector int)b, c);
}

static vector unsigned char __ATTRS_o_ai
vec_vperm(vector unsigned char a,
          vector unsigned char b,
          vector unsigned char c)
{
  return (vector unsigned char)
           __builtin_altivec_vperm_4si((vector int)a, (vector int)b, c);
}

static vector bool char __ATTRS_o_ai
vec_vperm(vector bool char a, vector bool char b, vector unsigned char c)
{
  return (vector bool char)
           __builtin_altivec_vperm_4si((vector int)a, (vector int)b, c);
}

static vector short __ATTRS_o_ai
vec_vperm(vector short a, vector short b, vector unsigned char c)
{
  return (vector short)
           __builtin_altivec_vperm_4si((vector int)a, (vector int)b, c);
}

static vector unsigned short __ATTRS_o_ai
vec_vperm(vector unsigned short a,
          vector unsigned short b,
          vector unsigned char c)
{
  return (vector unsigned short)
           __builtin_altivec_vperm_4si((vector int)a, (vector int)b, c);
}

static vector bool short __ATTRS_o_ai
vec_vperm(vector bool short a, vector bool short b, vector unsigned char c)
{
  return (vector bool short)
           __builtin_altivec_vperm_4si((vector int)a, (vector int)b, c);
}

static vector pixel __ATTRS_o_ai
vec_vperm(vector pixel a, vector pixel b, vector unsigned char c)
{
  return (vector pixel)
           __builtin_altivec_vperm_4si((vector int)a, (vector int)b, c);
}

static vector int __ATTRS_o_ai
vec_vperm(vector int a, vector int b, vector unsigned char c)
{
  return (vector int)__builtin_altivec_vperm_4si(a, b, c);
}

static vector unsigned int __ATTRS_o_ai
vec_vperm(vector unsigned int a, vector unsigned int b, vector unsigned char c)
{
  return (vector unsigned int)
           __builtin_altivec_vperm_4si((vector int)a, (vector int)b, c);
}

static vector bool int __ATTRS_o_ai
vec_vperm(vector bool int a, vector bool int b, vector unsigned char c)
{
  return (vector bool int)
           __builtin_altivec_vperm_4si((vector int)a, (vector int)b, c);
}

static vector float __ATTRS_o_ai
vec_vperm(vector float a, vector float b, vector unsigned char c)
{
  return (vector float)
           __builtin_altivec_vperm_4si((vector int)a, (vector int)b, c);
}

/* vec_re */

static vector float __attribute__((__always_inline__))
vec_re(vector float a)
{
  return __builtin_altivec_vrefp(a);
}

/* vec_vrefp */

static vector float __attribute__((__always_inline__))
vec_vrefp(vector float a)
{
  return __builtin_altivec_vrefp(a);
}

/* vec_rl */

static vector signed char __ATTRS_o_ai
vec_rl(vector signed char a, vector unsigned char b)
{
  return (vector signed char)__builtin_altivec_vrlb((vector char)a, b);
}

static vector unsigned char __ATTRS_o_ai
vec_rl(vector unsigned char a, vector unsigned char b)
{
  return (vector unsigned char)__builtin_altivec_vrlb((vector char)a, b);
}

static vector short __ATTRS_o_ai
vec_rl(vector short a, vector unsigned short b)
{
  return __builtin_altivec_vrlh(a, b);
}

static vector unsigned short __ATTRS_o_ai
vec_rl(vector unsigned short a, vector unsigned short b)
{
  return (vector unsigned short)__builtin_altivec_vrlh((vector short)a, b);
}

static vector int __ATTRS_o_ai
vec_rl(vector int a, vector unsigned int b)
{
  return __builtin_altivec_vrlw(a, b);
}

static vector unsigned int __ATTRS_o_ai
vec_rl(vector unsigned int a, vector unsigned int b)
{
  return (vector unsigned int)__builtin_altivec_vrlw((vector int)a, b);
}

/* vec_vrlb */

static vector signed char __ATTRS_o_ai
vec_vrlb(vector signed char a, vector unsigned char b)
{
  return (vector signed char)__builtin_altivec_vrlb((vector char)a, b);
}

static vector unsigned char __ATTRS_o_ai
vec_vrlb(vector unsigned char a, vector unsigned char b)
{
  return (vector unsigned char)__builtin_altivec_vrlb((vector char)a, b);
}

/* vec_vrlh */

static vector short __ATTRS_o_ai
vec_vrlh(vector short a, vector unsigned short b)
{
  return __builtin_altivec_vrlh(a, b);
}

static vector unsigned short __ATTRS_o_ai
vec_vrlh(vector unsigned short a, vector unsigned short b)
{
  return (vector unsigned short)__builtin_altivec_vrlh((vector short)a, b);
}

/* vec_vrlw */

static vector int __ATTRS_o_ai
vec_vrlw(vector int a, vector unsigned int b)
{
  return __builtin_altivec_vrlw(a, b);
}

static vector unsigned int __ATTRS_o_ai
vec_vrlw(vector unsigned int a, vector unsigned int b)
{
  return (vector unsigned int)__builtin_altivec_vrlw((vector int)a, b);
}

/* vec_round */

static vector float __attribute__((__always_inline__))
vec_round(vector float a)
{
  return __builtin_altivec_vrfin(a);
}

/* vec_vrfin */

static vector float __attribute__((__always_inline__))
vec_vrfin(vector float a)
{
  return __builtin_altivec_vrfin(a);
}

/* vec_rsqrte */

static __vector float __attribute__((__always_inline__))
vec_rsqrte(vector float a)
{
  return __builtin_altivec_vrsqrtefp(a);
}

/* vec_vrsqrtefp */

static __vector float __attribute__((__always_inline__))
vec_vrsqrtefp(vector float a)
{
  return __builtin_altivec_vrsqrtefp(a);
}

/* vec_sel */

#define __builtin_altivec_vsel_4si vec_sel

static vector signed char __ATTRS_o_ai
vec_sel(vector signed char a, vector signed char b, vector unsigned char c)
{
  return (a & ~(vector signed char)c) | (b & (vector signed char)c);
}

static vector signed char __ATTRS_o_ai
vec_sel(vector signed char a, vector signed char b, vector bool char c)
{
  return (a & ~(vector signed char)c) | (b & (vector signed char)c);
}

static vector unsigned char __ATTRS_o_ai
vec_sel(vector unsigned char a, vector unsigned char b, vector unsigned char c)
{
  return (a & ~c) | (b & c);
}

static vector unsigned char __ATTRS_o_ai
vec_sel(vector unsigned char a, vector unsigned char b, vector bool char c)
{
  return (a & ~(vector unsigned char)c) | (b & (vector unsigned char)c);
}

static vector bool char __ATTRS_o_ai
vec_sel(vector bool char a, vector bool char b, vector unsigned char c)
{
  return (a & ~(vector bool char)c) | (b & (vector bool char)c);
}

static vector bool char __ATTRS_o_ai
vec_sel(vector bool char a, vector bool char b, vector bool char c)
{
  return (a & ~c) | (b & c);
}

static vector short __ATTRS_o_ai
vec_sel(vector short a, vector short b, vector unsigned short c)
{
  return (a & ~(vector short)c) | (b & (vector short)c);
}

static vector short __ATTRS_o_ai
vec_sel(vector short a, vector short b, vector bool short c)
{
  return (a & ~(vector short)c) | (b & (vector short)c);
}

static vector unsigned short __ATTRS_o_ai
vec_sel(vector unsigned short a,
        vector unsigned short b,
        vector unsigned short c)
{
  return (a & ~c) | (b & c);
}

static vector unsigned short __ATTRS_o_ai
vec_sel(vector unsigned short a, vector unsigned short b, vector bool short c)
{
  return (a & ~(vector unsigned short)c) | (b & (vector unsigned short)c);
}

static vector bool short __ATTRS_o_ai
vec_sel(vector bool short a, vector bool short b, vector unsigned short c)
{
  return (a & ~(vector bool short)c) | (b & (vector bool short)c);
}

static vector bool short __ATTRS_o_ai
vec_sel(vector bool short a, vector bool short b, vector bool short c)
{
  return (a & ~c) | (b & c);
}

static vector int __ATTRS_o_ai
vec_sel(vector int a, vector int b, vector unsigned int c)
{
  return (a & ~(vector int)c) | (b & (vector int)c);
}

static vector int __ATTRS_o_ai
vec_sel(vector int a, vector int b, vector bool int c)
{
  return (a & ~(vector int)c) | (b & (vector int)c);
}

static vector unsigned int __ATTRS_o_ai
vec_sel(vector unsigned int a, vector unsigned int b, vector unsigned int c)
{
  return (a & ~c) | (b & c);
}

static vector unsigned int __ATTRS_o_ai
vec_sel(vector unsigned int a, vector unsigned int b, vector bool int c)
{
  return (a & ~(vector unsigned int)c) | (b & (vector unsigned int)c);
}

static vector bool int __ATTRS_o_ai
vec_sel(vector bool int a, vector bool int b, vector unsigned int c)
{
  return (a & ~(vector bool int)c) | (b & (vector bool int)c);
}

static vector bool int __ATTRS_o_ai
vec_sel(vector bool int a, vector bool int b, vector bool int c)
{
  return (a & ~c) | (b & c);
}

static vector float __ATTRS_o_ai
vec_sel(vector float a, vector float b, vector unsigned int c)
{
  vector int res = ((vector int)a & ~(vector int)c) 
                   | ((vector int)b & (vector int)c);
  return (vector float)res;
}

static vector float __ATTRS_o_ai
vec_sel(vector float a, vector float b, vector bool int c)
{
  vector int res = ((vector int)a & ~(vector int)c)
                   | ((vector int)b & (vector int)c);
  return (vector float)res;
}

/* vec_vsel */

static vector signed char __ATTRS_o_ai
vec_vsel(vector signed char a, vector signed char b, vector unsigned char c)
{
  return (a & ~(vector signed char)c) | (b & (vector signed char)c);
}

static vector signed char __ATTRS_o_ai
vec_vsel(vector signed char a, vector signed char b, vector bool char c)
{
  return (a & ~(vector signed char)c) | (b & (vector signed char)c);
}

static vector unsigned char __ATTRS_o_ai
vec_vsel(vector unsigned char a, vector unsigned char b, vector unsigned char c)
{
  return (a & ~c) | (b & c);
}

static vector unsigned char __ATTRS_o_ai
vec_vsel(vector unsigned char a, vector unsigned char b, vector bool char c)
{
  return (a & ~(vector unsigned char)c) | (b & (vector unsigned char)c);
}

static vector bool char __ATTRS_o_ai
vec_vsel(vector bool char a, vector bool char b, vector unsigned char c)
{
  return (a & ~(vector bool char)c) | (b & (vector bool char)c);
}

static vector bool char __ATTRS_o_ai
vec_vsel(vector bool char a, vector bool char b, vector bool char c)
{
  return (a & ~c) | (b & c);
}

static vector short __ATTRS_o_ai
vec_vsel(vector short a, vector short b, vector unsigned short c)
{
  return (a & ~(vector short)c) | (b & (vector short)c);
}

static vector short __ATTRS_o_ai
vec_vsel(vector short a, vector short b, vector bool short c)
{
  return (a & ~(vector short)c) | (b & (vector short)c);
}

static vector unsigned short __ATTRS_o_ai
vec_vsel(vector unsigned short a,
         vector unsigned short b,
         vector unsigned short c)
{
  return (a & ~c) | (b & c);
}

static vector unsigned short __ATTRS_o_ai
vec_vsel(vector unsigned short a, vector unsigned short b, vector bool short c)
{
  return (a & ~(vector unsigned short)c) | (b & (vector unsigned short)c);
}

static vector bool short __ATTRS_o_ai
vec_vsel(vector bool short a, vector bool short b, vector unsigned short c)
{
  return (a & ~(vector bool short)c) | (b & (vector bool short)c);
}

static vector bool short __ATTRS_o_ai
vec_vsel(vector bool short a, vector bool short b, vector bool short c)
{
  return (a & ~c) | (b & c);
}

static vector int __ATTRS_o_ai
vec_vsel(vector int a, vector int b, vector unsigned int c)
{
  return (a & ~(vector int)c) | (b & (vector int)c);
}

static vector int __ATTRS_o_ai
vec_vsel(vector int a, vector int b, vector bool int c)
{
  return (a & ~(vector int)c) | (b & (vector int)c);
}

static vector unsigned int __ATTRS_o_ai
vec_vsel(vector unsigned int a, vector unsigned int b, vector unsigned int c)
{
  return (a & ~c) | (b & c);
}

static vector unsigned int __ATTRS_o_ai
vec_vsel(vector unsigned int a, vector unsigned int b, vector bool int c)
{
  return (a & ~(vector unsigned int)c) | (b & (vector unsigned int)c);
}

static vector bool int __ATTRS_o_ai
vec_vsel(vector bool int a, vector bool int b, vector unsigned int c)
{
  return (a & ~(vector bool int)c) | (b & (vector bool int)c);
}

static vector bool int __ATTRS_o_ai
vec_vsel(vector bool int a, vector bool int b, vector bool int c)
{
  return (a & ~c) | (b & c);
}

static vector float __ATTRS_o_ai
vec_vsel(vector float a, vector float b, vector unsigned int c)
{
  vector int res = ((vector int)a & ~(vector int)c)
                   | ((vector int)b & (vector int)c);
  return (vector float)res;
}

static vector float __ATTRS_o_ai
vec_vsel(vector float a, vector float b, vector bool int c)
{
  vector int res = ((vector int)a & ~(vector int)c)
                   | ((vector int)b & (vector int)c);
  return (vector float)res;
}

/* vec_sl */

static vector signed char __ATTRS_o_ai
vec_sl(vector signed char a, vector unsigned char b)
{
  return a << (vector signed char)b;
}

static vector unsigned char __ATTRS_o_ai
vec_sl(vector unsigned char a, vector unsigned char b)
{
  return a << b;
}

static vector short __ATTRS_o_ai
vec_sl(vector short a, vector unsigned short b)
{
  return a << (vector short)b;
}

static vector unsigned short __ATTRS_o_ai
vec_sl(vector unsigned short a, vector unsigned short b)
{
  return a << b;
}

static vector int __ATTRS_o_ai
vec_sl(vector int a, vector unsigned int b)
{
  return a << (vector int)b;
}

static vector unsigned int __ATTRS_o_ai
vec_sl(vector unsigned int a, vector unsigned int b)
{
  return a << b;
}

/* vec_vslb */

#define __builtin_altivec_vslb vec_vslb

static vector signed char __ATTRS_o_ai
vec_vslb(vector signed char a, vector unsigned char b)
{
  return vec_sl(a, b);
}

static vector unsigned char __ATTRS_o_ai
vec_vslb(vector unsigned char a, vector unsigned char b)
{
  return vec_sl(a, b);
}

/* vec_vslh */

#define __builtin_altivec_vslh vec_vslh

static vector short __ATTRS_o_ai
vec_vslh(vector short a, vector unsigned short b)
{
  return vec_sl(a, b);
}

static vector unsigned short __ATTRS_o_ai
vec_vslh(vector unsigned short a, vector unsigned short b)
{
  return vec_sl(a, b);
}

/* vec_vslw */

#define __builtin_altivec_vslw vec_vslw

static vector int __ATTRS_o_ai
vec_vslw(vector int a, vector unsigned int b)
{
  return vec_sl(a, b);
}

static vector unsigned int __ATTRS_o_ai
vec_vslw(vector unsigned int a, vector unsigned int b)
{
  return vec_sl(a, b);
}

/* vec_sld */

#define __builtin_altivec_vsldoi_4si vec_sld

static vector signed char __ATTRS_o_ai
vec_sld(vector signed char a, vector signed char b, unsigned char c)
{
  return vec_perm(a, b, (vector unsigned char)
    (c,   c+1, c+2,  c+3,  c+4,  c+5,  c+6,  c+7, 
     c+8, c+9, c+10, c+11, c+12, c+13, c+14, c+15));
}

static vector unsigned char __ATTRS_o_ai
vec_sld(vector unsigned char a, vector unsigned char b, unsigned char c)
{
  return vec_perm(a, b, (vector unsigned char)
    (c,   c+1, c+2,  c+3,  c+4,  c+5,  c+6,  c+7, 
     c+8, c+9, c+10, c+11, c+12, c+13, c+14, c+15));
}

static vector short __ATTRS_o_ai
vec_sld(vector short a, vector short b, unsigned char c)
{
  return vec_perm(a, b, (vector unsigned char)
    (c,   c+1, c+2,  c+3,  c+4,  c+5,  c+6,  c+7, 
     c+8, c+9, c+10, c+11, c+12, c+13, c+14, c+15));
}

static vector unsigned short __ATTRS_o_ai
vec_sld(vector unsigned short a, vector unsigned short b, unsigned char c)
{
  return vec_perm(a, b, (vector unsigned char)
    (c,   c+1, c+2,  c+3,  c+4,  c+5,  c+6,  c+7, 
     c+8, c+9, c+10, c+11, c+12, c+13, c+14, c+15));
}

static vector pixel __ATTRS_o_ai
vec_sld(vector pixel a, vector pixel b, unsigned char c)
{
  return vec_perm(a, b, (vector unsigned char)
    (c,   c+1, c+2,  c+3,  c+4,  c+5,  c+6,  c+7, 
     c+8, c+9, c+10, c+11, c+12, c+13, c+14, c+15));
}

static vector int __ATTRS_o_ai
vec_sld(vector int a, vector int b, unsigned char c)
{
  return vec_perm(a, b, (vector unsigned char)
    (c,   c+1, c+2,  c+3,  c+4,  c+5,  c+6,  c+7, 
     c+8, c+9, c+10, c+11, c+12, c+13, c+14, c+15));
}

static vector unsigned int __ATTRS_o_ai
vec_sld(vector unsigned int a, vector unsigned int b, unsigned char c)
{
  return vec_perm(a, b, (vector unsigned char)
    (c,   c+1, c+2,  c+3,  c+4,  c+5,  c+6,  c+7, 
     c+8, c+9, c+10, c+11, c+12, c+13, c+14, c+15));
}

static vector float __ATTRS_o_ai
vec_sld(vector float a, vector float b, unsigned char c)
{
  return vec_perm(a, b, (vector unsigned char)
    (c,   c+1, c+2,  c+3,  c+4,  c+5,  c+6,  c+7, 
     c+8, c+9, c+10, c+11, c+12, c+13, c+14, c+15));
}

/* vec_vsldoi */

static vector signed char __ATTRS_o_ai
vec_vsldoi(vector signed char a, vector signed char b, unsigned char c)
{
  return vec_perm(a, b, (vector unsigned char)
    (c,   c+1, c+2,  c+3,  c+4,  c+5,  c+6,  c+7, 
     c+8, c+9, c+10, c+11, c+12, c+13, c+14, c+15));
}

static vector unsigned char __ATTRS_o_ai
vec_vsldoi(vector unsigned char a, vector unsigned char b, unsigned char c)
{
  return vec_perm(a, b, (vector unsigned char)
    (c,   c+1, c+2,  c+3,  c+4,  c+5,  c+6,  c+7, 
     c+8, c+9, c+10, c+11, c+12, c+13, c+14, c+15));
}

static vector short __ATTRS_o_ai
vec_vsldoi(vector short a, vector short b, unsigned char c)
{
  return vec_perm(a, b, (vector unsigned char)
    (c,   c+1, c+2,  c+3,  c+4,  c+5,  c+6,  c+7, 
     c+8, c+9, c+10, c+11, c+12, c+13, c+14, c+15));
}

static vector unsigned short __ATTRS_o_ai
vec_vsldoi(vector unsigned short a, vector unsigned short b, unsigned char c)
{
  return vec_perm(a, b, (vector unsigned char)
    (c,   c+1, c+2,  c+3,  c+4,  c+5,  c+6,  c+7, 
     c+8, c+9, c+10, c+11, c+12, c+13, c+14, c+15));
}

static vector pixel __ATTRS_o_ai
vec_vsldoi(vector pixel a, vector pixel b, unsigned char c)
{
  return vec_perm(a, b, (vector unsigned char)
    (c,   c+1, c+2,  c+3,  c+4,  c+5,  c+6,  c+7, 
     c+8, c+9, c+10, c+11, c+12, c+13, c+14, c+15));
}

static vector int __ATTRS_o_ai
vec_vsldoi(vector int a, vector int b, unsigned char c)
{
  return vec_perm(a, b, (vector unsigned char)
    (c,   c+1, c+2,  c+3,  c+4,  c+5,  c+6,  c+7, 
     c+8, c+9, c+10, c+11, c+12, c+13, c+14, c+15));
}

static vector unsigned int __ATTRS_o_ai
vec_vsldoi(vector unsigned int a, vector unsigned int b, unsigned char c)
{
  return vec_perm(a, b, (vector unsigned char)
    (c,   c+1, c+2,  c+3,  c+4,  c+5,  c+6,  c+7, 
     c+8, c+9, c+10, c+11, c+12, c+13, c+14, c+15));
}

static vector float __ATTRS_o_ai
vec_vsldoi(vector float a, vector float b, unsigned char c)
{
  return vec_perm(a, b, (vector unsigned char)
    (c,   c+1, c+2,  c+3,  c+4,  c+5,  c+6,  c+7, 
     c+8, c+9, c+10, c+11, c+12, c+13, c+14, c+15));
}

/* vec_sll */

static vector signed char __ATTRS_o_ai
vec_sll(vector signed char a, vector unsigned char b)
{
  return (vector signed char)
           __builtin_altivec_vsl((vector int)a, (vector int)b);
}

static vector signed char __ATTRS_o_ai
vec_sll(vector signed char a, vector unsigned short b)
{
  return (vector signed char)
           __builtin_altivec_vsl((vector int)a, (vector int)b);
}

static vector signed char __ATTRS_o_ai
vec_sll(vector signed char a, vector unsigned int b)
{
  return (vector signed char)
           __builtin_altivec_vsl((vector int)a, (vector int)b);
}

static vector unsigned char __ATTRS_o_ai
vec_sll(vector unsigned char a, vector unsigned char b)
{
  return (vector unsigned char)
           __builtin_altivec_vsl((vector int)a, (vector int)b);
}

static vector unsigned char __ATTRS_o_ai
vec_sll(vector unsigned char a, vector unsigned short b)
{
  return (vector unsigned char)
           __builtin_altivec_vsl((vector int)a, (vector int)b);
}

static vector unsigned char __ATTRS_o_ai
vec_sll(vector unsigned char a, vector unsigned int b)
{
  return (vector unsigned char)
           __builtin_altivec_vsl((vector int)a, (vector int)b);
}

static vector bool char __ATTRS_o_ai
vec_sll(vector bool char a, vector unsigned char b)
{
  return (vector bool char)__builtin_altivec_vsl((vector int)a, (vector int)b);
}

static vector bool char __ATTRS_o_ai
vec_sll(vector bool char a, vector unsigned short b)
{
  return (vector bool char)__builtin_altivec_vsl((vector int)a, (vector int)b);
}

static vector bool char __ATTRS_o_ai
vec_sll(vector bool char a, vector unsigned int b)
{
  return (vector bool char)__builtin_altivec_vsl((vector int)a, (vector int)b);
}

static vector short __ATTRS_o_ai
vec_sll(vector short a, vector unsigned char b)
{
  return (vector short)__builtin_altivec_vsl((vector int)a, (vector int)b);
}

static vector short __ATTRS_o_ai
vec_sll(vector short a, vector unsigned short b)
{
  return (vector short)__builtin_altivec_vsl((vector int)a, (vector int)b);
}

static vector short __ATTRS_o_ai
vec_sll(vector short a, vector unsigned int b)
{
  return (vector short)__builtin_altivec_vsl((vector int)a, (vector int)b);
}

static vector unsigned short __ATTRS_o_ai
vec_sll(vector unsigned short a, vector unsigned char b)
{
  return (vector unsigned short)
           __builtin_altivec_vsl((vector int)a, (vector int)b);
}

static vector unsigned short __ATTRS_o_ai
vec_sll(vector unsigned short a, vector unsigned short b)
{
  return (vector unsigned short)
           __builtin_altivec_vsl((vector int)a, (vector int)b);
}

static vector unsigned short __ATTRS_o_ai
vec_sll(vector unsigned short a, vector unsigned int b)
{
  return (vector unsigned short)
           __builtin_altivec_vsl((vector int)a, (vector int)b);
}

static vector bool short __ATTRS_o_ai
vec_sll(vector bool short a, vector unsigned char b)
{
  return (vector bool short)__builtin_altivec_vsl((vector int)a, (vector int)b);
}

static vector bool short __ATTRS_o_ai
vec_sll(vector bool short a, vector unsigned short b)
{
  return (vector bool short)__builtin_altivec_vsl((vector int)a, (vector int)b);
}

static vector bool short __ATTRS_o_ai
vec_sll(vector bool short a, vector unsigned int b)
{
  return (vector bool short)__builtin_altivec_vsl((vector int)a, (vector int)b);
}

static vector pixel __ATTRS_o_ai
vec_sll(vector pixel a, vector unsigned char b)
{
  return (vector pixel)__builtin_altivec_vsl((vector int)a, (vector int)b);
}

static vector pixel __ATTRS_o_ai
vec_sll(vector pixel a, vector unsigned short b)
{
  return (vector pixel)__builtin_altivec_vsl((vector int)a, (vector int)b);
}

static vector pixel __ATTRS_o_ai
vec_sll(vector pixel a, vector unsigned int b)
{
  return (vector pixel)__builtin_altivec_vsl((vector int)a, (vector int)b);
}

static vector int __ATTRS_o_ai
vec_sll(vector int a, vector unsigned char b)
{
  return (vector int)__builtin_altivec_vsl(a, (vector int)b);
}

static vector int __ATTRS_o_ai
vec_sll(vector int a, vector unsigned short b)
{
  return (vector int)__builtin_altivec_vsl(a, (vector int)b);
}

static vector int __ATTRS_o_ai
vec_sll(vector int a, vector unsigned int b)
{
  return (vector int)__builtin_altivec_vsl(a, (vector int)b);
}

static vector unsigned int __ATTRS_o_ai
vec_sll(vector unsigned int a, vector unsigned char b)
{
  return (vector unsigned int)
           __builtin_altivec_vsl((vector int)a, (vector int)b);
}

static vector unsigned int __ATTRS_o_ai
vec_sll(vector unsigned int a, vector unsigned short b)
{
  return (vector unsigned int)
           __builtin_altivec_vsl((vector int)a, (vector int)b);
}

static vector unsigned int __ATTRS_o_ai
vec_sll(vector unsigned int a, vector unsigned int b)
{
  return (vector unsigned int)
           __builtin_altivec_vsl((vector int)a, (vector int)b);
}

static vector bool int __ATTRS_o_ai
vec_sll(vector bool int a, vector unsigned char b)
{
  return (vector bool int)__builtin_altivec_vsl((vector int)a, (vector int)b);
}

static vector bool int __ATTRS_o_ai
vec_sll(vector bool int a, vector unsigned short b)
{
  return (vector bool int)__builtin_altivec_vsl((vector int)a, (vector int)b);
}

static vector bool int __ATTRS_o_ai
vec_sll(vector bool int a, vector unsigned int b)
{
  return (vector bool int)__builtin_altivec_vsl((vector int)a, (vector int)b);
}

/* vec_vsl */

static vector signed char __ATTRS_o_ai
vec_vsl(vector signed char a, vector unsigned char b)
{
  return (vector signed char)
           __builtin_altivec_vsl((vector int)a, (vector int)b);
}

static vector signed char __ATTRS_o_ai
vec_vsl(vector signed char a, vector unsigned short b)
{
  return (vector signed char)
           __builtin_altivec_vsl((vector int)a, (vector int)b);
}

static vector signed char __ATTRS_o_ai
vec_vsl(vector signed char a, vector unsigned int b)
{
  return (vector signed char)
           __builtin_altivec_vsl((vector int)a, (vector int)b);
}

static vector unsigned char __ATTRS_o_ai
vec_vsl(vector unsigned char a, vector unsigned char b)
{
  return (vector unsigned char)
           __builtin_altivec_vsl((vector int)a, (vector int)b);
}

static vector unsigned char __ATTRS_o_ai
vec_vsl(vector unsigned char a, vector unsigned short b)
{
  return (vector unsigned char)
           __builtin_altivec_vsl((vector int)a, (vector int)b);
}

static vector unsigned char __ATTRS_o_ai
vec_vsl(vector unsigned char a, vector unsigned int b)
{
  return (vector unsigned char)
           __builtin_altivec_vsl((vector int)a, (vector int)b);
}

static vector bool char __ATTRS_o_ai
vec_vsl(vector bool char a, vector unsigned char b)
{
  return (vector bool char)__builtin_altivec_vsl((vector int)a, (vector int)b);
}

static vector bool char __ATTRS_o_ai
vec_vsl(vector bool char a, vector unsigned short b)
{
  return (vector bool char)__builtin_altivec_vsl((vector int)a, (vector int)b);
}

static vector bool char __ATTRS_o_ai
vec_vsl(vector bool char a, vector unsigned int b)
{
  return (vector bool char)__builtin_altivec_vsl((vector int)a, (vector int)b);
}

static vector short __ATTRS_o_ai
vec_vsl(vector short a, vector unsigned char b)
{
  return (vector short)__builtin_altivec_vsl((vector int)a, (vector int)b);
}

static vector short __ATTRS_o_ai
vec_vsl(vector short a, vector unsigned short b)
{
  return (vector short)__builtin_altivec_vsl((vector int)a, (vector int)b);
}

static vector short __ATTRS_o_ai
vec_vsl(vector short a, vector unsigned int b)
{
  return (vector short)__builtin_altivec_vsl((vector int)a, (vector int)b);
}

static vector unsigned short __ATTRS_o_ai
vec_vsl(vector unsigned short a, vector unsigned char b)
{
  return (vector unsigned short)
           __builtin_altivec_vsl((vector int)a, (vector int)b);
}

static vector unsigned short __ATTRS_o_ai
vec_vsl(vector unsigned short a, vector unsigned short b)
{
  return (vector unsigned short)
           __builtin_altivec_vsl((vector int)a, (vector int)b);
}

static vector unsigned short __ATTRS_o_ai
vec_vsl(vector unsigned short a, vector unsigned int b)
{
  return (vector unsigned short)
           __builtin_altivec_vsl((vector int)a, (vector int)b);
}

static vector bool short __ATTRS_o_ai
vec_vsl(vector bool short a, vector unsigned char b)
{
  return (vector bool short)__builtin_altivec_vsl((vector int)a, (vector int)b);
}

static vector bool short __ATTRS_o_ai
vec_vsl(vector bool short a, vector unsigned short b)
{
  return (vector bool short)__builtin_altivec_vsl((vector int)a, (vector int)b);
}

static vector bool short __ATTRS_o_ai
vec_vsl(vector bool short a, vector unsigned int b)
{
  return (vector bool short)__builtin_altivec_vsl((vector int)a, (vector int)b);
}

static vector pixel __ATTRS_o_ai
vec_vsl(vector pixel a, vector unsigned char b)
{
  return (vector pixel)__builtin_altivec_vsl((vector int)a, (vector int)b);
}

static vector pixel __ATTRS_o_ai
vec_vsl(vector pixel a, vector unsigned short b)
{
  return (vector pixel)__builtin_altivec_vsl((vector int)a, (vector int)b);
}

static vector pixel __ATTRS_o_ai
vec_vsl(vector pixel a, vector unsigned int b)
{
  return (vector pixel)__builtin_altivec_vsl((vector int)a, (vector int)b);
}

static vector int __ATTRS_o_ai
vec_vsl(vector int a, vector unsigned char b)
{
  return (vector int)__builtin_altivec_vsl(a, (vector int)b);
}

static vector int __ATTRS_o_ai
vec_vsl(vector int a, vector unsigned short b)
{
  return (vector int)__builtin_altivec_vsl(a, (vector int)b);
}

static vector int __ATTRS_o_ai
vec_vsl(vector int a, vector unsigned int b)
{
  return (vector int)__builtin_altivec_vsl(a, (vector int)b);
}

static vector unsigned int __ATTRS_o_ai
vec_vsl(vector unsigned int a, vector unsigned char b)
{
  return (vector unsigned int)
           __builtin_altivec_vsl((vector int)a, (vector int)b);
}

static vector unsigned int __ATTRS_o_ai
vec_vsl(vector unsigned int a, vector unsigned short b)
{
  return (vector unsigned int)
           __builtin_altivec_vsl((vector int)a, (vector int)b);
}

static vector unsigned int __ATTRS_o_ai
vec_vsl(vector unsigned int a, vector unsigned int b)
{
  return (vector unsigned int)
           __builtin_altivec_vsl((vector int)a, (vector int)b);
}

static vector bool int __ATTRS_o_ai
vec_vsl(vector bool int a, vector unsigned char b)
{
  return (vector bool int)__builtin_altivec_vsl((vector int)a, (vector int)b);
}

static vector bool int __ATTRS_o_ai
vec_vsl(vector bool int a, vector unsigned short b)
{
  return (vector bool int)__builtin_altivec_vsl((vector int)a, (vector int)b);
}

static vector bool int __ATTRS_o_ai
vec_vsl(vector bool int a, vector unsigned int b)
{
  return (vector bool int)__builtin_altivec_vsl((vector int)a, (vector int)b);
}

/* vec_slo */

static vector signed char __ATTRS_o_ai
vec_slo(vector signed char a, vector signed char b)
{
  return (vector signed char)
           __builtin_altivec_vslo((vector int)a, (vector int)b);
}

static vector signed char __ATTRS_o_ai
vec_slo(vector signed char a, vector unsigned char b)
{
  return (vector signed char)
           __builtin_altivec_vslo((vector int)a, (vector int)b);
}

static vector unsigned char __ATTRS_o_ai
vec_slo(vector unsigned char a, vector signed char b)
{
  return (vector unsigned char)
           __builtin_altivec_vslo((vector int)a, (vector int)b);
}

static vector unsigned char __ATTRS_o_ai
vec_slo(vector unsigned char a, vector unsigned char b)
{
  return (vector unsigned char)
           __builtin_altivec_vslo((vector int)a, (vector int)b);
}

static vector short __ATTRS_o_ai
vec_slo(vector short a, vector signed char b)
{
  return (vector short)__builtin_altivec_vslo((vector int)a, (vector int)b);
}

static vector short __ATTRS_o_ai
vec_slo(vector short a, vector unsigned char b)
{
  return (vector short)__builtin_altivec_vslo((vector int)a, (vector int)b);
}

static vector unsigned short __ATTRS_o_ai
vec_slo(vector unsigned short a, vector signed char b)
{
  return (vector unsigned short)
           __builtin_altivec_vslo((vector int)a, (vector int)b);
}

static vector unsigned short __ATTRS_o_ai
vec_slo(vector unsigned short a, vector unsigned char b)
{
  return (vector unsigned short)
           __builtin_altivec_vslo((vector int)a, (vector int)b);
}

static vector pixel __ATTRS_o_ai
vec_slo(vector pixel a, vector signed char b)
{
  return (vector pixel)__builtin_altivec_vslo((vector int)a, (vector int)b);
}

static vector pixel __ATTRS_o_ai
vec_slo(vector pixel a, vector unsigned char b)
{
  return (vector pixel)__builtin_altivec_vslo((vector int)a, (vector int)b);
}

static vector int __ATTRS_o_ai
vec_slo(vector int a, vector signed char b)
{
  return (vector int)__builtin_altivec_vslo(a, (vector int)b);
}

static vector int __ATTRS_o_ai
vec_slo(vector int a, vector unsigned char b)
{
  return (vector int)__builtin_altivec_vslo(a, (vector int)b);
}

static vector unsigned int __ATTRS_o_ai
vec_slo(vector unsigned int a, vector signed char b)
{
  return (vector unsigned int)
           __builtin_altivec_vslo((vector int)a, (vector int)b);
}

static vector unsigned int __ATTRS_o_ai
vec_slo(vector unsigned int a, vector unsigned char b)
{
  return (vector unsigned int)
           __builtin_altivec_vslo((vector int)a, (vector int)b);
}

static vector float __ATTRS_o_ai
vec_slo(vector float a, vector signed char b)
{
  return (vector float)__builtin_altivec_vslo((vector int)a, (vector int)b);
}

static vector float __ATTRS_o_ai
vec_slo(vector float a, vector unsigned char b)
{
  return (vector float)__builtin_altivec_vslo((vector int)a, (vector int)b);
}

/* vec_vslo */

static vector signed char __ATTRS_o_ai
vec_vslo(vector signed char a, vector signed char b)
{
  return (vector signed char)
           __builtin_altivec_vslo((vector int)a, (vector int)b);
}

static vector signed char __ATTRS_o_ai
vec_vslo(vector signed char a, vector unsigned char b)
{
  return (vector signed char)
           __builtin_altivec_vslo((vector int)a, (vector int)b);
}

static vector unsigned char __ATTRS_o_ai
vec_vslo(vector unsigned char a, vector signed char b)
{
  return (vector unsigned char)
           __builtin_altivec_vslo((vector int)a, (vector int)b);
}

static vector unsigned char __ATTRS_o_ai
vec_vslo(vector unsigned char a, vector unsigned char b)
{
  return (vector unsigned char)
           __builtin_altivec_vslo((vector int)a, (vector int)b);
}

static vector short __ATTRS_o_ai
vec_vslo(vector short a, vector signed char b)
{
  return (vector short)__builtin_altivec_vslo((vector int)a, (vector int)b);
}

static vector short __ATTRS_o_ai
vec_vslo(vector short a, vector unsigned char b)
{
  return (vector short)__builtin_altivec_vslo((vector int)a, (vector int)b);
}

static vector unsigned short __ATTRS_o_ai
vec_vslo(vector unsigned short a, vector signed char b)
{
  return (vector unsigned short)
           __builtin_altivec_vslo((vector int)a, (vector int)b);
}

static vector unsigned short __ATTRS_o_ai
vec_vslo(vector unsigned short a, vector unsigned char b)
{
  return (vector unsigned short)
           __builtin_altivec_vslo((vector int)a, (vector int)b);
}

static vector pixel __ATTRS_o_ai
vec_vslo(vector pixel a, vector signed char b)
{
  return (vector pixel)__builtin_altivec_vslo((vector int)a, (vector int)b);
}

static vector pixel __ATTRS_o_ai
vec_vslo(vector pixel a, vector unsigned char b)
{
  return (vector pixel)__builtin_altivec_vslo((vector int)a, (vector int)b);
}

static vector int __ATTRS_o_ai
vec_vslo(vector int a, vector signed char b)
{
  return (vector int)__builtin_altivec_vslo(a, (vector int)b);
}

static vector int __ATTRS_o_ai
vec_vslo(vector int a, vector unsigned char b)
{
  return (vector int)__builtin_altivec_vslo(a, (vector int)b);
}

static vector unsigned int __ATTRS_o_ai
vec_vslo(vector unsigned int a, vector signed char b)
{
  return (vector unsigned int)
           __builtin_altivec_vslo((vector int)a, (vector int)b);
}

static vector unsigned int __ATTRS_o_ai
vec_vslo(vector unsigned int a, vector unsigned char b)
{
  return (vector unsigned int)
           __builtin_altivec_vslo((vector int)a, (vector int)b);
}

static vector float __ATTRS_o_ai
vec_vslo(vector float a, vector signed char b)
{
  return (vector float)__builtin_altivec_vslo((vector int)a, (vector int)b);
}

static vector float __ATTRS_o_ai
vec_vslo(vector float a, vector unsigned char b)
{
  return (vector float)__builtin_altivec_vslo((vector int)a, (vector int)b);
}

/* vec_splat */

static vector signed char __ATTRS_o_ai
vec_splat(vector signed char a, unsigned char b)
{
  return vec_perm(a, a, (vector unsigned char)(b));
}

static vector unsigned char __ATTRS_o_ai
vec_splat(vector unsigned char a, unsigned char b)
{
  return vec_perm(a, a, (vector unsigned char)(b));
}

static vector bool char __ATTRS_o_ai
vec_splat(vector bool char a, unsigned char b)
{
  return vec_perm(a, a, (vector unsigned char)(b));
}

static vector short __ATTRS_o_ai
vec_splat(vector short a, unsigned char b)
{ 
  b *= 2;
  unsigned char b1=b+1;
  return vec_perm(a, a, (vector unsigned char)
    (b, b1, b, b1, b, b1, b, b1, b, b1, b, b1, b, b1, b, b1));
}

static vector unsigned short __ATTRS_o_ai
vec_splat(vector unsigned short a, unsigned char b)
{ 
  b *= 2;
  unsigned char b1=b+1;
  return vec_perm(a, a, (vector unsigned char)
    (b, b1, b, b1, b, b1, b, b1, b, b1, b, b1, b, b1, b, b1));
}

static vector bool short __ATTRS_o_ai
vec_splat(vector bool short a, unsigned char b)
{ 
  b *= 2;
  unsigned char b1=b+1;
  return vec_perm(a, a, (vector unsigned char)
    (b, b1, b, b1, b, b1, b, b1, b, b1, b, b1, b, b1, b, b1));
}

static vector pixel __ATTRS_o_ai
vec_splat(vector pixel a, unsigned char b)
{ 
  b *= 2;
  unsigned char b1=b+1;
  return vec_perm(a, a, (vector unsigned char)
    (b, b1, b, b1, b, b1, b, b1, b, b1, b, b1, b, b1, b, b1));
}

static vector int __ATTRS_o_ai
vec_splat(vector int a, unsigned char b)
{ 
  b *= 4;
  unsigned char b1=b+1, b2=b+2, b3=b+3;
  return vec_perm(a, a, (vector unsigned char)
    (b, b1, b2, b3, b, b1, b2, b3, b, b1, b2, b3, b, b1, b2, b3));
}

static vector unsigned int __ATTRS_o_ai
vec_splat(vector unsigned int a, unsigned char b)
{ 
  b *= 4;
  unsigned char b1=b+1, b2=b+2, b3=b+3;
  return vec_perm(a, a, (vector unsigned char)
    (b, b1, b2, b3, b, b1, b2, b3, b, b1, b2, b3, b, b1, b2, b3));
}

static vector bool int __ATTRS_o_ai
vec_splat(vector bool int a, unsigned char b)
{ 
  b *= 4;
  unsigned char b1=b+1, b2=b+2, b3=b+3;
  return vec_perm(a, a, (vector unsigned char)
    (b, b1, b2, b3, b, b1, b2, b3, b, b1, b2, b3, b, b1, b2, b3));
}

static vector float __ATTRS_o_ai
vec_splat(vector float a, unsigned char b)
{ 
  b *= 4;
  unsigned char b1=b+1, b2=b+2, b3=b+3;
  return vec_perm(a, a, (vector unsigned char)
    (b, b1, b2, b3, b, b1, b2, b3, b, b1, b2, b3, b, b1, b2, b3));
}

/* vec_vspltb */

#define __builtin_altivec_vspltb vec_vspltb

static vector signed char __ATTRS_o_ai
vec_vspltb(vector signed char a, unsigned char b)
{
  return vec_perm(a, a, (vector unsigned char)(b));
}

static vector unsigned char __ATTRS_o_ai
vec_vspltb(vector unsigned char a, unsigned char b)
{
  return vec_perm(a, a, (vector unsigned char)(b));
}

static vector bool char __ATTRS_o_ai
vec_vspltb(vector bool char a, unsigned char b)
{
  return vec_perm(a, a, (vector unsigned char)(b));
}

/* vec_vsplth */

#define __builtin_altivec_vsplth vec_vsplth

static vector short __ATTRS_o_ai
vec_vsplth(vector short a, unsigned char b)
{
  b *= 2;
  unsigned char b1=b+1;
  return vec_perm(a, a, (vector unsigned char)
    (b, b1, b, b1, b, b1, b, b1, b, b1, b, b1, b, b1, b, b1));
}

static vector unsigned short __ATTRS_o_ai
vec_vsplth(vector unsigned short a, unsigned char b)
{
  b *= 2;
  unsigned char b1=b+1;
  return vec_perm(a, a, (vector unsigned char)
    (b, b1, b, b1, b, b1, b, b1, b, b1, b, b1, b, b1, b, b1));
}

static vector bool short __ATTRS_o_ai
vec_vsplth(vector bool short a, unsigned char b)
{
  b *= 2;
  unsigned char b1=b+1;
  return vec_perm(a, a, (vector unsigned char)
    (b, b1, b, b1, b, b1, b, b1, b, b1, b, b1, b, b1, b, b1));
}

static vector pixel __ATTRS_o_ai
vec_vsplth(vector pixel a, unsigned char b)
{
  b *= 2;
  unsigned char b1=b+1;
  return vec_perm(a, a, (vector unsigned char)
    (b, b1, b, b1, b, b1, b, b1, b, b1, b, b1, b, b1, b, b1));
}

/* vec_vspltw */

#define __builtin_altivec_vspltw vec_vspltw

static vector int __ATTRS_o_ai
vec_vspltw(vector int a, unsigned char b)
{
  b *= 4;
  unsigned char b1=b+1, b2=b+2, b3=b+3;
  return vec_perm(a, a, (vector unsigned char)
    (b, b1, b2, b3, b, b1, b2, b3, b, b1, b2, b3, b, b1, b2, b3));
}

static vector unsigned int __ATTRS_o_ai
vec_vspltw(vector unsigned int a, unsigned char b)
{
  b *= 4;
  unsigned char b1=b+1, b2=b+2, b3=b+3;
  return vec_perm(a, a, (vector unsigned char)
    (b, b1, b2, b3, b, b1, b2, b3, b, b1, b2, b3, b, b1, b2, b3));
}

static vector bool int __ATTRS_o_ai
vec_vspltw(vector bool int a, unsigned char b)
{
  b *= 4;
  unsigned char b1=b+1, b2=b+2, b3=b+3;
  return vec_perm(a, a, (vector unsigned char)
    (b, b1, b2, b3, b, b1, b2, b3, b, b1, b2, b3, b, b1, b2, b3));
}

static vector float __ATTRS_o_ai
vec_vspltw(vector float a, unsigned char b)
{
  b *= 4;
  unsigned char b1=b+1, b2=b+2, b3=b+3;
  return vec_perm(a, a, (vector unsigned char)
    (b, b1, b2, b3, b, b1, b2, b3, b, b1, b2, b3, b, b1, b2, b3));
}

/* vec_splat_s8 */

#define __builtin_altivec_vspltisb vec_splat_s8

// FIXME: parameter should be treated as 5-bit signed literal
static vector signed char __ATTRS_o_ai
vec_splat_s8(signed char a)
{
  return (vector signed char)(a);
}

/* vec_vspltisb */

// FIXME: parameter should be treated as 5-bit signed literal
static vector signed char __ATTRS_o_ai
vec_vspltisb(signed char a)
{
  return (vector signed char)(a);
}

/* vec_splat_s16 */

#define __builtin_altivec_vspltish vec_splat_s16

// FIXME: parameter should be treated as 5-bit signed literal
static vector short __ATTRS_o_ai
vec_splat_s16(signed char a)
{
  return (vector short)(a);
}

/* vec_vspltish */

// FIXME: parameter should be treated as 5-bit signed literal
static vector short __ATTRS_o_ai
vec_vspltish(signed char a)
{
  return (vector short)(a);
}

/* vec_splat_s32 */

#define __builtin_altivec_vspltisw vec_splat_s32

// FIXME: parameter should be treated as 5-bit signed literal
static vector int __ATTRS_o_ai
vec_splat_s32(signed char a)
{
  return (vector int)(a);
}

/* vec_vspltisw */

// FIXME: parameter should be treated as 5-bit signed literal
static vector int __ATTRS_o_ai
vec_vspltisw(signed char a)
{
  return (vector int)(a);
}

/* vec_splat_u8 */

// FIXME: parameter should be treated as 5-bit signed literal
static vector unsigned char __ATTRS_o_ai
vec_splat_u8(unsigned char a)
{
  return (vector unsigned char)(a);
}

/* vec_splat_u16 */

// FIXME: parameter should be treated as 5-bit signed literal
static vector unsigned short __ATTRS_o_ai
vec_splat_u16(signed char a)
{
  return (vector unsigned short)(a);
}

/* vec_splat_u32 */

// FIXME: parameter should be treated as 5-bit signed literal
static vector unsigned int __ATTRS_o_ai
vec_splat_u32(signed char a)
{
  return (vector unsigned int)(a);
}

/* vec_sr */

static vector signed char __ATTRS_o_ai
vec_sr(vector signed char a, vector unsigned char b)
{
  return a >> (vector signed char)b;
}

static vector unsigned char __ATTRS_o_ai
vec_sr(vector unsigned char a, vector unsigned char b)
{
  return a >> b;
}

static vector short __ATTRS_o_ai
vec_sr(vector short a, vector unsigned short b)
{
  return a >> (vector short)b;
}

static vector unsigned short __ATTRS_o_ai
vec_sr(vector unsigned short a, vector unsigned short b)
{
  return a >> b;
}

static vector int __ATTRS_o_ai
vec_sr(vector int a, vector unsigned int b)
{
  return a >> (vector int)b;
}

static vector unsigned int __ATTRS_o_ai
vec_sr(vector unsigned int a, vector unsigned int b)
{
  return a >> b;
}

/* vec_vsrb */

#define __builtin_altivec_vsrb vec_vsrb

static vector signed char __ATTRS_o_ai
vec_vsrb(vector signed char a, vector unsigned char b)
{
  return a >> (vector signed char)b;
}

static vector unsigned char __ATTRS_o_ai
vec_vsrb(vector unsigned char a, vector unsigned char b)
{
  return a >> b;
}

/* vec_vsrh */

#define __builtin_altivec_vsrh vec_vsrh

static vector short __ATTRS_o_ai
vec_vsrh(vector short a, vector unsigned short b)
{
  return a >> (vector short)b;
}

static vector unsigned short __ATTRS_o_ai
vec_vsrh(vector unsigned short a, vector unsigned short b)
{
  return a >> b;
}

/* vec_vsrw */

#define __builtin_altivec_vsrw vec_vsrw

static vector int __ATTRS_o_ai
vec_vsrw(vector int a, vector unsigned int b)
{
  return a >> (vector int)b;
}

static vector unsigned int __ATTRS_o_ai
vec_vsrw(vector unsigned int a, vector unsigned int b)
{
  return a >> b;
}

/* vec_sra */

static vector signed char __ATTRS_o_ai
vec_sra(vector signed char a, vector unsigned char b)
{
  return (vector signed char)__builtin_altivec_vsrab((vector char)a, b);
}

static vector unsigned char __ATTRS_o_ai
vec_sra(vector unsigned char a, vector unsigned char b)
{
  return (vector unsigned char)__builtin_altivec_vsrab((vector char)a, b);
}

static vector short __ATTRS_o_ai
vec_sra(vector short a, vector unsigned short b)
{
  return __builtin_altivec_vsrah(a, (vector unsigned short)b);
}

static vector unsigned short __ATTRS_o_ai
vec_sra(vector unsigned short a, vector unsigned short b)
{
  return (vector unsigned short)__builtin_altivec_vsrah((vector short)a, b);
}

static vector int __ATTRS_o_ai
vec_sra(vector int a, vector unsigned int b)
{
  return __builtin_altivec_vsraw(a, b);
}

static vector unsigned int __ATTRS_o_ai
vec_sra(vector unsigned int a, vector unsigned int b)
{
  return (vector unsigned int)__builtin_altivec_vsraw((vector int)a, b);
}

/* vec_vsrab */

static vector signed char __ATTRS_o_ai
vec_vsrab(vector signed char a, vector unsigned char b)
{
  return (vector signed char)__builtin_altivec_vsrab((vector char)a, b);
}

static vector unsigned char __ATTRS_o_ai
vec_vsrab(vector unsigned char a, vector unsigned char b)
{
  return (vector unsigned char)__builtin_altivec_vsrab((vector char)a, b);
}

/* vec_vsrah */

static vector short __ATTRS_o_ai
vec_vsrah(vector short a, vector unsigned short b)
{
  return __builtin_altivec_vsrah(a, (vector unsigned short)b);
}

static vector unsigned short __ATTRS_o_ai
vec_vsrah(vector unsigned short a, vector unsigned short b)
{
  return (vector unsigned short)__builtin_altivec_vsrah((vector short)a, b);
}

/* vec_vsraw */

static vector int __ATTRS_o_ai
vec_vsraw(vector int a, vector unsigned int b)
{
  return __builtin_altivec_vsraw(a, b);
}

static vector unsigned int __ATTRS_o_ai
vec_vsraw(vector unsigned int a, vector unsigned int b)
{
  return (vector unsigned int)__builtin_altivec_vsraw((vector int)a, b);
}

/* vec_srl */

static vector signed char __ATTRS_o_ai
vec_srl(vector signed char a, vector unsigned char b)
{
  return (vector signed char)
           __builtin_altivec_vsr((vector int)a, (vector int)b);
}

static vector signed char __ATTRS_o_ai
vec_srl(vector signed char a, vector unsigned short b)
{
  return (vector signed char)
           __builtin_altivec_vsr((vector int)a, (vector int)b);
}

static vector signed char __ATTRS_o_ai
vec_srl(vector signed char a, vector unsigned int b)
{
  return (vector signed char)
           __builtin_altivec_vsr((vector int)a, (vector int)b);
}

static vector unsigned char __ATTRS_o_ai
vec_srl(vector unsigned char a, vector unsigned char b)
{
  return (vector unsigned char)
           __builtin_altivec_vsr((vector int)a, (vector int)b);
}

static vector unsigned char __ATTRS_o_ai
vec_srl(vector unsigned char a, vector unsigned short b)
{
  return (vector unsigned char)
           __builtin_altivec_vsr((vector int)a, (vector int)b);
}

static vector unsigned char __ATTRS_o_ai
vec_srl(vector unsigned char a, vector unsigned int b)
{
  return (vector unsigned char)
           __builtin_altivec_vsr((vector int)a, (vector int)b);
}

static vector bool char __ATTRS_o_ai
vec_srl(vector bool char a, vector unsigned char b)
{
  return (vector bool char)__builtin_altivec_vsr((vector int)a, (vector int)b);
}

static vector bool char __ATTRS_o_ai
vec_srl(vector bool char a, vector unsigned short b)
{
  return (vector bool char)__builtin_altivec_vsr((vector int)a, (vector int)b);
}

static vector bool char __ATTRS_o_ai
vec_srl(vector bool char a, vector unsigned int b)
{
  return (vector bool char)__builtin_altivec_vsr((vector int)a, (vector int)b);
}

static vector short __ATTRS_o_ai
vec_srl(vector short a, vector unsigned char b)
{
  return (vector short)__builtin_altivec_vsr((vector int)a, (vector int)b);
}

static vector short __ATTRS_o_ai
vec_srl(vector short a, vector unsigned short b)
{
  return (vector short)__builtin_altivec_vsr((vector int)a, (vector int)b);
}

static vector short __ATTRS_o_ai
vec_srl(vector short a, vector unsigned int b)
{
  return (vector short)__builtin_altivec_vsr((vector int)a, (vector int)b);
}

static vector unsigned short __ATTRS_o_ai
vec_srl(vector unsigned short a, vector unsigned char b)
{
  return (vector unsigned short)
           __builtin_altivec_vsr((vector int)a, (vector int)b);
}

static vector unsigned short __ATTRS_o_ai
vec_srl(vector unsigned short a, vector unsigned short b)
{
  return (vector unsigned short)
           __builtin_altivec_vsr((vector int)a, (vector int)b);
}

static vector unsigned short __ATTRS_o_ai
vec_srl(vector unsigned short a, vector unsigned int b)
{
  return (vector unsigned short)
           __builtin_altivec_vsr((vector int)a, (vector int)b);
}

static vector bool short __ATTRS_o_ai
vec_srl(vector bool short a, vector unsigned char b)
{
  return (vector bool short)__builtin_altivec_vsr((vector int)a, (vector int)b);
}

static vector bool short __ATTRS_o_ai
vec_srl(vector bool short a, vector unsigned short b)
{
  return (vector bool short)__builtin_altivec_vsr((vector int)a, (vector int)b);
}

static vector bool short __ATTRS_o_ai
vec_srl(vector bool short a, vector unsigned int b)
{
  return (vector bool short)__builtin_altivec_vsr((vector int)a, (vector int)b);
}

static vector pixel __ATTRS_o_ai
vec_srl(vector pixel a, vector unsigned char b)
{
  return (vector pixel)__builtin_altivec_vsr((vector int)a, (vector int)b);
}

static vector pixel __ATTRS_o_ai
vec_srl(vector pixel a, vector unsigned short b)
{
  return (vector pixel)__builtin_altivec_vsr((vector int)a, (vector int)b);
}

static vector pixel __ATTRS_o_ai
vec_srl(vector pixel a, vector unsigned int b)
{
  return (vector pixel)__builtin_altivec_vsr((vector int)a, (vector int)b);
}

static vector int __ATTRS_o_ai
vec_srl(vector int a, vector unsigned char b)
{
  return (vector int)__builtin_altivec_vsr(a, (vector int)b);
}

static vector int __ATTRS_o_ai
vec_srl(vector int a, vector unsigned short b)
{
  return (vector int)__builtin_altivec_vsr(a, (vector int)b);
}

static vector int __ATTRS_o_ai
vec_srl(vector int a, vector unsigned int b)
{
  return (vector int)__builtin_altivec_vsr(a, (vector int)b);
}

static vector unsigned int __ATTRS_o_ai
vec_srl(vector unsigned int a, vector unsigned char b)
{
  return (vector unsigned int)
           __builtin_altivec_vsr((vector int)a, (vector int)b);
}

static vector unsigned int __ATTRS_o_ai
vec_srl(vector unsigned int a, vector unsigned short b)
{
  return (vector unsigned int)
           __builtin_altivec_vsr((vector int)a, (vector int)b);
}

static vector unsigned int __ATTRS_o_ai
vec_srl(vector unsigned int a, vector unsigned int b)
{
  return (vector unsigned int)
           __builtin_altivec_vsr((vector int)a, (vector int)b);
}

static vector bool int __ATTRS_o_ai
vec_srl(vector bool int a, vector unsigned char b)
{
  return (vector bool int)__builtin_altivec_vsr((vector int)a, (vector int)b);
}

static vector bool int __ATTRS_o_ai
vec_srl(vector bool int a, vector unsigned short b)
{
  return (vector bool int)__builtin_altivec_vsr((vector int)a, (vector int)b);
}

static vector bool int __ATTRS_o_ai
vec_srl(vector bool int a, vector unsigned int b)
{
  return (vector bool int)__builtin_altivec_vsr((vector int)a, (vector int)b);
}

/* vec_vsr */

static vector signed char __ATTRS_o_ai
vec_vsr(vector signed char a, vector unsigned char b)
{
  return (vector signed char)
           __builtin_altivec_vsr((vector int)a, (vector int)b);
}

static vector signed char __ATTRS_o_ai
vec_vsr(vector signed char a, vector unsigned short b)
{
  return (vector signed char)
           __builtin_altivec_vsr((vector int)a, (vector int)b);
}

static vector signed char __ATTRS_o_ai
vec_vsr(vector signed char a, vector unsigned int b)
{
  return (vector signed char)
           __builtin_altivec_vsr((vector int)a, (vector int)b);
}

static vector unsigned char __ATTRS_o_ai
vec_vsr(vector unsigned char a, vector unsigned char b)
{
  return (vector unsigned char)
           __builtin_altivec_vsr((vector int)a, (vector int)b);
}

static vector unsigned char __ATTRS_o_ai
vec_vsr(vector unsigned char a, vector unsigned short b)
{
  return (vector unsigned char)
           __builtin_altivec_vsr((vector int)a, (vector int)b);
}

static vector unsigned char __ATTRS_o_ai
vec_vsr(vector unsigned char a, vector unsigned int b)
{
  return (vector unsigned char)
           __builtin_altivec_vsr((vector int)a, (vector int)b);
}

static vector bool char __ATTRS_o_ai
vec_vsr(vector bool char a, vector unsigned char b)
{
  return (vector bool char)__builtin_altivec_vsr((vector int)a, (vector int)b);
}

static vector bool char __ATTRS_o_ai
vec_vsr(vector bool char a, vector unsigned short b)
{
  return (vector bool char)__builtin_altivec_vsr((vector int)a, (vector int)b);
}

static vector bool char __ATTRS_o_ai
vec_vsr(vector bool char a, vector unsigned int b)
{
  return (vector bool char)__builtin_altivec_vsr((vector int)a, (vector int)b);
}

static vector short __ATTRS_o_ai
vec_vsr(vector short a, vector unsigned char b)
{
  return (vector short)__builtin_altivec_vsr((vector int)a, (vector int)b);
}

static vector short __ATTRS_o_ai
vec_vsr(vector short a, vector unsigned short b)
{
  return (vector short)__builtin_altivec_vsr((vector int)a, (vector int)b);
}

static vector short __ATTRS_o_ai
vec_vsr(vector short a, vector unsigned int b)
{
  return (vector short)__builtin_altivec_vsr((vector int)a, (vector int)b);
}

static vector unsigned short __ATTRS_o_ai
vec_vsr(vector unsigned short a, vector unsigned char b)
{
  return (vector unsigned short)
           __builtin_altivec_vsr((vector int)a, (vector int)b);
}

static vector unsigned short __ATTRS_o_ai
vec_vsr(vector unsigned short a, vector unsigned short b)
{
  return (vector unsigned short)
           __builtin_altivec_vsr((vector int)a, (vector int)b);
}

static vector unsigned short __ATTRS_o_ai
vec_vsr(vector unsigned short a, vector unsigned int b)
{
  return (vector unsigned short)
           __builtin_altivec_vsr((vector int)a, (vector int)b);
}

static vector bool short __ATTRS_o_ai
vec_vsr(vector bool short a, vector unsigned char b)
{
  return (vector bool short)__builtin_altivec_vsr((vector int)a, (vector int)b);
}

static vector bool short __ATTRS_o_ai
vec_vsr(vector bool short a, vector unsigned short b)
{
  return (vector bool short)__builtin_altivec_vsr((vector int)a, (vector int)b);
}

static vector bool short __ATTRS_o_ai
vec_vsr(vector bool short a, vector unsigned int b)
{
  return (vector bool short)__builtin_altivec_vsr((vector int)a, (vector int)b);
}

static vector pixel __ATTRS_o_ai
vec_vsr(vector pixel a, vector unsigned char b)
{
  return (vector pixel)__builtin_altivec_vsr((vector int)a, (vector int)b);
}

static vector pixel __ATTRS_o_ai
vec_vsr(vector pixel a, vector unsigned short b)
{
  return (vector pixel)__builtin_altivec_vsr((vector int)a, (vector int)b);
}

static vector pixel __ATTRS_o_ai
vec_vsr(vector pixel a, vector unsigned int b)
{
  return (vector pixel)__builtin_altivec_vsr((vector int)a, (vector int)b);
}

static vector int __ATTRS_o_ai
vec_vsr(vector int a, vector unsigned char b)
{
  return (vector int)__builtin_altivec_vsr(a, (vector int)b);
}

static vector int __ATTRS_o_ai
vec_vsr(vector int a, vector unsigned short b)
{
  return (vector int)__builtin_altivec_vsr(a, (vector int)b);
}

static vector int __ATTRS_o_ai
vec_vsr(vector int a, vector unsigned int b)
{
  return (vector int)__builtin_altivec_vsr(a, (vector int)b);
}

static vector unsigned int __ATTRS_o_ai
vec_vsr(vector unsigned int a, vector unsigned char b)
{
  return (vector unsigned int)
           __builtin_altivec_vsr((vector int)a, (vector int)b);
}

static vector unsigned int __ATTRS_o_ai
vec_vsr(vector unsigned int a, vector unsigned short b)
{
  return (vector unsigned int)
           __builtin_altivec_vsr((vector int)a, (vector int)b);
}

static vector unsigned int __ATTRS_o_ai
vec_vsr(vector unsigned int a, vector unsigned int b)
{
  return (vector unsigned int)
           __builtin_altivec_vsr((vector int)a, (vector int)b);
}

static vector bool int __ATTRS_o_ai
vec_vsr(vector bool int a, vector unsigned char b)
{
  return (vector bool int)__builtin_altivec_vsr((vector int)a, (vector int)b);
}

static vector bool int __ATTRS_o_ai
vec_vsr(vector bool int a, vector unsigned short b)
{
  return (vector bool int)__builtin_altivec_vsr((vector int)a, (vector int)b);
}

static vector bool int __ATTRS_o_ai
vec_vsr(vector bool int a, vector unsigned int b)
{
  return (vector bool int)__builtin_altivec_vsr((vector int)a, (vector int)b);
}

/* vec_sro */

static vector signed char __ATTRS_o_ai
vec_sro(vector signed char a, vector signed char b)
{
  return (vector signed char)
           __builtin_altivec_vsro((vector int)a, (vector int)b);
}

static vector signed char __ATTRS_o_ai
vec_sro(vector signed char a, vector unsigned char b)
{
  return (vector signed char)
           __builtin_altivec_vsro((vector int)a, (vector int)b);
}

static vector unsigned char __ATTRS_o_ai
vec_sro(vector unsigned char a, vector signed char b)
{
  return (vector unsigned char)
           __builtin_altivec_vsro((vector int)a, (vector int)b);
}

static vector unsigned char __ATTRS_o_ai
vec_sro(vector unsigned char a, vector unsigned char b)
{
  return (vector unsigned char)
           __builtin_altivec_vsro((vector int)a, (vector int)b);
}

static vector short __ATTRS_o_ai
vec_sro(vector short a, vector signed char b)
{
  return (vector short)__builtin_altivec_vsro((vector int)a, (vector int)b);
}

static vector short __ATTRS_o_ai
vec_sro(vector short a, vector unsigned char b)
{
  return (vector short)__builtin_altivec_vsro((vector int)a, (vector int)b);
}

static vector unsigned short __ATTRS_o_ai
vec_sro(vector unsigned short a, vector signed char b)
{
  return (vector unsigned short)
           __builtin_altivec_vsro((vector int)a, (vector int)b);
}

static vector unsigned short __ATTRS_o_ai
vec_sro(vector unsigned short a, vector unsigned char b)
{
  return (vector unsigned short)
           __builtin_altivec_vsro((vector int)a, (vector int)b);
}

static vector pixel __ATTRS_o_ai
vec_sro(vector pixel a, vector signed char b)
{
  return (vector pixel)__builtin_altivec_vsro((vector int)a, (vector int)b);
}

static vector pixel __ATTRS_o_ai
vec_sro(vector pixel a, vector unsigned char b)
{
  return (vector pixel)__builtin_altivec_vsro((vector int)a, (vector int)b);
}

static vector int __ATTRS_o_ai
vec_sro(vector int a, vector signed char b)
{
  return (vector int)__builtin_altivec_vsro(a, (vector int)b);
}

static vector int __ATTRS_o_ai
vec_sro(vector int a, vector unsigned char b)
{
  return (vector int)__builtin_altivec_vsro(a, (vector int)b);
}

static vector unsigned int __ATTRS_o_ai
vec_sro(vector unsigned int a, vector signed char b)
{
  return (vector unsigned int)
           __builtin_altivec_vsro((vector int)a, (vector int)b);
}

static vector unsigned int __ATTRS_o_ai
vec_sro(vector unsigned int a, vector unsigned char b)
{
  return (vector unsigned int)
           __builtin_altivec_vsro((vector int)a, (vector int)b);
}

static vector float __ATTRS_o_ai
vec_sro(vector float a, vector signed char b)
{
  return (vector float)__builtin_altivec_vsro((vector int)a, (vector int)b);
}

static vector float __ATTRS_o_ai
vec_sro(vector float a, vector unsigned char b)
{
  return (vector float)__builtin_altivec_vsro((vector int)a, (vector int)b);
}

/* vec_vsro */

static vector signed char __ATTRS_o_ai
vec_vsro(vector signed char a, vector signed char b)
{
  return (vector signed char)
           __builtin_altivec_vsro((vector int)a, (vector int)b);
}

static vector signed char __ATTRS_o_ai
vec_vsro(vector signed char a, vector unsigned char b)
{
  return (vector signed char)
           __builtin_altivec_vsro((vector int)a, (vector int)b);
}

static vector unsigned char __ATTRS_o_ai
vec_vsro(vector unsigned char a, vector signed char b)
{
  return (vector unsigned char)
           __builtin_altivec_vsro((vector int)a, (vector int)b);
}

static vector unsigned char __ATTRS_o_ai
vec_vsro(vector unsigned char a, vector unsigned char b)
{
  return (vector unsigned char)
           __builtin_altivec_vsro((vector int)a, (vector int)b);
}

static vector short __ATTRS_o_ai
vec_vsro(vector short a, vector signed char b)
{
  return (vector short)__builtin_altivec_vsro((vector int)a, (vector int)b);
}

static vector short __ATTRS_o_ai
vec_vsro(vector short a, vector unsigned char b)
{
  return (vector short)__builtin_altivec_vsro((vector int)a, (vector int)b);
}

static vector unsigned short __ATTRS_o_ai
vec_vsro(vector unsigned short a, vector signed char b)
{
  return (vector unsigned short)
           __builtin_altivec_vsro((vector int)a, (vector int)b);
}

static vector unsigned short __ATTRS_o_ai
vec_vsro(vector unsigned short a, vector unsigned char b)
{
  return (vector unsigned short)
           __builtin_altivec_vsro((vector int)a, (vector int)b);
}

static vector pixel __ATTRS_o_ai
vec_vsro(vector pixel a, vector signed char b)
{
  return (vector pixel)__builtin_altivec_vsro((vector int)a, (vector int)b);
}

static vector pixel __ATTRS_o_ai
vec_vsro(vector pixel a, vector unsigned char b)
{
  return (vector pixel)__builtin_altivec_vsro((vector int)a, (vector int)b);
}

static vector int __ATTRS_o_ai
vec_vsro(vector int a, vector signed char b)
{
  return (vector int)__builtin_altivec_vsro(a, (vector int)b);
}

static vector int __ATTRS_o_ai
vec_vsro(vector int a, vector unsigned char b)
{
  return (vector int)__builtin_altivec_vsro(a, (vector int)b);
}

static vector unsigned int __ATTRS_o_ai
vec_vsro(vector unsigned int a, vector signed char b)
{
  return (vector unsigned int)
           __builtin_altivec_vsro((vector int)a, (vector int)b);
}

static vector unsigned int __ATTRS_o_ai
vec_vsro(vector unsigned int a, vector unsigned char b)
{
  return (vector unsigned int)
           __builtin_altivec_vsro((vector int)a, (vector int)b);
}

static vector float __ATTRS_o_ai
vec_vsro(vector float a, vector signed char b)
{
  return (vector float)__builtin_altivec_vsro((vector int)a, (vector int)b);
}

static vector float __ATTRS_o_ai
vec_vsro(vector float a, vector unsigned char b)
{
  return (vector float)__builtin_altivec_vsro((vector int)a, (vector int)b);
}

/* vec_st */

static void __ATTRS_o_ai
vec_st(vector signed char a, int b, vector signed char *c)
{
  __builtin_altivec_stvx((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_st(vector signed char a, int b, signed char *c)
{
  __builtin_altivec_stvx((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_st(vector unsigned char a, int b, vector unsigned char *c)
{
  __builtin_altivec_stvx((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_st(vector unsigned char a, int b, unsigned char *c)
{
  __builtin_altivec_stvx((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_st(vector bool char a, int b, signed char *c)
{
  __builtin_altivec_stvx((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_st(vector bool char a, int b, unsigned char *c)
{
  __builtin_altivec_stvx((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_st(vector bool char a, int b, vector bool char *c)
{
  __builtin_altivec_stvx((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_st(vector short a, int b, vector short *c)
{
  __builtin_altivec_stvx((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_st(vector short a, int b, short *c)
{
  __builtin_altivec_stvx((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_st(vector unsigned short a, int b, vector unsigned short *c)
{
  __builtin_altivec_stvx((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_st(vector unsigned short a, int b, unsigned short *c)
{
  __builtin_altivec_stvx((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_st(vector bool short a, int b, short *c)
{
  __builtin_altivec_stvx((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_st(vector bool short a, int b, unsigned short *c)
{
  __builtin_altivec_stvx((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_st(vector bool short a, int b, vector bool short *c)
{
  __builtin_altivec_stvx((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_st(vector pixel a, int b, short *c)
{
  __builtin_altivec_stvx((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_st(vector pixel a, int b, unsigned short *c)
{
  __builtin_altivec_stvx((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_st(vector pixel a, int b, vector pixel *c)
{
  __builtin_altivec_stvx((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_st(vector int a, int b, vector int *c)
{
  __builtin_altivec_stvx(a, b, c);
}

static void __ATTRS_o_ai
vec_st(vector int a, int b, int *c)
{
  __builtin_altivec_stvx(a, b, c);
}

static void __ATTRS_o_ai
vec_st(vector unsigned int a, int b, vector unsigned int *c)
{
  __builtin_altivec_stvx((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_st(vector unsigned int a, int b, unsigned int *c)
{
  __builtin_altivec_stvx((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_st(vector bool int a, int b, int *c)
{
  __builtin_altivec_stvx((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_st(vector bool int a, int b, unsigned int *c)
{
  __builtin_altivec_stvx((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_st(vector bool int a, int b, vector bool int *c)
{
  __builtin_altivec_stvx((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_st(vector float a, int b, vector float *c)
{
  __builtin_altivec_stvx((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_st(vector float a, int b, float *c)
{
  __builtin_altivec_stvx((vector int)a, b, c);
}

/* vec_stvx */

static void __ATTRS_o_ai
vec_stvx(vector signed char a, int b, vector signed char *c)
{
  __builtin_altivec_stvx((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_stvx(vector signed char a, int b, signed char *c)
{
  __builtin_altivec_stvx((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_stvx(vector unsigned char a, int b, vector unsigned char *c)
{
  __builtin_altivec_stvx((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_stvx(vector unsigned char a, int b, unsigned char *c)
{
  __builtin_altivec_stvx((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_stvx(vector bool char a, int b, signed char *c)
{
  __builtin_altivec_stvx((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_stvx(vector bool char a, int b, unsigned char *c)
{
  __builtin_altivec_stvx((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_stvx(vector bool char a, int b, vector bool char *c)
{
  __builtin_altivec_stvx((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_stvx(vector short a, int b, vector short *c)
{
  __builtin_altivec_stvx((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_stvx(vector short a, int b, short *c)
{
  __builtin_altivec_stvx((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_stvx(vector unsigned short a, int b, vector unsigned short *c)
{
  __builtin_altivec_stvx((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_stvx(vector unsigned short a, int b, unsigned short *c)
{
  __builtin_altivec_stvx((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_stvx(vector bool short a, int b, short *c)
{
  __builtin_altivec_stvx((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_stvx(vector bool short a, int b, unsigned short *c)
{
  __builtin_altivec_stvx((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_stvx(vector bool short a, int b, vector bool short *c)
{
  __builtin_altivec_stvx((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_stvx(vector pixel a, int b, short *c)
{
  __builtin_altivec_stvx((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_stvx(vector pixel a, int b, unsigned short *c)
{
  __builtin_altivec_stvx((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_stvx(vector pixel a, int b, vector pixel *c)
{
  __builtin_altivec_stvx((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_stvx(vector int a, int b, vector int *c)
{
  __builtin_altivec_stvx(a, b, c);
}

static void __ATTRS_o_ai
vec_stvx(vector int a, int b, int *c)
{
  __builtin_altivec_stvx(a, b, c);
}

static void __ATTRS_o_ai
vec_stvx(vector unsigned int a, int b, vector unsigned int *c)
{
  __builtin_altivec_stvx((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_stvx(vector unsigned int a, int b, unsigned int *c)
{
  __builtin_altivec_stvx((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_stvx(vector bool int a, int b, int *c)
{
  __builtin_altivec_stvx((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_stvx(vector bool int a, int b, unsigned int *c)
{
  __builtin_altivec_stvx((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_stvx(vector bool int a, int b, vector bool int *c)
{
  __builtin_altivec_stvx((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_stvx(vector float a, int b, vector float *c)
{
  __builtin_altivec_stvx((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_stvx(vector float a, int b, float *c)
{
  __builtin_altivec_stvx((vector int)a, b, c);
}

/* vec_ste */

static void __ATTRS_o_ai
vec_ste(vector signed char a, int b, signed char *c)
{
  __builtin_altivec_stvebx((vector char)a, b, c);
}

static void __ATTRS_o_ai
vec_ste(vector unsigned char a, int b, unsigned char *c)
{
  __builtin_altivec_stvebx((vector char)a, b, c);
}

static void __ATTRS_o_ai
vec_ste(vector bool char a, int b, signed char *c)
{
  __builtin_altivec_stvebx((vector char)a, b, c);
}

static void __ATTRS_o_ai
vec_ste(vector bool char a, int b, unsigned char *c)
{
  __builtin_altivec_stvebx((vector char)a, b, c);
}

static void __ATTRS_o_ai
vec_ste(vector short a, int b, short *c)
{
  __builtin_altivec_stvehx(a, b, c);
}

static void __ATTRS_o_ai
vec_ste(vector unsigned short a, int b, unsigned short *c)
{
  __builtin_altivec_stvehx((vector short)a, b, c);
}

static void __ATTRS_o_ai
vec_ste(vector bool short a, int b, short *c)
{
  __builtin_altivec_stvehx((vector short)a, b, c);
}

static void __ATTRS_o_ai
vec_ste(vector bool short a, int b, unsigned short *c)
{
  __builtin_altivec_stvehx((vector short)a, b, c);
}

static void __ATTRS_o_ai
vec_ste(vector pixel a, int b, short *c)
{
  __builtin_altivec_stvehx((vector short)a, b, c);
}

static void __ATTRS_o_ai
vec_ste(vector pixel a, int b, unsigned short *c)
{
  __builtin_altivec_stvehx((vector short)a, b, c);
}

static void __ATTRS_o_ai
vec_ste(vector int a, int b, int *c)
{
  __builtin_altivec_stvewx(a, b, c);
}

static void __ATTRS_o_ai
vec_ste(vector unsigned int a, int b, unsigned int *c)
{
  __builtin_altivec_stvewx((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_ste(vector bool int a, int b, int *c)
{
  __builtin_altivec_stvewx((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_ste(vector bool int a, int b, unsigned int *c)
{
  __builtin_altivec_stvewx((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_ste(vector float a, int b, float *c)
{
  __builtin_altivec_stvewx((vector int)a, b, c);
}

/* vec_stvebx */

static void __ATTRS_o_ai
vec_stvebx(vector signed char a, int b, signed char *c)
{
  __builtin_altivec_stvebx((vector char)a, b, c);
}

static void __ATTRS_o_ai
vec_stvebx(vector unsigned char a, int b, unsigned char *c)
{
  __builtin_altivec_stvebx((vector char)a, b, c);
}

static void __ATTRS_o_ai
vec_stvebx(vector bool char a, int b, signed char *c)
{
  __builtin_altivec_stvebx((vector char)a, b, c);
}

static void __ATTRS_o_ai
vec_stvebx(vector bool char a, int b, unsigned char *c)
{
  __builtin_altivec_stvebx((vector char)a, b, c);
}

/* vec_stvehx */

static void __ATTRS_o_ai
vec_stvehx(vector short a, int b, short *c)
{
  __builtin_altivec_stvehx(a, b, c);
}

static void __ATTRS_o_ai
vec_stvehx(vector unsigned short a, int b, unsigned short *c)
{
  __builtin_altivec_stvehx((vector short)a, b, c);
}

static void __ATTRS_o_ai
vec_stvehx(vector bool short a, int b, short *c)
{
  __builtin_altivec_stvehx((vector short)a, b, c);
}

static void __ATTRS_o_ai
vec_stvehx(vector bool short a, int b, unsigned short *c)
{
  __builtin_altivec_stvehx((vector short)a, b, c);
}

static void __ATTRS_o_ai
vec_stvehx(vector pixel a, int b, short *c)
{
  __builtin_altivec_stvehx((vector short)a, b, c);
}

static void __ATTRS_o_ai
vec_stvehx(vector pixel a, int b, unsigned short *c)
{
  __builtin_altivec_stvehx((vector short)a, b, c);
}

/* vec_stvewx */

static void __ATTRS_o_ai
vec_stvewx(vector int a, int b, int *c)
{
  __builtin_altivec_stvewx(a, b, c);
}

static void __ATTRS_o_ai
vec_stvewx(vector unsigned int a, int b, unsigned int *c)
{
  __builtin_altivec_stvewx((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_stvewx(vector bool int a, int b, int *c)
{
  __builtin_altivec_stvewx((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_stvewx(vector bool int a, int b, unsigned int *c)
{
  __builtin_altivec_stvewx((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_stvewx(vector float a, int b, float *c)
{
  __builtin_altivec_stvewx((vector int)a, b, c);
}

/* vec_stl */

static void __ATTRS_o_ai
vec_stl(vector signed char a, int b, vector signed char *c)
{
  __builtin_altivec_stvxl((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_stl(vector signed char a, int b, signed char *c)
{
  __builtin_altivec_stvxl((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_stl(vector unsigned char a, int b, vector unsigned char *c)
{
  __builtin_altivec_stvxl((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_stl(vector unsigned char a, int b, unsigned char *c)
{
  __builtin_altivec_stvxl((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_stl(vector bool char a, int b, signed char *c)
{
  __builtin_altivec_stvxl((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_stl(vector bool char a, int b, unsigned char *c)
{
  __builtin_altivec_stvxl((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_stl(vector bool char a, int b, vector bool char *c)
{
  __builtin_altivec_stvxl((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_stl(vector short a, int b, vector short *c)
{
  __builtin_altivec_stvxl((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_stl(vector short a, int b, short *c)
{
  __builtin_altivec_stvxl((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_stl(vector unsigned short a, int b, vector unsigned short *c)
{
  __builtin_altivec_stvxl((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_stl(vector unsigned short a, int b, unsigned short *c)
{
  __builtin_altivec_stvxl((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_stl(vector bool short a, int b, short *c)
{
  __builtin_altivec_stvxl((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_stl(vector bool short a, int b, unsigned short *c)
{
  __builtin_altivec_stvxl((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_stl(vector bool short a, int b, vector bool short *c)
{
  __builtin_altivec_stvxl((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_stl(vector pixel a, int b, short *c)
{
  __builtin_altivec_stvxl((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_stl(vector pixel a, int b, unsigned short *c)
{
  __builtin_altivec_stvxl((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_stl(vector pixel a, int b, vector pixel *c)
{
  __builtin_altivec_stvxl((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_stl(vector int a, int b, vector int *c)
{
  __builtin_altivec_stvxl(a, b, c);
}

static void __ATTRS_o_ai
vec_stl(vector int a, int b, int *c)
{
  __builtin_altivec_stvxl(a, b, c);
}

static void __ATTRS_o_ai
vec_stl(vector unsigned int a, int b, vector unsigned int *c)
{
  __builtin_altivec_stvxl((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_stl(vector unsigned int a, int b, unsigned int *c)
{
  __builtin_altivec_stvxl((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_stl(vector bool int a, int b, int *c)
{
  __builtin_altivec_stvxl((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_stl(vector bool int a, int b, unsigned int *c)
{
  __builtin_altivec_stvxl((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_stl(vector bool int a, int b, vector bool int *c)
{
  __builtin_altivec_stvxl((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_stl(vector float a, int b, vector float *c)
{
  __builtin_altivec_stvxl((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_stl(vector float a, int b, float *c)
{
  __builtin_altivec_stvxl((vector int)a, b, c);
}

/* vec_stvxl */

static void __ATTRS_o_ai
vec_stvxl(vector signed char a, int b, vector signed char *c)
{
  __builtin_altivec_stvxl((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_stvxl(vector signed char a, int b, signed char *c)
{
  __builtin_altivec_stvxl((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_stvxl(vector unsigned char a, int b, vector unsigned char *c)
{
  __builtin_altivec_stvxl((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_stvxl(vector unsigned char a, int b, unsigned char *c)
{
  __builtin_altivec_stvxl((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_stvxl(vector bool char a, int b, signed char *c)
{
  __builtin_altivec_stvxl((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_stvxl(vector bool char a, int b, unsigned char *c)
{
  __builtin_altivec_stvxl((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_stvxl(vector bool char a, int b, vector bool char *c)
{
  __builtin_altivec_stvxl((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_stvxl(vector short a, int b, vector short *c)
{
  __builtin_altivec_stvxl((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_stvxl(vector short a, int b, short *c)
{
  __builtin_altivec_stvxl((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_stvxl(vector unsigned short a, int b, vector unsigned short *c)
{
  __builtin_altivec_stvxl((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_stvxl(vector unsigned short a, int b, unsigned short *c)
{
  __builtin_altivec_stvxl((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_stvxl(vector bool short a, int b, short *c)
{
  __builtin_altivec_stvxl((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_stvxl(vector bool short a, int b, unsigned short *c)
{
  __builtin_altivec_stvxl((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_stvxl(vector bool short a, int b, vector bool short *c)
{
  __builtin_altivec_stvxl((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_stvxl(vector pixel a, int b, short *c)
{
  __builtin_altivec_stvxl((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_stvxl(vector pixel a, int b, unsigned short *c)
{
  __builtin_altivec_stvxl((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_stvxl(vector pixel a, int b, vector pixel *c)
{
  __builtin_altivec_stvxl((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_stvxl(vector int a, int b, vector int *c)
{
  __builtin_altivec_stvxl(a, b, c);
}

static void __ATTRS_o_ai
vec_stvxl(vector int a, int b, int *c)
{
  __builtin_altivec_stvxl(a, b, c);
}

static void __ATTRS_o_ai
vec_stvxl(vector unsigned int a, int b, vector unsigned int *c)
{
  __builtin_altivec_stvxl((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_stvxl(vector unsigned int a, int b, unsigned int *c)
{
  __builtin_altivec_stvxl((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_stvxl(vector bool int a, int b, int *c)
{
  __builtin_altivec_stvxl((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_stvxl(vector bool int a, int b, unsigned int *c)
{
  __builtin_altivec_stvxl((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_stvxl(vector bool int a, int b, vector bool int *c)
{
  __builtin_altivec_stvxl((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_stvxl(vector float a, int b, vector float *c)
{
  __builtin_altivec_stvxl((vector int)a, b, c);
}

static void __ATTRS_o_ai
vec_stvxl(vector float a, int b, float *c)
{
  __builtin_altivec_stvxl((vector int)a, b, c);
}

/* vec_sub */

static vector signed char __ATTRS_o_ai
vec_sub(vector signed char a, vector signed char b)
{
  return a - b;
}

static vector signed char __ATTRS_o_ai
vec_sub(vector bool char a, vector signed char b)
{
  return (vector signed char)a - b;
}

static vector signed char __ATTRS_o_ai
vec_sub(vector signed char a, vector bool char b)
{
  return a - (vector signed char)b;
}

static vector unsigned char __ATTRS_o_ai
vec_sub(vector unsigned char a, vector unsigned char b)
{
  return a - b;
}

static vector unsigned char __ATTRS_o_ai
vec_sub(vector bool char a, vector unsigned char b)
{
  return (vector unsigned char)a - b;
}

static vector unsigned char __ATTRS_o_ai
vec_sub(vector unsigned char a, vector bool char b)
{
  return a - (vector unsigned char)b;
}

static vector short __ATTRS_o_ai
vec_sub(vector short a, vector short b)
{
  return a - b;
}

static vector short __ATTRS_o_ai
vec_sub(vector bool short a, vector short b)
{
  return (vector short)a - b;
}

static vector short __ATTRS_o_ai
vec_sub(vector short a, vector bool short b)
{
  return a - (vector short)b;
}

static vector unsigned short __ATTRS_o_ai
vec_sub(vector unsigned short a, vector unsigned short b)
{
  return a - b;
}

static vector unsigned short __ATTRS_o_ai
vec_sub(vector bool short a, vector unsigned short b)
{
  return (vector unsigned short)a - b;
}

static vector unsigned short __ATTRS_o_ai
vec_sub(vector unsigned short a, vector bool short b)
{
  return a - (vector unsigned short)b;
}

static vector int __ATTRS_o_ai
vec_sub(vector int a, vector int b)
{
  return a - b;
}

static vector int __ATTRS_o_ai
vec_sub(vector bool int a, vector int b)
{
  return (vector int)a - b;
}

static vector int __ATTRS_o_ai
vec_sub(vector int a, vector bool int b)
{
  return a - (vector int)b;
}

static vector unsigned int __ATTRS_o_ai
vec_sub(vector unsigned int a, vector unsigned int b)
{
  return a - b;
}

static vector unsigned int __ATTRS_o_ai
vec_sub(vector bool int a, vector unsigned int b)
{
  return (vector unsigned int)a - b;
}

static vector unsigned int __ATTRS_o_ai
vec_sub(vector unsigned int a, vector bool int b)
{
  return a - (vector unsigned int)b;
}

static vector float __ATTRS_o_ai
vec_sub(vector float a, vector float b)
{
  return a - b;
}

/* vec_vsububm */

#define __builtin_altivec_vsububm vec_vsububm

static vector signed char __ATTRS_o_ai
vec_vsububm(vector signed char a, vector signed char b)
{
  return a - b;
}

static vector signed char __ATTRS_o_ai
vec_vsububm(vector bool char a, vector signed char b)
{
  return (vector signed char)a - b;
}

static vector signed char __ATTRS_o_ai
vec_vsububm(vector signed char a, vector bool char b)
{
  return a - (vector signed char)b;
}

static vector unsigned char __ATTRS_o_ai
vec_vsububm(vector unsigned char a, vector unsigned char b)
{
  return a - b;
}

static vector unsigned char __ATTRS_o_ai
vec_vsububm(vector bool char a, vector unsigned char b)
{
  return (vector unsigned char)a - b;
}

static vector unsigned char __ATTRS_o_ai
vec_vsububm(vector unsigned char a, vector bool char b)
{
  return a - (vector unsigned char)b;
}

/* vec_vsubuhm */

#define __builtin_altivec_vsubuhm vec_vsubuhm

static vector short __ATTRS_o_ai
vec_vsubuhm(vector short a, vector short b)
{
  return a - b;
}

static vector short __ATTRS_o_ai
vec_vsubuhm(vector bool short a, vector short b)
{
  return (vector short)a - b;
}

static vector short __ATTRS_o_ai
vec_vsubuhm(vector short a, vector bool short b)
{
  return a - (vector short)b;
}

static vector unsigned short __ATTRS_o_ai
vec_vsubuhm(vector unsigned short a, vector unsigned short b)
{
  return a - b;
}

static vector unsigned short __ATTRS_o_ai
vec_vsubuhm(vector bool short a, vector unsigned short b)
{
  return (vector unsigned short)a - b;
}

static vector unsigned short __ATTRS_o_ai
vec_vsubuhm(vector unsigned short a, vector bool short b)
{
  return a - (vector unsigned short)b;
}

/* vec_vsubuwm */

#define __builtin_altivec_vsubuwm vec_vsubuwm

static vector int __ATTRS_o_ai
vec_vsubuwm(vector int a, vector int b)
{
  return a - b;
}

static vector int __ATTRS_o_ai
vec_vsubuwm(vector bool int a, vector int b)
{
  return (vector int)a - b;
}

static vector int __ATTRS_o_ai
vec_vsubuwm(vector int a, vector bool int b)
{
  return a - (vector int)b;
}

static vector unsigned int __ATTRS_o_ai
vec_vsubuwm(vector unsigned int a, vector unsigned int b)
{
  return a - b;
}

static vector unsigned int __ATTRS_o_ai
vec_vsubuwm(vector bool int a, vector unsigned int b)
{
  return (vector unsigned int)a - b;
}

static vector unsigned int __ATTRS_o_ai
vec_vsubuwm(vector unsigned int a, vector bool int b)
{
  return a - (vector unsigned int)b;
}

/* vec_vsubfp */

#define __builtin_altivec_vsubfp vec_vsubfp

static vector float __attribute__((__always_inline__))
vec_vsubfp(vector float a, vector float b)
{
  return a - b;
}

/* vec_subc */

static vector unsigned int __attribute__((__always_inline__))
vec_subc(vector unsigned int a, vector unsigned int b)
{
  return __builtin_altivec_vsubcuw(a, b);
}

/* vec_vsubcuw */

static vector unsigned int __attribute__((__always_inline__))
vec_vsubcuw(vector unsigned int a, vector unsigned int b)
{
  return __builtin_altivec_vsubcuw(a, b);
}

/* vec_subs */

static vector signed char __ATTRS_o_ai
vec_subs(vector signed char a, vector signed char b)
{
  return __builtin_altivec_vsubsbs(a, b);
}

static vector signed char __ATTRS_o_ai
vec_subs(vector bool char a, vector signed char b)
{
  return __builtin_altivec_vsubsbs((vector signed char)a, b);
}

static vector signed char __ATTRS_o_ai
vec_subs(vector signed char a, vector bool char b)
{
  return __builtin_altivec_vsubsbs(a, (vector signed char)b);
}

static vector unsigned char __ATTRS_o_ai
vec_subs(vector unsigned char a, vector unsigned char b)
{
  return __builtin_altivec_vsububs(a, b);
}

static vector unsigned char __ATTRS_o_ai
vec_subs(vector bool char a, vector unsigned char b)
{
  return __builtin_altivec_vsububs((vector unsigned char)a, b);
}

static vector unsigned char __ATTRS_o_ai
vec_subs(vector unsigned char a, vector bool char b)
{
  return __builtin_altivec_vsububs(a, (vector unsigned char)b);
}

static vector short __ATTRS_o_ai
vec_subs(vector short a, vector short b)
{
  return __builtin_altivec_vsubshs(a, b);
}

static vector short __ATTRS_o_ai
vec_subs(vector bool short a, vector short b)
{
  return __builtin_altivec_vsubshs((vector short)a, b);
}

static vector short __ATTRS_o_ai
vec_subs(vector short a, vector bool short b)
{
  return __builtin_altivec_vsubshs(a, (vector short)b);
}

static vector unsigned short __ATTRS_o_ai
vec_subs(vector unsigned short a, vector unsigned short b)
{
  return __builtin_altivec_vsubuhs(a, b);
}

static vector unsigned short __ATTRS_o_ai
vec_subs(vector bool short a, vector unsigned short b)
{
  return __builtin_altivec_vsubuhs((vector unsigned short)a, b);
}

static vector unsigned short __ATTRS_o_ai
vec_subs(vector unsigned short a, vector bool short b)
{
  return __builtin_altivec_vsubuhs(a, (vector unsigned short)b);
}

static vector int __ATTRS_o_ai
vec_subs(vector int a, vector int b)
{
  return __builtin_altivec_vsubsws(a, b);
}

static vector int __ATTRS_o_ai
vec_subs(vector bool int a, vector int b)
{
  return __builtin_altivec_vsubsws((vector int)a, b);
}

static vector int __ATTRS_o_ai
vec_subs(vector int a, vector bool int b)
{
  return __builtin_altivec_vsubsws(a, (vector int)b);
}

static vector unsigned int __ATTRS_o_ai
vec_subs(vector unsigned int a, vector unsigned int b)
{
  return __builtin_altivec_vsubuws(a, b);
}

static vector unsigned int __ATTRS_o_ai
vec_subs(vector bool int a, vector unsigned int b)
{
  return __builtin_altivec_vsubuws((vector unsigned int)a, b);
}

static vector unsigned int __ATTRS_o_ai
vec_subs(vector unsigned int a, vector bool int b)
{
  return __builtin_altivec_vsubuws(a, (vector unsigned int)b);
}

/* vec_vsubsbs */

static vector signed char __ATTRS_o_ai
vec_vsubsbs(vector signed char a, vector signed char b)
{
  return __builtin_altivec_vsubsbs(a, b);
}

static vector signed char __ATTRS_o_ai
vec_vsubsbs(vector bool char a, vector signed char b)
{
  return __builtin_altivec_vsubsbs((vector signed char)a, b);
}

static vector signed char __ATTRS_o_ai
vec_vsubsbs(vector signed char a, vector bool char b)
{
  return __builtin_altivec_vsubsbs(a, (vector signed char)b);
}

/* vec_vsububs */

static vector unsigned char __ATTRS_o_ai
vec_vsububs(vector unsigned char a, vector unsigned char b)
{
  return __builtin_altivec_vsububs(a, b);
}

static vector unsigned char __ATTRS_o_ai
vec_vsububs(vector bool char a, vector unsigned char b)
{
  return __builtin_altivec_vsububs((vector unsigned char)a, b);
}

static vector unsigned char __ATTRS_o_ai
vec_vsububs(vector unsigned char a, vector bool char b)
{
  return __builtin_altivec_vsububs(a, (vector unsigned char)b);
}

/* vec_vsubshs */

static vector short __ATTRS_o_ai
vec_vsubshs(vector short a, vector short b)
{
  return __builtin_altivec_vsubshs(a, b);
}

static vector short __ATTRS_o_ai
vec_vsubshs(vector bool short a, vector short b)
{
  return __builtin_altivec_vsubshs((vector short)a, b);
}

static vector short __ATTRS_o_ai
vec_vsubshs(vector short a, vector bool short b)
{
  return __builtin_altivec_vsubshs(a, (vector short)b);
}

/* vec_vsubuhs */

static vector unsigned short __ATTRS_o_ai
vec_vsubuhs(vector unsigned short a, vector unsigned short b)
{
  return __builtin_altivec_vsubuhs(a, b);
}

static vector unsigned short __ATTRS_o_ai
vec_vsubuhs(vector bool short a, vector unsigned short b)
{
  return __builtin_altivec_vsubuhs((vector unsigned short)a, b);
}

static vector unsigned short __ATTRS_o_ai
vec_vsubuhs(vector unsigned short a, vector bool short b)
{
  return __builtin_altivec_vsubuhs(a, (vector unsigned short)b);
}

/* vec_vsubsws */

static vector int __ATTRS_o_ai
vec_vsubsws(vector int a, vector int b)
{
  return __builtin_altivec_vsubsws(a, b);
}

static vector int __ATTRS_o_ai
vec_vsubsws(vector bool int a, vector int b)
{
  return __builtin_altivec_vsubsws((vector int)a, b);
}

static vector int __ATTRS_o_ai
vec_vsubsws(vector int a, vector bool int b)
{
  return __builtin_altivec_vsubsws(a, (vector int)b);
}

/* vec_vsubuws */

static vector unsigned int __ATTRS_o_ai
vec_vsubuws(vector unsigned int a, vector unsigned int b)
{
  return __builtin_altivec_vsubuws(a, b);
}

static vector unsigned int __ATTRS_o_ai
vec_vsubuws(vector bool int a, vector unsigned int b)
{
  return __builtin_altivec_vsubuws((vector unsigned int)a, b);
}

static vector unsigned int __ATTRS_o_ai
vec_vsubuws(vector unsigned int a, vector bool int b)
{
  return __builtin_altivec_vsubuws(a, (vector unsigned int)b);
}

/* vec_sum4s */

static vector int __ATTRS_o_ai
vec_sum4s(vector signed char a, vector int b)
{
  return __builtin_altivec_vsum4sbs(a, b);
}

static vector unsigned int __ATTRS_o_ai
vec_sum4s(vector unsigned char a, vector unsigned int b)
{
  return __builtin_altivec_vsum4ubs(a, b);
}

static vector int __ATTRS_o_ai
vec_sum4s(vector signed short a, vector int b)
{
  return __builtin_altivec_vsum4shs(a, b);
}

/* vec_vsum4sbs */

static vector int __attribute__((__always_inline__))
vec_vsum4sbs(vector signed char a, vector int b)
{
  return __builtin_altivec_vsum4sbs(a, b);
}

/* vec_vsum4ubs */

static vector unsigned int __attribute__((__always_inline__))
vec_vsum4ubs(vector unsigned char a, vector unsigned int b)
{
  return __builtin_altivec_vsum4ubs(a, b);
}

/* vec_vsum4shs */

static vector int __attribute__((__always_inline__))
vec_vsum4shs(vector signed short a, vector int b)
{
  return __builtin_altivec_vsum4shs(a, b);
}

/* vec_sum2s */

static vector signed int __attribute__((__always_inline__))
vec_sum2s(vector int a, vector int b)
{
  return __builtin_altivec_vsum2sws(a, b);
}

/* vec_vsum2sws */

static vector signed int __attribute__((__always_inline__))
vec_vsum2sws(vector int a, vector int b)
{
  return __builtin_altivec_vsum2sws(a, b);
}

/* vec_sums */

static vector signed int __attribute__((__always_inline__))
vec_sums(vector signed int a, vector signed int b)
{
  return __builtin_altivec_vsumsws(a, b);
}

/* vec_vsumsws */

static vector signed int __attribute__((__always_inline__))
vec_vsumsws(vector signed int a, vector signed int b)
{
  return __builtin_altivec_vsumsws(a, b);
}

/* vec_trunc */

static vector float __attribute__((__always_inline__))
vec_trunc(vector float a)
{
  return __builtin_altivec_vrfiz(a);
}

/* vec_vrfiz */

static vector float __attribute__((__always_inline__))
vec_vrfiz(vector float a)
{
  return __builtin_altivec_vrfiz(a);
}

/* vec_unpackh */

static vector short __ATTRS_o_ai
vec_unpackh(vector signed char a)
{
  return __builtin_altivec_vupkhsb((vector char)a);
}

static vector bool short __ATTRS_o_ai
vec_unpackh(vector bool char a)
{
  return (vector bool short)__builtin_altivec_vupkhsb((vector char)a);
}

static vector int __ATTRS_o_ai
vec_unpackh(vector short a)
{
  return __builtin_altivec_vupkhsh(a);
}

static vector bool int __ATTRS_o_ai
vec_unpackh(vector bool short a)
{
  return (vector bool int)__builtin_altivec_vupkhsh((vector short)a);
}

static vector unsigned int __ATTRS_o_ai
vec_unpackh(vector pixel a)
{
  return (vector unsigned int)__builtin_altivec_vupkhsh((vector short)a);
}

/* vec_vupkhsb */

static vector short __ATTRS_o_ai
vec_vupkhsb(vector signed char a)
{
  return __builtin_altivec_vupkhsb((vector char)a);
}

static vector bool short __ATTRS_o_ai
vec_vupkhsb(vector bool char a)
{
  return (vector bool short)__builtin_altivec_vupkhsb((vector char)a);
}

/* vec_vupkhsh */

static vector int __ATTRS_o_ai
vec_vupkhsh(vector short a)
{
  return __builtin_altivec_vupkhsh(a);
}

static vector bool int __ATTRS_o_ai
vec_vupkhsh(vector bool short a)
{
  return (vector bool int)__builtin_altivec_vupkhsh((vector short)a);
}

static vector unsigned int __ATTRS_o_ai
vec_vupkhsh(vector pixel a)
{
  return (vector unsigned int)__builtin_altivec_vupkhsh((vector short)a);
}

/* vec_unpackl */

static vector short __ATTRS_o_ai
vec_unpackl(vector signed char a)
{
  return __builtin_altivec_vupklsb((vector char)a);
}

static vector bool short __ATTRS_o_ai
vec_unpackl(vector bool char a)
{
  return (vector bool short)__builtin_altivec_vupklsb((vector char)a);
}

static vector int __ATTRS_o_ai
vec_unpackl(vector short a)
{
  return __builtin_altivec_vupklsh(a);
}

static vector bool int __ATTRS_o_ai
vec_unpackl(vector bool short a)
{
  return (vector bool int)__builtin_altivec_vupklsh((vector short)a);
}

static vector unsigned int __ATTRS_o_ai
vec_unpackl(vector pixel a)
{
  return (vector unsigned int)__builtin_altivec_vupklsh((vector short)a);
}

/* vec_vupklsb */

static vector short __ATTRS_o_ai
vec_vupklsb(vector signed char a)
{
  return __builtin_altivec_vupklsb((vector char)a);
}

static vector bool short __ATTRS_o_ai
vec_vupklsb(vector bool char a)
{
  return (vector bool short)__builtin_altivec_vupklsb((vector char)a);
}

/* vec_vupklsh */

static vector int __ATTRS_o_ai
vec_vupklsh(vector short a)
{
  return __builtin_altivec_vupklsh(a);
}

static vector bool int __ATTRS_o_ai
vec_vupklsh(vector bool short a)
{
  return (vector bool int)__builtin_altivec_vupklsh((vector short)a);
}

static vector unsigned int __ATTRS_o_ai
vec_vupklsh(vector pixel a)
{
  return (vector unsigned int)__builtin_altivec_vupklsh((vector short)a);
}

/* vec_xor */

#define __builtin_altivec_vxor vec_xor

static vector signed char __ATTRS_o_ai
vec_xor(vector signed char a, vector signed char b)
{
  return a ^ b;
}

static vector signed char __ATTRS_o_ai
vec_xor(vector bool char a, vector signed char b)
{
  return (vector signed char)a ^ b;
}

static vector signed char __ATTRS_o_ai
vec_xor(vector signed char a, vector bool char b)
{
  return a ^ (vector signed char)b;
}

static vector unsigned char __ATTRS_o_ai
vec_xor(vector unsigned char a, vector unsigned char b)
{
  return a ^ b;
}

static vector unsigned char __ATTRS_o_ai
vec_xor(vector bool char a, vector unsigned char b)
{
  return (vector unsigned char)a ^ b;
}

static vector unsigned char __ATTRS_o_ai
vec_xor(vector unsigned char a, vector bool char b)
{
  return a ^ (vector unsigned char)b;
}

static vector bool char __ATTRS_o_ai
vec_xor(vector bool char a, vector bool char b)
{
  return a ^ b;
}

static vector short __ATTRS_o_ai
vec_xor(vector short a, vector short b)
{
  return a ^ b;
}

static vector short __ATTRS_o_ai
vec_xor(vector bool short a, vector short b)
{
  return (vector short)a ^ b;
}

static vector short __ATTRS_o_ai
vec_xor(vector short a, vector bool short b)
{
  return a ^ (vector short)b;
}

static vector unsigned short __ATTRS_o_ai
vec_xor(vector unsigned short a, vector unsigned short b)
{
  return a ^ b;
}

static vector unsigned short __ATTRS_o_ai
vec_xor(vector bool short a, vector unsigned short b)
{
  return (vector unsigned short)a ^ b;
}

static vector unsigned short __ATTRS_o_ai
vec_xor(vector unsigned short a, vector bool short b)
{
  return a ^ (vector unsigned short)b;
}

static vector bool short __ATTRS_o_ai
vec_xor(vector bool short a, vector bool short b)
{
  return a ^ b;
}

static vector int __ATTRS_o_ai
vec_xor(vector int a, vector int b)
{
  return a ^ b;
}

static vector int __ATTRS_o_ai
vec_xor(vector bool int a, vector int b)
{
  return (vector int)a ^ b;
}

static vector int __ATTRS_o_ai
vec_xor(vector int a, vector bool int b)
{
  return a ^ (vector int)b;
}

static vector unsigned int __ATTRS_o_ai
vec_xor(vector unsigned int a, vector unsigned int b)
{
  return a ^ b;
}

static vector unsigned int __ATTRS_o_ai
vec_xor(vector bool int a, vector unsigned int b)
{
  return (vector unsigned int)a ^ b;
}

static vector unsigned int __ATTRS_o_ai
vec_xor(vector unsigned int a, vector bool int b)
{
  return a ^ (vector unsigned int)b;
}

static vector bool int __ATTRS_o_ai
vec_xor(vector bool int a, vector bool int b)
{
  return a ^ b;
}

static vector float __ATTRS_o_ai
vec_xor(vector float a, vector float b)
{
  vector unsigned int res = (vector unsigned int)a ^ (vector unsigned int)b;
  return (vector float)res;
}

static vector float __ATTRS_o_ai
vec_xor(vector bool int a, vector float b)
{
  vector unsigned int res = (vector unsigned int)a ^ (vector unsigned int)b;
  return (vector float)res;
}

static vector float __ATTRS_o_ai
vec_xor(vector float a, vector bool int b)
{
  vector unsigned int res = (vector unsigned int)a ^ (vector unsigned int)b;
  return (vector float)res;
}

/* vec_vxor */

static vector signed char __ATTRS_o_ai
vec_vxor(vector signed char a, vector signed char b)
{
  return a ^ b;
}

static vector signed char __ATTRS_o_ai
vec_vxor(vector bool char a, vector signed char b)
{
  return (vector signed char)a ^ b;
}

static vector signed char __ATTRS_o_ai
vec_vxor(vector signed char a, vector bool char b)
{
  return a ^ (vector signed char)b;
}

static vector unsigned char __ATTRS_o_ai
vec_vxor(vector unsigned char a, vector unsigned char b)
{
  return a ^ b;
}

static vector unsigned char __ATTRS_o_ai
vec_vxor(vector bool char a, vector unsigned char b)
{
  return (vector unsigned char)a ^ b;
}

static vector unsigned char __ATTRS_o_ai
vec_vxor(vector unsigned char a, vector bool char b)
{
  return a ^ (vector unsigned char)b;
}

static vector bool char __ATTRS_o_ai
vec_vxor(vector bool char a, vector bool char b)
{
  return a ^ b;
}

static vector short __ATTRS_o_ai
vec_vxor(vector short a, vector short b)
{
  return a ^ b;
}

static vector short __ATTRS_o_ai
vec_vxor(vector bool short a, vector short b)
{
  return (vector short)a ^ b;
}

static vector short __ATTRS_o_ai
vec_vxor(vector short a, vector bool short b)
{
  return a ^ (vector short)b;
}

static vector unsigned short __ATTRS_o_ai
vec_vxor(vector unsigned short a, vector unsigned short b)
{
  return a ^ b;
}

static vector unsigned short __ATTRS_o_ai
vec_vxor(vector bool short a, vector unsigned short b)
{
  return (vector unsigned short)a ^ b;
}

static vector unsigned short __ATTRS_o_ai
vec_vxor(vector unsigned short a, vector bool short b)
{
  return a ^ (vector unsigned short)b;
}

static vector bool short __ATTRS_o_ai
vec_vxor(vector bool short a, vector bool short b)
{
  return a ^ b;
}

static vector int __ATTRS_o_ai
vec_vxor(vector int a, vector int b)
{
  return a ^ b;
}

static vector int __ATTRS_o_ai
vec_vxor(vector bool int a, vector int b)
{
  return (vector int)a ^ b;
}

static vector int __ATTRS_o_ai
vec_vxor(vector int a, vector bool int b)
{
  return a ^ (vector int)b;
}

static vector unsigned int __ATTRS_o_ai
vec_vxor(vector unsigned int a, vector unsigned int b)
{
  return a ^ b;
}

static vector unsigned int __ATTRS_o_ai
vec_vxor(vector bool int a, vector unsigned int b)
{
  return (vector unsigned int)a ^ b;
}

static vector unsigned int __ATTRS_o_ai
vec_vxor(vector unsigned int a, vector bool int b)
{
  return a ^ (vector unsigned int)b;
}

static vector bool int __ATTRS_o_ai
vec_vxor(vector bool int a, vector bool int b)
{
  return a ^ b;
}

static vector float __ATTRS_o_ai
vec_vxor(vector float a, vector float b)
{
  vector unsigned int res = (vector unsigned int)a ^ (vector unsigned int)b;
  return (vector float)res;
}

static vector float __ATTRS_o_ai
vec_vxor(vector bool int a, vector float b)
{
  vector unsigned int res = (vector unsigned int)a ^ (vector unsigned int)b;
  return (vector float)res;
}

static vector float __ATTRS_o_ai
vec_vxor(vector float a, vector bool int b)
{
  vector unsigned int res = (vector unsigned int)a ^ (vector unsigned int)b;
  return (vector float)res;
}

/* ------------------------ extensions for CBEA ----------------------------- */

/* vec_extract */

static signed char __ATTRS_o_ai
vec_extract(vector signed char a, int b)
{
  return a[b];
}

static unsigned char __ATTRS_o_ai
vec_extract(vector unsigned char a, int b)
{
  return a[b];
}

static short __ATTRS_o_ai
vec_extract(vector short a, int b)
{
  return a[b];
}

static unsigned short __ATTRS_o_ai
vec_extract(vector unsigned short a, int b)
{
  return a[b];
}

static int __ATTRS_o_ai
vec_extract(vector int a, int b)
{
  return a[b];
}

static unsigned int __ATTRS_o_ai
vec_extract(vector unsigned int a, int b)
{
  return a[b];
}

static float __ATTRS_o_ai
vec_extract(vector float a, int b)
{
  return a[b];
}

/* vec_insert */

static vector signed char __ATTRS_o_ai
vec_insert(signed char a, vector signed char b, int c)
{
  b[c] = a;
  return b;
}

static vector unsigned char __ATTRS_o_ai
vec_insert(unsigned char a, vector unsigned char b, int c)
{
  b[c] = a;
  return b;
}

static vector short __ATTRS_o_ai
vec_insert(short a, vector short b, int c)
{
  b[c] = a;
  return b;
}

static vector unsigned short __ATTRS_o_ai
vec_insert(unsigned short a, vector unsigned short b, int c)
{
  b[c] = a;
  return b;
}

static vector int __ATTRS_o_ai
vec_insert(int a, vector int b, int c)
{
  b[c] = a;
  return b;
}

static vector unsigned int __ATTRS_o_ai
vec_insert(unsigned int a, vector unsigned int b, int c)
{
  b[c] = a;
  return b;
}

static vector float __ATTRS_o_ai
vec_insert(float a, vector float b, int c)
{
  b[c] = a;
  return b;
}

/* vec_lvlx */

static vector signed char __ATTRS_o_ai
vec_lvlx(int a, const signed char *b)
{
  return vec_perm(vec_ld(a, b),
                  (vector signed char)(0),
                  vec_lvsl(a, b));
}

static vector signed char __ATTRS_o_ai
vec_lvlx(int a, const vector signed char *b)
{
  return vec_perm(vec_ld(a, b), 
                  (vector signed char)(0),
                  vec_lvsl(a, (unsigned char *)b));
}

static vector unsigned char __ATTRS_o_ai
vec_lvlx(int a, const unsigned char *b)
{
  return vec_perm(vec_ld(a, b),
                  (vector unsigned char)(0),
                  vec_lvsl(a, b));
}

static vector unsigned char __ATTRS_o_ai
vec_lvlx(int a, const vector unsigned char *b)
{
  return vec_perm(vec_ld(a, b), 
                  (vector unsigned char)(0),
                  vec_lvsl(a, (unsigned char *)b));
}

static vector bool char __ATTRS_o_ai
vec_lvlx(int a, const vector bool char *b)
{
  return vec_perm(vec_ld(a, b), 
                  (vector bool char)(0),
                  vec_lvsl(a, (unsigned char *)b));
}

static vector short __ATTRS_o_ai
vec_lvlx(int a, const short *b)
{
  return vec_perm(vec_ld(a, b),
                  (vector short)(0),
                  vec_lvsl(a, b));
}

static vector short __ATTRS_o_ai
vec_lvlx(int a, const vector short *b)
{
  return vec_perm(vec_ld(a, b),
                  (vector short)(0),
                  vec_lvsl(a, (unsigned char *)b));
}

static vector unsigned short __ATTRS_o_ai
vec_lvlx(int a, const unsigned short *b)
{
  return vec_perm(vec_ld(a, b),
                  (vector unsigned short)(0),
                  vec_lvsl(a, b));
}

static vector unsigned short __ATTRS_o_ai
vec_lvlx(int a, const vector unsigned short *b)
{
  return vec_perm(vec_ld(a, b), 
                  (vector unsigned short)(0),
                  vec_lvsl(a, (unsigned char *)b));
}

static vector bool short __ATTRS_o_ai
vec_lvlx(int a, const vector bool short *b)
{
  return vec_perm(vec_ld(a, b), 
                  (vector bool short)(0),
                  vec_lvsl(a, (unsigned char *)b));
}

static vector pixel __ATTRS_o_ai
vec_lvlx(int a, const vector pixel *b)
{
  return vec_perm(vec_ld(a, b), 
                  (vector pixel)(0),
                  vec_lvsl(a, (unsigned char *)b));
}

static vector int __ATTRS_o_ai
vec_lvlx(int a, const int *b)
{
  return vec_perm(vec_ld(a, b),
                  (vector int)(0),
                  vec_lvsl(a, b));
}

static vector int __ATTRS_o_ai
vec_lvlx(int a, const vector int *b)
{
  return vec_perm(vec_ld(a, b),
                  (vector int)(0),
                  vec_lvsl(a, (unsigned char *)b));
}

static vector unsigned int __ATTRS_o_ai
vec_lvlx(int a, const unsigned int *b)
{
  return vec_perm(vec_ld(a, b),
                  (vector unsigned int)(0),
                  vec_lvsl(a, b));
}

static vector unsigned int __ATTRS_o_ai
vec_lvlx(int a, const vector unsigned int *b)
{
  return vec_perm(vec_ld(a, b), 
                  (vector unsigned int)(0),
                  vec_lvsl(a, (unsigned char *)b));
}

static vector bool int __ATTRS_o_ai
vec_lvlx(int a, const vector bool int *b)
{
  return vec_perm(vec_ld(a, b), 
                  (vector bool int)(0),
                  vec_lvsl(a, (unsigned char *)b));
}

static vector float __ATTRS_o_ai
vec_lvlx(int a, const float *b)
{
  return vec_perm(vec_ld(a, b),
                  (vector float)(0),
                  vec_lvsl(a, b));
}

static vector float __ATTRS_o_ai
vec_lvlx(int a, const vector float *b)
{
  return vec_perm(vec_ld(a, b),
                  (vector float)(0),
                  vec_lvsl(a, (unsigned char *)b));
}

/* vec_lvlxl */

static vector signed char __ATTRS_o_ai
vec_lvlxl(int a, const signed char *b)
{
  return vec_perm(vec_ldl(a, b),
                  (vector signed char)(0),
                  vec_lvsl(a, b));
}

static vector signed char __ATTRS_o_ai
vec_lvlxl(int a, const vector signed char *b)
{
  return vec_perm(vec_ldl(a, b), 
                  (vector signed char)(0),
                  vec_lvsl(a, (unsigned char *)b));
}

static vector unsigned char __ATTRS_o_ai
vec_lvlxl(int a, const unsigned char *b)
{
  return vec_perm(vec_ldl(a, b),
                  (vector unsigned char)(0),
                  vec_lvsl(a, b));
}

static vector unsigned char __ATTRS_o_ai
vec_lvlxl(int a, const vector unsigned char *b)
{
  return vec_perm(vec_ldl(a, b), 
                  (vector unsigned char)(0),
                  vec_lvsl(a, (unsigned char *)b));
}

static vector bool char __ATTRS_o_ai
vec_lvlxl(int a, const vector bool char *b)
{
  return vec_perm(vec_ldl(a, b), 
                  (vector bool char)(0),
                  vec_lvsl(a, (unsigned char *)b));
}

static vector short __ATTRS_o_ai
vec_lvlxl(int a, const short *b)
{
  return vec_perm(vec_ldl(a, b),
                  (vector short)(0),
                  vec_lvsl(a, b));
}

static vector short __ATTRS_o_ai
vec_lvlxl(int a, const vector short *b)
{
  return vec_perm(vec_ldl(a, b),
                  (vector short)(0),
                  vec_lvsl(a, (unsigned char *)b));
}

static vector unsigned short __ATTRS_o_ai
vec_lvlxl(int a, const unsigned short *b)
{
  return vec_perm(vec_ldl(a, b),
                  (vector unsigned short)(0),
                  vec_lvsl(a, b));
}

static vector unsigned short __ATTRS_o_ai
vec_lvlxl(int a, const vector unsigned short *b)
{
  return vec_perm(vec_ldl(a, b), 
                  (vector unsigned short)(0),
                  vec_lvsl(a, (unsigned char *)b));
}

static vector bool short __ATTRS_o_ai
vec_lvlxl(int a, const vector bool short *b)
{
  return vec_perm(vec_ldl(a, b), 
                  (vector bool short)(0),
                  vec_lvsl(a, (unsigned char *)b));
}

static vector pixel __ATTRS_o_ai
vec_lvlxl(int a, const vector pixel *b)
{
  return vec_perm(vec_ldl(a, b), 
                  (vector pixel)(0),
                  vec_lvsl(a, (unsigned char *)b));
}

static vector int __ATTRS_o_ai
vec_lvlxl(int a, const int *b)
{
  return vec_perm(vec_ldl(a, b),
                  (vector int)(0),
                  vec_lvsl(a, b));
}

static vector int __ATTRS_o_ai
vec_lvlxl(int a, const vector int *b)
{
  return vec_perm(vec_ldl(a, b),
                  (vector int)(0),
                  vec_lvsl(a, (unsigned char *)b));
}

static vector unsigned int __ATTRS_o_ai
vec_lvlxl(int a, const unsigned int *b)
{
  return vec_perm(vec_ldl(a, b),
                  (vector unsigned int)(0),
                  vec_lvsl(a, b));
}

static vector unsigned int __ATTRS_o_ai
vec_lvlxl(int a, const vector unsigned int *b)
{
  return vec_perm(vec_ldl(a, b), 
                  (vector unsigned int)(0),
                  vec_lvsl(a, (unsigned char *)b));
}

static vector bool int __ATTRS_o_ai
vec_lvlxl(int a, const vector bool int *b)
{
  return vec_perm(vec_ldl(a, b), 
                  (vector bool int)(0),
                  vec_lvsl(a, (unsigned char *)b));
}

static vector float __ATTRS_o_ai
vec_lvlxl(int a, const float *b)
{
  return vec_perm(vec_ldl(a, b),
                  (vector float)(0),
                  vec_lvsl(a, b));
}

static vector float __ATTRS_o_ai
vec_lvlxl(int a, vector float *b)
{
  return vec_perm(vec_ldl(a, b),
                  (vector float)(0),
                  vec_lvsl(a, (unsigned char *)b));
}

/* vec_lvrx */

static vector signed char __ATTRS_o_ai
vec_lvrx(int a, const signed char *b)
{
  return vec_perm((vector signed char)(0),
                  vec_ld(a, b),
                  vec_lvsl(a, b));
}

static vector signed char __ATTRS_o_ai
vec_lvrx(int a, const vector signed char *b)
{
  return vec_perm((vector signed char)(0),
                  vec_ld(a, b), 
                  vec_lvsl(a, (unsigned char *)b));
}

static vector unsigned char __ATTRS_o_ai
vec_lvrx(int a, const unsigned char *b)
{
  return vec_perm((vector unsigned char)(0),
                  vec_ld(a, b),
                  vec_lvsl(a, b));
}

static vector unsigned char __ATTRS_o_ai
vec_lvrx(int a, const vector unsigned char *b)
{
  return vec_perm((vector unsigned char)(0),
                  vec_ld(a, b), 
                  vec_lvsl(a, (unsigned char *)b));
}

static vector bool char __ATTRS_o_ai
vec_lvrx(int a, const vector bool char *b)
{
  return vec_perm((vector bool char)(0),
                  vec_ld(a, b), 
                  vec_lvsl(a, (unsigned char *)b));
}

static vector short __ATTRS_o_ai
vec_lvrx(int a, const short *b)
{
  return vec_perm((vector short)(0),
                  vec_ld(a, b),
                  vec_lvsl(a, b));
}

static vector short __ATTRS_o_ai
vec_lvrx(int a, const vector short *b)
{
  return vec_perm((vector short)(0),
                  vec_ld(a, b),
                  vec_lvsl(a, (unsigned char *)b));
}

static vector unsigned short __ATTRS_o_ai
vec_lvrx(int a, const unsigned short *b)
{
  return vec_perm((vector unsigned short)(0),
                  vec_ld(a, b),
                  vec_lvsl(a, b));
}

static vector unsigned short __ATTRS_o_ai
vec_lvrx(int a, const vector unsigned short *b)
{
  return vec_perm((vector unsigned short)(0),
                  vec_ld(a, b), 
                  vec_lvsl(a, (unsigned char *)b));
}

static vector bool short __ATTRS_o_ai
vec_lvrx(int a, const vector bool short *b)
{
  return vec_perm((vector bool short)(0),
                  vec_ld(a, b), 
                  vec_lvsl(a, (unsigned char *)b));
}

static vector pixel __ATTRS_o_ai
vec_lvrx(int a, const vector pixel *b)
{
  return vec_perm((vector pixel)(0),
                  vec_ld(a, b), 
                  vec_lvsl(a, (unsigned char *)b));
}

static vector int __ATTRS_o_ai
vec_lvrx(int a, const int *b)
{
  return vec_perm((vector int)(0),
                  vec_ld(a, b),
                  vec_lvsl(a, b));
}

static vector int __ATTRS_o_ai
vec_lvrx(int a, const vector int *b)
{
  return vec_perm((vector int)(0),
                  vec_ld(a, b),
                  vec_lvsl(a, (unsigned char *)b));
}

static vector unsigned int __ATTRS_o_ai
vec_lvrx(int a, const unsigned int *b)
{
  return vec_perm((vector unsigned int)(0),
                  vec_ld(a, b),
                  vec_lvsl(a, b));
}

static vector unsigned int __ATTRS_o_ai
vec_lvrx(int a, const vector unsigned int *b)
{
  return vec_perm((vector unsigned int)(0),
                  vec_ld(a, b), 
                  vec_lvsl(a, (unsigned char *)b));
}

static vector bool int __ATTRS_o_ai
vec_lvrx(int a, const vector bool int *b)
{
  return vec_perm((vector bool int)(0),
                  vec_ld(a, b), 
                  vec_lvsl(a, (unsigned char *)b));
}

static vector float __ATTRS_o_ai
vec_lvrx(int a, const float *b)
{
  return vec_perm((vector float)(0),
                  vec_ld(a, b),
                  vec_lvsl(a, b));
}

static vector float __ATTRS_o_ai
vec_lvrx(int a, const vector float *b)
{
  return vec_perm((vector float)(0),
                  vec_ld(a, b),
                  vec_lvsl(a, (unsigned char *)b));
}

/* vec_lvrxl */

static vector signed char __ATTRS_o_ai
vec_lvrxl(int a, const signed char *b)
{
  return vec_perm((vector signed char)(0),
                  vec_ldl(a, b),
                  vec_lvsl(a, b));
}

static vector signed char __ATTRS_o_ai
vec_lvrxl(int a, const vector signed char *b)
{
  return vec_perm((vector signed char)(0),
                  vec_ldl(a, b), 
                  vec_lvsl(a, (unsigned char *)b));
}

static vector unsigned char __ATTRS_o_ai
vec_lvrxl(int a, const unsigned char *b)
{
  return vec_perm((vector unsigned char)(0),
                  vec_ldl(a, b),
                  vec_lvsl(a, b));
}

static vector unsigned char __ATTRS_o_ai
vec_lvrxl(int a, const vector unsigned char *b)
{
  return vec_perm((vector unsigned char)(0),
                  vec_ldl(a, b), 
                  vec_lvsl(a, (unsigned char *)b));
}

static vector bool char __ATTRS_o_ai
vec_lvrxl(int a, const vector bool char *b)
{
  return vec_perm((vector bool char)(0),
                  vec_ldl(a, b), 
                  vec_lvsl(a, (unsigned char *)b));
}

static vector short __ATTRS_o_ai
vec_lvrxl(int a, const short *b)
{
  return vec_perm((vector short)(0),
                  vec_ldl(a, b),
                  vec_lvsl(a, b));
}

static vector short __ATTRS_o_ai
vec_lvrxl(int a, const vector short *b)
{
  return vec_perm((vector short)(0),
                  vec_ldl(a, b),
                  vec_lvsl(a, (unsigned char *)b));
}

static vector unsigned short __ATTRS_o_ai
vec_lvrxl(int a, const unsigned short *b)
{
  return vec_perm((vector unsigned short)(0),
                  vec_ldl(a, b),
                  vec_lvsl(a, b));
}

static vector unsigned short __ATTRS_o_ai
vec_lvrxl(int a, const vector unsigned short *b)
{
  return vec_perm((vector unsigned short)(0),
                  vec_ldl(a, b), 
                  vec_lvsl(a, (unsigned char *)b));
}

static vector bool short __ATTRS_o_ai
vec_lvrxl(int a, const vector bool short *b)
{
  return vec_perm((vector bool short)(0),
                  vec_ldl(a, b), 
                  vec_lvsl(a, (unsigned char *)b));
}

static vector pixel __ATTRS_o_ai
vec_lvrxl(int a, const vector pixel *b)
{
  return vec_perm((vector pixel)(0),
                  vec_ldl(a, b), 
                  vec_lvsl(a, (unsigned char *)b));
}

static vector int __ATTRS_o_ai
vec_lvrxl(int a, const int *b)
{
  return vec_perm((vector int)(0),
                  vec_ldl(a, b),
                  vec_lvsl(a, b));
}

static vector int __ATTRS_o_ai
vec_lvrxl(int a, const vector int *b)
{
  return vec_perm((vector int)(0),
                  vec_ldl(a, b),
                  vec_lvsl(a, (unsigned char *)b));
}

static vector unsigned int __ATTRS_o_ai
vec_lvrxl(int a, const unsigned int *b)
{
  return vec_perm((vector unsigned int)(0),
                  vec_ldl(a, b),
                  vec_lvsl(a, b));
}

static vector unsigned int __ATTRS_o_ai
vec_lvrxl(int a, const vector unsigned int *b)
{
  return vec_perm((vector unsigned int)(0),
                  vec_ldl(a, b), 
                  vec_lvsl(a, (unsigned char *)b));
}

static vector bool int __ATTRS_o_ai
vec_lvrxl(int a, const vector bool int *b)
{
  return vec_perm((vector bool int)(0),
                  vec_ldl(a, b), 
                  vec_lvsl(a, (unsigned char *)b));
}

static vector float __ATTRS_o_ai
vec_lvrxl(int a, const float *b)
{
  return vec_perm((vector float)(0),
                  vec_ldl(a, b),
                  vec_lvsl(a, b));
}

static vector float __ATTRS_o_ai
vec_lvrxl(int a, const vector float *b)
{
  return vec_perm((vector float)(0),
                  vec_ldl(a, b),
                  vec_lvsl(a, (unsigned char *)b));
}

/* vec_stvlx */

static void __ATTRS_o_ai
vec_stvlx(vector signed char a, int b, signed char *c)
{
  return vec_st(vec_perm(vec_lvrx(b, c),
                         a,
                         vec_lvsr(b, c)),
                b, c);
}

static void __ATTRS_o_ai
vec_stvlx(vector signed char a, int b, vector signed char *c)
{
  return vec_st(vec_perm(vec_lvrx(b, c),
                         a,
                         vec_lvsr(b, (unsigned char *)c)),
                b, c);
}

static void __ATTRS_o_ai
vec_stvlx(vector unsigned char a, int b, unsigned char *c)
{
  return vec_st(vec_perm(vec_lvrx(b, c),
                         a,
                         vec_lvsr(b, c)),
                b, c);
}

static void __ATTRS_o_ai
vec_stvlx(vector unsigned char a, int b, vector unsigned char *c)
{
  return vec_st(vec_perm(vec_lvrx(b, c),
                         a,
                         vec_lvsr(b, (unsigned char *)c)),
                b, c);
}

static void __ATTRS_o_ai
vec_stvlx(vector bool char a, int b, vector bool char *c)
{
  return vec_st(vec_perm(vec_lvrx(b, c),
                         a,
                         vec_lvsr(b, (unsigned char *)c)),
                b, c);
}

static void __ATTRS_o_ai
vec_stvlx(vector short a, int b, short *c)
{
  return vec_st(vec_perm(vec_lvrx(b, c),
                         a,
                         vec_lvsr(b, c)),
                b, c);
}

static void __ATTRS_o_ai
vec_stvlx(vector short a, int b, vector short *c)
{
  return vec_st(vec_perm(vec_lvrx(b, c),
                         a,
                         vec_lvsr(b, (unsigned char *)c)),
                b, c);
}

static void __ATTRS_o_ai
vec_stvlx(vector unsigned short a, int b, unsigned short *c)
{
  return vec_st(vec_perm(vec_lvrx(b, c),
                         a,
                         vec_lvsr(b, c)),
                b, c);
}

static void __ATTRS_o_ai
vec_stvlx(vector unsigned short a, int b, vector unsigned short *c)
{
  return vec_st(vec_perm(vec_lvrx(b, c),
                         a,
                         vec_lvsr(b, (unsigned char *)c)),
                b, c);
}

static void __ATTRS_o_ai
vec_stvlx(vector bool short a, int b, vector bool short *c)
{
  return vec_st(vec_perm(vec_lvrx(b, c),
                         a,
                         vec_lvsr(b, (unsigned char *)c)),
                b, c);
}

static void __ATTRS_o_ai
vec_stvlx(vector pixel a, int b, vector pixel *c)
{
  return vec_st(vec_perm(vec_lvrx(b, c),
                         a,
                         vec_lvsr(b, (unsigned char *)c)),
                b, c);
}

static void __ATTRS_o_ai
vec_stvlx(vector int a, int b, int *c)
{
  return vec_st(vec_perm(vec_lvrx(b, c),
                         a,
                         vec_lvsr(b, c)),
                b, c);
}

static void __ATTRS_o_ai
vec_stvlx(vector int a, int b, vector int *c)
{
  return vec_st(vec_perm(vec_lvrx(b, c),
                         a,
                         vec_lvsr(b, (unsigned char *)c)),
                b, c);
}

static void __ATTRS_o_ai
vec_stvlx(vector unsigned int a, int b, unsigned int *c)
{
  return vec_st(vec_perm(vec_lvrx(b, c),
                         a,
                         vec_lvsr(b, c)),
                b, c);
}

static void __ATTRS_o_ai
vec_stvlx(vector unsigned int a, int b, vector unsigned int *c)
{
  return vec_st(vec_perm(vec_lvrx(b, c),
                         a,
                         vec_lvsr(b, (unsigned char *)c)),
                b, c);
}

static void __ATTRS_o_ai
vec_stvlx(vector bool int a, int b, vector bool int *c)
{
  return vec_st(vec_perm(vec_lvrx(b, c),
                         a,
                         vec_lvsr(b, (unsigned char *)c)),
                b, c);
}

static void __ATTRS_o_ai
vec_stvlx(vector float a, int b, vector float *c)
{
  return vec_st(vec_perm(vec_lvrx(b, c),
                         a,
                         vec_lvsr(b, (unsigned char *)c)),
                b, c);
}

/* vec_stvlxl */

static void __ATTRS_o_ai
vec_stvlxl(vector signed char a, int b, signed char *c)
{
  return vec_stl(vec_perm(vec_lvrx(b, c),
                          a,
                          vec_lvsr(b, c)),
                 b, c);
}

static void __ATTRS_o_ai
vec_stvlxl(vector signed char a, int b, vector signed char *c)
{
  return vec_stl(vec_perm(vec_lvrx(b, c),
                          a,
                          vec_lvsr(b, (unsigned char *)c)),
                 b, c);
}

static void __ATTRS_o_ai
vec_stvlxl(vector unsigned char a, int b, unsigned char *c)
{
  return vec_stl(vec_perm(vec_lvrx(b, c),
                          a,
                          vec_lvsr(b, c)),
                 b, c);
}

static void __ATTRS_o_ai
vec_stvlxl(vector unsigned char a, int b, vector unsigned char *c)
{
  return vec_stl(vec_perm(vec_lvrx(b, c),
                          a,
                          vec_lvsr(b, (unsigned char *)c)),
                 b, c);
}

static void __ATTRS_o_ai
vec_stvlxl(vector bool char a, int b, vector bool char *c)
{
  return vec_stl(vec_perm(vec_lvrx(b, c),
                          a,
                          vec_lvsr(b, (unsigned char *)c)),
                 b, c);
}

static void __ATTRS_o_ai
vec_stvlxl(vector short a, int b, short *c)
{
  return vec_stl(vec_perm(vec_lvrx(b, c),
                          a,
                          vec_lvsr(b, c)),
                 b, c);
}

static void __ATTRS_o_ai
vec_stvlxl(vector short a, int b, vector short *c)
{
  return vec_stl(vec_perm(vec_lvrx(b, c),
                          a,
                          vec_lvsr(b, (unsigned char *)c)),
                 b, c);
}

static void __ATTRS_o_ai
vec_stvlxl(vector unsigned short a, int b, unsigned short *c)
{
  return vec_stl(vec_perm(vec_lvrx(b, c),
                          a,
                          vec_lvsr(b, c)),
                 b, c);
}

static void __ATTRS_o_ai
vec_stvlxl(vector unsigned short a, int b, vector unsigned short *c)
{
  return vec_stl(vec_perm(vec_lvrx(b, c),
                          a,
                          vec_lvsr(b, (unsigned char *)c)),
                 b, c);
}

static void __ATTRS_o_ai
vec_stvlxl(vector bool short a, int b, vector bool short *c)
{
  return vec_stl(vec_perm(vec_lvrx(b, c),
                          a,
                          vec_lvsr(b, (unsigned char *)c)),
                 b, c);
}

static void __ATTRS_o_ai
vec_stvlxl(vector pixel a, int b, vector pixel *c)
{
  return vec_stl(vec_perm(vec_lvrx(b, c),
                          a,
                          vec_lvsr(b, (unsigned char *)c)),
                 b, c);
}

static void __ATTRS_o_ai
vec_stvlxl(vector int a, int b, int *c)
{
  return vec_stl(vec_perm(vec_lvrx(b, c),
                          a,
                          vec_lvsr(b, c)),
                 b, c);
}

static void __ATTRS_o_ai
vec_stvlxl(vector int a, int b, vector int *c)
{
  return vec_stl(vec_perm(vec_lvrx(b, c),
                          a,
                          vec_lvsr(b, (unsigned char *)c)),
                 b, c);
}

static void __ATTRS_o_ai
vec_stvlxl(vector unsigned int a, int b, unsigned int *c)
{
  return vec_stl(vec_perm(vec_lvrx(b, c),
                          a,
                          vec_lvsr(b, c)),
                 b, c);
}

static void __ATTRS_o_ai
vec_stvlxl(vector unsigned int a, int b, vector unsigned int *c)
{
  return vec_stl(vec_perm(vec_lvrx(b, c),
                          a,
                          vec_lvsr(b, (unsigned char *)c)),
                 b, c);
}

static void __ATTRS_o_ai
vec_stvlxl(vector bool int a, int b, vector bool int *c)
{
  return vec_stl(vec_perm(vec_lvrx(b, c),
                          a,
                          vec_lvsr(b, (unsigned char *)c)),
                 b, c);
}

static void __ATTRS_o_ai
vec_stvlxl(vector float a, int b, vector float *c)
{
  return vec_stl(vec_perm(vec_lvrx(b, c),
                          a,
                          vec_lvsr(b, (unsigned char *)c)),
                 b, c);
}

/* vec_stvrx */

static void __ATTRS_o_ai
vec_stvrx(vector signed char a, int b, signed char *c)
{
  return vec_st(vec_perm(a,
                         vec_lvlx(b, c),
                         vec_lvsr(b, c)),
                b, c);
}

static void __ATTRS_o_ai
vec_stvrx(vector signed char a, int b, vector signed char *c)
{
  return vec_st(vec_perm(a,
                         vec_lvlx(b, c),
                         vec_lvsr(b, (unsigned char *)c)),
                b, c);
}

static void __ATTRS_o_ai
vec_stvrx(vector unsigned char a, int b, unsigned char *c)
{
  return vec_st(vec_perm(a,
                         vec_lvlx(b, c),
                         vec_lvsr(b, c)),
                b, c);
}

static void __ATTRS_o_ai
vec_stvrx(vector unsigned char a, int b, vector unsigned char *c)
{
  return vec_st(vec_perm(a,
                         vec_lvlx(b, c),
                         vec_lvsr(b, (unsigned char *)c)),
                b, c);
}

static void __ATTRS_o_ai
vec_stvrx(vector bool char a, int b, vector bool char *c)
{
  return vec_st(vec_perm(a,
                         vec_lvlx(b, c),
                         vec_lvsr(b, (unsigned char *)c)),
                b, c);
}

static void __ATTRS_o_ai
vec_stvrx(vector short a, int b, short *c)
{
  return vec_st(vec_perm(a,
                         vec_lvlx(b, c),
                         vec_lvsr(b, c)),
                b, c);
}

static void __ATTRS_o_ai
vec_stvrx(vector short a, int b, vector short *c)
{
  return vec_st(vec_perm(a,
                         vec_lvlx(b, c),
                         vec_lvsr(b, (unsigned char *)c)),
                b, c);
}

static void __ATTRS_o_ai
vec_stvrx(vector unsigned short a, int b, unsigned short *c)
{
  return vec_st(vec_perm(a,
                         vec_lvlx(b, c),
                         vec_lvsr(b, c)),
                b, c);
}

static void __ATTRS_o_ai
vec_stvrx(vector unsigned short a, int b, vector unsigned short *c)
{
  return vec_st(vec_perm(a,
                         vec_lvlx(b, c),
                         vec_lvsr(b, (unsigned char *)c)),
                b, c);
}

static void __ATTRS_o_ai
vec_stvrx(vector bool short a, int b, vector bool short *c)
{
  return vec_st(vec_perm(a,
                         vec_lvlx(b, c),
                         vec_lvsr(b, (unsigned char *)c)),
                b, c);
}

static void __ATTRS_o_ai
vec_stvrx(vector pixel a, int b, vector pixel *c)
{
  return vec_st(vec_perm(a,
                         vec_lvlx(b, c),
                         vec_lvsr(b, (unsigned char *)c)),
                b, c);
}

static void __ATTRS_o_ai
vec_stvrx(vector int a, int b, int *c)
{
  return vec_st(vec_perm(a,
                         vec_lvlx(b, c),
                         vec_lvsr(b, c)),
                b, c);
}

static void __ATTRS_o_ai
vec_stvrx(vector int a, int b, vector int *c)
{
  return vec_st(vec_perm(a,
                         vec_lvlx(b, c),
                         vec_lvsr(b, (unsigned char *)c)),
                b, c);
}

static void __ATTRS_o_ai
vec_stvrx(vector unsigned int a, int b, unsigned int *c)
{
  return vec_st(vec_perm(a,
                         vec_lvlx(b, c),
                         vec_lvsr(b, c)),
                b, c);
}

static void __ATTRS_o_ai
vec_stvrx(vector unsigned int a, int b, vector unsigned int *c)
{
  return vec_st(vec_perm(a,
                         vec_lvlx(b, c),
                         vec_lvsr(b, (unsigned char *)c)),
                b, c);
}

static void __ATTRS_o_ai
vec_stvrx(vector bool int a, int b, vector bool int *c)
{
  return vec_st(vec_perm(a,
                         vec_lvlx(b, c),
                         vec_lvsr(b, (unsigned char *)c)),
                b, c);
}

static void __ATTRS_o_ai
vec_stvrx(vector float a, int b, vector float *c)
{
  return vec_st(vec_perm(a,
                         vec_lvlx(b, c),
                         vec_lvsr(b, (unsigned char *)c)),
                b, c);
}

/* vec_stvrxl */

static void __ATTRS_o_ai
vec_stvrxl(vector signed char a, int b, signed char *c)
{
  return vec_stl(vec_perm(a,
                          vec_lvlx(b, c),
                          vec_lvsr(b, c)),
                 b, c);
}

static void __ATTRS_o_ai
vec_stvrxl(vector signed char a, int b, vector signed char *c)
{
  return vec_stl(vec_perm(a,
                          vec_lvlx(b, c),
                          vec_lvsr(b, (unsigned char *)c)),
                 b, c);
}

static void __ATTRS_o_ai
vec_stvrxl(vector unsigned char a, int b, unsigned char *c)
{
  return vec_stl(vec_perm(a,
                          vec_lvlx(b, c),
                          vec_lvsr(b, c)),
                 b, c);
}

static void __ATTRS_o_ai
vec_stvrxl(vector unsigned char a, int b, vector unsigned char *c)
{
  return vec_stl(vec_perm(a,
                          vec_lvlx(b, c),
                          vec_lvsr(b, (unsigned char *)c)),
                 b, c);
}

static void __ATTRS_o_ai
vec_stvrxl(vector bool char a, int b, vector bool char *c)
{
  return vec_stl(vec_perm(a,
                          vec_lvlx(b, c),
                          vec_lvsr(b, (unsigned char *)c)),
                 b, c);
}

static void __ATTRS_o_ai
vec_stvrxl(vector short a, int b, short *c)
{
  return vec_stl(vec_perm(a,
                          vec_lvlx(b, c),
                          vec_lvsr(b, c)),
                 b, c);
}

static void __ATTRS_o_ai
vec_stvrxl(vector short a, int b, vector short *c)
{
  return vec_stl(vec_perm(a,
                          vec_lvlx(b, c),
                          vec_lvsr(b, (unsigned char *)c)),
                 b, c);
}

static void __ATTRS_o_ai
vec_stvrxl(vector unsigned short a, int b, unsigned short *c)
{
  return vec_stl(vec_perm(a,
                          vec_lvlx(b, c),
                          vec_lvsr(b, c)),
                 b, c);
}

static void __ATTRS_o_ai
vec_stvrxl(vector unsigned short a, int b, vector unsigned short *c)
{
  return vec_stl(vec_perm(a,
                          vec_lvlx(b, c),
                          vec_lvsr(b, (unsigned char *)c)),
                 b, c);
}

static void __ATTRS_o_ai
vec_stvrxl(vector bool short a, int b, vector bool short *c)
{
  return vec_stl(vec_perm(a,
                          vec_lvlx(b, c),
                          vec_lvsr(b, (unsigned char *)c)),
                 b, c);
}

static void __ATTRS_o_ai
vec_stvrxl(vector pixel a, int b, vector pixel *c)
{
  return vec_stl(vec_perm(a,
                          vec_lvlx(b, c),
                          vec_lvsr(b, (unsigned char *)c)),
                 b, c);
}

static void __ATTRS_o_ai
vec_stvrxl(vector int a, int b, int *c)
{
  return vec_stl(vec_perm(a,
                          vec_lvlx(b, c),
                          vec_lvsr(b, c)),
                 b, c);
}

static void __ATTRS_o_ai
vec_stvrxl(vector int a, int b, vector int *c)
{
  return vec_stl(vec_perm(a,
                          vec_lvlx(b, c),
                          vec_lvsr(b, (unsigned char *)c)),
                 b, c);
}

static void __ATTRS_o_ai
vec_stvrxl(vector unsigned int a, int b, unsigned int *c)
{
  return vec_stl(vec_perm(a,
                          vec_lvlx(b, c),
                          vec_lvsr(b, c)),
                 b, c);
}

static void __ATTRS_o_ai
vec_stvrxl(vector unsigned int a, int b, vector unsigned int *c)
{
  return vec_stl(vec_perm(a,
                          vec_lvlx(b, c),
                          vec_lvsr(b, (unsigned char *)c)),
                 b, c);
}

static void __ATTRS_o_ai
vec_stvrxl(vector bool int a, int b, vector bool int *c)
{
  return vec_stl(vec_perm(a,
                          vec_lvlx(b, c),
                          vec_lvsr(b, (unsigned char *)c)),
                 b, c);
}

static void __ATTRS_o_ai
vec_stvrxl(vector float a, int b, vector float *c)
{
  return vec_stl(vec_perm(a,
                          vec_lvlx(b, c),
                          vec_lvsr(b, (unsigned char *)c)),
                 b, c);
}

/* vec_promote */

static vector signed char __ATTRS_o_ai
vec_promote(signed char a, int b)
{
  vector signed char res = (vector signed char)(0);
  res[b] = a;
  return res;
}

static vector unsigned char __ATTRS_o_ai
vec_promote(unsigned char a, int b)
{
  vector unsigned char res = (vector unsigned char)(0);
  res[b] = a;
  return res;
}

static vector short __ATTRS_o_ai
vec_promote(short a, int b)
{
  vector short res = (vector short)(0);
  res[b] = a;
  return res;
}

static vector unsigned short __ATTRS_o_ai
vec_promote(unsigned short a, int b)
{
  vector unsigned short res = (vector unsigned short)(0);
  res[b] = a;
  return res;
}

static vector int __ATTRS_o_ai
vec_promote(int a, int b)
{
  vector int res = (vector int)(0);
  res[b] = a;
  return res;
}

static vector unsigned int __ATTRS_o_ai
vec_promote(unsigned int a, int b)
{
  vector unsigned int res = (vector unsigned int)(0);
  res[b] = a;
  return res;
}

static vector float __ATTRS_o_ai
vec_promote(float a, int b)
{
  vector float res = (vector float)(0);
  res[b] = a;
  return res;
}

/* vec_splats */

static vector signed char __ATTRS_o_ai
vec_splats(signed char a)
{
  return (vector signed char)(a);
}

static vector unsigned char __ATTRS_o_ai
vec_splats(unsigned char a)
{
  return (vector unsigned char)(a);
}

static vector short __ATTRS_o_ai
vec_splats(short a)
{
  return (vector short)(a);
}

static vector unsigned short __ATTRS_o_ai
vec_splats(unsigned short a)
{
  return (vector unsigned short)(a);
}

static vector int __ATTRS_o_ai
vec_splats(int a)
{
  return (vector int)(a);
}

static vector unsigned int __ATTRS_o_ai
vec_splats(unsigned int a)
{
  return (vector unsigned int)(a);
}

static vector float __ATTRS_o_ai
vec_splats(float a)
{
  return (vector float)(a);
}

/* ----------------------------- predicates --------------------------------- */

/* vec_all_eq */

static int __ATTRS_o_ai
vec_all_eq(vector signed char a, vector signed char b)
{
  return __builtin_altivec_vcmpequb_p(__CR6_LT, (vector char)a, (vector char)b);
}

static int __ATTRS_o_ai
vec_all_eq(vector signed char a, vector bool char b)
{
  return __builtin_altivec_vcmpequb_p(__CR6_LT, (vector char)a, (vector char)b);
}

static int __ATTRS_o_ai
vec_all_eq(vector unsigned char a, vector unsigned char b)
{
  return __builtin_altivec_vcmpequb_p(__CR6_LT, (vector char)a, (vector char)b);
}

static int __ATTRS_o_ai
vec_all_eq(vector unsigned char a, vector bool char b)
{
  return __builtin_altivec_vcmpequb_p(__CR6_LT, (vector char)a, (vector char)b);
}

static int __ATTRS_o_ai
vec_all_eq(vector bool char a, vector signed char b)
{
  return __builtin_altivec_vcmpequb_p(__CR6_LT, (vector char)a, (vector char)b);
}

static int __ATTRS_o_ai
vec_all_eq(vector bool char a, vector unsigned char b)
{
  return __builtin_altivec_vcmpequb_p(__CR6_LT, (vector char)a, (vector char)b);
}

static int __ATTRS_o_ai
vec_all_eq(vector bool char a, vector bool char b)
{
  return __builtin_altivec_vcmpequb_p(__CR6_LT, (vector char)a, (vector char)b);
}

static int __ATTRS_o_ai
vec_all_eq(vector short a, vector short b)
{
  return __builtin_altivec_vcmpequh_p(__CR6_LT, a, b);
}

static int __ATTRS_o_ai
vec_all_eq(vector short a, vector bool short b)
{
  return __builtin_altivec_vcmpequh_p(__CR6_LT, a, (vector short)b);
}

static int __ATTRS_o_ai
vec_all_eq(vector unsigned short a, vector unsigned short b)
{
  return
    __builtin_altivec_vcmpequh_p(__CR6_LT, (vector short)a, (vector short)b);
}

static int __ATTRS_o_ai
vec_all_eq(vector unsigned short a, vector bool short b)
{
  return
    __builtin_altivec_vcmpequh_p(__CR6_LT, (vector short)a, (vector short)b);
}

static int __ATTRS_o_ai
vec_all_eq(vector bool short a, vector short b)
{
  return
    __builtin_altivec_vcmpequh_p(__CR6_LT, (vector short)a, (vector short)b);
}

static int __ATTRS_o_ai
vec_all_eq(vector bool short a, vector unsigned short b)
{
  return
    __builtin_altivec_vcmpequh_p(__CR6_LT, (vector short)a, (vector short)b);
}

static int __ATTRS_o_ai
vec_all_eq(vector bool short a, vector bool short b)
{
  return
    __builtin_altivec_vcmpequh_p(__CR6_LT, (vector short)a, (vector short)b);
}

static int __ATTRS_o_ai
vec_all_eq(vector pixel a, vector pixel b)
{
  return
    __builtin_altivec_vcmpequh_p(__CR6_LT, (vector short)a, (vector short)b);
}

static int __ATTRS_o_ai
vec_all_eq(vector int a, vector int b)
{
  return __builtin_altivec_vcmpequw_p(__CR6_LT, a, b);
}

static int __ATTRS_o_ai
vec_all_eq(vector int a, vector bool int b)
{
  return __builtin_altivec_vcmpequw_p(__CR6_LT, a, (vector int)b);
}

static int __ATTRS_o_ai
vec_all_eq(vector unsigned int a, vector unsigned int b)
{
  return __builtin_altivec_vcmpequw_p(__CR6_LT, (vector int)a, (vector int)b);
}

static int __ATTRS_o_ai
vec_all_eq(vector unsigned int a, vector bool int b)
{
  return __builtin_altivec_vcmpequw_p(__CR6_LT, (vector int)a, (vector int)b);
}

static int __ATTRS_o_ai
vec_all_eq(vector bool int a, vector int b)
{
  return __builtin_altivec_vcmpequw_p(__CR6_LT, (vector int)a, (vector int)b);
}

static int __ATTRS_o_ai
vec_all_eq(vector bool int a, vector unsigned int b)
{
  return __builtin_altivec_vcmpequw_p(__CR6_LT, (vector int)a, (vector int)b);
}

static int __ATTRS_o_ai
vec_all_eq(vector bool int a, vector bool int b)
{
  return __builtin_altivec_vcmpequw_p(__CR6_LT, (vector int)a, (vector int)b);
}

static int __ATTRS_o_ai
vec_all_eq(vector float a, vector float b)
{
  return __builtin_altivec_vcmpeqfp_p(__CR6_LT, a, b);
}

/* vec_all_ge */

static int __ATTRS_o_ai
vec_all_ge(vector signed char a, vector signed char b)
{
  return __builtin_altivec_vcmpgtsb_p(__CR6_EQ, b, a);
}

static int __ATTRS_o_ai
vec_all_ge(vector signed char a, vector bool char b)
{
  return __builtin_altivec_vcmpgtsb_p(__CR6_EQ, (vector signed char)b, a);
}

static int __ATTRS_o_ai
vec_all_ge(vector unsigned char a, vector unsigned char b)
{
  return __builtin_altivec_vcmpgtub_p(__CR6_EQ, b, a);
}

static int __ATTRS_o_ai
vec_all_ge(vector unsigned char a, vector bool char b)
{
  return __builtin_altivec_vcmpgtub_p(__CR6_EQ, (vector unsigned char)b, a);
}

static int __ATTRS_o_ai
vec_all_ge(vector bool char a, vector signed char b)
{
  return __builtin_altivec_vcmpgtub_p(__CR6_EQ,
                                      (vector unsigned char)b,
                                      (vector unsigned char)a);
}

static int __ATTRS_o_ai
vec_all_ge(vector bool char a, vector unsigned char b)
{
  return __builtin_altivec_vcmpgtub_p(__CR6_EQ, b, (vector unsigned char)a);
}

static int __ATTRS_o_ai
vec_all_ge(vector bool char a, vector bool char b)
{
  return __builtin_altivec_vcmpgtub_p(__CR6_EQ,
                                      (vector unsigned char)b,
                                      (vector unsigned char)a);
}

static int __ATTRS_o_ai
vec_all_ge(vector short a, vector short b)
{
  return __builtin_altivec_vcmpgtsh_p(__CR6_EQ, b, a);
}

static int __ATTRS_o_ai
vec_all_ge(vector short a, vector bool short b)
{
  return __builtin_altivec_vcmpgtsh_p(__CR6_EQ, (vector short)b, a);
}

static int __ATTRS_o_ai
vec_all_ge(vector unsigned short a, vector unsigned short b)
{
  return __builtin_altivec_vcmpgtuh_p(__CR6_EQ, b, a);
}

static int __ATTRS_o_ai
vec_all_ge(vector unsigned short a, vector bool short b)
{
  return __builtin_altivec_vcmpgtuh_p(__CR6_EQ, (vector unsigned short)b, a);
}

static int __ATTRS_o_ai
vec_all_ge(vector bool short a, vector short b)
{
  return __builtin_altivec_vcmpgtuh_p(__CR6_EQ,
                                      (vector unsigned short)b,
                                      (vector unsigned short)a);
}

static int __ATTRS_o_ai
vec_all_ge(vector bool short a, vector unsigned short b)
{
  return __builtin_altivec_vcmpgtuh_p(__CR6_EQ, b, (vector unsigned short)a);
}

static int __ATTRS_o_ai
vec_all_ge(vector bool short a, vector bool short b)
{
  return __builtin_altivec_vcmpgtuh_p(__CR6_EQ,
                                      (vector unsigned short)b,
                                      (vector unsigned short)a);
}

static int __ATTRS_o_ai
vec_all_ge(vector int a, vector int b)
{
  return __builtin_altivec_vcmpgtsw_p(__CR6_EQ, b, a);
}

static int __ATTRS_o_ai
vec_all_ge(vector int a, vector bool int b)
{
  return __builtin_altivec_vcmpgtsw_p(__CR6_EQ, (vector int)b, a);
}

static int __ATTRS_o_ai
vec_all_ge(vector unsigned int a, vector unsigned int b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_EQ, b, a);
}

static int __ATTRS_o_ai
vec_all_ge(vector unsigned int a, vector bool int b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_EQ, (vector unsigned int)b, a);
}

static int __ATTRS_o_ai
vec_all_ge(vector bool int a, vector int b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_EQ,
                                      (vector unsigned int)b,
                                      (vector unsigned int)a);
}

static int __ATTRS_o_ai
vec_all_ge(vector bool int a, vector unsigned int b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_EQ, b, (vector unsigned int)a);
}

static int __ATTRS_o_ai
vec_all_ge(vector bool int a, vector bool int b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_EQ,
                                      (vector unsigned int)b,
                                      (vector unsigned int)a);
}

static int __ATTRS_o_ai
vec_all_ge(vector float a, vector float b)
{
  return __builtin_altivec_vcmpgefp_p(__CR6_LT, a, b);
}

/* vec_all_gt */

static int __ATTRS_o_ai
vec_all_gt(vector signed char a, vector signed char b)
{
  return __builtin_altivec_vcmpgtsb_p(__CR6_LT, a, b);
}

static int __ATTRS_o_ai
vec_all_gt(vector signed char a, vector bool char b)
{
  return __builtin_altivec_vcmpgtsb_p(__CR6_LT, a, (vector signed char)b);
}

static int __ATTRS_o_ai
vec_all_gt(vector unsigned char a, vector unsigned char b)
{
  return __builtin_altivec_vcmpgtub_p(__CR6_LT, a, b);
}

static int __ATTRS_o_ai
vec_all_gt(vector unsigned char a, vector bool char b)
{
  return __builtin_altivec_vcmpgtub_p(__CR6_LT, a, (vector unsigned char)b);
}

static int __ATTRS_o_ai
vec_all_gt(vector bool char a, vector signed char b)
{
  return __builtin_altivec_vcmpgtub_p(__CR6_LT,
                                      (vector unsigned char)a,
                                      (vector unsigned char)b);
}

static int __ATTRS_o_ai
vec_all_gt(vector bool char a, vector unsigned char b)
{
  return __builtin_altivec_vcmpgtub_p(__CR6_LT, (vector unsigned char)a, b);
}

static int __ATTRS_o_ai
vec_all_gt(vector bool char a, vector bool char b)
{
  return __builtin_altivec_vcmpgtub_p(__CR6_LT,
                                      (vector unsigned char)a,
                                      (vector unsigned char)b);
}

static int __ATTRS_o_ai
vec_all_gt(vector short a, vector short b)
{
  return __builtin_altivec_vcmpgtsh_p(__CR6_LT, a, b);
}

static int __ATTRS_o_ai
vec_all_gt(vector short a, vector bool short b)
{
  return __builtin_altivec_vcmpgtsh_p(__CR6_LT, a, (vector short)b);
}

static int __ATTRS_o_ai
vec_all_gt(vector unsigned short a, vector unsigned short b)
{
  return __builtin_altivec_vcmpgtuh_p(__CR6_LT, a, b);
}

static int __ATTRS_o_ai
vec_all_gt(vector unsigned short a, vector bool short b)
{
  return __builtin_altivec_vcmpgtuh_p(__CR6_LT, a, (vector unsigned short)b);
}

static int __ATTRS_o_ai
vec_all_gt(vector bool short a, vector short b)
{
  return __builtin_altivec_vcmpgtuh_p(__CR6_LT,
                                      (vector unsigned short)a,
                                      (vector unsigned short)b);
}

static int __ATTRS_o_ai
vec_all_gt(vector bool short a, vector unsigned short b)
{
  return __builtin_altivec_vcmpgtuh_p(__CR6_LT, (vector unsigned short)a, b);
}

static int __ATTRS_o_ai
vec_all_gt(vector bool short a, vector bool short b)
{
  return __builtin_altivec_vcmpgtuh_p(__CR6_LT,
                                      (vector unsigned short)a,
                                      (vector unsigned short)b);
}

static int __ATTRS_o_ai
vec_all_gt(vector int a, vector int b)
{
  return __builtin_altivec_vcmpgtsw_p(__CR6_LT, a, b);
}

static int __ATTRS_o_ai
vec_all_gt(vector int a, vector bool int b)
{
  return __builtin_altivec_vcmpgtsw_p(__CR6_LT, a, (vector int)b);
}

static int __ATTRS_o_ai
vec_all_gt(vector unsigned int a, vector unsigned int b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_LT, a, b);
}

static int __ATTRS_o_ai
vec_all_gt(vector unsigned int a, vector bool int b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_LT, a, (vector unsigned int)b);
}

static int __ATTRS_o_ai
vec_all_gt(vector bool int a, vector int b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_LT,
                                      (vector unsigned int)a,
                                      (vector unsigned int)b);
}

static int __ATTRS_o_ai
vec_all_gt(vector bool int a, vector unsigned int b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_LT, (vector unsigned int)a, b);
}

static int __ATTRS_o_ai
vec_all_gt(vector bool int a, vector bool int b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_LT,
                                      (vector unsigned int)a,
                                      (vector unsigned int)b);
}

static int __ATTRS_o_ai
vec_all_gt(vector float a, vector float b)
{
  return __builtin_altivec_vcmpgtfp_p(__CR6_LT, a, b);
}

/* vec_all_in */

static int __attribute__((__always_inline__))
vec_all_in(vector float a, vector float b)
{
  return __builtin_altivec_vcmpbfp_p(__CR6_EQ, a, b);
}

/* vec_all_le */

static int __ATTRS_o_ai
vec_all_le(vector signed char a, vector signed char b)
{
  return __builtin_altivec_vcmpgtsb_p(__CR6_EQ, a, b);
}

static int __ATTRS_o_ai
vec_all_le(vector signed char a, vector bool char b)
{
  return __builtin_altivec_vcmpgtsb_p(__CR6_EQ, a, (vector signed char)b);
}

static int __ATTRS_o_ai
vec_all_le(vector unsigned char a, vector unsigned char b)
{
  return __builtin_altivec_vcmpgtub_p(__CR6_EQ, a, b);
}

static int __ATTRS_o_ai
vec_all_le(vector unsigned char a, vector bool char b)
{
  return __builtin_altivec_vcmpgtub_p(__CR6_EQ, a, (vector unsigned char)b);
}

static int __ATTRS_o_ai
vec_all_le(vector bool char a, vector signed char b)
{
  return __builtin_altivec_vcmpgtub_p(__CR6_EQ,
                                      (vector unsigned char)a,
                                      (vector unsigned char)b);
}

static int __ATTRS_o_ai
vec_all_le(vector bool char a, vector unsigned char b)
{
  return __builtin_altivec_vcmpgtub_p(__CR6_EQ, (vector unsigned char)a, b);
}

static int __ATTRS_o_ai
vec_all_le(vector bool char a, vector bool char b)
{
  return __builtin_altivec_vcmpgtub_p(__CR6_EQ,
                                      (vector unsigned char)a,
                                      (vector unsigned char)b);
}

static int __ATTRS_o_ai
vec_all_le(vector short a, vector short b)
{
  return __builtin_altivec_vcmpgtsh_p(__CR6_EQ, a, b);
}

static int __ATTRS_o_ai
vec_all_le(vector short a, vector bool short b)
{
  return __builtin_altivec_vcmpgtsh_p(__CR6_EQ, a, (vector short)b);
}

static int __ATTRS_o_ai
vec_all_le(vector unsigned short a, vector unsigned short b)
{
  return __builtin_altivec_vcmpgtuh_p(__CR6_EQ, a, b);
}

static int __ATTRS_o_ai
vec_all_le(vector unsigned short a, vector bool short b)
{
  return __builtin_altivec_vcmpgtuh_p(__CR6_EQ, a, (vector unsigned short)b);
}

static int __ATTRS_o_ai
vec_all_le(vector bool short a, vector short b)
{
  return __builtin_altivec_vcmpgtuh_p(__CR6_EQ,
                                      (vector unsigned short)a,
                                      (vector unsigned short)b);
}

static int __ATTRS_o_ai
vec_all_le(vector bool short a, vector unsigned short b)
{
  return __builtin_altivec_vcmpgtuh_p(__CR6_EQ, (vector unsigned short)a, b);
}

static int __ATTRS_o_ai
vec_all_le(vector bool short a, vector bool short b)
{
  return __builtin_altivec_vcmpgtuh_p(__CR6_EQ,
                                      (vector unsigned short)a,
                                      (vector unsigned short)b);
}

static int __ATTRS_o_ai
vec_all_le(vector int a, vector int b)
{
  return __builtin_altivec_vcmpgtsw_p(__CR6_EQ, a, b);
}

static int __ATTRS_o_ai
vec_all_le(vector int a, vector bool int b)
{
  return __builtin_altivec_vcmpgtsw_p(__CR6_EQ, a, (vector int)b);
}

static int __ATTRS_o_ai
vec_all_le(vector unsigned int a, vector unsigned int b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_EQ, a, b);
}

static int __ATTRS_o_ai
vec_all_le(vector unsigned int a, vector bool int b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_EQ, a, (vector unsigned int)b);
}

static int __ATTRS_o_ai
vec_all_le(vector bool int a, vector int b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_EQ,
                                      (vector unsigned int)a,
                                      (vector unsigned int)b);
}

static int __ATTRS_o_ai
vec_all_le(vector bool int a, vector unsigned int b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_EQ, (vector unsigned int)a, b);
}

static int __ATTRS_o_ai
vec_all_le(vector bool int a, vector bool int b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_EQ,
                                      (vector unsigned int)a,
                                      (vector unsigned int)b);
}

static int __ATTRS_o_ai
vec_all_le(vector float a, vector float b)
{
  return __builtin_altivec_vcmpgefp_p(__CR6_LT, b, a);
}

/* vec_all_lt */

static int __ATTRS_o_ai
vec_all_lt(vector signed char a, vector signed char b)
{
  return __builtin_altivec_vcmpgtsb_p(__CR6_LT, b, a);
}

static int __ATTRS_o_ai
vec_all_lt(vector signed char a, vector bool char b)
{
  return __builtin_altivec_vcmpgtsb_p(__CR6_LT, (vector signed char)b, a);
}

static int __ATTRS_o_ai
vec_all_lt(vector unsigned char a, vector unsigned char b)
{
  return __builtin_altivec_vcmpgtub_p(__CR6_LT, b, a);
}

static int __ATTRS_o_ai
vec_all_lt(vector unsigned char a, vector bool char b)
{
  return __builtin_altivec_vcmpgtub_p(__CR6_LT, (vector unsigned char)b, a);
}

static int __ATTRS_o_ai
vec_all_lt(vector bool char a, vector signed char b)
{
  return __builtin_altivec_vcmpgtub_p(__CR6_LT,
                                      (vector unsigned char)b,
                                      (vector unsigned char)a);
}

static int __ATTRS_o_ai
vec_all_lt(vector bool char a, vector unsigned char b)
{
  return __builtin_altivec_vcmpgtub_p(__CR6_LT, b, (vector unsigned char)a);
}

static int __ATTRS_o_ai
vec_all_lt(vector bool char a, vector bool char b)
{
  return __builtin_altivec_vcmpgtub_p(__CR6_LT,
                                      (vector unsigned char)b,
                                      (vector unsigned char)a);
}

static int __ATTRS_o_ai
vec_all_lt(vector short a, vector short b)
{
  return __builtin_altivec_vcmpgtsh_p(__CR6_LT, b, a);
}

static int __ATTRS_o_ai
vec_all_lt(vector short a, vector bool short b)
{
  return __builtin_altivec_vcmpgtsh_p(__CR6_LT, (vector short)b, a);
}

static int __ATTRS_o_ai
vec_all_lt(vector unsigned short a, vector unsigned short b)
{
  return __builtin_altivec_vcmpgtuh_p(__CR6_LT, b, a);
}

static int __ATTRS_o_ai
vec_all_lt(vector unsigned short a, vector bool short b)
{
  return __builtin_altivec_vcmpgtuh_p(__CR6_LT, (vector unsigned short)b, a);
}

static int __ATTRS_o_ai
vec_all_lt(vector bool short a, vector short b)
{
  return __builtin_altivec_vcmpgtuh_p(__CR6_LT,
                                      (vector unsigned short)b,
                                      (vector unsigned short)a);
}

static int __ATTRS_o_ai
vec_all_lt(vector bool short a, vector unsigned short b)
{
  return __builtin_altivec_vcmpgtuh_p(__CR6_LT, b, (vector unsigned short)a);
}

static int __ATTRS_o_ai
vec_all_lt(vector bool short a, vector bool short b)
{
  return __builtin_altivec_vcmpgtuh_p(__CR6_LT,
                                      (vector unsigned short)b,
                                      (vector unsigned short)a);
}

static int __ATTRS_o_ai
vec_all_lt(vector int a, vector int b)
{
  return __builtin_altivec_vcmpgtsw_p(__CR6_LT, b, a);
}

static int __ATTRS_o_ai
vec_all_lt(vector int a, vector bool int b)
{
  return __builtin_altivec_vcmpgtsw_p(__CR6_LT, (vector int)b, a);
}

static int __ATTRS_o_ai
vec_all_lt(vector unsigned int a, vector unsigned int b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_LT, b, a);
}

static int __ATTRS_o_ai
vec_all_lt(vector unsigned int a, vector bool int b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_LT, (vector unsigned int)b, a);
}

static int __ATTRS_o_ai
vec_all_lt(vector bool int a, vector int b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_LT,
                                      (vector unsigned int)b,
                                      (vector unsigned int)a);
}

static int __ATTRS_o_ai
vec_all_lt(vector bool int a, vector unsigned int b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_LT, b, (vector unsigned int)a);
}

static int __ATTRS_o_ai
vec_all_lt(vector bool int a, vector bool int b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_LT,
                                      (vector unsigned int)b,
                                      (vector unsigned int)a);
}

static int __ATTRS_o_ai
vec_all_lt(vector float a, vector float b)
{
  return __builtin_altivec_vcmpgtfp_p(__CR6_LT, b, a);
}

/* vec_all_nan */

static int __attribute__((__always_inline__))
vec_all_nan(vector float a)
{
  return __builtin_altivec_vcmpeqfp_p(__CR6_EQ, a, a);
}

/* vec_all_ne */

static int __ATTRS_o_ai
vec_all_ne(vector signed char a, vector signed char b)
{
  return __builtin_altivec_vcmpequb_p(__CR6_EQ, (vector char)a, (vector char)b);
}

static int __ATTRS_o_ai
vec_all_ne(vector signed char a, vector bool char b)
{
  return __builtin_altivec_vcmpequb_p(__CR6_EQ, (vector char)a, (vector char)b);
}

static int __ATTRS_o_ai
vec_all_ne(vector unsigned char a, vector unsigned char b)
{
  return __builtin_altivec_vcmpequb_p(__CR6_EQ, (vector char)a, (vector char)b);
}

static int __ATTRS_o_ai
vec_all_ne(vector unsigned char a, vector bool char b)
{
  return __builtin_altivec_vcmpequb_p(__CR6_EQ, (vector char)a, (vector char)b);
}

static int __ATTRS_o_ai
vec_all_ne(vector bool char a, vector signed char b)
{
  return __builtin_altivec_vcmpequb_p(__CR6_EQ, (vector char)a, (vector char)b);
}

static int __ATTRS_o_ai
vec_all_ne(vector bool char a, vector unsigned char b)
{
  return __builtin_altivec_vcmpequb_p(__CR6_EQ, (vector char)a, (vector char)b);
}

static int __ATTRS_o_ai
vec_all_ne(vector bool char a, vector bool char b)
{
  return __builtin_altivec_vcmpequb_p(__CR6_EQ, (vector char)a, (vector char)b);
}

static int __ATTRS_o_ai
vec_all_ne(vector short a, vector short b)
{
  return __builtin_altivec_vcmpequh_p(__CR6_EQ, a, b);
}

static int __ATTRS_o_ai
vec_all_ne(vector short a, vector bool short b)
{
  return __builtin_altivec_vcmpequh_p(__CR6_EQ, a, (vector short)b);
}

static int __ATTRS_o_ai
vec_all_ne(vector unsigned short a, vector unsigned short b)
{
  return
    __builtin_altivec_vcmpequh_p(__CR6_EQ, (vector short)a, (vector short)b);
}

static int __ATTRS_o_ai
vec_all_ne(vector unsigned short a, vector bool short b)
{
  return
    __builtin_altivec_vcmpequh_p(__CR6_EQ, (vector short)a, (vector short)b);
}

static int __ATTRS_o_ai
vec_all_ne(vector bool short a, vector short b)
{
  return
    __builtin_altivec_vcmpequh_p(__CR6_EQ, (vector short)a, (vector short)b);
}

static int __ATTRS_o_ai
vec_all_ne(vector bool short a, vector unsigned short b)
{
  return
    __builtin_altivec_vcmpequh_p(__CR6_EQ, (vector short)a, (vector short)b);
}

static int __ATTRS_o_ai
vec_all_ne(vector bool short a, vector bool short b)
{
  return
    __builtin_altivec_vcmpequh_p(__CR6_EQ, (vector short)a, (vector short)b);
}

static int __ATTRS_o_ai
vec_all_ne(vector pixel a, vector pixel b)
{
  return
    __builtin_altivec_vcmpequh_p(__CR6_EQ, (vector short)a, (vector short)b);
}

static int __ATTRS_o_ai
vec_all_ne(vector int a, vector int b)
{
  return __builtin_altivec_vcmpequw_p(__CR6_EQ, a, b);
}

static int __ATTRS_o_ai
vec_all_ne(vector int a, vector bool int b)
{
  return __builtin_altivec_vcmpequw_p(__CR6_EQ, a, (vector int)b);
}

static int __ATTRS_o_ai
vec_all_ne(vector unsigned int a, vector unsigned int b)
{
  return __builtin_altivec_vcmpequw_p(__CR6_EQ, (vector int)a, (vector int)b);
}

static int __ATTRS_o_ai
vec_all_ne(vector unsigned int a, vector bool int b)
{
  return __builtin_altivec_vcmpequw_p(__CR6_EQ, (vector int)a, (vector int)b);
}

static int __ATTRS_o_ai
vec_all_ne(vector bool int a, vector int b)
{
  return __builtin_altivec_vcmpequw_p(__CR6_EQ, (vector int)a, (vector int)b);
}

static int __ATTRS_o_ai
vec_all_ne(vector bool int a, vector unsigned int b)
{
  return __builtin_altivec_vcmpequw_p(__CR6_EQ, (vector int)a, (vector int)b);
}

static int __ATTRS_o_ai
vec_all_ne(vector bool int a, vector bool int b)
{
  return __builtin_altivec_vcmpequw_p(__CR6_EQ, (vector int)a, (vector int)b);
}

static int __ATTRS_o_ai
vec_all_ne(vector float a, vector float b)
{
  return __builtin_altivec_vcmpeqfp_p(__CR6_EQ, a, b);
}

/* vec_all_nge */

static int __attribute__((__always_inline__))
vec_all_nge(vector float a, vector float b)
{
  return __builtin_altivec_vcmpgefp_p(__CR6_EQ, a, b);
}

/* vec_all_ngt */

static int __attribute__((__always_inline__))
vec_all_ngt(vector float a, vector float b)
{
  return __builtin_altivec_vcmpgtfp_p(__CR6_EQ, a, b);
}

/* vec_all_nle */

static int __attribute__((__always_inline__))
vec_all_nle(vector float a, vector float b)
{
  return __builtin_altivec_vcmpgefp_p(__CR6_EQ, b, a);
}

/* vec_all_nlt */

static int __attribute__((__always_inline__))
vec_all_nlt(vector float a, vector float b)
{
  return __builtin_altivec_vcmpgtfp_p(__CR6_EQ, b, a);
}

/* vec_all_numeric */

static int __attribute__((__always_inline__))
vec_all_numeric(vector float a)
{
  return __builtin_altivec_vcmpeqfp_p(__CR6_LT, a, a);
}

/* vec_any_eq */

static int __ATTRS_o_ai
vec_any_eq(vector signed char a, vector signed char b)
{
  return
    __builtin_altivec_vcmpequb_p(__CR6_EQ_REV, (vector char)a, (vector char)b);
}

static int __ATTRS_o_ai
vec_any_eq(vector signed char a, vector bool char b)
{
  return
    __builtin_altivec_vcmpequb_p(__CR6_EQ_REV, (vector char)a, (vector char)b);
}

static int __ATTRS_o_ai
vec_any_eq(vector unsigned char a, vector unsigned char b)
{
  return
    __builtin_altivec_vcmpequb_p(__CR6_EQ_REV, (vector char)a, (vector char)b);
}

static int __ATTRS_o_ai
vec_any_eq(vector unsigned char a, vector bool char b)
{
  return
    __builtin_altivec_vcmpequb_p(__CR6_EQ_REV, (vector char)a, (vector char)b);
}

static int __ATTRS_o_ai
vec_any_eq(vector bool char a, vector signed char b)
{
  return
    __builtin_altivec_vcmpequb_p(__CR6_EQ_REV, (vector char)a, (vector char)b);
}

static int __ATTRS_o_ai
vec_any_eq(vector bool char a, vector unsigned char b)
{
  return
    __builtin_altivec_vcmpequb_p(__CR6_EQ_REV, (vector char)a, (vector char)b);
}

static int __ATTRS_o_ai
vec_any_eq(vector bool char a, vector bool char b)
{
  return
    __builtin_altivec_vcmpequb_p(__CR6_EQ_REV, (vector char)a, (vector char)b);
}

static int __ATTRS_o_ai
vec_any_eq(vector short a, vector short b)
{
  return __builtin_altivec_vcmpequh_p(__CR6_EQ_REV, a, b);
}

static int __ATTRS_o_ai
vec_any_eq(vector short a, vector bool short b)
{
  return __builtin_altivec_vcmpequh_p(__CR6_EQ_REV, a, (vector short)b);
}

static int __ATTRS_o_ai
vec_any_eq(vector unsigned short a, vector unsigned short b)
{
  return __builtin_altivec_vcmpequh_p(__CR6_EQ_REV, 
                                      (vector short)a,
                                      (vector short)b);
}

static int __ATTRS_o_ai
vec_any_eq(vector unsigned short a, vector bool short b)
{
  return __builtin_altivec_vcmpequh_p(__CR6_EQ_REV, 
                                      (vector short)a,
                                      (vector short)b);
}

static int __ATTRS_o_ai
vec_any_eq(vector bool short a, vector short b)
{
  return __builtin_altivec_vcmpequh_p(__CR6_EQ_REV,
                                      (vector short)a,
                                      (vector short)b);
}

static int __ATTRS_o_ai
vec_any_eq(vector bool short a, vector unsigned short b)
{
  return __builtin_altivec_vcmpequh_p(__CR6_EQ_REV,
                                      (vector short)a,
                                      (vector short)b);
}

static int __ATTRS_o_ai
vec_any_eq(vector bool short a, vector bool short b)
{
  return __builtin_altivec_vcmpequh_p(__CR6_EQ_REV,
                                      (vector short)a,
                                      (vector short)b);
}

static int __ATTRS_o_ai
vec_any_eq(vector pixel a, vector pixel b)
{
  return __builtin_altivec_vcmpequh_p(__CR6_EQ_REV, 
                                      (vector short)a,
                                      (vector short)b);
}

static int __ATTRS_o_ai
vec_any_eq(vector int a, vector int b)
{
  return __builtin_altivec_vcmpequw_p(__CR6_EQ_REV, a, b);
}

static int __ATTRS_o_ai
vec_any_eq(vector int a, vector bool int b)
{
  return __builtin_altivec_vcmpequw_p(__CR6_EQ_REV, a, (vector int)b);
}

static int __ATTRS_o_ai
vec_any_eq(vector unsigned int a, vector unsigned int b)
{
  return
    __builtin_altivec_vcmpequw_p(__CR6_EQ_REV, (vector int)a, (vector int)b);
}

static int __ATTRS_o_ai
vec_any_eq(vector unsigned int a, vector bool int b)
{
  return
    __builtin_altivec_vcmpequw_p(__CR6_EQ_REV, (vector int)a, (vector int)b);
}

static int __ATTRS_o_ai
vec_any_eq(vector bool int a, vector int b)
{
  return
    __builtin_altivec_vcmpequw_p(__CR6_EQ_REV, (vector int)a, (vector int)b);
}

static int __ATTRS_o_ai
vec_any_eq(vector bool int a, vector unsigned int b)
{
  return
    __builtin_altivec_vcmpequw_p(__CR6_EQ_REV, (vector int)a, (vector int)b);
}

static int __ATTRS_o_ai
vec_any_eq(vector bool int a, vector bool int b)
{
  return
    __builtin_altivec_vcmpequw_p(__CR6_EQ_REV, (vector int)a, (vector int)b);
}

static int __ATTRS_o_ai
vec_any_eq(vector float a, vector float b)
{
  return __builtin_altivec_vcmpeqfp_p(__CR6_EQ_REV, a, b);
}

/* vec_any_ge */

static int __ATTRS_o_ai
vec_any_ge(vector signed char a, vector signed char b)
{
  return __builtin_altivec_vcmpgtsb_p(__CR6_LT_REV, b, a);
}

static int __ATTRS_o_ai
vec_any_ge(vector signed char a, vector bool char b)
{
  return __builtin_altivec_vcmpgtsb_p(__CR6_LT_REV, (vector signed char)b, a);
}

static int __ATTRS_o_ai
vec_any_ge(vector unsigned char a, vector unsigned char b)
{
  return __builtin_altivec_vcmpgtub_p(__CR6_LT_REV, b, a);
}

static int __ATTRS_o_ai
vec_any_ge(vector unsigned char a, vector bool char b)
{
  return __builtin_altivec_vcmpgtub_p(__CR6_LT_REV, (vector unsigned char)b, a);
}

static int __ATTRS_o_ai
vec_any_ge(vector bool char a, vector signed char b)
{
  return __builtin_altivec_vcmpgtub_p(__CR6_LT_REV,
                                      (vector unsigned char)b,
                                      (vector unsigned char)a);
}

static int __ATTRS_o_ai
vec_any_ge(vector bool char a, vector unsigned char b)
{
  return __builtin_altivec_vcmpgtub_p(__CR6_LT_REV, b, (vector unsigned char)a);
}

static int __ATTRS_o_ai
vec_any_ge(vector bool char a, vector bool char b)
{
  return __builtin_altivec_vcmpgtub_p(__CR6_LT_REV,
                                      (vector unsigned char)b,
                                      (vector unsigned char)a);
}

static int __ATTRS_o_ai
vec_any_ge(vector short a, vector short b)
{
  return __builtin_altivec_vcmpgtsh_p(__CR6_LT_REV, b, a);
}

static int __ATTRS_o_ai
vec_any_ge(vector short a, vector bool short b)
{
  return __builtin_altivec_vcmpgtsh_p(__CR6_LT_REV, (vector short)b, a);
}

static int __ATTRS_o_ai
vec_any_ge(vector unsigned short a, vector unsigned short b)
{
  return __builtin_altivec_vcmpgtuh_p(__CR6_LT_REV, b, a);
}

static int __ATTRS_o_ai
vec_any_ge(vector unsigned short a, vector bool short b)
{
  return
    __builtin_altivec_vcmpgtuh_p(__CR6_LT_REV, (vector unsigned short)b, a);
}

static int __ATTRS_o_ai
vec_any_ge(vector bool short a, vector short b)
{
  return __builtin_altivec_vcmpgtuh_p(__CR6_LT_REV,
                                      (vector unsigned short)b,
                                      (vector unsigned short)a);
}

static int __ATTRS_o_ai
vec_any_ge(vector bool short a, vector unsigned short b)
{
  return 
    __builtin_altivec_vcmpgtuh_p(__CR6_LT_REV, b, (vector unsigned short)a);
}

static int __ATTRS_o_ai
vec_any_ge(vector bool short a, vector bool short b)
{
  return __builtin_altivec_vcmpgtuh_p(__CR6_LT_REV,
                                      (vector unsigned short)b,
                                      (vector unsigned short)a);
}

static int __ATTRS_o_ai
vec_any_ge(vector int a, vector int b)
{
  return __builtin_altivec_vcmpgtsw_p(__CR6_LT_REV, b, a);
}

static int __ATTRS_o_ai
vec_any_ge(vector int a, vector bool int b)
{
  return __builtin_altivec_vcmpgtsw_p(__CR6_LT_REV, (vector int)b, a);
}

static int __ATTRS_o_ai
vec_any_ge(vector unsigned int a, vector unsigned int b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_LT_REV, b, a);
}

static int __ATTRS_o_ai
vec_any_ge(vector unsigned int a, vector bool int b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_LT_REV, (vector unsigned int)b, a);
}

static int __ATTRS_o_ai
vec_any_ge(vector bool int a, vector int b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_LT_REV,
                                      (vector unsigned int)b,
                                      (vector unsigned int)a);
}

static int __ATTRS_o_ai
vec_any_ge(vector bool int a, vector unsigned int b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_LT_REV, b, (vector unsigned int)a);
}

static int __ATTRS_o_ai
vec_any_ge(vector bool int a, vector bool int b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_LT_REV,
                                      (vector unsigned int)b,
                                      (vector unsigned int)a);
}

static int __ATTRS_o_ai
vec_any_ge(vector float a, vector float b)
{
  return __builtin_altivec_vcmpgefp_p(__CR6_EQ_REV, a, b);
}

/* vec_any_gt */

static int __ATTRS_o_ai
vec_any_gt(vector signed char a, vector signed char b)
{
  return __builtin_altivec_vcmpgtsb_p(__CR6_EQ_REV, a, b);
}

static int __ATTRS_o_ai
vec_any_gt(vector signed char a, vector bool char b)
{
  return __builtin_altivec_vcmpgtsb_p(__CR6_EQ_REV, a, (vector signed char)b);
}

static int __ATTRS_o_ai
vec_any_gt(vector unsigned char a, vector unsigned char b)
{
  return __builtin_altivec_vcmpgtub_p(__CR6_EQ_REV, a, b);
}

static int __ATTRS_o_ai
vec_any_gt(vector unsigned char a, vector bool char b)
{
  return 
    __builtin_altivec_vcmpgtub_p(__CR6_EQ_REV, a, (vector unsigned char)b);
}

static int __ATTRS_o_ai
vec_any_gt(vector bool char a, vector signed char b)
{
  return __builtin_altivec_vcmpgtub_p(__CR6_EQ_REV,
                                      (vector unsigned char)a,
                                      (vector unsigned char)b);
}

static int __ATTRS_o_ai
vec_any_gt(vector bool char a, vector unsigned char b)
{
  return 
    __builtin_altivec_vcmpgtub_p(__CR6_EQ_REV, (vector unsigned char)a, b);
}

static int __ATTRS_o_ai
vec_any_gt(vector bool char a, vector bool char b)
{
  return __builtin_altivec_vcmpgtub_p(__CR6_EQ_REV,
                                      (vector unsigned char)a,
                                      (vector unsigned char)b);
}

static int __ATTRS_o_ai
vec_any_gt(vector short a, vector short b)
{
  return __builtin_altivec_vcmpgtsh_p(__CR6_EQ_REV, a, b);
}

static int __ATTRS_o_ai
vec_any_gt(vector short a, vector bool short b)
{
  return __builtin_altivec_vcmpgtsh_p(__CR6_EQ_REV, a, (vector short)b);
}

static int __ATTRS_o_ai
vec_any_gt(vector unsigned short a, vector unsigned short b)
{
  return __builtin_altivec_vcmpgtuh_p(__CR6_EQ_REV, a, b);
}

static int __ATTRS_o_ai
vec_any_gt(vector unsigned short a, vector bool short b)
{
  return 
    __builtin_altivec_vcmpgtuh_p(__CR6_EQ_REV, a, (vector unsigned short)b);
}

static int __ATTRS_o_ai
vec_any_gt(vector bool short a, vector short b)
{
  return __builtin_altivec_vcmpgtuh_p(__CR6_EQ_REV,
                                      (vector unsigned short)a,
                                      (vector unsigned short)b);
}

static int __ATTRS_o_ai
vec_any_gt(vector bool short a, vector unsigned short b)
{
  return
    __builtin_altivec_vcmpgtuh_p(__CR6_EQ_REV, (vector unsigned short)a, b);
}

static int __ATTRS_o_ai
vec_any_gt(vector bool short a, vector bool short b)
{
  return __builtin_altivec_vcmpgtuh_p(__CR6_EQ_REV,
                                      (vector unsigned short)a,
                                      (vector unsigned short)b);
}

static int __ATTRS_o_ai
vec_any_gt(vector int a, vector int b)
{
  return __builtin_altivec_vcmpgtsw_p(__CR6_EQ_REV, a, b);
}

static int __ATTRS_o_ai
vec_any_gt(vector int a, vector bool int b)
{
  return __builtin_altivec_vcmpgtsw_p(__CR6_EQ_REV, a, (vector int)b);
}

static int __ATTRS_o_ai
vec_any_gt(vector unsigned int a, vector unsigned int b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_EQ_REV, a, b);
}

static int __ATTRS_o_ai
vec_any_gt(vector unsigned int a, vector bool int b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_EQ_REV, a, (vector unsigned int)b);
}

static int __ATTRS_o_ai
vec_any_gt(vector bool int a, vector int b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_EQ_REV,
                                      (vector unsigned int)a,
                                      (vector unsigned int)b);
}

static int __ATTRS_o_ai
vec_any_gt(vector bool int a, vector unsigned int b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_EQ_REV, (vector unsigned int)a, b);
}

static int __ATTRS_o_ai
vec_any_gt(vector bool int a, vector bool int b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_EQ_REV,
                                      (vector unsigned int)a,
                                      (vector unsigned int)b);
}

static int __ATTRS_o_ai
vec_any_gt(vector float a, vector float b)
{
  return __builtin_altivec_vcmpgtfp_p(__CR6_EQ_REV, a, b);
}

/* vec_any_le */

static int __ATTRS_o_ai
vec_any_le(vector signed char a, vector signed char b)
{
  return __builtin_altivec_vcmpgtsb_p(__CR6_LT_REV, a, b);
}

static int __ATTRS_o_ai
vec_any_le(vector signed char a, vector bool char b)
{
  return __builtin_altivec_vcmpgtsb_p(__CR6_LT_REV, a, (vector signed char)b);
}

static int __ATTRS_o_ai
vec_any_le(vector unsigned char a, vector unsigned char b)
{
  return __builtin_altivec_vcmpgtub_p(__CR6_LT_REV, a, b);
}

static int __ATTRS_o_ai
vec_any_le(vector unsigned char a, vector bool char b)
{
  return 
    __builtin_altivec_vcmpgtub_p(__CR6_LT_REV, a, (vector unsigned char)b);
}

static int __ATTRS_o_ai
vec_any_le(vector bool char a, vector signed char b)
{
  return __builtin_altivec_vcmpgtub_p(__CR6_LT_REV,
                                      (vector unsigned char)a,
                                      (vector unsigned char)b);
}

static int __ATTRS_o_ai
vec_any_le(vector bool char a, vector unsigned char b)
{
  return 
    __builtin_altivec_vcmpgtub_p(__CR6_LT_REV, (vector unsigned char)a, b);
}

static int __ATTRS_o_ai
vec_any_le(vector bool char a, vector bool char b)
{
  return __builtin_altivec_vcmpgtub_p(__CR6_LT_REV,
                                      (vector unsigned char)a,
                                      (vector unsigned char)b);
}

static int __ATTRS_o_ai
vec_any_le(vector short a, vector short b)
{
  return __builtin_altivec_vcmpgtsh_p(__CR6_LT_REV, a, b);
}

static int __ATTRS_o_ai
vec_any_le(vector short a, vector bool short b)
{
  return __builtin_altivec_vcmpgtsh_p(__CR6_LT_REV, a, (vector short)b);
}

static int __ATTRS_o_ai
vec_any_le(vector unsigned short a, vector unsigned short b)
{
  return __builtin_altivec_vcmpgtuh_p(__CR6_LT_REV, a, b);
}

static int __ATTRS_o_ai
vec_any_le(vector unsigned short a, vector bool short b)
{
  return 
    __builtin_altivec_vcmpgtuh_p(__CR6_LT_REV, a, (vector unsigned short)b);
}

static int __ATTRS_o_ai
vec_any_le(vector bool short a, vector short b)
{
  return __builtin_altivec_vcmpgtuh_p(__CR6_LT_REV,
                                      (vector unsigned short)a,
                                      (vector unsigned short)b);
}

static int __ATTRS_o_ai
vec_any_le(vector bool short a, vector unsigned short b)
{
  return 
    __builtin_altivec_vcmpgtuh_p(__CR6_LT_REV, (vector unsigned short)a, b);
}

static int __ATTRS_o_ai
vec_any_le(vector bool short a, vector bool short b)
{
  return __builtin_altivec_vcmpgtuh_p(__CR6_LT_REV,
                                      (vector unsigned short)a,
                                      (vector unsigned short)b);
}

static int __ATTRS_o_ai
vec_any_le(vector int a, vector int b)
{
  return __builtin_altivec_vcmpgtsw_p(__CR6_LT_REV, a, b);
}

static int __ATTRS_o_ai
vec_any_le(vector int a, vector bool int b)
{
  return __builtin_altivec_vcmpgtsw_p(__CR6_LT_REV, a, (vector int)b);
}

static int __ATTRS_o_ai
vec_any_le(vector unsigned int a, vector unsigned int b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_LT_REV, a, b);
}

static int __ATTRS_o_ai
vec_any_le(vector unsigned int a, vector bool int b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_LT_REV, a, (vector unsigned int)b);
}

static int __ATTRS_o_ai
vec_any_le(vector bool int a, vector int b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_LT_REV,
                                      (vector unsigned int)a,
                                      (vector unsigned int)b);
}

static int __ATTRS_o_ai
vec_any_le(vector bool int a, vector unsigned int b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_LT_REV, (vector unsigned int)a, b);
}

static int __ATTRS_o_ai
vec_any_le(vector bool int a, vector bool int b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_LT_REV,
                                      (vector unsigned int)a,
                                      (vector unsigned int)b);
}

static int __ATTRS_o_ai
vec_any_le(vector float a, vector float b)
{
  return __builtin_altivec_vcmpgefp_p(__CR6_EQ_REV, b, a);
}

/* vec_any_lt */

static int __ATTRS_o_ai
vec_any_lt(vector signed char a, vector signed char b)
{
  return __builtin_altivec_vcmpgtsb_p(__CR6_EQ_REV, b, a);
}

static int __ATTRS_o_ai
vec_any_lt(vector signed char a, vector bool char b)
{
  return __builtin_altivec_vcmpgtsb_p(__CR6_EQ_REV, (vector signed char)b, a);
}

static int __ATTRS_o_ai
vec_any_lt(vector unsigned char a, vector unsigned char b)
{
  return __builtin_altivec_vcmpgtub_p(__CR6_EQ_REV, b, a);
}

static int __ATTRS_o_ai
vec_any_lt(vector unsigned char a, vector bool char b)
{
  return 
    __builtin_altivec_vcmpgtub_p(__CR6_EQ_REV, (vector unsigned char)b, a);
}

static int __ATTRS_o_ai
vec_any_lt(vector bool char a, vector signed char b)
{
  return __builtin_altivec_vcmpgtub_p(__CR6_EQ_REV,
                                      (vector unsigned char)b,
                                      (vector unsigned char)a);
}

static int __ATTRS_o_ai
vec_any_lt(vector bool char a, vector unsigned char b)
{
  return 
    __builtin_altivec_vcmpgtub_p(__CR6_EQ_REV, b, (vector unsigned char)a);
}

static int __ATTRS_o_ai
vec_any_lt(vector bool char a, vector bool char b)
{
  return __builtin_altivec_vcmpgtub_p(__CR6_EQ_REV,
                                      (vector unsigned char)b,
                                      (vector unsigned char)a);
}

static int __ATTRS_o_ai
vec_any_lt(vector short a, vector short b)
{
  return __builtin_altivec_vcmpgtsh_p(__CR6_EQ_REV, b, a);
}

static int __ATTRS_o_ai
vec_any_lt(vector short a, vector bool short b)
{
  return __builtin_altivec_vcmpgtsh_p(__CR6_EQ_REV, (vector short)b, a);
}

static int __ATTRS_o_ai
vec_any_lt(vector unsigned short a, vector unsigned short b)
{
  return __builtin_altivec_vcmpgtuh_p(__CR6_EQ_REV, b, a);
}

static int __ATTRS_o_ai
vec_any_lt(vector unsigned short a, vector bool short b)
{
  return 
    __builtin_altivec_vcmpgtuh_p(__CR6_EQ_REV, (vector unsigned short)b, a);
}

static int __ATTRS_o_ai
vec_any_lt(vector bool short a, vector short b)
{
  return __builtin_altivec_vcmpgtuh_p(__CR6_EQ_REV,
                                      (vector unsigned short)b,
                                      (vector unsigned short)a);
}

static int __ATTRS_o_ai
vec_any_lt(vector bool short a, vector unsigned short b)
{
  return 
    __builtin_altivec_vcmpgtuh_p(__CR6_EQ_REV, b, (vector unsigned short)a);
}

static int __ATTRS_o_ai
vec_any_lt(vector bool short a, vector bool short b)
{
  return __builtin_altivec_vcmpgtuh_p(__CR6_EQ_REV,
                                      (vector unsigned short)b,
                                      (vector unsigned short)a);
}

static int __ATTRS_o_ai
vec_any_lt(vector int a, vector int b)
{
  return __builtin_altivec_vcmpgtsw_p(__CR6_EQ_REV, b, a);
}

static int __ATTRS_o_ai
vec_any_lt(vector int a, vector bool int b)
{
  return __builtin_altivec_vcmpgtsw_p(__CR6_EQ_REV, (vector int)b, a);
}

static int __ATTRS_o_ai
vec_any_lt(vector unsigned int a, vector unsigned int b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_EQ_REV, b, a);
}

static int __ATTRS_o_ai
vec_any_lt(vector unsigned int a, vector bool int b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_EQ_REV, (vector unsigned int)b, a);
}

static int __ATTRS_o_ai
vec_any_lt(vector bool int a, vector int b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_EQ_REV,
                                      (vector unsigned int)b,
                                      (vector unsigned int)a);
}

static int __ATTRS_o_ai
vec_any_lt(vector bool int a, vector unsigned int b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_EQ_REV, b, (vector unsigned int)a);
}

static int __ATTRS_o_ai
vec_any_lt(vector bool int a, vector bool int b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_EQ_REV,
                                      (vector unsigned int)b,
                                      (vector unsigned int)a);
}

static int __ATTRS_o_ai
vec_any_lt(vector float a, vector float b)
{
  return __builtin_altivec_vcmpgtfp_p(__CR6_EQ_REV, b, a);
}

/* vec_any_nan */

static int __attribute__((__always_inline__))
vec_any_nan(vector float a)
{
  return __builtin_altivec_vcmpeqfp_p(__CR6_LT_REV, a, a);
}

/* vec_any_ne */

static int __ATTRS_o_ai
vec_any_ne(vector signed char a, vector signed char b)
{
  return
    __builtin_altivec_vcmpequb_p(__CR6_LT_REV, (vector char)a, (vector char)b);
}

static int __ATTRS_o_ai
vec_any_ne(vector signed char a, vector bool char b)
{
  return
    __builtin_altivec_vcmpequb_p(__CR6_LT_REV, (vector char)a, (vector char)b);
}

static int __ATTRS_o_ai
vec_any_ne(vector unsigned char a, vector unsigned char b)
{
  return
    __builtin_altivec_vcmpequb_p(__CR6_LT_REV, (vector char)a, (vector char)b);
}

static int __ATTRS_o_ai
vec_any_ne(vector unsigned char a, vector bool char b)
{
  return
    __builtin_altivec_vcmpequb_p(__CR6_LT_REV, (vector char)a, (vector char)b);
}

static int __ATTRS_o_ai
vec_any_ne(vector bool char a, vector signed char b)
{
  return
    __builtin_altivec_vcmpequb_p(__CR6_LT_REV, (vector char)a, (vector char)b);
}

static int __ATTRS_o_ai
vec_any_ne(vector bool char a, vector unsigned char b)
{
  return
    __builtin_altivec_vcmpequb_p(__CR6_LT_REV, (vector char)a, (vector char)b);
}

static int __ATTRS_o_ai
vec_any_ne(vector bool char a, vector bool char b)
{
  return
    __builtin_altivec_vcmpequb_p(__CR6_LT_REV, (vector char)a, (vector char)b);
}

static int __ATTRS_o_ai
vec_any_ne(vector short a, vector short b)
{
  return __builtin_altivec_vcmpequh_p(__CR6_LT_REV, a, b);
}

static int __ATTRS_o_ai
vec_any_ne(vector short a, vector bool short b)
{
  return __builtin_altivec_vcmpequh_p(__CR6_LT_REV, a, (vector short)b);
}

static int __ATTRS_o_ai
vec_any_ne(vector unsigned short a, vector unsigned short b)
{
  return __builtin_altivec_vcmpequh_p(__CR6_LT_REV, 
                                      (vector short)a,
                                      (vector short)b);
}

static int __ATTRS_o_ai
vec_any_ne(vector unsigned short a, vector bool short b)
{
  return __builtin_altivec_vcmpequh_p(__CR6_LT_REV,
                                      (vector short)a,
                                      (vector short)b);
}

static int __ATTRS_o_ai
vec_any_ne(vector bool short a, vector short b)
{
  return __builtin_altivec_vcmpequh_p(__CR6_LT_REV,
                                      (vector short)a,
                                      (vector short)b);
}

static int __ATTRS_o_ai
vec_any_ne(vector bool short a, vector unsigned short b)
{
  return __builtin_altivec_vcmpequh_p(__CR6_LT_REV,
                                      (vector short)a,
                                      (vector short)b);
}

static int __ATTRS_o_ai
vec_any_ne(vector bool short a, vector bool short b)
{
  return __builtin_altivec_vcmpequh_p(__CR6_LT_REV,
                                      (vector short)a,
                                      (vector short)b);
}

static int __ATTRS_o_ai
vec_any_ne(vector pixel a, vector pixel b)
{
  return __builtin_altivec_vcmpequh_p(__CR6_LT_REV,
                                      (vector short)a,
                                      (vector short)b);
}

static int __ATTRS_o_ai
vec_any_ne(vector int a, vector int b)
{
  return __builtin_altivec_vcmpequw_p(__CR6_LT_REV, a, b);
}

static int __ATTRS_o_ai
vec_any_ne(vector int a, vector bool int b)
{
  return __builtin_altivec_vcmpequw_p(__CR6_LT_REV, a, (vector int)b);
}

static int __ATTRS_o_ai
vec_any_ne(vector unsigned int a, vector unsigned int b)
{
  return
    __builtin_altivec_vcmpequw_p(__CR6_LT_REV, (vector int)a, (vector int)b);
}

static int __ATTRS_o_ai
vec_any_ne(vector unsigned int a, vector bool int b)
{
  return
    __builtin_altivec_vcmpequw_p(__CR6_LT_REV, (vector int)a, (vector int)b);
}

static int __ATTRS_o_ai
vec_any_ne(vector bool int a, vector int b)
{
  return
    __builtin_altivec_vcmpequw_p(__CR6_LT_REV, (vector int)a, (vector int)b);
}

static int __ATTRS_o_ai
vec_any_ne(vector bool int a, vector unsigned int b)
{
  return
    __builtin_altivec_vcmpequw_p(__CR6_LT_REV, (vector int)a, (vector int)b);
}

static int __ATTRS_o_ai
vec_any_ne(vector bool int a, vector bool int b)
{
  return
    __builtin_altivec_vcmpequw_p(__CR6_LT_REV, (vector int)a, (vector int)b);
}

static int __ATTRS_o_ai
vec_any_ne(vector float a, vector float b)
{
  return __builtin_altivec_vcmpeqfp_p(__CR6_LT_REV, a, b);
}

/* vec_any_nge */

static int __attribute__((__always_inline__))
vec_any_nge(vector float a, vector float b)
{
  return __builtin_altivec_vcmpgefp_p(__CR6_LT_REV, a, b);
}

/* vec_any_ngt */

static int __attribute__((__always_inline__))
vec_any_ngt(vector float a, vector float b)
{
  return __builtin_altivec_vcmpgtfp_p(__CR6_LT_REV, a, b);
}

/* vec_any_nle */

static int __attribute__((__always_inline__))
vec_any_nle(vector float a, vector float b)
{
  return __builtin_altivec_vcmpgefp_p(__CR6_LT_REV, b, a);
}

/* vec_any_nlt */

static int __attribute__((__always_inline__))
vec_any_nlt(vector float a, vector float b)
{
  return __builtin_altivec_vcmpgtfp_p(__CR6_LT_REV, b, a);
}

/* vec_any_numeric */

static int __attribute__((__always_inline__))
vec_any_numeric(vector float a)
{
  return __builtin_altivec_vcmpeqfp_p(__CR6_EQ_REV, a, a);
}

/* vec_any_out */

static int __attribute__((__always_inline__))
vec_any_out(vector float a, vector float b)
{
  return __builtin_altivec_vcmpbfp_p(__CR6_EQ_REV, a, b);
}

#undef __ATTRS_o_ai

#endif /* __ALTIVEC_H */
