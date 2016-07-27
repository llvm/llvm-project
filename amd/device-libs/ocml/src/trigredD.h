/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

extern int MATH_PRIVATE(trigredsmall)(__private double *r, __private double *rr, double x);
extern int MATH_PRIVATE(trigredlarge)(__private double *r, __private double *rr, double x);
extern int MATH_PRIVATE(trigred)(__private double *r, __private double *rr, double x);

extern double MATH_PRIVATE(sincosred)(double x, __private double *cp);
extern double MATH_PRIVATE(sincosred2)(double x, double y, __private double *cp);

extern CONSTATTR double MATH_PRIVATE(tanred2)(double x, double xx, int sel);

