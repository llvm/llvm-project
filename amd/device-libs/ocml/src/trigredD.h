/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

struct redret {
    double lo;
    double hi;
    int i;
};

struct scret {
    double s;
    double c;
};

extern CONSTATTR struct redret MATH_PRIVATE(trigredsmall)(double x);
extern CONSTATTR struct redret MATH_PRIVATE(trigredlarge)(double x);
extern CONSTATTR struct redret MATH_PRIVATE(trigred)(double x);

extern CONSTATTR struct scret MATH_PRIVATE(sincosred)(double x);
extern CONSTATTR struct scret MATH_PRIVATE(sincosred2)(double x, double y);

extern CONSTATTR double MATH_PRIVATE(tanred2)(double x, double xx, int sel);

