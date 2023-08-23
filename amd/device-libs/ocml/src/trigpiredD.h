/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

struct redret {
    double hi;
    int i;
};

struct scret {
    double c;
    double s;
};

extern CONSTATTR struct redret MATH_PRIVATE(trigpired)(double x);
extern CONSTATTR struct scret MATH_PRIVATE(sincospired)(double x);
extern CONSTATTR double MATH_PRIVATE(tanpired)(double x, int i);

