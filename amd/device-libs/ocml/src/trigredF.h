/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#define SMALL_BOUND 0x1.0p+17f

#if defined EXTRA_PRECISION
struct redret {
    float hi;
    float lo;
    int i;
};
#else
struct redret {
    float hi;
    int i;
};
#endif

struct scret {
    float s;
    float c;
};

extern CONSTATTR struct redret MATH_PRIVATE(trigredsmall)(float x);
extern CONSTATTR struct redret MATH_PRIVATE(trigredlarge)(float x);
extern CONSTATTR struct redret MATH_PRIVATE(trigred)(float x);


#if defined EXTRA_PRECISION
extern CONSTATTR struct scret  MATH_PRIVATE(sincosred2)(float x, float y);
#else
extern CONSTATTR struct scret  MATH_PRIVATE(sincosred)(float x);
#endif

extern CONSTATTR float MATH_PRIVATE(tanred)(float x, int regn);

