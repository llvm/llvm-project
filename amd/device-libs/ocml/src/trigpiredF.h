/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

struct redret {
    float hi;
    int i;
};

struct scret {
    float s;
    float c;
};

extern CONSTATTR struct redret MATH_PRIVATE(trigpired)(float x);
extern CONSTATTR struct scret MATH_PRIVATE(sincospired)(float x);
extern CONSTATTR float MATH_PRIVATE(tanpired)(float x, int i);

