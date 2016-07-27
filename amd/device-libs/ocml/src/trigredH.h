/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

extern int MATH_PRIVATE(trigred)(__private half *r, half x);
extern half MATH_PRIVATE(sincosred)(half x, __private half *cp);
extern CONSTATTR half MATH_PRIVATE(tanred)(half x, int regn);

