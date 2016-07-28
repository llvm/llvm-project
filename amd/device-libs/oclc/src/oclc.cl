/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "oclc.h"

__attribute__((always_inline, const, weak)) int __oclc_finite_only_opt(void) { return 0; }
__attribute__((always_inline, const, weak)) int __oclc_unsafe_math_opt(void) { return 0; }
__attribute__((always_inline, const, weak)) int __oclc_daz_opt(void) { return 1; }
__attribute__((always_inline, const, weak)) int __oclc_amd_opt(void) { return 1; }
__attribute__((always_inline, const, weak)) int __oclc_correctly_rounded_sqrt32(void) { return 0; }
__attribute__((always_inline, const, weak)) int __oclc_ISA_version(void) { return 800; }

