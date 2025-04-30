/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#define TABLE_TARGET
#include "relaxedmathc.h"
#undef TABLE_TARGET

#define TARGET_VEX_OR_FMA       vex
#include "relaxedmathc.h"
#undef  TARGET_VEX_OR_FMA

#define TARGET_FMA
#define TARGET_VEX_OR_FMA       fma4
#define VFMA_IS_FMA3_OR_FMA4    FMA4
#include "relaxedmathc.h"
#undef  TARGET_VEX_OR_FMA
#undef  VFMA_IS_FMA3_OR_FMA4

#define TARGET_FMA
#define TARGET_VEX_OR_FMA       avx2
#define VFMA_IS_FMA3_OR_FMA4    FMA3
#include "relaxedmathc.h"
#undef  TARGET_VEX_OR_FMA
#undef  VFMA_IS_FMA3_OR_FMA4
