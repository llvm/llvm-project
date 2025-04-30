
/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "directives.h"

#define _SX0 0
#define _SX1 4
#define _SX2 8
#define _SX3 12

#define _SY0 16
#define _SY1 20
#define _SY2 24
#define _SY3 28

#define _SR0 32
#define _SR1 36
#define _SR2 40
#define _SR3 44

#ifndef LNUM
#define LNUM 001
#endif
#define NNN LNUM

#define LNUM 001
#include "relaxedtable.h"
#undef LNUM

#undef TARGET_FMA
#define VEX_TARGET 1
#define	TARGET_VEX_OR_FMA	vex
#define LNUM 004
#include "relaxedmath_vex.h"
#include "relaxedmath_vex_mask.h"
#undef VEX_TARGET 
#undef	TARGET_VEX_OR_FMA
#undef LNUM


#define TARGET_FMA 1
#define	TARGET_VEX_OR_FMA	fma4
#define	VFMA_IS_FMA3_OR_FMA4	FMA4
#define LNUM 005
#include "relaxedmath_vex.h"
#include "relaxedmath_vex_mask.h"
#undef	TARGET_FMA
#undef	TARGET_VEX_OR_FMA
#undef	VFMA_IS_FMA3_OR_FMA4
#undef	LNUM

#define TARGET_FMA 1
#define	TARGET_VEX_OR_FMA	avx2
#define	VFMA_IS_FMA3_OR_FMA4	FMA3
#define LNUM 007
#include "relaxedmath_vex.h"
#include "relaxedmath_vex_mask.h"
#undef	TARGET_FMA
#undef	TARGET_VEX_OR_FMA
#undef	VFMA_IS_FMA3_OR_FMA4
#undef	LNUM
