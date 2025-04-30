/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

/*
 *	Global to all include passes.
 */
#include "tmp.mthdecls.h"

#ifdef PGFLANG
#include "directives.h"
#else
#include "directives.h"
#endif

#define	FMATH_EXCEPTIONS

/*
 *	First pass.
 */
#define	TABLE_TARGET
#define	LNUM	001
#include "fastmath.h"
#undef	TABLE_TARGET
#undef	LNUM


/*
 *	Second pass.
 */
#define	LNUM	002
#include "fastmath.h"
#undef	LNUM


/*
 *	Third pass.
 */
#define	GH_TARGET
#define	LNUM	003
#include "fastmath.h"
#undef	GH_TARGET
#undef	LNUM


/*
 *	Fourth pass.
 */
#define	VEX_TARGET
#define	TARGET_VEX_OR_FMA	vex
#define	LNUM	004
#include "fastmath.h"
#undef	VEX_TARGET
#undef	TARGET_VEX_OR_FMA
#undef	LNUM


/*
 *	Fifth pass.
 */
#define	VEX_TARGET
#define	TARGET_FMA
#define	TARGET_VEX_OR_FMA	fma4
#define	VFMA_IS_FMA3_OR_FMA4	FMA4
#define	LNUM	005
#include "fastmath.h"
#undef	VEX_TARGET
#undef	TARGET_FMA
#undef	TARGET_VEX_OR_FMA
#undef	VFMA_IS_FMA3_OR_FMA4
#undef	LNUM


/*
 *	Sixth pass.
 */
#define	HELPER_TARGET
#define	LNUM	006
#include "fastmath.h"
#undef	HELPER_TARGET
#undef	LNUM

/*
 *	Seventh pass.
 */
#define	VEX_TARGET
#define	TARGET_FMA
#define	TARGET_VEX_OR_FMA	avx2
#define	VFMA_IS_FMA3_OR_FMA4	FMA3
#define	LNUM	007
#include "fastmath.h"
#undef	VEX_TARGET
#undef	TARGET_FMA
#undef	TARGET_VEX_OR_FMA
#undef	VFMA_IS_FMA3_OR_FMA4
#undef	LNUM
