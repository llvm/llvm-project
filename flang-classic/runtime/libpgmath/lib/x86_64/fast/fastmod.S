/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "directives.h"

#define TABLE_TARGET
#define LNUM 001
#include "fastmod.h"


#undef TABLE_TARGET
#undef LNUM
#define LNUM 002
#include "fastmod.h"


#undef LNUM
#define LNUM 003
#define GH_TARGET
#include "fastmod.h"


#undef LNUM
#define LNUM 004
#undef GH_TARGET
#define	TARGET_VEX_OR_FMA	vex
#include "fastmod.h"
#undef	TARGET_VEX_OR_FMA


#undef LNUM
#define LNUM 005
#define	TARGET_VEX_OR_FMA	fma4
#include "fastmod.h"
#undef	TARGET_VEX_OR_FMA
