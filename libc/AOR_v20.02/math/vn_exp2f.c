/*
 * AdvSIMD vector PCS variant of __v_exp2f.
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */
#include "mathlib.h"
#ifdef __vpcs
#define VPCS 1
#define VPCS_ALIAS strong_alias (__vn_exp2f, _ZGVnN4v_exp2f)
#include "v_exp2f.c"
#endif
