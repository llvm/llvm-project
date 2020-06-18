/*
 * AdvSIMD vector PCS variant of __v_exp2f_1u.
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */
#include "mathlib.h"
#ifdef __vpcs
#define VPCS 1
#include "v_exp2f_1u.c"
#endif
