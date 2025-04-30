/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */


/**
 *  \file
 *  Declare routines that access the machine registers
 */

#include "flangrti_config.h"

#if defined(HAVE_GREGSET_T)
#include <sys/ucontext.h>
#define FLANGRTI_GREGSET_T gregset_t
#define FLANGRTI_UCONTEXT_T ucontext_t
#else
#define FLANGRTI_GREGSET_T void
#define FLANGRTI_UCONTEXT_T void
#endif

void dumpregs(FLANGRTI_GREGSET_T *regs);
FLANGRTI_GREGSET_T *getRegs(FLANGRTI_UCONTEXT_T *u);
