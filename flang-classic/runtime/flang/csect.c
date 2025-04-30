/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*
 * Define the support of an i/o statement to be a critical section:
 *    ENTF90IO(BEGIN, begin)() - marks the begin of an i/o statement
 *    ENTF90IO(END, end)()   - marks the end of an i/o statement
 * Routines are called by the generated code if the option -x 125 1 is
 * selected.  Functions to be completed by users who need this facility.
 */

#include "fioMacros.h"

void ENTF90IO(BEGIN, begin)() {}

void ENTF90IO(END, end)() {}

void ENTCRF90IO(BEGIN, begin)() {}

void ENTCRF90IO(END, end)() {}
