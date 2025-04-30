/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/** \file
 * \brief Stubs
 */

#include "stdioInterf.h"
#include "fioMacros.h"

static int tid = 0;

const char *__fort_transnam = "rpm1";

#if defined(_WIN64)

/* pg access routines for data shared between windows dlls */

const char *
__get_fort_transnam(void)
{
  return __fort_transnam;
}

#endif

/** \brief Abort the whole mess */
void
__fort_abortx(void)
{
  __fort_traceback();
  __abort(1, NULL);
}

/** \brief End of parallel program */
void
__fort_endpar(void)
{
}

/** \brief Begin parallel program */
void
__fort_begpar(int ncpus)
{
  SET_DIST_TCPUS(1);
  SET_DIST_LCPU(0);
  SET_DIST_TIDS(&tid);

  __fort_procargs();
  if (GET_DIST_TCPUS != 1) {
    fprintf(__io_stderr(),
            "0: RPM1 uses only 1 processor, using 1 processor\n");
    SET_DIST_TCPUS(1);
  }
  __fort_sethand();
}

void
__fort_barrier()
{
}

void ENTFTN(BARRIER, barrier)() {}

__INT_T
ENTFTN(TID, tid)(__INT_T *lcpu) { return (*lcpu); }
