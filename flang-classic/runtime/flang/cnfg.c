/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* cnfg.c - contains the definitions & routines which control
   configurable behavior */

#include "cnfg.h"
#include <string.h>
#include <stdlib.h>
#include "stdioInterf.h"

/* WARNING: cnfg.c generates two objects, cnfg.o and pgfcnfg.o
 * define global variable which contains the configurable items:
 *
 * default_name - sprintf string used to construct the default name of a
 *                file upon an open of unit which does not contain a file
 *                specifier and is not a scratch file.
 *
 * true_mask    - mask value used to determine if a value is true or false;
 *                defined so that if the expression (true_mask & val) is
 *                non-zero, the value is true.
 *                    1 => bottom bit is tested (odd => true)
 *                   -1 => non-zero value is true
 *                Default is 1 (VAX-style)
 */

FIO_CNFG __fortio_cnfg_ = { /* ending '_' so it can be accessed by user */
    "fort.%d",              /* default_name */
                            /* vax-style */
    1,                      /* odd => true */
    -1,                     /* internal value of .TRUE. */
};

/* fio access routines */

const char *
__get_fio_cnfg_default_name(void)
{
  return __fortio_cnfg_.default_name;
}

int
__get_fio_cnfg_true_mask(void)
{
  return __fortio_cnfg_.true_mask;
}

int *
__get_fio_cnfg_true_mask_addr(void)
{
  return &__fortio_cnfg_.true_mask;
}

int
__get_fio_cnfg_ftn_true(void)
{
  return __fortio_cnfg_.ftn_true;
}

int *
__get_fio_cnfg_ftn_true_addr(void)
{
  return &__fortio_cnfg_.ftn_true;
}

void
__fortio_scratch_name(char *filename, int unit)
/* generate the name of an unnamed scracth file */
{
  char *nm;

#if defined(_WIN64)
  if (getenv("TMP") == 0)
    nm = __io_tempnam("C:\\", "FTN");
  else
    nm = __io_tempnam((char *)0, "FTN");
  strcpy(filename, nm);
  if (nm)
    free(nm);
#else /*_WIN64*/

  nm = __io_tempnam((char *)0, "FTN");
  strcpy(filename, nm);
  if (nm)
    free(nm);

#endif /*_WIN64*/

}
