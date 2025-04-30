/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include <stdioInterf.h>

/*
 * On linux, linking -g77libs (=> libf2c) may result in an 'undefined
 * reference to MAIN__' which is g77's name for a fortran main program.
 * Just define the function in libpgc.a and hope it's never called.
 */
int
MAIN__()
{
  fprintf(__io_stderr(),
          "MAIN__ called -- missing g77-compiled main program\n");
  return 0;
}
