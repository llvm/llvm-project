/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file
 * \brief Initialization and error handling functions for Fortran I/O
 */

void set_gbl_newunit(bool newunit);
bool get_gbl_newunit();
void __fortio_errinit(__INT_T unit, __INT_T bitv, __INT_T *iostat, const char *str);
void __fortio_errinit03(__INT_T unit, __INT_T bitv, __INT_T *iostat, const char *str);
void __fortio_errend03();
void __fortio_fmtinit();
void __fortio_fmtend(void);
int __fortio_error(int errval);
const char * __fortio_errmsg(int errval);
int __fortio_eoferr(int errval);
int __fortio_eorerr(int errval);
int __fortio_check_format(void);
int __fortio_eor_crlf(void);

int __fortio_no_minus_zero(void);

