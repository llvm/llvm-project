/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file
 *  Function declarations for Fortran list IO (files ldread.c/ldwrite.c).
 */

/** \brief
 * list-directed external file read initialization (defined in ldread.c) 
 *
 * \param type data  type (as defined in pghpft.h) 
 * \param length  # items of type to read
 * \param stride   distance in bytes between items
 * \param item   where to transfer data to
 * \param itemlen
 */
int __f90io_ldr(int type, long length, int stride, char *item, __CLEN_T itemlen);

/** \brief
 *  list-directed external file write initializations (defined in ldwrite.c) 
 *
 * \param type     data type (as defined in pghpft.h)
 * \param length  # items of type to write. May be <= 0 
 * \param stride   distance in bytes between items 
 * \param item   where to transfer data from 
 * \param item_length
 */
int
__f90io_ldw(int type, long length, int stride, char *item, __CLEN_T item_length);
