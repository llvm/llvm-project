/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/**
 * \file
 * \brief Fortran RTE name build and entry symbol macros
 */

/* TODO FOR FLANG: resolve/merge w/ent3f.h??? */

#ifndef _PGHPFENT_H_
#define _PGHPFENT_H_

/* Alternate Fortran entry symbol formats */
#if defined(DESC_I8)
#define ENTF90IO(UC, LC) f90io_##LC##_i8
#define ENTF90(UC, LC) f90_##LC##_i8
#define ENTFTN(UC, LC) fort_##LC##_i8
#define ENTRY(UC, LC) LC##_i8
#define ENTCRF90IO(UC, LC) crf90io_##LC##_i8 /* FIXME: HPF, delete all with this prefix*/
#define ENTFTNIO(UC, LC) ftnio_##LC##64
#define ENTCRFTNIO(UC, LC) crftnio_##LC##_i8 /* FIXME: HPF, delete all with this prefix*/
#define F90_MATMUL(s) f90_mm_##s##_i8_
#define F90_NORM2(s) f90_norm2_##s##_i8_
#else /* !defined(DESC_I8) */
#define ENTF90IO(UC, LC) f90io_##LC
#define ENTF90(UC, LC) f90_##LC
#define ENTFTN(UC, LC) fort_##LC
#define ENTRY(UC, LC) LC
#define ENTCRF90IO(UC, LC) crf90io_##LC	/* FIXME: HPF, delete all with this prefix*/
#define ENTFTNIO(UC, LC) ftnio_##LC
#define ENTCRFTNIO(UC, LC) crftnio_##LC	/* FIXME: HPF, delete all with this prefix*/
#define F90_MATMUL(s) f90_mm_##s##_
#define F90_NORM2(s) f90_norm2_##s##_
#endif /* defined(DESC_I8) */
#define ENTF90COMN(UC, LC) pgf90_##LC
#define ENTCRF90(UC, LC) crf90_##LC	/* FIXME: HPF, delete all with this prefix*/
#define ENTCRFTN(UC, LC) crftn_##LC	/* FIXME: HPF, delete all with this prefix*/ 
#define ENTCOMN(UC, LC) ftn_##LC##_	/* FIXME: common blocks */

#if defined(DESC_I8)
#define I8(s) s##_i8
#define I8_(s) s##i8_
#define F90_I8(s) s##_i8_
#else
#define I8(s) s
#define I8_(s) s
#define F90_I8(s) s##_
#endif

/* macros to put character length arguments in their place.
   DCHAR declares a character pointer argument.
   DCLEN declares a character length argument. Since DCLEN may have an
   empty definition, no commas should be used before or after a DCLEN
   reference in a dummy argument list.
   CADR gets the character pointer.
   CLEN gets the character length.  */

#define __CLEN_T size_t
#define DCHAR(ARG) char *ARG##_adr
#define DCLEN(ARG) , int ARG##_len
#define DCLEN64(ARG) , __CLEN_T ARG##_len
#define CADR(ARG) (ARG##_adr)
#define CLEN(ARG) (ARG##_len)

#if defined(_WIN64) && defined(_DLL)
   #define WIN_API extern __declspec(dllexport)
#else
  #define WIN_API extern
#endif

#define CORMEM ENTCOMN(0L, 0l)
#define LINENO ENTCOMN(LINENO, lineno)

#define LOCAL_MODE 0

/* SUBGROUP_MODE used to indicate communication between a subset of
 * processors ...
 * Used in __fort_global_reduce (reduct.c)
 */

#define SUBGROUP_MODE 0

/* declare a variable private to a thread (taskcommon) */

#define PRIVGLOB(type, var) type var
#define PRIVSTAT(type, var) static type var
#define PRIVXTRN(type, var) extern type var

#endif
