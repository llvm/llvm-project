/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file
 * Macro definitions and function declarations for Fortran formatted IO.
 */

/* define number of decimal digit characters to represent exponent       */
#define REAL4_E 2
#define REAL8_E 3
#define REAL16_E 4

/* define number of decimal digit characters to represent mantissa       */
/* appears to be <FLT/DBL/LDBL>_DIG  + 1    from appropriate c float.h   */
#define REAL4_D 7
#define REAL8_D 16
#define REAL16_D 34
/* compatible with gfortran */
#define G_REAL16_D 36

/* define number of decimal digit characters to represent a real         */
/* appears to be  W  ==  E + D + 6                                       */
/*      to print  real of form  -O.xxxxE-yyy                             */
/*       additional digits:     123    45       6?                       */
#define REAL4_W 15
#define REAL8_W 25
#define REAL16_W 44
/* compatible with gfortran */
#define G_REAL16_W 45

/* Only use these for converting BIGreals */
#define BIGREAL_E REAL8_E
#define BIGREAL_D REAL8_D
#define BIGREAL_W REAL8_W

/* ***  fmtconv.c  *****/

/** \brief Generate a formated REAL*8 or REAL*16 string */
void __fortio_printbigreal(__BIGREAL_T);

/** \brief Drive the generation of output strings based on format strings */
char *__fortio_default_convert(char *, int, int, int *, bool, bool, int);

/** \brief Generate a formated INTEGER string */
char *__fortio_fmt_i(__BIGINT_T, int, int, bool);

/** \brief Generate a formated INTEGER*8 string */
char *__fortio_fmt_i8(DBLINT64, int, int, bool);

/** \brief Generate a string for a 'D' format characer */
char *__fortio_fmt_d(__BIGREAL_T, int, int, int, int, bool, int);

/** \brief Generate a string for a 'G' format characer */
char *__fortio_fmt_g(__BIGREAL_T, int, int, int, int, int, bool, bool, bool, int,
                     int);

/** \brief Generate a string for an 'F' format characer */
char *__fortio_fmt_f(__BIGREAL_T, int, int, int, bool, bool, int);

/** \brief Generate a string for an 'E' format characer */
char *__fortio_fmt_e(__BIGREAL_T, int, int, int, int, int, bool, bool, bool,
                     int, int);

/** \brief Convert REAL*4 to REAL4*8 */
__BIGREAL_T __fortio_chk_f(__REAL4_T *);

/* ***  fmtgetnum.c  *****/

/** \brief Extract integer or __BIGREAL_T scalar values from a string. */
int __fortio_getnum(char *, int *, void *, int *, bool);

/* ***  fmtwrite.c  *****/

/** \brief Low level Fortran formatted write routine
 *
 *  \param type data type (as defined in pghpft.h)
 *  \param length # items of type to read. May be <= 0
 *  \param stride distance in bytes between items
 *  \param *item where to transfer data from
 *  \param item_length byte length
 */
int __f90io_fmt_write(int type, long length, int stride, char *item,
                      __CLEN_T item_length);

/* ***  fmtread.c  *****/

/** \brief Low level Fortran formatted read routine
 *
 *  \param type data type (as defined in pghpft.h)
 *  \param length # items of type to read. May be <= 0
 *  \param stride distance in bytes between items
 *  \param item where to transfer data to
 *  \param item_length byte length
 */
int __f90io_fmt_read(int type, long length, int stride, char *item,
                     __CLEN_T item_length);

/* ***  utilsi64.c  *****/
/** \brief char string to 32-bit integer.
 *
 *  \param Input string containing number to be converted
 *         (string is NOT null terminated.)
 *  \param Pointer to returned integer value
 *  \param Number of chars from str to convert
 *  \param Radix of conversion -- 2, 8, 10, 16.  If base is 16, then the digits\
 * *         a-f or A-F are allowed.
 *
 *  \return
 *          -1      str does not represent a valid number in radix base
 *          -2      overflow occurred on conversion
 *           0      No error -- *ival contains the converted number.
 */

int __fort_atoxi32(char *s, INT *i, int n, int base);

/** \brief Converts [un]signed 64 integer into a char string of
 *   the selected radix.
 *
 *  \param input integer value
 *  \param output string
 *  \param number of chars in to
 *  \param 0 = signed conversion; 1 = unsigned
 *  \param conversion radix: 2, 8, 10, 16
 */

void __fort_i64toax(DBLINT64 from, char *to, int count, int sign, int radix);

/** \brief char string to 64-bit integer.
 *
 *  \param Input string containing number to be converted
 *         (string is NOT null terminated.)
 *  \param DBLUINT64 output value
 *  \param Number of chars from str to convert
 *  \param Radix of conversion -- 2, 8, 10, 16.  If base is 16, then the
 *   digits a-f
 *
 *  \return
 *    -1      str does not represent a valid number in radix base
 *    -2      overflow occurred on conversion
 *     0      No error -- *ir contains the converted number.
 */
int __fort_atoxi64(char *s, DBLINT64 ir, int n, int radix);
