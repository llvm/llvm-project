/**
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */
/** \file
 * \brief LR utility header files
 *
 *       This file declares all constants and varaiables global to all
 *       LR utils routines, also contains all the routine declarations.
 *
 */

#ifndef LRUTILS_H
#define LRUTILS_H

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <errno.h>
#include <setjmp.h>
#include <signal.h>
#include <ctype.h>

/* Constants */

#define ABORT 1
#define CONTINUE 0
#define FALSE 0
#define FPS_ERR 0
#define MAX_LINE_LEN 1024
#define SYS_ERR 1
#define TRUE 1

/* cli */
#define ARG_TABLE_COLS 65
#define ARG_TABLE_ROWS 256
#define END_OF_TABLE_MARK 0
#define OVERFLOW_MARK -3
#define UNSWITCHED_ARG_MARK -1
#define UNEXPECTED_ARG_MARK -2

/* file */
#define BINARY_FILES 0x3a0e50
#define ERR_FCLOSE EOF
#define ERR_FILE_DOESNT_EXIST 2
#define EXT_READ_WRITE_ACCESS 3
#define EXT_SIZE 7
#define FILNAM_LEN 128
#define FSEEK_OK 0
#define MAX_FILTYP 22
#define MAX_PREFIXES 2
#define NFILE 10
#define NO_OP 0
#define PREFIX_SIZE 29
#define STANDARD_INPUT_OUTPUT 1

/* Macros */
/**
 *  INSET     In Set Macro
 *
 *       INSET receives two (2) numeric parameters. The first (set)
 *       is a bit map for a set, with each bit being a flag which
 *       indicates if the number corresponding to the bit's position
 *       in the word is in the set. The second parameter (number)
 *       is a number which must be in the range 0 to 31. The macro
 *       returns 1 if 'number' is in 'set', 0 otherwise.
 */
#define INSET(set, number) (((set) >> (number)) & 1)

#define INT int
#define UINT unsigned int

/*  Function Declarations */

/* char */
void a1tos1(INT *from, INT *to, INT count);
void a4tos1(const char *from, INT *to, INT count);
void i32tos(INT *from, INT *to, INT count, INT sign, INT radix, INT format);
void s1toa1(INT *from, INT *to, INT *count);
void s1toa4(INT *from, INT *to, INT *count);

/* file */
INT find(INT *swtab, INT arg_table[ARG_TABLE_ROWS][ARG_TABLE_COLS],
         INT cur_arg);
INT rdline(FILE *fp, INT *buffer, INT count);
void wtline(FILE *fp, INT *buffer, INT count);
INT wtpage(FILE *fp);

/* misc */
void add_arg_entry(INT *argtab, INT arglen, INT *argtab_end_ptr,
                   INT arg_table[ARG_TABLE_ROWS][ARG_TABLE_COLS], INT type,
                   INT first_lbl, INT lbl_cnt);
void cli(INT *swtab, INT *argtab, INT arglen, INT caller);
void exitf(INT errcnd);
void getarg(INT *n, char *s, INT ls);
INT iargc(void);
void lower_to_upper(INT cur_arg, INT arg_table[ARG_TABLE_ROWS][ARG_TABLE_COLS]);
INT upper_to_lower(INT *name, INT col);

void util_error(const char *, INT, INT);

#ifdef TRACE
void trace(char *);
#endif

#endif
