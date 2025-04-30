/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file   scan.h
    \brief  data declarations for those items which are set by
            the scanner for use by the parser or semantic analyzer.
*/
/* check radix format  */
#define RADIX2_FMT_ERR(r, n) (r == 2 && (n > 64))
#define RADIX8_FMT_ERR(r, n, cp) (r == 8 && (n > 22 || (n == 22 && (*cp - '0' > 1))))
#define RADIX16_FMT_ERR(r, n) (r == 16 && (n > 16))

typedef struct {
  int stmtyp;
  int currlab;
  INT labno;
  LOGICAL end_program_unit; /* end of program unit seen */
  LOGICAL is_hpf;           /* true if current statement began with the
                             * '!hpf$' prefix.
                             */
  LOGICAL multiple_stmts;   /* stmts separated by ';' */
  char *directive;          /* malloc'd area containing a directive string
                             * to be passed thru as a comment string.  The
                             * string includes the the necessary prefix.
                             */
  char *options;            /* malloc'd area containing the string after
                             * 'options' in the options statement.
                             */
  struct {
    char *name;
    int avl;
    int size;
  } id;
} SCN;

/* File Records:
 *
 * Each record in the ast source file (astb.astfil) begins with a
 * 4-byte type field.  In most cases, the remaining portion of the
 * field is textual information in the form of a line (terminated by
 * '\n'.
 */
typedef enum {
  FR_SRC = -1,
  FR_B_INCL = -2,
  FR_E_INCL = -3,
  FR_END = -4,
  FR_LINENO = -5,
  FR_PRAGMA = -6,
  FR_STMT = -7,
  FR_B_HDR = -8,
  FR_E_HDR = -9,
  FR_LABEL = -98,
  FR_TOKEN = -99
} FR_TYPE;

extern SCN scn;

void scan_init(FILE *);
void scan_reset(void);
void scan_fini(void);
int get_token(INT *);
void scan_include(char *);
void scan_opt_restore(void);
int get_named_stmtyp(void);
int getCurrColumn(void);
void scan_options(void);
void fe_save_state(void);
void fe_init(void);
void fe_restart(void);
char * get_src_line(int line, char **src_file, int col, int *srcCol, 
                    int *contNo);

LOGICAL is_executable(int); /* parser.c */
void parser(void);          /* parser.c */
