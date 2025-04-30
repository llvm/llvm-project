/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/**
    \file dinit.h
    \brief Fortran declarations needed to use dinitutil.c module.
*/

typedef struct {/* dinit file record */
  int dtype;    /*  also sptr  */
  INT conval;   /*  also offset */
} DREC;

#define DINIT_STARTARY -95
#define DINIT_ENDARY -94
#define DINIT_TYPEDEF -93
#define DINIT_ENDTYPE -92
#define DINIT_LOC -99
#define DINIT_REPEAT -88
#define DINIT_OFFSET -77
#define DINIT_LABEL -33
#define DINIT_STR -67
#define DINIT_ZEROES -66
#define DINIT_FMT -98
#define DINIT_NML -97
#define DINIT_END -96 /* end of a data statement */

extern void dinit_init(void);
extern void dinit_put(int, INT);
extern DREC *dinit_read(void);
extern void dinit_fseek(long);
extern void dinit_fseek_end(void);
extern long dinit_ftell(void);
extern void dinit_end(void);
extern void dinit_newfile(FILE *);
extern void dinit_save(void);
extern void dinit_restore(void);
extern void df_dinit_end(void);
void dinit(VAR *, ACL *);
void dinit_no_dinitp(VAR *, ACL *);
void do_dinit(void);
void df_dinit(VAR *, ACL *);
INT dinit_eval(int);
LOGICAL dinit_ok(int);
