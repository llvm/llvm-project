/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file
 * \brief SCFTN data initialization file utilities.
 */

#include "dinitutl.h"
#include "error.h"
#include "global.h"
#include "dinit.h"
#include "ilm.h"
#include "dtypeutl.h"

/*
 * mode == ' ' means file is not open
 * mode == 'r' means file is open for read
 * mode == 'w' means file is open for write
 * mode == 'e' means file was open for read but had reached end of file
 */
static char mode = ' ';
static FILE *df = NULL;
static void dump_buff(char);
static DREC t;

/*****************************************************************/

void
dinit_init(void)
{
    mode = ' '; /* neither read nor write */
  if (df) {
    mode = 'e';
    fseek(df, 0L, 0);
  }
}

/*****************************************************************/

void
dinit_put(DTYPE dtype, ISZ_T conval)
{
  int n;

  if (mode == 'e') {
    mode = 'w';
  } else if (mode == ' ') {
    if ((df = tmpfile()) == NULL)
      errfatal(F_0005_Unable_to_open_temporary_file);
    mode = 'w';
  } else if (mode != 'w') {
    error(F_0010_File_write_error_occurred_OP1, ERR_Fatal, 0,
          "(data init file)", CNULL);
  }

  t.dtype = dtype;
  t.conval = conval;
  if (DBGBIT(6, 1))
    dump_buff(mode);

  n = fwrite((char *)&t, sizeof(t), 1, df);
  if (n != 1)
    error(F_0010_File_write_error_occurred_OP1, ERR_Fatal, 0,
          "(data init file)", CNULL);
}

/*
 * must appear after DINIT_STRING record, so the file must be open in mode 'w'
 */
void
dinit_put_string(ISZ_T len, char *str)
{
  int n;
  if (df == NULL || mode != 'w')
    error(F_0010_File_write_error_occurred_OP1, ERR_Fatal, 0, "(data init file)", CNULL);
  if (DBGBIT(6, 1))
    fprintf(gbl.dbgfil, "    string(%d)\n", (int)len);

  n = fwrite(str, 1, len, df);
  if (n != len)
    error(F_0010_File_write_error_occurred_OP1, ERR_Fatal, 0, "(data init file)", CNULL);
} /* dinit_put_string */

/*******************************************************/

DREC *
dinit_read(void)
{
  register int n;

  if (mode == ' ' || mode == 'e' || df == NULL)
    return NULL;
  if (mode == 'w') {
    t.dtype = DINIT_ENDFILE;
    t.conval = 0;
    n = fwrite((char *)&t, sizeof(t), 1, df);
    n = fseek(df, 0L, 0);
    assert(n == 0, "dinit_read:bad rewind", n, ERR_Fatal);
    mode = 'r';
  }

  n = fread((char *)&t, sizeof(t), 1, df);
  if (n == 0) { /* end of file */
    mode = 'e';
    return NULL;
  }
  if (t.dtype == DINIT_ENDFILE) {
    mode = 'e';
    return NULL;
  }

  if (DBGBIT(6, 1))
    dump_buff(mode);
  return &t;
}

void
dinit_read_string(ISZ_T len, char *str)
{
  int n;
  n = fread(str, 1, len, df);
  if (n != len) { /* end of file */
    mode = 'e';
  }
} /* dinit_read_string */

/********************************************************/

static void
dump_buff(char mode)
{
  char buf[32];

  fprintf(gbl.dbgfil, "   %c ", mode);
  if (t.dtype > 0 && t.dtype < stb.dt.stg_avail) {
    getdtype(t.dtype, buf);
    fprintf(gbl.dbgfil, "dtype: %s  conval: %ld\n", buf, t.conval);
  } else if (t.dtype == 0) {
  }
#ifdef DINIT_MODE
  else if (t.dtype == DINIT_MODE)
    fprintf(gbl.dbgfil, " mode %d\n", (int)t.conval);
#endif
#ifdef DINIT_FUNCCOUNT
  else if (t.dtype == DINIT_FUNCCOUNT)
    fprintf(gbl.dbgfil, " funccount %d\n", (int)t.conval);
#endif
  else {
    if (t.dtype == DINIT_LOC)
      fprintf(gbl.dbgfil, "DINIT_LOC  %s  ", getprint((int)t.conval));
    else if (t.dtype == DINIT_SLOC)
      fprintf(gbl.dbgfil, "DINIT_SLOC  %s  ", getprint((int)t.conval));
    else if (t.dtype == DINIT_REPEAT)
      fprintf(gbl.dbgfil, "DINIT_REPEAT   ");
    else if (t.dtype == DINIT_OFFSET)
      fprintf(gbl.dbgfil, "DINIT_OFFSET   ");
    else if (t.dtype == DINIT_LABEL)
      fprintf(gbl.dbgfil, "DINIT_LABEL %s ", getprint((int)t.conval));
    else if (t.dtype == DINIT_ZEROES)
      fprintf(gbl.dbgfil, "DINIT_ZEROES   ");
    else if (t.dtype == DINIT_SECT)
      fprintf(gbl.dbgfil, "DINIT_SECT     ");
    else if (t.dtype == DINIT_DATASECT)
      fprintf(gbl.dbgfil, "DINIT_DATASECT ");
#ifdef DINIT_STRING
    else if (t.dtype == DINIT_STRING)
      fprintf(gbl.dbgfil, " string len=%d\n", (int)t.conval);
#endif
    else
      fprintf(gbl.dbgfil, "dtype: %4d    ", t.dtype);
    fprintf(gbl.dbgfil, "   conval: %10" ISZ_PF "d  (0x%08" ISZ_PF "X)\n",
            t.conval, t.conval);
  }
}

/*****************************************************************/

long
dinit_ftell(void)
{
  return (ftell(df));
}

/*****************************************************************/

void
dinit_fskip(long off)
{
  int n;

  mode = 'r';
  n = fseek(df, off, SEEK_CUR);
  assert(n == 0, "dinit_fskip:bad seek", n, ERR_Fatal);
} /* dinit_fskip */

void
dinit_fseek(long off)
{
  int n;

  mode = 'r';
  n = fseek(df, off, 0);
  assert(n == 0, "dinit_fseek:bad seek", n, ERR_Fatal);
}

/*****************************************************************/

void
dinit_end(void)
{
  if (df) {
    fclose(df);
    df = NULL;
  }
  /* if this is block data, need to free the ilmb memory that
     would ordinarily be freed in expand.  purify MLK (memory
     leak) error was being reported. */
  if (gbl.rutype == RU_BDATA && ilmb.ilm_base) {
    FREE(ilmb.ilm_base);
    ilmb.ilm_base = NULL;
  }
  mode = ' '; /* no file */
}

static char savemode;
static long savepos;

/*
 * save and restore position and mode of the dinit file
 */
void
dinit_save(void)
{
  savemode = mode;
  savepos = 0;
  if (df) {
    savepos = ftell(df);
  }
} /* dinit_save */

void
dinit_restore(void)
{
  mode = savemode;
  if (df) {
    fseek(df, savepos, 0);
  }
} /* dinit_restore */

bool
df_is_open(void)
{
  return (df != NULL);
}
