/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* lz.c - compression */

#include <stdio.h>
#include "gbldefs.h"
#if !defined(HOST_WIN)
#include <unistd.h>
#include <sys/types.h>
#endif

#ifndef HOST_WIN
#define USE_GETLINE 1
#endif

#include "lz.h"

void
lzreinit(lzhandle *lzh)
{
} /* lzreinit */

void
lzreset(lzhandle *lzh)
{
    lzreinit(lzh);
} /* lzreset */

static void
lzfini(lzhandle *lzh)
{
  if (lzh->buff)
    free(lzh->buff);
  free(lzh);
} /* lzfini */

/*
 * encode a single line
 */
void
lz(lzhandle *lzh, char *line, int linelen)
{
  fputs(line, lzh->file);
} /* lz */

/*
 * save the file, allocate buffer
 */
lzhandle *
lzinitfile(FILE *out, int compress)
{
  lzhandle *lzh = (lzhandle *)malloc(sizeof(lzhandle));
  if (lzh == NULL) {
    fprintf(stderr, "Ran out of memory\n");
    exit(1);
  }
  memset(lzh, 0, sizeof(lzhandle));
  lzh->file = out;
  lzh->inout = 1;
  return lzh;
} /* lzinitfile */

/*
 * flush any remaining output, free buffer, call lzfini
 */
void
lzfinifile(lzhandle *lzh)
{
  fflush(lzh->file);
  lzfini(lzh);
} /* lzfinifile */

/*
 * called like printf
 */
extern void
lzprintf(lzhandle *lzh, const char *fmt, ...)
{
  va_list argptr;
  va_start(argptr, fmt);
  vfprintf(lzh->file, fmt, argptr);
  va_end(argptr);
} /* lzprintf */

/*
 * initialize line for decoder
 *  call before calling ulz at all
 */
lzhandle *
ulzinit(FILE *in, int compress)
{
  lzhandle *lzh = (lzhandle *)malloc(sizeof(lzhandle));
  if (lzh == NULL) {
    fprintf(stderr, "Ran out of memory\n");
    exit(1);
  }
  memset(lzh, 0, sizeof(lzhandle));
  lzh->file = in;
  lzh->inout = 0;
  lzh->buffsize = 4096;
  lzh->bufflen = 0;
  lzh->buff = (char *)malloc(sizeof(char) * lzh->buffsize);
  if (lzh->buff == NULL) {
    fprintf(stderr, "Ran out of memory\n");
    exit(1);
  }
  return lzh;
} /* ulzinit */

/*
 * free up line
 *  call when done calling ulz
 */
void
ulzfini(lzhandle *lzh)
{
  lzfini(lzh);
} /* ulzfini */

/*
 * decoder, line at a time
 */
char *
ulz(lzhandle *lzh)
{
#ifndef USE_GETLINE
  int ch;
#endif
  lzh->bufflen = 0;
#ifdef USE_GETLINE
    int res = getline(&lzh->buff, &lzh->buffsize, lzh->file);
    if (res > 0) {
      if (lzh->buff[res-1] == '\n')
        --res;
      lzh->bufflen = res;
    }
#else
    /* read in chars one by one to endline or EOF */
    while ((ch = getc(lzh->file)) != '\n' && ch != EOF) {
      if (lzh->bufflen + 3 >= lzh->buffsize) {
        lzh->buffsize = lzh->buffsize * 2;
        lzh->buff = realloc(lzh->buff, lzh->buffsize);
        if (lzh->buff == NULL) {
          fprintf(stderr, "Ran out of memory\n");
          exit(1);
        }
      }
      lzh->buff[lzh->bufflen++] = ch;
    }
#endif
  lzh->buff[lzh->bufflen] = '\0';
#ifdef ZDEBUGLZ
  printf("ULZ:%d %s\n", lzh->compress, lzh->buff);
#endif
  return lzh->buff;
} /* ulz */

/*
* decoder, char at a time
*/
char
ulzgetc(lzhandle *lzh)
{
  return getc(lzh->file);
} /* ulzgetc */

/*
 * save position of lzh->file, and current dictionary size
 */
void
lzsave(lzhandle *lzh)
{
  lzh->savefile = ftell(lzh->file);
} /* lzsave */

/*
 * restore file to saved position;
 * restore dictionary to saved size
 */
void
lzrestore(lzhandle *lzh)
{
  fseek(lzh->file, lzh->savefile, SEEK_SET);
#if !defined(HOST_WIN)
  if (lzh->inout) {
    ftruncate(fileno(lzh->file), lzh->savefile);
  }
#endif
} /* lzrestore */
