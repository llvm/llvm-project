/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#if !defined(__PGSTDINIT_H__)
#define __PGSTDINIT_H__

#include <stdio.h>  /* TODO: try moving to  pgstdio.h */
#include <string.h>
#ifndef _WIN64
#include <unistd.h>
#endif
#include <stdlib.h>

/* defines to use real host stdio routines */

#define __io_fclose(fp) fclose(fp)
#define __io_fflush(fp) fflush(fp)
#define __io_fgetc(fp) fgetc(fp)
#define __io_fgets(ptr, n, fp) fgets(ptr, n, fp)

#define __io_fopen(file, typ) fopen(file, typ)

#define __io_fprintf fprintf
#define __io_fputs(ptr, fp) fputs(ptr, fp)
#define __io_fread(ptr, size, nitems, fp) fread(ptr, size, nitems, fp)
#define __io_freopen(file, typ, fp) freopen(file, typ, fp)
#define __io_fscanf fscanf

#define __io_fputc(c, fp) fputc(c, fp)

typedef long seekoff_t;
#define __io_fseek(fp, off, wh) fseek(fp, off, wh)
#define __io_ftell(fp) ftell(fp)
typedef long long seekoff64_t;
#define __io_fseek64(fp, off, wh) fseek(fp, (long)off, wh)
#define __io_ftell64(fp) (long long) ftell(fp)
typedef long seekoffx_t;
#define __io_fseekx(fp, off, wh) fseek(fp, off, wh)
#define __io_ftellx(fp) ftell(fp)

#define __io_gets(ptr) gets(ptr)
#define __io_perror(ptr) perror(ptr)
#define __io_printf printf
#define __io_puts(ptr) puts(ptr)
#define __io_remove(ptr) remove(ptr)
#define __io_rename(ptr1, ptr2) rename(ptr1, ptr2)
#define __io_rewind(fp) rewind(fp)
#define __io_scanf scanf
#define __io_setbuf(fp, ptr) setbuf(fp, ptr)
#define __io_setvbuf(fp, ptr, typ, size) setvbuf(fp, ptr, typ, size)
#define __io_sprintf sprintf
#define __io_sscanf sscanf
#define __io_tmpfile() tmpfile()
#define __io_tmpnam(ptr) tmpnam(ptr)
#define __io_ungetc(c, fp) ungetc(c, fp)

/* some conversions */

#define __io_strtod(p, ep) __fortio_strtod(p, ep)
#define __io_ecvt(v, w, n, d, s, r, q) __fortio_ecvt(v, w, n, d, s, r, q)
#define __io_fcvt(v, w, n, sf, d, s, r, q)                                     \
  __fortio_fcvt(v, w, n, sf, d, s, r, q)

#define __io_strtold(p, ep) __fortio_strtold(p, ep)

/* and defines for other routines */
#define __fort_getfd(fp) __io_getfd(fp)
#define __fort_isatty(fd) __io_isatty(fd)

#define __fort_truncate(name, len) truncate(name, len)
#define __fort_ftruncate(fd, len) ftruncate(fd, len)
#define __fort_access(path, mode) access(path, mode)
#define __fort_unlink(path) unlink(path)
#define __fort_getenv(name) getenv(name)
#define __io_abort() exit(1)
#define __fortio_ispipe(fp) __io_ispipe(fp)

/* finally the prototypes */


int __io_errno(void);
void __io_set_errno(int);
FILE *__io_stdin(void);
FILE *__io_stdout(void);
FILE *__io_stderr(void);
int __io_getc(FILE *);
int __io_putc(int x, FILE *);
int __io_getchar(void);
int __io_putchar(int x);
void __io_clearerr(FILE *);
int __io_getfd(FILE *);
int __io_isatty(int fd);
int __io_binary_mode(FILE *);
int __io_setmode_binary(FILE *);
int __io_ispipe(FILE *);
int __io_feof(FILE *);
int __io_ferror(FILE *);
size_t __io_fwrite(const void *, size_t, size_t, FILE *);
int __io_timezone(void *);
int fclose(FILE *);
int fflush(FILE *);
int __io_fputc(int, FILE *);
FILE *tmpfile(void);
char *tmpnam(char *);
char *__io_tempnam(const char *, const char *);

extern void *__aligned_malloc(size_t, size_t); /* pgmemalign.c */
extern void __aligned_free(void *);
extern int __fenv_fegetzerodenorm(void);

void __abort(int sv, const char *msg);
void __abort_trace(int skip);
void __abort_sig_init(void);

/* FIXME: delete after these have been cleanout of pgftn/f90_global.h and 
 *        others ???
 */
#define WIN_CDECL
#define WIN_MSVCRT_IMP extern

#endif
