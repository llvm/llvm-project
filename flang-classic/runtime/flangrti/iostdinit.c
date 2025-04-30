/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include <stdio.h>
#if !defined(_WIN64) && !defined(ST100)
#include <sys/stat.h>
#include <unistd.h>
#endif
#include <time.h>
#include <errno.h>

/* get environ */

#if defined(_WIN64)
char * * * __cdecl __p__environ(void);
/*
 * enclose _fileno within parens to ensure calling the function rather than
 * the _fileno function macro (if/when it exists).
 */
#define fileno(x) (_fileno)(x)
#endif

#if   defined(_WIN64)
#include <stdlib.h>
extern char **environ;
#elif defined(TARGET_OSX)
#include <crt_externs.h>
#else
extern char **environ;
#endif

char **
__io_environ()
{
#if defined(TARGET_WIN)
  return(*__p__environ());
#elif !defined(TARGET_OSX)
  return (environ);
#else
  return (*_NSGetEnviron());
#endif
}

/* get errno value */

int
__io_errno()
{
  return (errno);
}

/* set errno value */

void
__io_set_errno(int value)
{
  errno = value;
}

/* return standard fp's */

void *
__io_stdin(void)
{
  return ((void *)stdin);
}

void *
__io_stdout(void)
{
  return ((void *)stdout);
}

void *
__io_stderr(void)
{
  return ((void *)stderr);
}

/* convert macros to routines */

#if defined(TARGET_WIN) || defined(_WIN64)
#include <stdio.h>
int
__io_fgetc(FILE *p)
{
  return _fgetc_nolock(p);
}

int
__io_ungetc(int x, FILE *p)
{
  return (_ungetc_nolock(x, (FILE *)p));
}

int
__io_fputc(int x, FILE *p)
{
  return (_putc_nolock(x, (FILE *)p));
}

#else

int
__io_getc(void *p)
{
  return (getc((FILE *)p));
}

int
__io_putc(int x, void *p)
{
  return (putc(x, (FILE *)p));
}
#endif

int
__io_getchar(void)
{
  return (getchar());
}

int
__io_putchar(int x)
{
  return (putchar(x));
}

void
__io_clearerr(void *p)
{
  clearerr((FILE *)p);
}

int
__io_feof(void *p)
{
  return (feof((FILE *)p));
}

int
__io_ferror(void *p)
{
  return (ferror((FILE *)p));
}

/* get fd from fp */

int
__io_getfd(void *fp)
{
  return (fileno((FILE *)fp));
}

/* is a tty? */

int
__io_isatty(int fd)
{
  return (isatty(fd));
}

/* some NT stuff */

int
__io_binary_mode(void *fp)
{
#if defined(_WIN64)
#include <fcntl.h>

  int mode;

  mode = setmode(fileno((FILE *)fp), O_BINARY);
  if (mode == -1) {
    /* The mode argument is clearly legal, so this should not
     * happen.  But, in a console app, setmode will fail on
     * the fd representing stdout.
     */
    return 0;
  }
  (void)setmode(fileno((FILE *)fp), mode);
  return (mode & O_BINARY);
#else
  return 1;
#endif
}

int
__io_setmode_binary(void *fp)
{
#if defined(_WIN64)
#include <fcntl.h>

  int mode;

  return setmode(fileno((FILE *)fp), O_BINARY);
#else
  return 0; /* NOTE: -1 is error */
#endif
}

int
__io_ispipe(void *f)
{
#if !defined(_WIN64) && !defined(ST100)
  struct stat st;

  fstat(fileno((FILE *)f), &st);
  if (S_ISCHR(st.st_mode) || S_ISFIFO(st.st_mode))
    return 1;
#endif
  return 0;
}

/*
 * On AT&T SysV R4, Release 1.0, the fwrite function does not correctly
 * handle line-buffered files.  If the file is line-buffered, then we
 * just to putc's, else do fwrite directly.
 *
 * This is o.k. to be ANSI since pgcc is always used to compile it.
 */

size_t
__io_fwrite(char *ptr, size_t size, size_t nitems, FILE *stream)
{
#ifdef BROKEN_FWRITE
  int i, c;

  if (stream->_base)
    if (!(stream->_flag & _IOLBF))
      return fwrite(ptr, size, nitems, stream);

  /* first time, or line buffered, force putc */
  /* line buffered */
  for (i = size * nitems; i > 0; --i)
    putc(*ptr++, stream);
  if (ferror(stream))
    return 0;
  return nitems;
#else
  return (fwrite(ptr, size, nitems, stream));
#endif
}

#if defined(_WIN64)

#if   defined(PGI_CRTDLL)
extern long *_imp___timezone_dll; /* for crtdll.dll */
#define timezone (*_imp___timezone_dll)
#elif defined(PGI_CYGNUS)
#define timezone _timezone /* cygnus, timezone is usually a function */
#endif

#elif !defined(DEC) && !defined(IBM) && !defined(ST100_V1_2) && !defined(OSX86) /* !defined(_WIN64) */
extern time_t timezone; /* for the rest */
#endif

int
__io_timezone(void *tm)
{
#if defined(SUN4) || defined(PPC) || defined(OSX86)
  return ((struct tm *)tm)->tm_gmtoff;
#elif defined(_WIN64)
  return (0);
#else
  return -(timezone - (((struct tm *)tm)->tm_isdst ? 3600 : 0));
#endif
}

#if defined(_WIN64)
/* wrappers for stderr, stdin, stdout : include
  pgc/port/pgi_iobuf.h after stdio.h 
 */
void * 
_pgi_get_iob(int xx) {
	 return __acrt_iob_func (xx);
}
#endif
