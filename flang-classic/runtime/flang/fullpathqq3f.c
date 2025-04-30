/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*	fullpathqq3f.c - Implements DFLIB fullpathqq subprogram.  */

#include <string.h>
#include <stdlib.h>
/* must include ent3f.h AFTER io3f.h */
#include "io3f.h"
#include "ent3f.h"
#include "utils3f.h"

#if defined(_WIN64)

int ENT3F(FULLPATHQQ, fullpathqq)(DCHAR(fname),
                                  DCHAR(fpath) DCLEN(fname) DCLEN(fpath))
{
  char *path, *name;
  int rslt = 0;

  path = __fstr2cstr(CADR(fpath), CLEN(fpath));
  name = __fstr2cstr(CADR(fname), CLEN(fname));

  if (!path || !name) {
    __io_errno();
    goto rtn;
  }

  /*
    char *_fullpath(
    char *absPath,
    const char *relPath,
    size_t maxLength
    );
  */

  if (_fullpath(path, name, CLEN(fpath)) != NULL) {
    rslt = strlen(path);
    __fcp_cstr(CADR(fpath), CLEN(fpath), path);
  }

rtn:
  __cstr_free(path);
  __cstr_free(name);

  return rslt;
}
#else
int ENT3F(FULLPATHQQ, fullpathqq)(DCHAR(fname),
                                  DCHAR(fpath) DCLEN(fname) DCLEN(fpath))
{
  fprintf(__io_stderr(), "fullpathqq() not implemented on this target\n");
  return 0;
}

#endif
