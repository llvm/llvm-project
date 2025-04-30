/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/** \file
 *  Implements DFLIB findfileqq subprogram.  */

#include <string.h>
#include <stdlib.h>
/* must include ent3f.h AFTER io3f.h */
#include "io3f.h"
#include "ent3f.h"
#include "utils3f.h"

#if defined(_WIN64)

int ENT3F(FINDFILEQQ, findfileqq)(DCHAR(fname), DCHAR(fvarname),
                                  DCHAR(fpath) DCLEN(fname) DCLEN(fvarname)
                                      DCLEN(fpath))
{
  char *path, *name, *varname;
  int rslt = 0;

  path = __fstr2cstr(CADR(fpath), CLEN(fpath));
  name = __fstr2cstr(CADR(fname), CLEN(fname));
  varname = __fstr2cstr(CADR(fvarname), CLEN(fvarname));

  if (!path || !name || !varname) {
    __io_errno();
    goto rtn;
  }

  /*
errno_t _searchenv_s(
   const char *filename,
   const char *varname,
   char *pathname,
   size_t numberOfElements
);
  */

  if (_searchenv_s(name, varname, path, CLEN(fpath)) == 0) {
    rslt = strlen(path);
    __fcp_cstr(CADR(fpath), CLEN(fpath), path);
  }

rtn:
  __cstr_free(path);
  __cstr_free(name);
  __cstr_free(varname);

  return rslt;
}
#else
int ENT3F(FINDFILEQQ, findfileqq)(DCHAR(fname), DCHAR(fvarname),
                                  DCHAR(fpath) DCLEN(fname) DCLEN(fvarname)
                                      DCLEN(fpath))
{
  fprintf(__io_stderr(), "findfileqq() not implemented on this target\n");
  return 0;
}

#endif
