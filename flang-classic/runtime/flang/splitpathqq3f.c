/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*	splitpathqq3f.c - Implements DFLIB splitpathqq subprogram.  */

#include <string.h>
#include <stdlib.h>
/* must include ent3f.h AFTER io3f.h */
#include "io3f.h"
#include "ent3f.h"
#include "utils3f.h"

#if defined(_WIN64)

int ENT3F(SPLITPATHQQ, splitpathqq)(DCHAR(fpath), DCHAR(fdrive), DCHAR(fdir),
                                    DCHAR(fname),
                                    DCHAR(fext) DCLEN(fpath) DCLEN(fdrive)
                                        DCLEN(fdir) DCLEN(fname) DCLEN(fext))
{
  char *path, *ext, *name, *dir, *drive;
  errno_t err;
  int rslt = 0;

  path = __fstr2cstr(CADR(fpath), CLEN(fpath));
  drive = __fstr2cstr(CADR(fdrive), CLEN(fdrive));
  dir = __fstr2cstr(CADR(fdir), CLEN(fdir));
  name = __fstr2cstr(CADR(fname), CLEN(fname));
  ext = __fstr2cstr(CADR(fext), CLEN(fext));

  if (!path || !drive || !dir || !name || !ext) {
    __io_errno();
    goto rtn;
  }

  /* errno_t _splitpath_s(
     const char * path,
     char * drive,
     size_t driveSizeInCharacters,
     char * dir,
     size_t dirSizeInCharacters,
     char * fname,
     size_t nameSizeInCharacters,
     char * ext,
     size_t extSizeInBytes
     );
  */

  err = _splitpath_s(path, drive, CLEN(fdrive), dir, CLEN(fdir), name,
                     CLEN(fname), ext, CLEN(fext));
  if (err) {
    __io_errno();
    goto rtn;
  }

  rslt = strlen(dir);
  __fcp_cstr(CADR(fext), CLEN(fext), ext);
  __fcp_cstr(CADR(fdrive), CLEN(fdrive), drive);
  __fcp_cstr(CADR(fdir), CLEN(fdir), dir);
  __fcp_cstr(CADR(fname), CLEN(fname), name);

rtn:
  __cstr_free(path);
  __cstr_free(drive);
  __cstr_free(dir);
  __cstr_free(name);
  __cstr_free(ext);

  return rslt;
}
#else
int ENT3F(SPLITPATHQQ, splitpathqq)(DCHAR(fpath), DCHAR(fdrive), DCHAR(fdir),
                                    DCHAR(fname),
                                    DCHAR(fext) DCLEN(fpath) DCLEN(fdrive)
                                        DCLEN(fdir) DCLEN(fname) DCLEN(fext))
{
  fprintf(__io_stderr(), "splitpathqq() not implemented on this target\n");
  return 0;
}

#endif
