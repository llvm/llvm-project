/*===- InstrProfilingUtil.h - Support library for PGO instrumentation -----===*\
|*
|*                     The LLVM Compiler Infrastructure
|*
|* This file is distributed under the University of Illinois Open Source
|* License. See LICENSE.TXT for details.
|*
\*===----------------------------------------------------------------------===*/

#ifndef PROFILE_INSTRPROFILINGUTIL_H
#define PROFILE_INSTRPROFILINGUTIL_H

#include <stddef.h>

/*! \brief Create a directory tree. */
void __llvm_profile_recursive_mkdir(char *Pathname);

/* PS4 doesn't have getenv. Define a shim. */
#if __PS4__
static inline char *getenv(const char *name) { return NULL; }
#endif /* #if __PS4__ */

int lprofGetHostName(char *Name, int Len);

unsigned BoolCmpXchg(void **Ptr, void *OldV, void *NewV);

#endif  /* PROFILE_INSTRPROFILINGUTIL_H */
