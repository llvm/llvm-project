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
#include <stdio.h>

/*! \brief Create a directory tree. */
void __llvm_profile_recursive_mkdir(char *Pathname);

/*! Open file \c Filename for read+write with write
 * lock for exclusive access. The caller will block
 * if the lock is already held by another process. */
FILE *lprofOpenFileEx(const char *Filename);
/* PS4 doesn't have getenv. Define a shim. */
#if __ORBIS__
static inline char *getenv(const char *name) { return NULL; }
#endif /* #if __ORBIS__ */

int lprofGetHostName(char *Name, int Len);

unsigned lprofBoolCmpXchg(void **Ptr, void *OldV, void *NewV);
void *lprofPtrFetchAdd(void **Mem, long ByteIncr);

#endif /* PROFILE_INSTRPROFILINGUTIL_H */
