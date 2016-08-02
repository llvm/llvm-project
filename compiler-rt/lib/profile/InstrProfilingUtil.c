/*===- InstrProfilingUtil.c - Support library for PGO instrumentation -----===*\
|*
|*                     The LLVM Compiler Infrastructure
|*
|* This file is distributed under the University of Illinois Open Source
|* License. See LICENSE.TXT for details.
|*
\*===----------------------------------------------------------------------===*/

#include "InstrProfilingUtil.h"
#include "InstrProfiling.h"

#ifdef _WIN32
#include <direct.h>
#elif I386_FREEBSD
int mkdir(const char*, unsigned short);
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif

COMPILER_RT_VISIBILITY
void __llvm_profile_recursive_mkdir(char *path) {
  int i;

  for (i = 1; path[i] != '\0'; ++i) {
    if (path[i] != '/') continue;
    path[i] = '\0';
#ifdef _WIN32
    _mkdir(path);
#else
    mkdir(path, 0755);  /* Some of these will fail, ignore it. */
#endif
    path[i] = '/';
  }
}
