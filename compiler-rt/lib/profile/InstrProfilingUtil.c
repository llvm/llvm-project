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
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif

#ifdef COMPILER_RT_HAS_UNAME
#include <sys/utsname.h>
#endif

#include <string.h>

COMPILER_RT_VISIBILITY
void __llvm_profile_recursive_mkdir(char *path) {
  int i;

  for (i = 1; path[i] != '\0'; ++i) {
    char save = path[i];
    if (!(path[i] == '/' || path[i] == '\\'))
      continue;
    path[i] = '\0';
#ifdef _WIN32
    _mkdir(path);
#else
    mkdir(path, 0755);  /* Some of these will fail, ignore it. */
#endif
    path[i] = save;
  }
}

#if COMPILER_RT_HAS_ATOMICS != 1
COMPILER_RT_VISIBILITY
uint32_t lprofBoolCmpXchg(void **Ptr, void *OldV, void *NewV) {
  void *R = *Ptr;
  if (R == OldV) {
    *Ptr = NewV;
    return 1;
  }
  return 0;
}
COMPILER_RT_VISIBILITY
void *lprofPtrFetchAdd(void **Mem, long ByteIncr) {
  void *Old = *Mem;
  *((char **)Mem) += ByteIncr;
  return Old;
}

#endif

#ifdef COMPILER_RT_HAS_UNAME
int lprofGetHostName(char *Name, int Len) {
  struct utsname N;
  int R;
  if (!(R = uname(&N)))
    strncpy(Name, N.nodename, Len);
  return R;
}
#endif


