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
#include <windows.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#if defined(__linux__)
#include <unistd.h>
#endif
#include <fcntl.h>
#include <errno.h>
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
    mkdir(path, 0755); /* Some of these will fail, ignore it. */
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

FILE *lprofOpenFileEx(const char *ProfileName) {
  FILE *f;
  int fd;
#ifdef COMPILER_RT_HAS_FCNTL_LCK
  struct flock s_flock;

  s_flock.l_whence = SEEK_SET;
  s_flock.l_start = 0;
  s_flock.l_len = 0; /* Until EOF.  */
  s_flock.l_pid = getpid();

  s_flock.l_type = F_WRLCK;
  fd = open(ProfileName, O_RDWR | O_CREAT, 0666);
  if (fd < 0)
    return 0;

  while (fcntl(fd, F_SETLKW, &s_flock) && errno == EINTR)
    continue;

  f = fdopen(fd, "r+b");
#elif defined(_WIN32)
  HANDLE h = CreateFile(ProfileName, GENERIC_READ | GENERIC_WRITE, 0, 0,
                        OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, 0);
  if (h == INVALID_HANDLE_VALUE)
    return 0;

  fd = _open_osfhandle((intptr_t)h, 0);
  if (fd == -1) {
    CloseHandle(h);
    return 0;
  }

  f = _fdopen(fd, "r+b");
  if (f == 0) {
    CloseHandle(h);
    return 0;
  }
#else
  /* Worst case no locking applied.  */
  PROF_WARN("Concurrent file access is not supported : %s\n", "lack file locking");
  fd = open(ProfileName, O_RDWR | O_CREAT, 0666);
  if (fd < 0)
    return 0;
  f = fdopen(fd, "r+b");
#endif

  return f;
}
