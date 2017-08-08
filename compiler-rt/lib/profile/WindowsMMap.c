/*
 * This code is derived from uClibc (original license follows).
 * https://git.uclibc.org/uClibc/tree/utils/mmap-windows.c
 */
 /* mmap() replacement for Windows
 *
 * Author: Mike Frysinger <vapier@gentoo.org>
 * Placed into the public domain
 */

/* References:
 * CreateFileMapping: http://msdn.microsoft.com/en-us/library/aa366537(VS.85).aspx
 * CloseHandle:       http://msdn.microsoft.com/en-us/library/ms724211(VS.85).aspx
 * MapViewOfFile:     http://msdn.microsoft.com/en-us/library/aa366761(VS.85).aspx
 * UnmapViewOfFile:   http://msdn.microsoft.com/en-us/library/aa366882(VS.85).aspx
 */

#if defined(_WIN32)

#include "WindowsMMap.h"
#include "InstrProfiling.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#ifdef __USE_FILE_OFFSET64
# define DWORD_HI(x) (x >> 32)
# define DWORD_LO(x) ((x) & 0xffffffff)
#else
# define DWORD_HI(x) (0)
# define DWORD_LO(x) (x)
#endif

COMPILER_RT_VISIBILITY
void *mmap(void *start, size_t length, int prot, int flags, int fd, off_t offset)
{
  if (prot & ~(PROT_READ | PROT_WRITE | PROT_EXEC))
    return MAP_FAILED;
  if (fd == -1) {
    if (!(flags & MAP_ANON) || offset)
      return MAP_FAILED;
  } else if (flags & MAP_ANON)
    return MAP_FAILED;

  DWORD flProtect;
  if (prot & PROT_WRITE) {
    if (prot & PROT_EXEC)
      flProtect = PAGE_EXECUTE_READWRITE;
    else
      flProtect = PAGE_READWRITE;
  } else if (prot & PROT_EXEC) {
    if (prot & PROT_READ)
      flProtect = PAGE_EXECUTE_READ;
    else if (prot & PROT_EXEC)
      flProtect = PAGE_EXECUTE;
  } else
    flProtect = PAGE_READONLY;

  off_t end = length + offset;
  HANDLE mmap_fd, h;
  if (fd == -1)
    mmap_fd = INVALID_HANDLE_VALUE;
  else
    mmap_fd = (HANDLE)_get_osfhandle(fd);
  h = CreateFileMapping(mmap_fd, NULL, flProtect, DWORD_HI(end), DWORD_LO(end), NULL);
  if (h == NULL)
    return MAP_FAILED;

  DWORD dwDesiredAccess;
  if (prot & PROT_WRITE)
    dwDesiredAccess = FILE_MAP_WRITE;
  else
    dwDesiredAccess = FILE_MAP_READ;
  if (prot & PROT_EXEC)
    dwDesiredAccess |= FILE_MAP_EXECUTE;
  if (flags & MAP_PRIVATE)
    dwDesiredAccess |= FILE_MAP_COPY;
  void *ret = MapViewOfFile(h, dwDesiredAccess, DWORD_HI(offset), DWORD_LO(offset), length);
  if (ret == NULL) {
    CloseHandle(h);
    ret = MAP_FAILED;
  }
  return ret;
}

COMPILER_RT_VISIBILITY
void munmap(void *addr, size_t length)
{
  UnmapViewOfFile(addr);
  /* ruh-ro, we leaked handle from CreateFileMapping() ... */
}

COMPILER_RT_VISIBILITY
int msync(void *addr, size_t length, int flags)
{
  if (flags & MS_INVALIDATE)
    return -1; /* Not supported. */

  /* Exactly one of MS_ASYNC or MS_SYNC must be specified. */
  switch (flags & (MS_ASYNC | MS_SYNC)) {
    case MS_SYNC:
    case MS_ASYNC:
      break;
    default:
      return -1;
  }

  if (!FlushViewOfFile(addr, length))
    return -1;

  if (flags & MS_SYNC) {
    /* FIXME: No longer have access to handle from CreateFileMapping(). */
    /*
     * if (!FlushFileBuffers(h))
     *   return -1;
     */
  }

  return 0;
}

COMPILER_RT_VISIBILITY
int flock(int fd, int operation)
{
  return -1; /* Not supported. */
}

#undef DWORD_HI
#undef DWORD_LO

#endif /* _WIN32 */
