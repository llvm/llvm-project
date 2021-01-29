//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//
// POSIX-like portability helper functions.
//
// These generally behave like the proper posix functions, with these
// exceptions:
// On Windows, they take paths in wchar_t* form, instead of char* form.
//
// These are provided within an anonymous namespace within the detail
// namespace - callers need to include this header and call them as
// detail::function(), regardless of platform.
//

#ifndef POSIX_COMPAT_H
#define POSIX_COMPAT_H

#include "filesystem"

#include "filesystem_common.h"

#if defined(_LIBCPP_WIN32API)
# define WIN32_LEAN_AND_MEAN
# define NOMINMAX
# include <windows.h>
# include <io.h>
#else
# include <unistd.h>
# include <sys/stat.h>
# include <sys/statvfs.h>
#endif
#include <time.h>

_LIBCPP_BEGIN_NAMESPACE_FILESYSTEM

namespace detail {
namespace {

#if defined(_LIBCPP_WIN32API)

// Various C runtime header sets provide more or less of these. As we
// provide our own implementation, undef all potential defines from the
// C runtime headers and provide a complete set of macros of our own.

#undef _S_IFMT
#undef _S_IFDIR
#undef _S_IFCHR
#undef _S_IFIFO
#undef _S_IFREG
#undef _S_IFBLK
#undef _S_IFLNK
#undef _S_IFSOCK

#define _S_IFMT   0xF000
#define _S_IFDIR  0x4000
#define _S_IFCHR  0x2000
#define _S_IFIFO  0x1000
#define _S_IFREG  0x8000
#define _S_IFBLK  0x6000
#define _S_IFLNK  0xA000
#define _S_IFSOCK 0xC000

#undef S_ISDIR
#undef S_ISFIFO
#undef S_ISCHR
#undef S_ISREG
#undef S_ISLNK
#undef S_ISBLK
#undef S_ISSOCK

#define S_ISDIR(m)      (((m) & _S_IFMT) == _S_IFDIR)
#define S_ISCHR(m)      (((m) & _S_IFMT) == _S_IFCHR)
#define S_ISFIFO(m)     (((m) & _S_IFMT) == _S_IFIFO)
#define S_ISREG(m)      (((m) & _S_IFMT) == _S_IFREG)
#define S_ISBLK(m)      (((m) & _S_IFMT) == _S_IFBLK)
#define S_ISLNK(m)      (((m) & _S_IFMT) == _S_IFLNK)
#define S_ISSOCK(m)     (((m) & _S_IFMT) == _S_IFSOCK)


// There were 369 years and 89 leap days from the Windows epoch
// (1601) to the Unix epoch (1970).
#define FILE_TIME_OFFSET_SECS (uint64_t(369 * 365 + 89) * (24 * 60 * 60))

TimeSpec filetime_to_timespec(LARGE_INTEGER li) {
  TimeSpec ret;
  ret.tv_sec = li.QuadPart / 10000000 - FILE_TIME_OFFSET_SECS;
  ret.tv_nsec = (li.QuadPart % 10000000) * 100;
  return ret;
}

TimeSpec filetime_to_timespec(FILETIME ft) {
  LARGE_INTEGER li;
  li.LowPart = ft.dwLowDateTime;
  li.HighPart = ft.dwHighDateTime;
  return filetime_to_timespec(li);
}

FILETIME timespec_to_filetime(TimeSpec ts) {
  LARGE_INTEGER li;
  li.QuadPart =
      ts.tv_nsec / 100 + (ts.tv_sec + FILE_TIME_OFFSET_SECS) * 10000000;
  FILETIME ft;
  ft.dwLowDateTime = li.LowPart;
  ft.dwHighDateTime = li.HighPart;
  return ft;
}

int set_errno(int e = GetLastError()) {
  errno = static_cast<int>(__win_err_to_errc(e));
  return -1;
}

class WinHandle {
public:
  WinHandle(const wchar_t *p, DWORD access, DWORD flags) {
    h = CreateFileW(
        p, access, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
        nullptr, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS | flags, nullptr);
  }
  ~WinHandle() {
    if (h != INVALID_HANDLE_VALUE)
      CloseHandle(h);
  }
  operator HANDLE() const { return h; }
  operator bool() const { return h != INVALID_HANDLE_VALUE; }

private:
  HANDLE h;
};

int stat_handle(HANDLE h, StatT *buf) {
  FILE_BASIC_INFO basic;
  if (!GetFileInformationByHandleEx(h, FileBasicInfo, &basic, sizeof(basic)))
    return set_errno();
  memset(buf, 0, sizeof(*buf));
  buf->st_mtim = filetime_to_timespec(basic.LastWriteTime);
  buf->st_atim = filetime_to_timespec(basic.LastAccessTime);
  buf->st_mode = 0555; // Read-only
  if (!(basic.FileAttributes & FILE_ATTRIBUTE_READONLY))
    buf->st_mode |= 0222; // Write
  if (basic.FileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
    buf->st_mode |= _S_IFDIR;
  } else {
    buf->st_mode |= _S_IFREG;
  }
  if (basic.FileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) {
    FILE_ATTRIBUTE_TAG_INFO tag;
    if (!GetFileInformationByHandleEx(h, FileAttributeTagInfo, &tag,
                                      sizeof(tag)))
      return set_errno();
    if (tag.ReparseTag == IO_REPARSE_TAG_SYMLINK)
      buf->st_mode = (buf->st_mode & ~_S_IFMT) | _S_IFLNK;
  }
  FILE_STANDARD_INFO standard;
  if (!GetFileInformationByHandleEx(h, FileStandardInfo, &standard,
                                    sizeof(standard)))
    return set_errno();
  buf->st_nlink = standard.NumberOfLinks;
  buf->st_size = standard.EndOfFile.QuadPart;
  BY_HANDLE_FILE_INFORMATION info;
  if (!GetFileInformationByHandle(h, &info))
    return set_errno();
  buf->st_dev = info.dwVolumeSerialNumber;
  memcpy(&buf->st_ino.id[0], &info.nFileIndexHigh, 4);
  memcpy(&buf->st_ino.id[4], &info.nFileIndexLow, 4);
  return 0;
}

int stat_file(const wchar_t *path, StatT *buf, DWORD flags) {
  WinHandle h(path, FILE_READ_ATTRIBUTES, flags);
  if (!h)
    return set_errno();
  int ret = stat_handle(h, buf);
  return ret;
}

int stat(const wchar_t *path, StatT *buf) { return stat_file(path, buf, 0); }

int lstat(const wchar_t *path, StatT *buf) {
  return stat_file(path, buf, FILE_FLAG_OPEN_REPARSE_POINT);
}

int fstat(int fd, StatT *buf) {
  HANDLE h = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
  return stat_handle(h, buf);
}
#else
using ::fstat;
using ::lstat;
using ::stat;
#endif

} // namespace
} // end namespace detail

_LIBCPP_END_NAMESPACE_FILESYSTEM

#endif // POSIX_COMPAT_H
