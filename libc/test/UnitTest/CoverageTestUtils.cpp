//===-- Implementation of coverage test utilities -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/fcntl_macros.h"
#include "hdr/types/FILE.h"
#include "hdr/types/mode_t.h"
#include "hdr/types/off_t.h"
#include "hdr/types/pid_t.h"
#include "hdr/types/struct_utsname.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/fcntl/fcntl.h"
#include "src/fcntl/open.h"
#include "src/stdio/fclose.h"
#include "src/stdio/fdopen.h"
#include "src/stdio/feof.h"
#include "src/stdio/fflush.h"
#include "src/stdio/fileno.h"
#include "src/stdio/fopen.h"
#include "src/stdio/fread.h"
#include "src/stdio/fseek.h"
#include "src/stdio/ftell.h"
#include "src/stdio/fwrite.h"
#include "src/stdio/stderr.h"
#include "src/stdio/vfprintf.h"
#include "src/stdio/vsnprintf.h"
#include "src/stdlib/getenv.h"
#include "src/stdlib/setenv.h"
#include "src/stdlib/strtol.h"
#include "src/string/strchr.h"
#include "src/string/strcmp.h"
#include "src/string/strdup.h"
#include "src/string/strerror.h"
#include "src/string/strlen.h"
#include "src/string/strncpy.h"
#include "src/string/strrchr.h"
#include "src/sys/mman/madvise.h"
#include "src/sys/mman/mmap.h"
#include "src/sys/mman/munmap.h"
#include "src/sys/prctl/prctl.h"
#include "src/sys/stat/mkdir.h"
#include "src/sys/utsname/uname.h"
#include "src/unistd/ftruncate.h"
#include "src/unistd/getpagesize.h"
#include "src/unistd/getpid.h"

#include <stdarg.h>
#include <stddef.h>

namespace LIBC_NAMESPACE_DECL {
void *memset(void *ptr, int value, size_t count);
extern FILE *stderr;
} // namespace LIBC_NAMESPACE_DECL

extern "C" {

void *malloc(size_t);

void *calloc(size_t num, size_t size) {
  size_t total;
  if (__builtin_mul_overflow(num, size, &total))
    return nullptr;
  void *mem = malloc(total);
  if (mem != nullptr)
    LIBC_NAMESPACE::memset(mem, 0, total);
  return mem;
}

int *__llvm_libc_errno() noexcept;
int *__errno_location() { return __llvm_libc_errno(); }

FILE *stderr = nullptr;

[[gnu::constructor(101)]] static void init_coverage_stderr() {
  stderr = LIBC_NAMESPACE::stderr;
}

int fclose(FILE *stream) { return LIBC_NAMESPACE::fclose(stream); }

FILE *fdopen(int fd, const char *mode) {
  return LIBC_NAMESPACE::fdopen(fd, mode);
}

int feof(FILE *stream) { return LIBC_NAMESPACE::feof(stream); }

int fflush(FILE *stream) { return LIBC_NAMESPACE::fflush(stream); }

int fileno(FILE *stream) { return LIBC_NAMESPACE::fileno(stream); }

FILE *fopen(const char *path, const char *mode) {
  return LIBC_NAMESPACE::fopen(path, mode);
}

size_t fread(void *ptr, size_t size, size_t nmemb, FILE *stream) {
  return LIBC_NAMESPACE::fread(ptr, size, nmemb, stream);
}

int fseek(FILE *stream, long offset, int whence) {
  return LIBC_NAMESPACE::fseek(stream, offset, whence);
}

long ftell(FILE *stream) { return LIBC_NAMESPACE::ftell(stream); }

size_t fwrite(const void *ptr, size_t size, size_t nmemb, FILE *stream) {
  return LIBC_NAMESPACE::fwrite(ptr, size, nmemb, stream);
}

int fprintf(FILE *stream, const char *format, ...) {
  va_list vlist;
  va_start(vlist, format);
  int ret = LIBC_NAMESPACE::vfprintf(stream, format, vlist);
  va_end(vlist);
  return ret;
}

int snprintf(char *buffer, size_t buffsz, const char *format, ...) {
  va_list vlist;
  va_start(vlist, format);
  int ret = LIBC_NAMESPACE::vsnprintf(buffer, buffsz, format, vlist);
  va_end(vlist);
  return ret;
}

int fcntl(int fd, int cmd, ...) {
  va_list varargs;
  va_start(varargs, cmd);
  void *arg = va_arg(varargs, void *);
  va_end(varargs);
  return LIBC_NAMESPACE::fcntl(fd, cmd, arg);
}

int open(const char *path, int flags, ...) {
  mode_t mode = 0;
  if ((flags & O_CREAT) || (flags & O_TMPFILE) == O_TMPFILE) {
    va_list varargs;
    va_start(varargs, flags);
    mode = va_arg(varargs, mode_t);
    va_end(varargs);
  }
  return LIBC_NAMESPACE::open(path, flags, mode);
}

int mkdir(const char *path, mode_t mode) {
  return LIBC_NAMESPACE::mkdir(path, mode);
}

void *mmap(void *addr, size_t size, int prot, int flags, int fd, off_t offset) {
  return LIBC_NAMESPACE::mmap(addr, size, prot, flags, fd, offset);
}

int munmap(void *addr, size_t size) {
  return LIBC_NAMESPACE::munmap(addr, size);
}

int madvise(void *addr, size_t size, int advice) {
  return LIBC_NAMESPACE::madvise(addr, size, advice);
}

int ftruncate(int fd, off_t length) {
  return LIBC_NAMESPACE::ftruncate(fd, length);
}

int getpagesize() { return LIBC_NAMESPACE::getpagesize(); }

pid_t getpid() { return LIBC_NAMESPACE::getpid(); }

int prctl(int option, ...) {
  va_list vargs;
  va_start(vargs, option);
  unsigned long arg2 = va_arg(vargs, unsigned long);
  unsigned long arg3 = va_arg(vargs, unsigned long);
  unsigned long arg4 = va_arg(vargs, unsigned long);
  unsigned long arg5 = va_arg(vargs, unsigned long);
  va_end(vargs);
  return LIBC_NAMESPACE::prctl(option, arg2, arg3, arg4, arg5);
}

char *strchr(const char *src, int c) { return LIBC_NAMESPACE::strchr(src, c); }

char *strrchr(const char *src, int c) {
  return LIBC_NAMESPACE::strrchr(src, c);
}

int strcmp(const char *left, const char *right) {
  return LIBC_NAMESPACE::strcmp(left, right);
}

char *strdup(const char *src) { return LIBC_NAMESPACE::strdup(src); }

char *strerror(int err_num) { return LIBC_NAMESPACE::strerror(err_num); }

size_t strlen(const char *src) { return LIBC_NAMESPACE::strlen(src); }

char *strncpy(char *dest, const char *src, size_t count) {
  return LIBC_NAMESPACE::strncpy(dest, src, count);
}

char *getenv(const char *name) { return LIBC_NAMESPACE::getenv(name); }

int setenv(const char *name, const char *value, int overwrite) {
  return LIBC_NAMESPACE::setenv(name, value, overwrite);
}

long strtol(const char *str, char **str_end, int base) {
  return LIBC_NAMESPACE::strtol(str, str_end, base);
}

int uname(struct utsname *name) { return LIBC_NAMESPACE::uname(name); }

} // extern "C"
