// RUN: %clangxx_msan -O0 %s -o %t && %run %t %p 2>&1
// RUN: %clangxx_msan -O0 -D_FILE_OFFSET_BITS=64 %s -o %t && %run %t %p 2>&1
// RUN: %clangxx_msan -O3 %s -o %t && %run %t %p 2>&1

#include <argz.h>
#include <assert.h>
#include <sys/types.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sanitizer/msan_interface.h>

// Do not depend on libattr headers.
#ifndef ENOATTR
#define ENOATTR ENODATA
#endif

extern "C" {
ssize_t listxattr(const char *path, char *list, size_t size);
ssize_t llistxattr(const char *path, char *list, size_t size);
ssize_t flistxattr(int fd, char *list, size_t size);
ssize_t getxattr(const char *path, const char *name, void *value, size_t size);
ssize_t lgetxattr(const char *path, const char *name, void *value, size_t size);
ssize_t fgetxattr(int fd, const char *name, void *value, size_t size);
}

char g_path[1024];
int g_fd;

// Life before closures...
ssize_t listxattr_wrapper(char *buf, size_t size) {
  return listxattr(g_path, buf, size);
}

ssize_t llistxattr_wrapper(char *buf, size_t size) {
  return llistxattr(g_path, buf, size);
}

ssize_t flistxattr_wrapper(char *buf, size_t size) {
  return flistxattr(g_fd, buf, size);
}

ssize_t getxattr_wrapper(const char *name, char *buf, size_t size) {
  return getxattr(g_path, name, buf, size);
}

ssize_t lgetxattr_wrapper(const char *name, char *buf, size_t size) {
  return lgetxattr(g_path, name, buf, size);
}

ssize_t fgetxattr_wrapper(const char *name, char *buf, size_t size) {
  return fgetxattr(g_fd, name, buf, size);
}

size_t test_list(ssize_t fun(char*, size_t), char **buf) {
  int buf_size = 1024;
  while (true) {
    *buf = (char *)malloc(buf_size);
    assert(__msan_test_shadow(*buf, buf_size) != -1);
    ssize_t res = fun(*buf, buf_size);
    if (res >= 0) {
      assert(__msan_test_shadow(*buf, buf_size) == res);
      return res;
    }
    if (errno == ENOTSUP) {
      printf("Extended attributes are disabled. *xattr test is a no-op.\n");
      exit(0);
    }
    assert(errno == ERANGE);
    free(*buf);
    buf_size *= 2;
  }
}

// True means success. False means result inconclusive because we don't have
// access to this attribute.
bool test_get_single_attr(ssize_t fun(const char *, char *, size_t),
                          const char *attr_name) {
  char *buf;
  int buf_size = 1024;
  while (true) {
    buf = (char *)malloc(buf_size);
    assert(__msan_test_shadow(buf, buf_size) != -1);
    ssize_t res = fun(attr_name, buf, buf_size);
    if (res >= 0) {
      assert(__msan_test_shadow(buf, buf_size) == res);
      free(buf);
      return true;
    }
    if (errno == ENOTSUP) {
      printf("Extended attributes are disabled. *xattr test is a no-op.\n");
      exit(0);
    }
    if (errno == ENOATTR)
      return false;
    assert(errno == ERANGE);
    free(buf);
    buf_size *= 2;
  }
}

void test_get(ssize_t fun(const char *, char *, size_t), const char *attr_list,
              size_t attr_list_size) {
  // Try every attribute, until we see one we can access. Attribute names are
  // null-separated strings in attr_list.
  size_t attr_list_len = argz_count(attr_list, attr_list_size);
  size_t argv_size = (attr_list_len + 1) * sizeof(char *);
  char **attrs = (char **)malloc(argv_size);
  argz_extract(attr_list, attr_list_size, attrs);
  // TODO(smatveev): we need proper argz_* interceptors
  __msan_unpoison(attrs, argv_size);
  for (size_t i = 0; (i < attr_list_len) && attrs[i]; i++) {
    if (test_get_single_attr(fun, attrs[i]))
      return;
  }
  printf("*xattr test could not access any attributes.\n");
}

// TODO: set some attributes before trying to retrieve them with *getxattr.
// Currently the list is empty, so *getxattr is not tested.
int main(int argc, char *argv[]) {
  assert(argc == 2);
  snprintf(g_path, sizeof(g_path), "%s/%s", argv[1], "xattr_test_root/a");

  g_fd = open(g_path, O_RDONLY);
  assert(g_fd);

  char *attr_list;
  size_t attr_list_size;
  attr_list_size = test_list(listxattr_wrapper, &attr_list);
  free(attr_list);
  attr_list_size = test_list(llistxattr_wrapper, &attr_list);
  free(attr_list);
  attr_list_size = test_list(flistxattr_wrapper, &attr_list);

  test_get(getxattr_wrapper, attr_list, attr_list_size);
  test_get(lgetxattr_wrapper, attr_list, attr_list_size);
  test_get(fgetxattr_wrapper, attr_list, attr_list_size);

  free(attr_list);
  return 0;
}
