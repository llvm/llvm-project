// RUN: %clangxx -O0 %s -o %t && %run %t
// UNSUPPORTED: android

#include <assert.h>
#include <fcntl.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>

int main(int argc, char **argv) {
  int mount_id;
  struct file_handle *handle = reinterpret_cast<struct file_handle *>(
      malloc(sizeof(*handle) + MAX_HANDLE_SZ));

  handle->handle_bytes = MAX_HANDLE_SZ;
  int res = name_to_handle_at(AT_FDCWD, "/dev/null", handle, &mount_id, 0);
  assert(!res);

  free(handle);
  return 0;
}
