// RUN: %clangxx_msan -std=c++11 -O0 -g %s -o %t && %run %t

#include <assert.h>
#include <fcntl.h>
#include <sanitizer/msan_interface.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

int main(void) {
  struct file_handle *handle = reinterpret_cast<struct file_handle *>(
      malloc(sizeof(*handle) + MAX_HANDLE_SZ));
  handle->handle_bytes = MAX_HANDLE_SZ;

  int mount_id;
  int res = name_to_handle_at(AT_FDCWD, "/dev/null", handle, &mount_id, 0);
  assert(!res);
  __msan_check_mem_is_initialized(&mount_id, sizeof(mount_id));
  __msan_check_mem_is_initialized(&handle->handle_bytes,
                                  sizeof(handle->handle_bytes));
  __msan_check_mem_is_initialized(&handle->handle_type,
                                  sizeof(handle->handle_type));
  __msan_check_mem_is_initialized(&handle->f_handle, handle->handle_bytes);

  free(handle);
  return 0;
}
