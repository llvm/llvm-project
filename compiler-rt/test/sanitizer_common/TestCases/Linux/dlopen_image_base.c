// RUN: %clang -g %s -o %t
// RUN: %clang -g %s -DBUILD_SO -fPIC -o %t-so.so -shared -fuse-ld=lld -Wl,--image-base=0x4000000
// RUN: %run %t 2>&1

// REQUIRES: lld-available
// REQUIRES: glibc

// Regression test for #84482 and #21068
// XFAIL: msan, dfsan

#ifndef BUILD_SO
#  define _GNU_SOURCE
#  include <assert.h>
#  include <dlfcn.h>
#  include <link.h>
#  include <stdbool.h>
#  include <stdio.h>
#  include <unistd.h>

int main(int argc, char *argv[]) {
  char path[4096];
  snprintf(path, sizeof(path), "%s-so.so", argv[0]);

  void *handle = dlopen(path, RTLD_LAZY);
  if (!handle) {
    fprintf(stderr, "dlopen failed: %s\n", dlerror());
    return 1;
  }

  struct link_map *map = NULL;
  dlinfo(handle, RTLD_DI_LINKMAP, &map);
  if (map) {
    printf("DSO link_map name: %s\n", map->l_name);
    printf("DSO link_map l_addr: %p\n", (void *)map->l_addr);
    int pipefd[2];
    bool readable = false;
    if (pipe(pipefd) == 0) {
      if (write(pipefd[1], (void *)map->l_addr, 1) == 1) {
        readable = true;
      }
      close(pipefd[0]);
      close(pipefd[1]);
    }
    printf("DSO l_addr readable: %d\n", readable);
  }

  void (*fn)() = (void (*)())dlsym(handle, "fn");
  assert(fn != NULL);
  fn();

  dlclose(handle);
  return 0;
}
#else // BUILD_SO
#  include <stdio.h>
void fn() { printf("DSO function called successfully\n"); }
#endif
