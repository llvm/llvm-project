// RUN: %clangxx_tsan -O1 %s -DBUILD_SO -fPIC -shared -o %t-so.so
// RUN: %clangxx_tsan -O1 %s %link_libcxx_tsan -o %t && %run %t 2>&1 | FileCheck %s

// dl_iterate_phdr doesn't exist on OS X.
// UNSUPPORTED: darwin

#ifdef BUILD_SO

#include "test.h"

int exported_var = 0;

#else  // BUILD_SO

#include "test.h"
#include <dlfcn.h>
#include <link.h>
#include <string.h>
#include <string>

static int callback(struct dl_phdr_info *info, size_t size, void *data) {
  if (info->dlpi_name[0] == '\0')
    info->dlpi_name = "/proc/self/exe";
  return !strcmp(info->dlpi_name, "non existent module");
}

void *thread(void *unused) {
  for (int i = 0; i < 1000; i++) {
    barrier_wait(&barrier);
    dl_iterate_phdr(callback, 0);
  }
  return 0;
}

int main(int argc, char *argv[]) {
  barrier_init(&barrier, 2);
  std::string path = std::string(argv[0]) + std::string("-so.so");
  pthread_t th;
  pthread_create(&th, 0, thread, 0);
  for (int i = 0; i < 1000; i++) {
    barrier_wait(&barrier);
    void *lib = dlopen(path.c_str(), RTLD_NOW);
    if (!lib) {
      printf("error in dlopen: %s\n", dlerror());
      return 1;
    }
    dlclose(lib);
  }
  pthread_join(th, 0);
  fprintf(stderr, "DONE\n");
  return 0;
}

#endif  // BUILD_SO

// CHECK-NOT: WARNING: ThreadSanitizer: data race
// CHECK: DONE
