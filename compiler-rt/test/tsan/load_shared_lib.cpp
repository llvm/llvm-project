// Check that if the list of shared libraries changes between the two race
// reports, the second report occurring in a new shared library is still
// symbolized correctly.

// RUN: %clangxx_tsan -O1 %s -DBUILD_SO -fPIC -shared -o %t-so.so
// RUN: %clangxx_tsan -O1 %s %link_libcxx_tsan -o %t -rdynamic && %deflake %run %t | FileCheck %s

#ifdef BUILD_SO

#include "test.h"

int GLOB_SHARED = 0;

extern "C"
void init_so() {
  barrier_init(&barrier, 2);
}

extern "C"
void *write_from_so(void *unused) {
  if (unused == 0)
    barrier_wait(&barrier);
  GLOB_SHARED++;
  if (unused != 0)
    barrier_wait(&barrier);
  return NULL;
}

#else  // BUILD_SO

#include "test.h"
#include <dlfcn.h>
#include <string>

int GLOB = 0;

void *write_glob(void *unused) {
  if (unused == 0)
    barrier_wait(&barrier);
  GLOB++;
  if (unused != 0)
    barrier_wait(&barrier);
  return NULL;
}

void race_two_threads(void *(*access_callback)(void *unused)) {
  pthread_t t1, t2;
  pthread_create(&t1, NULL, access_callback, (void*)1);
  pthread_create(&t2, NULL, access_callback, NULL);
  pthread_join(t1, NULL);
  pthread_join(t2, NULL);
}

int main(int argc, char *argv[]) {
  barrier_init(&barrier, 2);
  std::string path = std::string(argv[0]) + std::string("-so.so");
  race_two_threads(write_glob);
  // CHECK: write_glob
  void *lib = dlopen(path.c_str(), RTLD_NOW);
    if (!lib) {
    printf("error in dlopen(): %s\n", dlerror());
    return 1;
  }
  void (*init_so)();
  *(void **)&init_so = dlsym(lib, "init_so");
  init_so();
  void *(*write_from_so)(void *unused);
  *(void **)&write_from_so = dlsym(lib, "write_from_so");
  race_two_threads(write_from_so);
  // CHECK: write_from_so
  return 0;
}

#endif  // BUILD_SO
