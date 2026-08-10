// Build a library with origin tracking and an executable w/o origin tracking.
// Test that origin tracking is enabled at runtime.
// RUN: %clangxx_lsan -O0 %s -DBUILD_SO -fPIC -shared -o %t-so.so
// RUN: %clangxx_lsan -O0 %s -ldl -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_lsan -O0 %s -ldl -o %t -DSUPPRESS_LEAK && %run %t

#ifdef BUILD_SO

#  include <stdlib.h>

extern "C" {
void *my_alloc(unsigned sz) { return malloc(sz); }
} // extern "C"

#else // BUILD_SO

#  include <assert.h>
#  include <dlfcn.h>
#  include <stdlib.h>
#  include <string>

#  ifdef SUPPRESS_LEAK
extern "C" const char *__lsan_default_suppressions() {
  return "leak:^<unknown module>$";
}
#  endif

int main(int argc, char **argv) {

  std::string path = std::string(argv[0]) + "-so.so";

  dlerror();

  void *handle = dlopen(path.c_str(), RTLD_LAZY);
  assert(handle != 0);
  typedef void *(*fn)(unsigned sz);
  fn my_alloc = (fn)dlsym(handle, "my_alloc");

  for (int i = 0; i < 100; ++i)
    my_alloc(i);

  dlclose(handle);
  return 0;
}

#endif // BUILD_SO

// CHECK: Direct leak
