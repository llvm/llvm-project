// RUN: %clangxx %s -o %t -ldl
// RUN: %clangxx_hwasan -shared %s -o %t.so -DSHARED_LIB -shared-libsan -Wl,-rpath,%compiler_rt_libdir
// RUN: %env_hwasan_opts=disable_allocator_tagging=0 %run %t

#include <stddef.h>

// Test that allocations made by the system allocator can be realloc'd and freed
// by the hwasan allocator.

typedef void run_test_fn(void *(*system_malloc)(size_t size));

#ifdef SHARED_LIB

// Call the __sanitizer_ versions of these functions so that the test
// doesn't require the Android dynamic loader.
extern "C" void *__sanitizer_realloc(void *ptr, size_t size);
extern "C" void __sanitizer_free(void *ptr);

extern "C" run_test_fn run_test;
void run_test(void *(*system_malloc)(size_t size)) {
  void *mem = system_malloc(64);
  mem = __sanitizer_realloc(mem, 128);
  __sanitizer_free(mem);
}

#else

#include <dlfcn.h>
#include <stdlib.h>
#include <string>

int main(int argc, char **argv) {
  std::string path = argv[0];
  path += ".so";
  void *lib = dlopen(path.c_str(), RTLD_NOW);
  if (!lib) {
    printf("error in dlopen(): %s\n", dlerror());
    return 1;
  }

  auto run_test = reinterpret_cast<run_test_fn *>(dlsym(lib, "run_test"));
  if (!run_test) {
    printf("failed dlsym\n");
    return 1;
  }

  run_test(malloc);
}

#endif
