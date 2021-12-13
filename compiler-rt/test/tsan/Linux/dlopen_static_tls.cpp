// RUN: %clangxx_tsan -O1 %s -DBUILD_SO -fPIC -shared -o %t-so.so
// RUN: %clangxx_tsan -O1 %s %link_libcxx_tsan -o %t && %run %t 2>&1 | FileCheck %s

// A test for loading a dynamic library with static TLS.
// Such static TLS is a hack that allows a dynamic library to have faster TLS,
// but it can be loaded only iff all threads happened to allocate some excess
// of static TLS space for whatever reason. If it's not the case loading fails with:
// dlopen: cannot load any more object with static TLS
// We used to produce a false positive because dlopen will write into TLS
// of all existing threads to initialize/zero TLS region for the loaded library.
// And this appears to be racing with initialization of TLS in the thread
// since we model a write into the whole static TLS region (we don't know what part
// of it is currently unused):
// WARNING: ThreadSanitizer: data race (pid=2317365)
//   Write of size 1 at 0x7f1fa9bfcdd7 by main thread:
//     #0 memset
//     #1 init_one_static_tls
//     #2 __pthread_init_static_tls
//     [[ this is where main calls dlopen ]]
//     #3 main
//   Previous write of size 8 at 0x7f1fa9bfcdd0 by thread T1:
//     #0 __tsan_tls_initialization

// Failing on bots:
// https://lab.llvm.org/buildbot#builders/184/builds/1580
// https://lab.llvm.org/buildbot#builders/18/builds/3167
// UNSUPPORTED: aarch64, powerpc64, powerpc64le

#ifdef BUILD_SO

__attribute__((tls_model("initial-exec"))) __thread char x = 42;
__attribute__((tls_model("initial-exec"))) __thread char y;

extern "C" int sofunc() { return ++x + ++y; }

#else // BUILD_SO

#  include "../test.h"
#  include <dlfcn.h>
#  include <string>

__thread int x[1023];

void *lib;
void (*func)();
int ready;

void *thread(void *arg) {
  barrier_wait(&barrier);
  if (__atomic_load_n(&ready, __ATOMIC_ACQUIRE))
    func();
  barrier_wait(&barrier);
  if (dlclose(lib)) {
    printf("error in dlclose: %s\n", dlerror());
    exit(1);
  }
  return 0;
}

int main(int argc, char *argv[]) {
  barrier_init(&barrier, 2);
  pthread_t th;
  pthread_create(&th, 0, thread, 0);
  lib = dlopen((std::string(argv[0]) + "-so.so").c_str(), RTLD_NOW);
  if (lib == 0) {
    printf("error in dlopen: %s\n", dlerror());
    return 1;
  }
  func = (void (*)())dlsym(lib, "sofunc");
  if (func == 0) {
    printf("error in dlsym: %s\n", dlerror());
    return 1;
  }
  __atomic_store_n(&ready, 1, __ATOMIC_RELEASE);
  barrier_wait(&barrier);
  func();
  barrier_wait(&barrier);
  pthread_join(th, 0);
  fprintf(stderr, "DONE\n");
  return 0;
}

#endif // BUILD_SO

// CHECK: DONE
