// Test that indirect call instrumentation works if background threads are
// spawned early during application startup. This can be the case with jemalloc,
// when libstdcxx.so invokes global constructors that call malloc.

// REQUIRES: system-linux
// RUN: %clangxx %cxxflags -Wl,-q %s -o %t.exe -D_GNU_SOURCE -g
// RUN: llvm-bolt -instrument %t.exe -o %t.bolt.exe
// RUN: %t.bolt.exe

#include <atomic>
#include <dlfcn.h>
#include <pthread.h>

static pthread_mutex_t M = PTHREAD_MUTEX_INITIALIZER;
static std::atomic_bool Init;
static void *(*MallocFn)(size_t);
static pthread_t Thread;

static void target() {}
static void (*IndTarget)() = target;

static void *threadFun(void *) {
  for (int I = 0; I < 1000; ++I)
    IndTarget();
  return nullptr;
}

extern "C" void *malloc(size_t Size) {
  if (!Init) {
    pthread_mutex_lock(&M);
    if (!Init) {
      MallocFn =
          reinterpret_cast<decltype(MallocFn)>(dlsym(RTLD_NEXT, "malloc"));
      Init = true;
      pthread_create(&Thread, nullptr, threadFun, nullptr);
    }
    pthread_mutex_unlock(&M);
  }
  return MallocFn(Size);
}

int main() {
  pthread_join(Thread, nullptr);
  return 0;
}
