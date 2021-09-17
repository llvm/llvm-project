// Test that dyld interposition works in the presence of DYLD_INSERT_LIBRARIES.
// Additionally, the injected library also has a pthread introspection hook that
// calls intercepted APIs before and after calling through to the TSan hook.
// This mirrors what libBacktraceRecording.dylib (Xcode 'Queue Debugging'
// feature) does.

// RUN: %clang_tsan %s -o %t
// RUN: %clang_tsan %s -o %t.dylib -fno-sanitize=thread -dynamiclib -DSHARED_LIB
//
// RUN: env DYLD_INSERT_LIBRARIES=%t.dylib %run %t 2>&1 | FileCheck %s --implicit-check-not='ThreadSanitizer'
//
// XFAIL: ios

#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>

#if defined(SHARED_LIB)
enum {
  PTHREAD_INTROSPECTION_THREAD_CREATE = 1,
  PTHREAD_INTROSPECTION_THREAD_START,
  PTHREAD_INTROSPECTION_THREAD_TERMINATE,
  PTHREAD_INTROSPECTION_THREAD_DESTROY,
};
typedef void (*pthread_introspection_hook_t)(unsigned int event,
                                             pthread_t thread, void *addr,
                                             size_t size);
extern pthread_introspection_hook_t pthread_introspection_hook_install(
    pthread_introspection_hook_t hook);

static pthread_introspection_hook_t previous_pthread_hook;
static void pthread_introspection_hook(unsigned int event, pthread_t thread, void *addr, size_t size) {
  pthread_t self;
  const unsigned k_max_thread_name_size = 64;
  char name[k_max_thread_name_size];

  // Use some intercepted APIs *before* TSan hook runs.
  {
    self = pthread_self();
    pthread_getname_np(self, name, k_max_thread_name_size);
    if (strlen(name) == 0) {
      strlcpy(name, "n/a", 4);
    }
  }

  // This calls through to the TSan-installed hook, because the injected library
  // constructor (see __library_initializer() below) runs after the TSan
  // initializer.  It replaces and forward to the previously-installed TSan
  // introspection hook (very similar to what libBacktraceRecording.dylib does).
  assert(previous_pthread_hook);
  previous_pthread_hook(event, thread, addr, size);

  // Use some intercepted APIs *after* TSan hook runs.
  {
    assert(self == pthread_self());
    char name2[k_max_thread_name_size];
    pthread_getname_np(self, name2, k_max_thread_name_size);
    if (strlen(name2) == 0) {
      strlcpy(name2, "n/a", 4);
    }
    assert(strcmp(name, name2) == 0);
  }

  switch (event) {
  case PTHREAD_INTROSPECTION_THREAD_CREATE:
    fprintf(stderr, "THREAD_CREATE    %p, self: %p, name: %s\n", thread, self, name);
    break;
  case PTHREAD_INTROSPECTION_THREAD_START:
    fprintf(stderr, "THREAD_START     %p, self: %p, name: %s\n", thread, self, name);
    break;
  case PTHREAD_INTROSPECTION_THREAD_TERMINATE:
    fprintf(stderr, "THREAD_TERMINATE %p, self: %p, name: %s\n", thread, self, name);
    break;
  case PTHREAD_INTROSPECTION_THREAD_DESTROY:
    fprintf(stderr, "THREAD_DESTROY   %p, self: %p, name: %s\n", thread, self, name);
    break;
  }
}

__attribute__((constructor))
static void __library_initializer(void) {
  fprintf(stderr, "__library_initializer\n");
  previous_pthread_hook = pthread_introspection_hook_install(pthread_introspection_hook);
}

#else  // defined(SHARED_LIB)

void *Thread(void *a) {
  pthread_setname_np("child thread");
  fprintf(stderr, "Hello from pthread\n");
  return NULL;
}

int main() {
  fprintf(stderr, "main\n");
  pthread_t t;
  pthread_create(&t, NULL, Thread, NULL);
  pthread_join(t, NULL);
  fprintf(stderr, "Done.\n");
}
#endif  // defined(SHARED_LIB)

// CHECK: __library_initializer
// CHECK: main
// Ignore TSan background thread.
// CHECK: THREAD_CREATE
// CHECK: THREAD_CREATE    [[CHILD:0x[0-9a-f]+]], self: [[MAIN:0x[0-9a-f]+]], name: n/a
// CHECK: THREAD_START     [[CHILD]], self: [[CHILD]], name: n/a
// CHECK: Hello from pthread
// CHECK: THREAD_TERMINATE [[CHILD]], self: [[CHILD]], name: child thread
// CHECK: THREAD_DESTROY   [[CHILD]], self: [[MAIN]]
// CHECK: Done.
