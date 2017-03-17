#include <pthread.h>
#include <signal.h>

void set_thread_name(const char *name) {
#if defined(__APPLE__)
  ::pthread_setname_np(name);
#elif defined(__FreeBSD__)
  ::pthread_set_name_np(::pthread_self(), name);
#elif defined(__linux__)
  ::pthread_setname_np(::pthread_self(), name);
#elif defined(__NetBSD__)
  ::pthread_setname_np(::pthread_self(), "%s", name);
#endif
}

int main() {
  set_thread_name("hello world");
  raise(SIGINT);
  set_thread_name("goodbye world");
  raise(SIGINT);
  return 0;
}
