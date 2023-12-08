// RUN: clang-tidy %s --checks=-*,bugprone-bad-signal-to-kill-thread -- | count 0

#define SIGTERM 15
#undef SIGTERM // no-crash
using pthread_t = int;
int pthread_kill(pthread_t thread, int sig);

int func() {
  pthread_t thread;
  return pthread_kill(thread, 0);
}
