// RUN: clang-tidy %s --checks=-*,bugprone-bad-signal-to-kill-thread -- | count 0

#define SIGTERM ((unsigned)15) // no-crash
using pthread_t = int;
int pthread_kill(pthread_t thread, int sig);

int func() {
  pthread_t thread;
  return pthread_kill(thread, 0);
}
