// RUN: %check_clang_tidy %s bugprone-bad-signal-to-kill-thread %t

#define SIGTERM 15
#define SIGINT 2
using pthread_t = int;
using pthread_attr_t = int;

int pthread_create(pthread_t *thread, const pthread_attr_t *attr,
                   void *(*start_routine)(void *), void *arg);

int pthread_kill(pthread_t thread, int sig);

int pthread_cancel(pthread_t thread);

void *test_func_return_a_pointer(void *foo);

int main() {
  int result;
  pthread_t thread;

  if ((result = pthread_create(&thread, nullptr, test_func_return_a_pointer, 0)) != 0) {
  }
  if ((result = pthread_kill(thread, SIGTERM)) != 0) {
    // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: thread should not be terminated by raising the 'SIGTERM' signal [bugprone-bad-signal-to-kill-thread]
  }

  //compliant solution
  if ((result = pthread_cancel(thread)) != 0) {
  }

  if ((result = pthread_kill(thread, SIGINT)) != 0) {
  }
  if ((result = pthread_kill(thread, 0xF)) != 0) {
    // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: thread should not be terminated by raising the 'SIGTERM' signal [bugprone-bad-signal-to-kill-thread]
  }

  return 0;
}
