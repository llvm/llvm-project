// RUN: %clang_safestack %s -pthread -o %t
// RUN: %run %t

#include <assert.h>
#include <pthread.h>
#include <signal.h>
#include <stddef.h>
#include <sys/mman.h>

// Test that safe stack works with sigaltstack.
int puts(const char *);

extern void *__get_unsafe_stack_ptr();

extern int unsafe_sigaltstack(size_t ss_size);
extern void *__get_unsafe_sigalt_stack_ptr();
extern void *__get_unsafe_sigalt_stack_top();
extern void *__get_unsafe_sigalt_stack_bottom();

__thread int signal_handlers_called = 0;
__thread int sigaltstack_called = 0;

void signal_handler(int signo) {
  signal_handlers_called += 1;
  assert(__get_unsafe_stack_ptr() > __get_unsafe_sigalt_stack_top() ||
         __get_unsafe_stack_ptr() < __get_unsafe_sigalt_stack_bottom());
}

void signal_sigalt_handler(int signo) {
  signal_handlers_called += 1;
  if (sigaltstack_called) {
    assert(__get_unsafe_stack_ptr() <= __get_unsafe_sigalt_stack_top() &&
           __get_unsafe_stack_ptr() >= __get_unsafe_sigalt_stack_bottom());
  } else {
    assert(__get_unsafe_stack_ptr() > __get_unsafe_sigalt_stack_top() ||
           __get_unsafe_stack_ptr() < __get_unsafe_sigalt_stack_bottom());
  }
}

void signal_sigaction(int signo, siginfo_t *si, void *uc) {
  signal_handlers_called += 1;
  assert(__get_unsafe_stack_ptr() > __get_unsafe_sigalt_stack_top() ||
         __get_unsafe_stack_ptr() < __get_unsafe_sigalt_stack_bottom());
}

void signal_sigalt_sigaction(int signo, siginfo_t *si, void *uc) {
  signal_handlers_called += 1;
  if (sigaltstack_called) {
    assert(__get_unsafe_stack_ptr() <= __get_unsafe_sigalt_stack_top() &&
           __get_unsafe_stack_ptr() >= __get_unsafe_sigalt_stack_bottom());
  } else {
    assert(__get_unsafe_stack_ptr() > __get_unsafe_sigalt_stack_top() ||
           __get_unsafe_stack_ptr() < __get_unsafe_sigalt_stack_bottom());
  }
}

void *t1_start(void *ptr) {
  // Test that since we didn't allocate a sigaltstack yet, we are not using any
  // unsafe sigalt stack.
  raise(SIGUSR1);
  raise(SIGUSR2);

  stack_t sigstk = {};
  size_t ss_size = 4096 * 4;
  void *ss_sp = mmap(NULL, ss_size, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_STACK, -1, 0);
  assert(ss_sp);
  sigstk.ss_size = ss_size;
  sigstk.ss_sp = ss_sp;

  unsafe_sigaltstack(sigstk.ss_size);
  sigaltstack(&sigstk, NULL);
  sigaltstack_called = 1;

  // Test that after sigaltstack is set, it automatically uses the sigaltstack.
  raise(SIGUSR1);
  raise(SIGUSR2);

  assert(signal_handlers_called == 4);

  return NULL;
}

int main() {
  char c[] = "hello world";
  puts(c);

  // Make sure no sigaltstack is allocated by default.
  assert(!__get_unsafe_sigalt_stack_ptr());

  stack_t sigstk = {};
  size_t ss_size = 4096 * 4;
  void *ss_sp = mmap(NULL, sigstk.ss_size, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_STACK, -1, 0);
  sigstk.ss_size = ss_size;
  sigstk.ss_sp = ss_sp;

  unsafe_sigaltstack(sigstk.ss_size);
  sigaltstack(&sigstk, NULL);
  sigaltstack_called = 1;

  // Make sure sigaltstack is intercepted correctly, setting the unsafe sigalt
  // stack in the correct size.
  assert(__get_unsafe_sigalt_stack_ptr());
  assert((__get_unsafe_sigalt_stack_top() -
          __get_unsafe_sigalt_stack_bottom()) == sigstk.ss_size);

  // Make sure retrieving the sigaltstack works without problems.
  sigaltstack(NULL, &sigstk);

  stack_t new_sigstk = {};
  new_sigstk.ss_size = 4096 * 8;
  new_sigstk.ss_sp = mmap(NULL, new_sigstk.ss_size, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANONYMOUS | MAP_STACK, -1, 0);

  unsafe_sigaltstack(new_sigstk.ss_size);
  sigaltstack(&new_sigstk, NULL);
  munmap(ss_sp, ss_size);

  // Make sure updating the size of the unsafe sigaltstack also updates when
  // setting a new sigaltstack.
  assert(__get_unsafe_sigalt_stack_ptr());
  assert((__get_unsafe_sigalt_stack_top() -
          __get_unsafe_sigalt_stack_bottom()) == new_sigstk.ss_size);

  struct sigaction sa;
  sa.sa_handler = signal_handler;
  sigemptyset(&sa.sa_mask);
  assert(sigaction(SIGUSR1, &sa, NULL) != -1);

  sa.sa_sigaction = signal_sigaction;
  sa.sa_flags = SA_SIGINFO;
  sigemptyset(&sa.sa_mask);
  assert(sigaction(SIGUSR2, &sa, NULL) != -1);

  // Test that we do not use the unsafe sigaltstack if SA_ONSTACK is not set.
  raise(SIGUSR1);
  raise(SIGUSR2);

  sa.sa_handler = signal_sigalt_handler;
  sa.sa_flags = SA_ONSTACK;
  sigemptyset(&sa.sa_mask);
  assert(sigaction(SIGUSR1, &sa, NULL) != -1);

  sa.sa_sigaction = signal_sigalt_sigaction;
  sa.sa_flags = SA_SIGINFO | SA_ONSTACK;
  sigemptyset(&sa.sa_mask);
  assert(sigaction(SIGUSR2, &sa, NULL) != -1);

  // Test that we do do use the unsafe sigaltstack if SA_ONSTACK is set.
  raise(SIGUSR1);
  raise(SIGUSR2);

  // Check that unsafe stack was correctly reset to the normal unsafe stack.
  assert(__get_unsafe_stack_ptr() > __get_unsafe_sigalt_stack_top() ||
         __get_unsafe_stack_ptr() < __get_unsafe_sigalt_stack_bottom());

  assert(signal_handlers_called == 4);

  // Now check if a sigaction set to use sigaltstack works on a thread that did
  // not call sigaltstack().
  pthread_t t1;
  assert(!pthread_create(&t1, NULL, t1_start, NULL));
  assert(!pthread_join(t1, NULL));

  return 0;
}
