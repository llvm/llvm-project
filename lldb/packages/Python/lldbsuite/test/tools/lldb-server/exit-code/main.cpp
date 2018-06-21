#include <cstdlib>
#include <cstring>
#include <errno.h>
#include <inttypes.h>
#include <memory>
#include <pthread.h>
#include <setjmp.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <vector>

#if defined(__APPLE__)
__OSX_AVAILABLE_STARTING(__MAC_10_6, __IPHONE_3_2)
int pthread_threadid_np(pthread_t, __uint64_t *);
#elif defined(__linux__)
#include <sys/syscall.h>
#elif defined(__NetBSD__)
#include <lwp.h>
#endif

static const char *const RETVAL_PREFIX = "retval:";
static const char *const SLEEP_PREFIX = "sleep:";
static const char *const STDERR_PREFIX = "stderr:";
static const char *const SET_MESSAGE_PREFIX = "set-message:";
static const char *const PRINT_MESSAGE_COMMAND = "print-message:";
static const char *const GET_DATA_ADDRESS_PREFIX = "get-data-address-hex:";
static const char *const GET_STACK_ADDRESS_COMMAND = "get-stack-address-hex:";
static const char *const GET_HEAP_ADDRESS_COMMAND = "get-heap-address-hex:";

static const char *const GET_CODE_ADDRESS_PREFIX = "get-code-address-hex:";
static const char *const CALL_FUNCTION_PREFIX = "call-function:";

static const char *const THREAD_PREFIX = "thread:";
static const char *const THREAD_COMMAND_NEW = "new";
static const char *const THREAD_COMMAND_PRINT_IDS = "print-ids";
static const char *const THREAD_COMMAND_SEGFAULT = "segfault";

static bool g_print_thread_ids = false;
static pthread_mutex_t g_print_mutex = PTHREAD_MUTEX_INITIALIZER;
static bool g_threads_do_segfault = false;

static pthread_mutex_t g_jump_buffer_mutex = PTHREAD_MUTEX_INITIALIZER;
static jmp_buf g_jump_buffer;
static bool g_is_segfaulting = false;

static char g_message[256];

static volatile char g_c1 = '0';
static volatile char g_c2 = '1';

static void print_thread_id() {
// Put in the right magic here for your platform to spit out the thread id (tid)
// that debugserver/lldb-gdbserver would see as a TID. Otherwise, let the else
// clause print out the unsupported text so that the unit test knows to skip
// verifying thread ids.
#if defined(__APPLE__)
  __uint64_t tid = 0;
  pthread_threadid_np(pthread_self(), &tid);
  printf("%" PRIx64, tid);
#elif defined(__linux__)
  // This is a call to gettid() via syscall.
  printf("%" PRIx64, static_cast<uint64_t>(syscall(__NR_gettid)));
#elif defined(__NetBSD__)
  // Technically lwpid_t is 32-bit signed integer
  printf("%" PRIx64, static_cast<uint64_t>(_lwp_self()));
#else
  printf("{no-tid-support}");
#endif
}

static void signal_handler(int signo) {
  const char *signal_name = nullptr;
  switch (signo) {
  case SIGUSR1:
    signal_name = "SIGUSR1";
    break;
  case SIGSEGV:
    signal_name = "SIGSEGV";
    break;
  default:
    signal_name = nullptr;
  }

  // Print notice that we received the signal on a given thread.
  pthread_mutex_lock(&g_print_mutex);
  if (signal_name)
    printf("received %s on thread id: ", signal_name);
  else
    printf("received signo %d (%s) on thread id: ", signo, strsignal(signo));
  print_thread_id();
  printf("\n");
  pthread_mutex_unlock(&g_print_mutex);

  // Reset the signal handler if we're one of the expected signal handlers.
  switch (signo) {
  case SIGSEGV:
    if (g_is_segfaulting) {
      // Fix up the pointer we're writing to.  This needs to happen if nothing
      // intercepts the SIGSEGV (i.e. if somebody runs this from the command
      // line).
      longjmp(g_jump_buffer, 1);
    }
    break;
  case SIGUSR1:
    if (g_is_segfaulting) {
      // Fix up the pointer we're writing to.  This is used to test gdb remote
      // signal delivery. A SIGSEGV will be raised when the thread is created,
      // switched out for a SIGUSR1, and then this code still needs to fix the
      // seg fault. (i.e. if somebody runs this from the command line).
      longjmp(g_jump_buffer, 1);
    }
    break;
  }

  // Reset the signal handler.
  sig_t sig_result = signal(signo, signal_handler);
  if (sig_result == SIG_ERR) {
    fprintf(stderr, "failed to set signal handler: errno=%d\n", errno);
    exit(1);
  }
}

static void swap_chars() {
  g_c1 = '1';
  g_c2 = '0';

  g_c1 = '0';
  g_c2 = '1';
}

static void hello() {
  pthread_mutex_lock(&g_print_mutex);
  printf("hello, world\n");
  pthread_mutex_unlock(&g_print_mutex);
}

static void *thread_func(void *arg) {
  static pthread_mutex_t s_thread_index_mutex = PTHREAD_MUTEX_INITIALIZER;
  static int s_thread_index = 1;

  pthread_mutex_lock(&s_thread_index_mutex);
  const int this_thread_index = s_thread_index++;
  pthread_mutex_unlock(&s_thread_index_mutex);

  if (g_print_thread_ids) {
    pthread_mutex_lock(&g_print_mutex);
    printf("thread %d id: ", this_thread_index);
    print_thread_id();
    printf("\n");
    pthread_mutex_unlock(&g_print_mutex);
  }

  if (g_threads_do_segfault) {
    // Sleep for a number of seconds based on the thread index.
    // TODO add ability to send commands to test exe so we can
    // handle timing more precisely.  This is clunky.  All we're
    // trying to do is add predictability as to the timing of
    // signal generation by created threads.
    int sleep_seconds = 2 * (this_thread_index - 1);
    while (sleep_seconds > 0)
      sleep_seconds = sleep(sleep_seconds);

    // Test creating a SEGV.
    pthread_mutex_lock(&g_jump_buffer_mutex);
    g_is_segfaulting = true;
    int *bad_p = nullptr;
    if (setjmp(g_jump_buffer) == 0) {
      // Force a seg fault signal on this thread.
      *bad_p = 0;
    } else {
      // Tell the system we're no longer seg faulting.
      // Used by the SIGUSR1 signal handler that we inject
      // in place of the SIGSEGV so it only tries to
      // recover from the SIGSEGV if this seg fault code
      // was in play.
      g_is_segfaulting = false;
    }
    pthread_mutex_unlock(&g_jump_buffer_mutex);

    pthread_mutex_lock(&g_print_mutex);
    printf("thread ");
    print_thread_id();
    printf(": past SIGSEGV\n");
    pthread_mutex_unlock(&g_print_mutex);
  }

  int sleep_seconds_remaining = 60;
  while (sleep_seconds_remaining > 0) {
    sleep_seconds_remaining = sleep(sleep_seconds_remaining);
  }

  return nullptr;
}

int main(int argc, char **argv) {
  lldb_enable_attach();

  std::vector<pthread_t> threads;
  std::unique_ptr<uint8_t[]> heap_array_up;
  int return_value = 0;

  // Set the signal handler.
  sig_t sig_result = signal(SIGALRM, signal_handler);
  if (sig_result == SIG_ERR) {
    fprintf(stderr, "failed to set SIGALRM signal handler: errno=%d\n", errno);
    exit(1);
  }

  sig_result = signal(SIGUSR1, signal_handler);
  if (sig_result == SIG_ERR) {
    fprintf(stderr, "failed to set SIGUSR1 handler: errno=%d\n", errno);
    exit(1);
  }

  sig_result = signal(SIGSEGV, signal_handler);
  if (sig_result == SIG_ERR) {
    fprintf(stderr, "failed to set SIGUSR1 handler: errno=%d\n", errno);
    exit(1);
  }

  // Process command line args.
  for (int i = 1; i < argc; ++i) {
    if (std::strstr(argv[i], STDERR_PREFIX)) {
      // Treat remainder as text to go to stderr.
      fprintf(stderr, "%s\n", (argv[i] + strlen(STDERR_PREFIX)));
    } else if (std::strstr(argv[i], RETVAL_PREFIX)) {
      // Treat as the return value for the program.
      return_value = std::atoi(argv[i] + strlen(RETVAL_PREFIX));
    } else if (std::strstr(argv[i], SLEEP_PREFIX)) {
      // Treat as the amount of time to have this process sleep (in seconds).
      int sleep_seconds_remaining = std::atoi(argv[i] + strlen(SLEEP_PREFIX));

      // Loop around, sleeping until all sleep time is used up.  Note that
      // signals will cause sleep to end early with the number of seconds
      // remaining.
      for (int i = 0; sleep_seconds_remaining > 0; ++i) {
        sleep_seconds_remaining = sleep(sleep_seconds_remaining);
        // std::cout << "sleep result (call " << i << "): " <<
        // sleep_seconds_remaining << std::endl;
      }
    } else if (std::strstr(argv[i], SET_MESSAGE_PREFIX)) {
      // Copy the contents after "set-message:" to the g_message buffer.
      // Used for reading inferior memory and verifying contents match
      // expectations.
      strncpy(g_message, argv[i] + strlen(SET_MESSAGE_PREFIX),
              sizeof(g_message));

      // Ensure we're null terminated.
      g_message[sizeof(g_message) - 1] = '\0';

    } else if (std::strstr(argv[i], PRINT_MESSAGE_COMMAND)) {
      pthread_mutex_lock(&g_print_mutex);
      printf("message: %s\n", g_message);
      pthread_mutex_unlock(&g_print_mutex);
    } else if (std::strstr(argv[i], GET_DATA_ADDRESS_PREFIX)) {
      volatile void *data_p = nullptr;

      if (std::strstr(argv[i] + strlen(GET_DATA_ADDRESS_PREFIX), "g_message"))
        data_p = &g_message[0];
      else if (std::strstr(argv[i] + strlen(GET_DATA_ADDRESS_PREFIX), "g_c1"))
        data_p = &g_c1;
      else if (std::strstr(argv[i] + strlen(GET_DATA_ADDRESS_PREFIX), "g_c2"))
        data_p = &g_c2;

      pthread_mutex_lock(&g_print_mutex);
      printf("data address: %p\n", data_p);
      pthread_mutex_unlock(&g_print_mutex);
    } else if (std::strstr(argv[i], GET_HEAP_ADDRESS_COMMAND)) {
      // Create a byte array if not already present.
      if (!heap_array_up)
        heap_array_up.reset(new uint8_t[32]);

      pthread_mutex_lock(&g_print_mutex);
      printf("heap address: %p\n", heap_array_up.get());
      pthread_mutex_unlock(&g_print_mutex);
    } else if (std::strstr(argv[i], GET_STACK_ADDRESS_COMMAND)) {
      pthread_mutex_lock(&g_print_mutex);
      printf("stack address: %p\n", &return_value);
      pthread_mutex_unlock(&g_print_mutex);
    } else if (std::strstr(argv[i], GET_CODE_ADDRESS_PREFIX)) {
      void (*func_p)() = nullptr;

      if (std::strstr(argv[i] + strlen(GET_CODE_ADDRESS_PREFIX), "hello"))
        func_p = hello;
      else if (std::strstr(argv[i] + strlen(GET_CODE_ADDRESS_PREFIX),
                           "swap_chars"))
        func_p = swap_chars;

      pthread_mutex_lock(&g_print_mutex);
      printf("code address: %p\n", func_p);
      pthread_mutex_unlock(&g_print_mutex);
    } else if (std::strstr(argv[i], CALL_FUNCTION_PREFIX)) {
      // Defaut to providing the address of main.
      if (std::strcmp(argv[i] + strlen(CALL_FUNCTION_PREFIX), "hello") == 0)
        hello();
      else if (std::strcmp(argv[i] + strlen(CALL_FUNCTION_PREFIX),
                           "swap_chars") == 0)
        swap_chars();
      else {
        pthread_mutex_lock(&g_print_mutex);
        printf("unknown function: %s\n",
               argv[i] + strlen(CALL_FUNCTION_PREFIX));
        pthread_mutex_unlock(&g_print_mutex);
      }
    } else if (std::strstr(argv[i], THREAD_PREFIX)) {
      // Check if we're creating a new thread.
      if (std::strstr(argv[i] + strlen(THREAD_PREFIX), THREAD_COMMAND_NEW)) {
        // Create a new thread.
        pthread_t new_thread;
        const int err =
            ::pthread_create(&new_thread, nullptr, thread_func, nullptr);
        if (err) {
          fprintf(stderr, "pthread_create() failed with error code %d\n", err);
          exit(err);
        }
        threads.push_back(new_thread);
      } else if (std::strstr(argv[i] + strlen(THREAD_PREFIX),
                             THREAD_COMMAND_PRINT_IDS)) {
        // Turn on thread id announcing.
        g_print_thread_ids = true;

        // And announce us.
        pthread_mutex_lock(&g_print_mutex);
        printf("thread 0 id: ");
        print_thread_id();
        printf("\n");
        pthread_mutex_unlock(&g_print_mutex);
      } else if (std::strstr(argv[i] + strlen(THREAD_PREFIX),
                             THREAD_COMMAND_SEGFAULT)) {
        g_threads_do_segfault = true;
      } else {
        // At this point we don't do anything else with threads.
        // Later use thread index and send command to thread.
      }
    } else {
      // Treat the argument as text for stdout.
      printf("%s\n", argv[i]);
    }
  }

  // If we launched any threads, join them
  for (std::vector<pthread_t>::iterator it = threads.begin();
       it != threads.end(); ++it) {
    void *thread_retval = nullptr;
    const int err = ::pthread_join(*it, &thread_retval);
    if (err != 0)
      fprintf(stderr, "pthread_join() failed with error code %d\n", err);
  }

  return return_value;
}
