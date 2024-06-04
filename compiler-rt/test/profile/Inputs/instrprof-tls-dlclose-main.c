#include <dlfcn.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

struct thread_arg {
  uint64_t buf_size;
  char const *buf;
  uint64_t iteration_counter;
  uint64_t output;
};

#ifndef DLOPEN_FUNC_DIR
unsigned char determine_value_dyn(unsigned char);
#endif

void *thread_fn(void *arg_ptr) {
#ifdef DLOPEN_FUNC_DIR

  unsigned char (*determine_value_dyn)(unsigned char) = NULL;

  const char *dynlib_name = DLOPEN_FUNC_DIR "/lib.shared";
  const char *dynlib_sym = "determine_value_dyn";
  void *handle = dlopen(dynlib_name, DLOPEN_FLAGS);
  if (handle == NULL) {
    fprintf(stderr, "dlopen error on: %s: %s\n", dynlib_name, dlerror());
    exit(EXIT_FAILURE);
  }

  determine_value_dyn = dlsym(handle, dynlib_sym);
  if (handle == NULL) {
    fprintf(stderr, "dlsym error on: %s : %s\n", dynlib_name, dynlib_sym);
    exit(EXIT_FAILURE);
  }
#endif

  struct thread_arg *arg = (struct thread_arg *)arg_ptr;
  for (uint64_t i = 0; i < arg->buf_size; i++) {
    unsigned char c = (unsigned char)arg->buf[i];
    arg->output += determine_value_dyn(c);
    arg->iteration_counter++;
  }

  // This should unload the thread local counters region for this module,
  // causing an expected failure for -fprofile-thread-local
#ifdef DLOPEN_FUNC_DIR
#  ifndef DONT_CLOSE
  dlclose(handle);
#  endif
#endif
  return NULL;
}

int main() {
  const uint64_t len = 40000;

  char *example_string = (char *)malloc(sizeof(char) * len);
  int high = 0;
  for (uint64_t i = 0; i < len; i++) {
    if (high == 2) {
      example_string[i] = 0xff;
      high = 0;
    } else {
      example_string[i] = 0x0;
      high++;
    }
  }

  pthread_t thread;
  struct thread_arg arg = {
      len,
      example_string,
      0,
      0,
  };
  if (pthread_create(&thread, NULL, thread_fn, &arg) != 0) {
    fprintf(stderr, "Failed to spawn thread, exiting\n");
    exit(EXIT_SUCCESS);
  }

  if (pthread_join(thread, NULL) != 0) {
    fprintf(stderr, "Failed to join thread, continuing\n");
    return EXIT_FAILURE;
  }

  printf("Thread output:\n"
         "iteration_counter: %lu\n"
         "output: %lx\n\n",
         arg.iteration_counter, arg.output);

  return EXIT_SUCCESS;
}
