#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef DLOPEN_FUNC_DIR
#  include <dlfcn.h>
int8_t (*func)(int8_t) = NULL;
int8_t (*func2)(int8_t) = NULL;
#else
int8_t func(int8_t);
int8_t func2(int8_t);
#endif

struct thread_arg {
  uint64_t buf_size;
  char const *buf;
  uint64_t output;
};

void *thread_fn(void *arg_ptr) {
  struct thread_arg *arg = (struct thread_arg *)arg_ptr;
  for (uint64_t i = 0; i < arg->buf_size; i++) {
    int8_t c = (int8_t)arg->buf[i];
    arg->output += func(c);
    arg->output += func2(c);
  }
  return NULL;
}

int main() {
#define n_threads 10
#define len 40000

#ifdef DLOPEN_FUNC_DIR
  const char *dynlib_path = DLOPEN_FUNC_DIR "/func.shared";
  const char *dynlib_sym = "func";
  void *handle = dlopen(dynlib_path, RTLD_LAZY);
  if (handle == NULL) {
    fprintf(stderr, "dlopen error on: %s: %s\n", dynlib_path, dlerror());
    return EXIT_FAILURE;
  }

  func = dlsym(handle, dynlib_sym);
  if (func == NULL) {
    fprintf(stderr, "dlsym error on: %s : %s\n", dynlib_path, dynlib_sym);
    return EXIT_FAILURE;
  }

  const char *dynlib_path2 = DLOPEN_FUNC_DIR "/func2.shared";
  const char *dynlib_sym2 = "func2";
  void *handle2 = dlopen(dynlib_path2, RTLD_LAZY);
  if (handle2 == NULL) {
    fprintf(stderr, "dlopen error on: %s: %s\n", dynlib_path2, dlerror());
    return EXIT_FAILURE;
  }

  func2 = dlsym(handle2, dynlib_sym2);
  if (func2 == NULL) {
    fprintf(stderr, "dlsym error on: %s : %s\n", dynlib_path2, dynlib_sym2);
    return EXIT_FAILURE;
  }
#endif

  pthread_t threads[n_threads] = {0};
  struct thread_arg args[n_threads] = {0};
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

  for (uint64_t i = 0; i < n_threads; i++) {
    struct thread_arg a = {
        len,
        example_string,
        0,
    };
    args[i] = a;
    if (pthread_create(&threads[i], NULL, thread_fn, &args[i]) != 0) {
      fprintf(stderr, "Failed to spawn thread %lu, exiting\n", i);
      return EXIT_FAILURE;
    }
  }

  int rc = EXIT_SUCCESS;
  for (uint64_t i = 0; i < n_threads; i++) {
    void *retval = NULL;
    if (pthread_join(threads[i], &retval) != 0) {
      printf("Failed to join thread %lu, continuing\n", i);
      rc = EXIT_FAILURE;
    }

    printf("Thread %lu output:\n"
           "output: %lx\n\n",
           i, args[i].output);
  }
  return rc;
}
