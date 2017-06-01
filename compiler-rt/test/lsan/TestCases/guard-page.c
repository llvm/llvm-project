// Check that if LSan finds that SP doesn't point into thread stack (e.g.
// if swapcontext is used), LSan will not hit the guard page.
// RUN: %clang_lsan %s -o %t && %run %t
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <ucontext.h>

pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
int ctxfunc_started = 0;

static void die(const char* msg, int err) {
  if (err == 0)
    err = errno;
  fprintf(stderr, "%s: %s\n", msg, strerror(err));
  exit(EXIT_FAILURE);
}

static void ctxfunc() {
  pthread_mutex_lock(&mutex);
  ctxfunc_started = 1;
  // printf("ctxfunc\n");
  pthread_cond_signal(&cond);
  pthread_mutex_unlock(&mutex);
  // Leave this context alive when the program exits.
  for (;;);
}

static void* thread(void* arg) {
  (void)arg;
  ucontext_t ctx;
  void* stack;

  if (getcontext(&ctx) < 0)
    die("getcontext", 0);
  stack = malloc(1 << 11);
  if (stack == NULL)
    die("malloc", 0);
  ctx.uc_stack.ss_sp = stack;
  ctx.uc_stack.ss_size = 1 << 11;
  makecontext(&ctx, ctxfunc, 0);
  setcontext(&ctx);
  die("setcontext", 0);
  return NULL;
}

int main() {
  pthread_t tid;
  int i;

  pthread_mutex_lock(&mutex);
  i = pthread_create(&tid, NULL, thread, NULL);
  if (i != 0)
    die("pthread_create", i);
  while (!ctxfunc_started) pthread_cond_wait(&cond, &mutex);
  pthread_mutex_unlock(&mutex);
  return 0;
}
