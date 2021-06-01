// RUN: %clang_scudo %s -O2 -o %t
// RUN: %env_scudo_opts="QuarantineChunksUpToSize=0" %run %t 2>&1

// This test attempts to reproduce a race condition in the deallocation path
// when bypassing the Quarantine. The old behavior was to zero-out the chunk
// header after checking its checksum, state & various other things, but that
// left a window during which 2 (or more) threads could deallocate the same
// chunk, with a net result of having said chunk present in those distinct
// thread caches.

// A passing test means all the children died with an error. The failing
// scenario involves winning a race, so repro can be scarce.

#include <pthread.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

const int kNumThreads = 2;
pthread_t tid[kNumThreads];

pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
char go = 0;

// Frees the pointer passed when signaled to.
void *thread_free(void *p) {
  pthread_mutex_lock(&mutex);
  while (!go)
    pthread_cond_wait(&cond, &mutex);
  pthread_mutex_unlock(&mutex);
  free(p);
  return 0;
}

// Allocates a chunk, and attempts to free it "simultaneously" by 2 threads.
void child(void) {
  void *p = malloc(16);
  for (int i = 0; i < kNumThreads; i++)
    pthread_create(&tid[i], 0, thread_free, p);
  pthread_mutex_lock(&mutex);
  go = 1;
  pthread_cond_broadcast(&cond);
  pthread_mutex_unlock(&mutex);
  for (int i = 0; i < kNumThreads; i++)
    pthread_join(tid[i], 0);
}

int main(int argc, char **argv) {
  const int kChildren = 40;
  pid_t pid;
  for (int i = 0; i < kChildren; ++i) {
    pid = fork();
    if (pid < 0) {
      exit(1);
    } else if (pid == 0) {
      child();
      exit(0);
    } else {
      int status;
      wait(&status);
      // A 0 status means the child didn't die with an error. The race was won.
      if (status == 0)
        exit(1);
    }
  }
  return 0;
}
