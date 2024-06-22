#include <pthread.h>
#include <semaphore.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void *exit_thread(void *arg_ptr) {
  sem_t *s_p = (sem_t *)arg_ptr;
  printf("Exit thread waiting...\n");
  if (sem_wait(s_p)) {
    fprintf(stderr, "Failed to wait on signal from main thread\n");
    exit(EXIT_FAILURE);
  }
  printf("Exit thread activated\n");
  exit(0);
  return NULL;
}

int main() {
  pthread_t exit;
  sem_t s;
  sem_init(&s, 0, 0);
  if (pthread_create(&exit, NULL, exit_thread, &s) != 0) {
    fprintf(stderr, "Failed to spawn exit thread\n");
    return EXIT_FAILURE;
  }
  if (sem_post(&s)) {
    fprintf(stderr, "Failed to send signal to exit thread\n");
    return EXIT_FAILURE;
  }
  if (pthread_join(exit, NULL)) {
    fprintf(stderr, "Failed to join exit thread\n");
    return EXIT_FAILURE;
  }
  fprintf(stderr, "Child thread should have called exit()\n");
  return EXIT_FAILURE;
}
