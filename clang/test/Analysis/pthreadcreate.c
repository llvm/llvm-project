//
// Created by MaxSa on 11/14/2024.
//

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>


void* thread_function(void* arg) {
  printf("thread_function start\n");
  return nullptr;
}

int main() {
  pthread_t thread;
  int arg = 42;

  if (pthread_create(&thread, NULL, thread_function, &arg)) {
    perror("pthread_create");
    exit(1);
  }

  if (pthread_join(thread, nullptr)) {
    perror("pthread_join");
    exit(1);
  }

  printf("thread exit\n");
  return 0;
}