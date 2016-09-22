// RUN: %clangxx -O0 -g %s -lutil -o %t && %run %t
// This test depends on the glibc layout of struct sem_t and checks that we
// don't leave sem_t::private uninitialized.
// UNSUPPORTED: android
#include <features.h>
#include <assert.h>
#include <semaphore.h>
#include <string.h>
#include <stdint.h>

// This condition needs to correspond to __HAVE_64B_ATOMICS macro in glibc.
#if (defined(__x86_64__) || defined(__aarch64__) || defined(__powerpc64__) || \
     defined(__s390x__) || defined(__sparc64__) || defined(__alpha__) || \
     defined(__ia64__) || defined(__m68k__)) && __GLIBC_PREREQ(2, 21)
typedef uint64_t semval_t;
#else
typedef unsigned semval_t;
#endif

void my_sem_init(bool priv, int value, semval_t *a, unsigned char *b) {
  sem_t sem;
  memset(&sem, 0xAB, sizeof(sem));
  sem_init(&sem, priv, value);

  char *p = (char *)&sem;
  memcpy(a, p, sizeof(semval_t));
  memcpy(b, p + sizeof(semval_t), sizeof(char));

  sem_destroy(&sem);
}

int main() {
  semval_t a;
  unsigned char b;

  my_sem_init(false, 42, &a, &b);
  assert(a == 42);
  assert(b != 0xAB);

  my_sem_init(true, 43, &a, &b);
  assert(a == 43);
  assert(b != 0xAB);
}
