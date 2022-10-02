// RUN: %clangxx -O0 -g %s -lutil -o %t && %run %t
// This test depends on the glibc layout of struct sem_t and checks that we
// don't leave sem_t::private uninitialized.
// UNSUPPORTED: android, lsan-x86, ubsan
#include <features.h>
#include <assert.h>
#include <semaphore.h>
#include <string.h>
#include <stdint.h>

// musl and glibc's __HAVE_64B_ATOMICS==0 ports (e.g. arm, i386) use 32-bit sem
// values. 64-bit glibc ports defining sem_init@GLIBC_2.0 (mips64) use 32-bit as
// well, if the sem_init interceptor picks the oldest versioned symbol
// (glibc<2.36, see https://sourceware.org/PR14932).
#if !defined(__GLIBC__) || defined(__ILP32__) ||                               \
    !__GLIBC_PREREQ(2, 36) && defined(__mips64__)
typedef unsigned semval_t;
#else
typedef uint64_t semval_t;
#endif

// glibc __HAVE_64B_ATOMICS==0 ports define a sem_init which shifts the value by
// 1 (https://sourceware.org/PR12674 glibc 2.21). The version is picked if
// either glibc>=2.36 or sem_init@GLIBC_2.0 is absent (arm and newer ports).
//
// The __GLIBC_PREREQ check is brittle in that it requires matched
// __GLIBC_PREREQ values for build time and run time.
#if defined(__GLIBC__) && defined(__ILP32__) &&                                \
    (__GLIBC_PREREQ(2, 36) || (__GLIBC_PREREQ(2, 21) && !defined(__i386__) &&  \
                               !defined(__mips__) && !defined(__powerpc__)))
#  define GET_SEM_VALUE(V) ((V) >> 1)
#else
#  define GET_SEM_VALUE(V) (V)
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
  assert(GET_SEM_VALUE(a) == 42);
  assert(b != 0xAB);

  my_sem_init(true, 43, &a, &b);
  assert(GET_SEM_VALUE(a) == 43);
  assert(b != 0xAB);
}
