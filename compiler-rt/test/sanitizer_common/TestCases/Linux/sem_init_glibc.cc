// RUN: %clangxx -O0 -g %s -lutil -o %t && %run %t
// This test depends on the glibc layout of struct sem_t and checks that we
// don't leave sem_t::private uninitialized.
// UNSUPPORTED: android, lsan-x86, ubsan
#include <features.h>
#include <assert.h>
#include <semaphore.h>
#include <string.h>
#include <stdint.h>

// On powerpc64be semval_t must be 64 bits even with "old" versions of glibc.
#if __PPC64__ && __BIG_ENDIAN__
typedef uint64_t semval_t;

// This condition needs to correspond to __HAVE_64B_ATOMICS macro in glibc.
#elif (defined(__x86_64__) || defined(__aarch64__) || defined(__powerpc64__) || \
     defined(__s390x__) || defined(__sparc64__) || defined(__alpha__) || \
     defined(__ia64__) || defined(__m68k__)) && __GLIBC_PREREQ(2, 21)
typedef uint64_t semval_t;
#else
typedef unsigned semval_t;
#endif

// glibc 2.21 has introduced some changes in the way the semaphore value is
// handled for 32-bit platforms, but since these changes are not ABI-breaking
// they are not versioned. On newer platforms such as ARM, there is only one
// version of the symbol, so it's enough to check the glibc version. However,
// for old platforms such as i386, glibc contains two or even three versions of
// the sem_init symbol, and the sanitizers always pick the oldest one.
// Therefore, it is not enough to rely on the __GLIBC_PREREQ macro - we should
// instead check the platform as well to make sure we only expect the new
// behavior on platforms where the older symbols do not exist.
#if defined(__arm__) && __GLIBC_PREREQ(2, 21)
#define GET_SEM_VALUE(V) ((V) >> 1)
#else
#define GET_SEM_VALUE(V) (V)
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
