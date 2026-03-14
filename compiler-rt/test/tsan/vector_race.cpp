// RUN: %clangxx_tsan -O1 %s -o %t && %deflake %run %t 2>&1 | FileCheck %s
#include "test.h"

__m128i data[20];

__m128i load(__m128i *v) {
#if TSAN_VECTORIZE
  return _mm_load_si128(v);
#else
  return *v;
#endif
}

void store(__m128i *v, __m128i a) {
#if TSAN_VECTORIZE
  _mm_store_si128(v, a);
#else
  *v = a;
#endif
}

void *Thread(void *arg);

int main() {
  barrier_init(&barrier, 2);
  pthread_t th;
  pthread_create(&th, NULL, Thread, NULL);
  barrier_wait(&barrier);

  print_address("addr0:", 2, &data[0], &data[0]);
  auto v0 = load(&data[1]);
  store(&data[0], v0);
  // CHECK: addr0:[[ADDR0_0:0x[0-9,a-f]+]] [[ADDR0_1:0x[0-9,a-f]+]]
  // CHECK: WARNING: ThreadSanitizer: data race
  // CHECK:   Write of size 8 at [[ADDR0_0]] by main thread:
  // CHECK:   Previous read of size 8 at [[ADDR0_1]] by thread T1:

  print_address("addr1:", 2, (char *)&data[2] + 8, (char *)&data[2] + 8);
  ((volatile unsigned long long *)(&data[2]))[1] = 42;
  // CHECK: addr1:[[ADDR1_0:0x[0-9,a-f]+]] [[ADDR1_1:0x[0-9,a-f]+]]
  // CHECK: WARNING: ThreadSanitizer: data race
  // CHECK:   Write of size 8 at [[ADDR1_0]] by main thread:
  // CHECK:   Previous read of size 8 at [[ADDR1_1]] by thread T1:

  print_address("addr2:", 2, (char *)&data[4] + 15, (char *)&data[4] + 8);
  ((volatile char *)(&data[4]))[15] = 42;
  // CHECK: addr2:[[ADDR2_0:0x[0-9,a-f]+]] [[ADDR2_1:0x[0-9,a-f]+]]
  // CHECK: WARNING: ThreadSanitizer: data race
  // CHECK:   Write of size 1 at [[ADDR2_0]] by main thread:
  // CHECK:   Previous read of size 8 at [[ADDR2_1]] by thread T1:

  store(&data[12], v0);
  ((volatile unsigned long long *)(&data[14]))[1] = 42;
  ((volatile char *)(&data[16]))[15] = 42;
  barrier_wait(&barrier);
  pthread_join(th, NULL);
  return 0;
}

void *Thread(void *arg) {
  // Use only even indexes so that compiler does not insert memcpy.
  auto v0 = load(&data[0]);
  auto v1 = load(&data[2]);
  auto v2 = load(&data[4]);
  store(&data[6], v0);
  store(&data[8], v1);
  store(&data[10], v2);
  barrier_wait(&barrier);
  barrier_wait(&barrier);

  print_address("addr3:", 2, &data[12], &data[12]);
  store(&data[12], v0);
  // CHECK: addr3:[[ADDR3_0:0x[0-9,a-f]+]] [[ADDR3_1:0x[0-9,a-f]+]]
  // CHECK: WARNING: ThreadSanitizer: data race
  // CHECK:   Write of size 8 at [[ADDR3_0]] by thread T1:
  // CHECK:   Previous write of size 8 at [[ADDR3_1]] by main thread:

  print_address("addr4:", 2, (char *)&data[14] + 8, (char *)&data[14] + 8);
  store(&data[14], v0);
  // CHECK: addr4:[[ADDR4_0:0x[0-9,a-f]+]] [[ADDR4_1:0x[0-9,a-f]+]]
  // CHECK: WARNING: ThreadSanitizer: data race
  // CHECK:   Write of size 8 at [[ADDR4_0]] by thread T1:
  // CHECK:   Previous write of size 8 at [[ADDR4_1]] by main thread:

  print_address("addr5:", 2, (char *)&data[16] + 8, (char *)&data[16] + 15);
  store(&data[16], v0);
  // CHECK: addr5:[[ADDR5_0:0x[0-9,a-f]+]] [[ADDR5_1:0x[0-9,a-f]+]]
  // CHECK: WARNING: ThreadSanitizer: data race
  // CHECK:   Write of size 8 at [[ADDR5_0]] by thread T1:
  // CHECK:   Previous write of size 1 at [[ADDR5_1]] by main thread:
  return NULL;
}
