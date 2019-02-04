// RUN: %clangxx -O0 -g %s -o %t && %run %t 2>&1 | FileCheck %s

#include <sys/param.h>

#include <assert.h>
#include <endian.h>
#include <md4.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void test1() {
  MD4_CTX ctx;
  uint8_t entropy[] = {0x11, 0x22, 0x33, 0x44, 0x55, 0x66};
  uint8_t digest[MD4_DIGEST_LENGTH];

  MD4Init(&ctx);
  MD4Update(&ctx, entropy, __arraycount(entropy));
  MD4Final(digest, &ctx);

  printf("test1: '");
  for (size_t i = 0; i < __arraycount(digest); i++)
    printf("%02x", digest[i]);
  printf("'\n");
}

void test2() {
  MD4_CTX ctx;
  uint8_t entropy[] = {0x11, 0x22, 0x33, 0x44, 0x55, 0x66};
  char digest[MD4_DIGEST_STRING_LENGTH];

  MD4Init(&ctx);
  MD4Update(&ctx, entropy, __arraycount(entropy));
  char *p = MD4End(&ctx, digest);
  assert(p == digest);

  printf("test2: '%s'\n", digest);
}

void test3() {
  MD4_CTX ctx;
  uint8_t entropy[] = {0x11, 0x22, 0x33, 0x44, 0x55, 0x66};

  MD4Init(&ctx);
  MD4Update(&ctx, entropy, __arraycount(entropy));
  char *p = MD4End(&ctx, NULL);
  assert(strlen(p) == MD4_DIGEST_STRING_LENGTH - 1);

  printf("test3: '%s'\n", p);

  free(p);
}

void test4() {
  char digest[MD4_DIGEST_STRING_LENGTH];

  char *p = MD4File("/etc/fstab", digest);
  assert(p == digest);

  printf("test4: '%s'\n", p);
}

void test5() {
  char *p = MD4File("/etc/fstab", NULL);
  assert(strlen(p) == MD4_DIGEST_STRING_LENGTH - 1);

  printf("test5: '%s'\n", p);

  free(p);
}

void test6() {
  uint8_t entropy[] = {0x11, 0x22, 0x33, 0x44, 0x55, 0x66};
  char digest[MD4_DIGEST_STRING_LENGTH];

  char *p = MD4Data(entropy, __arraycount(entropy), digest);
  assert(p == digest);

  printf("test6: '%s'\n", p);
}

void test7() {
  uint8_t entropy[] = {0x11, 0x22, 0x33, 0x44, 0x55, 0x66};

  char *p = MD4Data(entropy, __arraycount(entropy), NULL);
  assert(strlen(p) == MD4_DIGEST_STRING_LENGTH - 1);

  printf("test7: '%s'\n", p);

  free(p);
}

int main(void) {
  printf("MD4\n");

  test1();
  test2();
  test3();
  test4();
  test5();
  test6();
  test7();

  // CHECK: MD4
  // CHECK: test1: 'bf78fda2ca35eb7a026bfcdd3d17283d'
  // CHECK: test2: 'bf78fda2ca35eb7a026bfcdd3d17283d'
  // CHECK: test3: 'bf78fda2ca35eb7a026bfcdd3d17283d'
  // CHECK: test4: '{{.*}}'
  // CHECK: test5: '{{.*}}'
  // CHECK: test6: 'bf78fda2ca35eb7a026bfcdd3d17283d'
  // CHECK: test7: 'bf78fda2ca35eb7a026bfcdd3d17283d'

  return 0;
}
