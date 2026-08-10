// RUN: %clangxx -O0 -g %s -o %t && %run %t 2>&1 | FileCheck %s

#include <sys/param.h>

#include <assert.h>
#include <endian.h>
#include <md2.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void test1() {
  MD2_CTX ctx;
  uint8_t entropy[] = {0x11, 0x22, 0x33, 0x44, 0x55, 0x66};
  uint8_t digest[MD2_DIGEST_LENGTH];

  MD2Init(&ctx);
  MD2Update(&ctx, entropy, __arraycount(entropy));
  MD2Final(digest, &ctx);

  printf("test1: '");
  for (size_t i = 0; i < __arraycount(digest); i++)
    printf("%02x", digest[i]);
  printf("'\n");
}

void test2() {
  MD2_CTX ctx;
  uint8_t entropy[] = {0x11, 0x22, 0x33, 0x44, 0x55, 0x66};
  char digest[MD2_DIGEST_STRING_LENGTH];

  MD2Init(&ctx);
  MD2Update(&ctx, entropy, __arraycount(entropy));
  char *p = MD2End(&ctx, digest);
  assert(p == digest);

  printf("test2: '%s'\n", digest);
}

void test3() {
  MD2_CTX ctx;
  uint8_t entropy[] = {0x11, 0x22, 0x33, 0x44, 0x55, 0x66};

  MD2Init(&ctx);
  MD2Update(&ctx, entropy, __arraycount(entropy));
  char *p = MD2End(&ctx, NULL);
  assert(strlen(p) == MD2_DIGEST_STRING_LENGTH - 1);

  printf("test3: '%s'\n", p);

  free(p);
}

void test4() {
  char digest[MD2_DIGEST_STRING_LENGTH];

  char *p = MD2File("/etc/fstab", digest);
  assert(p == digest);

  printf("test4: '%s'\n", p);
}

void test5() {
  char *p = MD2File("/etc/fstab", NULL);
  assert(strlen(p) == MD2_DIGEST_STRING_LENGTH - 1);

  printf("test5: '%s'\n", p);

  free(p);
}

void test6() {
  uint8_t entropy[] = {0x11, 0x22, 0x33, 0x44, 0x55, 0x66};
  char digest[MD2_DIGEST_STRING_LENGTH];

  char *p = MD2Data(entropy, __arraycount(entropy), digest);
  assert(p == digest);

  printf("test6: '%s'\n", p);
}

void test7() {
  uint8_t entropy[] = {0x11, 0x22, 0x33, 0x44, 0x55, 0x66};

  char *p = MD2Data(entropy, __arraycount(entropy), NULL);
  assert(strlen(p) == MD2_DIGEST_STRING_LENGTH - 1);

  printf("test7: '%s'\n", p);

  free(p);
}

int main(void) {
  printf("MD2\n");

  test1();
  test2();
  test3();
  test4();
  test5();
  test6();
  test7();

  // CHECK: MD2
  // CHECK: test1: 'e303e49b34f981c2740cdf809200d51b'
  // CHECK: test2: 'e303e49b34f981c2740cdf809200d51b'
  // CHECK: test3: 'e303e49b34f981c2740cdf809200d51b'
  // CHECK: test4: '{{.*}}'
  // CHECK: test5: '{{.*}}'
  // CHECK: test6: 'e303e49b34f981c2740cdf809200d51b'
  // CHECK: test7: 'e303e49b34f981c2740cdf809200d51b'

  return 0;
}
