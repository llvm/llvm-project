// RUN: %clangxx -O0 -g %s -o %t && %run %t 2>&1 | FileCheck %s

#include <assert.h>
#include <rmd160.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void test1() {
  RMD160_CTX ctx;
  uint8_t entropy[] = { 0x11, 0x22, 0x33, 0x44, 0x55, 0x66 };
  uint8_t digest[RMD160_DIGEST_LENGTH];

  RMD160Init(&ctx);
  RMD160Update(&ctx, entropy, __arraycount(entropy));
  RMD160Final(digest, &ctx);

  printf("test1: '");
  for (size_t i = 0; i < __arraycount(digest); i++)
    printf("%02x", digest[i]);
  printf("'\n");
}

void test2() {
  RMD160_CTX ctx;
  uint8_t entropy[] = { 0x11, 0x22, 0x33, 0x44, 0x55, 0x66 };
  char digest[RMD160_DIGEST_STRING_LENGTH];

  RMD160Init(&ctx);
  RMD160Update(&ctx, entropy, __arraycount(entropy));
  char *p = RMD160End(&ctx, digest);
  assert(p == digest);

  printf("test2: '%s'\n", digest);
}

void test3() {
  RMD160_CTX ctx;
  uint8_t entropy[] = { 0x11, 0x22, 0x33, 0x44, 0x55, 0x66 };

  RMD160Init(&ctx);
  RMD160Update(&ctx, entropy, __arraycount(entropy));
  char *p = RMD160End(&ctx, NULL);
  assert(strlen(p) == RMD160_DIGEST_STRING_LENGTH - 1);

  printf("test3: '%s'\n", p);

  free(p);
}

void test4() {
  char digest[RMD160_DIGEST_STRING_LENGTH];

  char *p = RMD160File("/etc/fstab", digest);
  assert(p == digest);

  printf("test4: '%s'\n", p);
}

void test5() {
  char *p = RMD160File("/etc/fstab", NULL);
  assert(strlen(p) == RMD160_DIGEST_STRING_LENGTH - 1);

  printf("test5: '%s'\n", p);

  free(p);
}

void test6() {
  char digest[RMD160_DIGEST_STRING_LENGTH];

  char *p = RMD160FileChunk("/etc/fstab", digest, 10, 20);
  assert(p == digest);

  printf("test6: '%s'\n", p);
}

void test7() {
  char *p = RMD160FileChunk("/etc/fstab", NULL, 10, 20);
  assert(strlen(p) == RMD160_DIGEST_STRING_LENGTH - 1);

  printf("test7: '%s'\n", p);

  free(p);
}

void test8() {
  uint8_t entropy[] = { 0x11, 0x22, 0x33, 0x44, 0x55, 0x66 };
  char digest[RMD160_DIGEST_STRING_LENGTH];

  char *p = RMD160Data(entropy, __arraycount(entropy), digest);
  assert(p == digest);

  printf("test8: '%s'\n", p);
}

void test9() {
  uint8_t entropy[] = { 0x11, 0x22, 0x33, 0x44, 0x55, 0x66 };

  char *p = RMD160Data(entropy, __arraycount(entropy), NULL);
  assert(strlen(p) == RMD160_DIGEST_STRING_LENGTH - 1);

  printf("test9: '%s'\n", p);

  free(p);
}

int main(void) {
  printf("RMD160\n");

  test1();
  test2();
  test3();
  test4();
  test5();
  test6();
  test7();
  test8();
  test9();

  // CHECK: RMD160
  // CHECK: test1: '2787e5a006365df6e8e799315b669dc34866783c'
  // CHECK: test2: '2787e5a006365df6e8e799315b669dc34866783c'
  // CHECK: test3: '2787e5a006365df6e8e799315b669dc34866783c'
  // CHECK: test4: '{{.*}}'
  // CHECK: test5: '{{.*}}'
  // CHECK: test6: '{{.*}}'
  // CHECK: test7: '{{.*}}'
  // CHECK: test8: '2787e5a006365df6e8e799315b669dc34866783c'
  // CHECK: test9: '2787e5a006365df6e8e799315b669dc34866783c'

  return 0;
}
