// RUN: %clangxx -O0 -g %s -o %t -lmd && %run %t 2>&1 | FileCheck %s

#include <sys/param.h>

#include <assert.h>
#include <md5.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void test1() {
  MD5_CTX ctx;
  uint8_t entropy[] = {0x11, 0x22, 0x33, 0x44, 0x55, 0x66};
  uint8_t digest[MD5_DIGEST_LENGTH];
  size_t entropysz = sizeof(entropy);
  size_t digestsz = sizeof(digest);

  MD5Init(&ctx);
  MD5Update(&ctx, entropy, entropysz);
  MD5Final(digest, &ctx);

  printf("test1: '");
  for (size_t i = 0; i < digestsz; i++)
    printf("%02x", digest[i]);
  printf("'\n");
}

void test2() {
  MD5_CTX ctx;
  uint8_t entropy[] = {0x11, 0x22, 0x33, 0x44, 0x55, 0x66};
  char digest[MD5_DIGEST_STRING_LENGTH];
  size_t entropysz = sizeof(entropy);

  MD5Init(&ctx);
  MD5Update(&ctx, entropy, entropysz);
  char *p = MD5End(&ctx, digest);
  assert(p);

  printf("test2: '%s'\n", digest);
}

void test3() {
  MD5_CTX ctx;
  uint8_t entropy[] = {0x11, 0x22, 0x33, 0x44, 0x55, 0x66};
  size_t entropysz = sizeof(entropy);

  MD5Init(&ctx);
  MD5Update(&ctx, entropy, entropysz);
  char *p = MD5End(&ctx, NULL);
  assert(strlen(p) == MD5_DIGEST_STRING_LENGTH - 1);

  printf("test3: '%s'\n", p);

  free(p);
}

void test4() {
  char digest[MD5_DIGEST_STRING_LENGTH];

  char *p = MD5File("/etc/fstab", digest);
  assert(p == digest);

  printf("test4: '%s'\n", p);
}

void test5() {
  char *p = MD5File("/etc/fstab", NULL);
  assert(strlen(p) == MD5_DIGEST_STRING_LENGTH - 1);

  printf("test5: '%s'\n", p);

  free(p);
}

void test6() {
  uint8_t entropy[] = {0x11, 0x22, 0x33, 0x44, 0x55, 0x66};
  char digest[MD5_DIGEST_STRING_LENGTH];
  size_t entropysz = sizeof(entropy);

  char *p = MD5Data(entropy, entropysz, digest);
  assert(p == digest);

  printf("test6: '%s'\n", p);
}

void test7() {
  uint8_t entropy[] = {0x11, 0x22, 0x33, 0x44, 0x55, 0x66};
  size_t entropysz = sizeof(entropy);

  char *p = MD5Data(entropy, entropysz, NULL);
  assert(strlen(p) == MD5_DIGEST_STRING_LENGTH - 1);

  printf("test7: '%s'\n", p);

  free(p);
}

int main(void) {
  printf("MD5\n");

  test1();
  test2();
  test3();
  test4();
  test5();
  test6();
  test7();

  // CHECK: MD5
  // CHECK: test1: '86e65b1ef4a830af347ac05ab4f0e999'
  // CHECK: test2: '86e65b1ef4a830af347ac05ab4f0e999'
  // CHECK: test3: '86e65b1ef4a830af347ac05ab4f0e999'
  // CHECK: test4: '{{.*}}'
  // CHECK: test5: '{{.*}}'
  // CHECK: test6: '86e65b1ef4a830af347ac05ab4f0e999'
  // CHECK: test7: '86e65b1ef4a830af347ac05ab4f0e999'

  return 0;
}
