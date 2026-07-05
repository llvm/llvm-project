// RUN: %clangxx -O0 -g %s -o %t && %run %t 2>&1 | FileCheck %s

#include <assert.h>
#include <sha1.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void test1() {
  SHA1_CTX ctx;
  uint8_t entropy[] = { 0x11, 0x22, 0x33, 0x44, 0x55, 0x66 };
  uint8_t digest[SHA1_DIGEST_LENGTH];

  SHA1Init(&ctx);
  SHA1Update(&ctx, entropy, __arraycount(entropy));
  SHA1Final(digest, &ctx);

  printf("test1: '");
  for (size_t i = 0; i < __arraycount(digest); i++)
    printf("%02x", digest[i]);
  printf("'\n");
}

void local_SHA1Update(SHA1_CTX *context, const uint8_t *data, unsigned int len)
{
    unsigned int a, b;

    b = context->count[0];
    context->count[0] += len << 3;
    if (context->count[0] < b)
        context->count[1] += (len >> 29) + 1;
    b = (b >> 3) & 63;
    if ((b + len) > 63) {
        memcpy(&context->buffer[b], data, (a = 64 - b));
        SHA1Transform(context->state, context->buffer);
        for ( ; a + 63 < len; a += 64)
            SHA1Transform(context->state, &data[a]);
        b = 0;
    } else {
        a = 0;
    }
    memcpy(&context->buffer[b], &data[a], len - a);
}

void test2() {
  SHA1_CTX ctx;
  uint8_t entropy[] = { 0x11, 0x22, 0x33, 0x44, 0x55, 0x66 };
  uint8_t digest[SHA1_DIGEST_LENGTH];

  SHA1Init(&ctx);
  local_SHA1Update(&ctx, entropy, __arraycount(entropy));
  SHA1Final(digest, &ctx);

  printf("test2: '");
  for (size_t i = 0; i < __arraycount(digest); i++)
    printf("%02x", digest[i]);
  printf("'\n");
}

void test3() {
  SHA1_CTX ctx;
  uint8_t entropy[] = { 0x11, 0x22, 0x33, 0x44, 0x55, 0x66 };
  char digest[SHA1_DIGEST_STRING_LENGTH];

  SHA1Init(&ctx);
  SHA1Update(&ctx, entropy, __arraycount(entropy));
  char *p = SHA1End(&ctx, digest);
  assert(p == digest);

  printf("test3: '%s'\n", digest);
}

void test4() {
  SHA1_CTX ctx;
  uint8_t entropy[] = { 0x11, 0x22, 0x33, 0x44, 0x55, 0x66 };

  SHA1Init(&ctx);
  SHA1Update(&ctx, entropy, __arraycount(entropy));
  char *p = SHA1End(&ctx, NULL);
  assert(strlen(p) == SHA1_DIGEST_STRING_LENGTH - 1);

  printf("test4: '%s'\n", p);

  free(p);
}

void test5() {
  char digest[SHA1_DIGEST_STRING_LENGTH];

  char *p = SHA1File("/etc/fstab", digest);
  assert(p == digest);

  printf("test5: '%s'\n", p);
}

void test6() {
  char *p = SHA1File("/etc/fstab", NULL);
  assert(strlen(p) == SHA1_DIGEST_STRING_LENGTH - 1);

  printf("test6: '%s'\n", p);

  free(p);
}

void test7() {
  char digest[SHA1_DIGEST_STRING_LENGTH];

  char *p = SHA1FileChunk("/etc/fstab", digest, 10, 20);
  assert(p == digest);

  printf("test7: '%s'\n", p);
}

void test8() {
  char *p = SHA1FileChunk("/etc/fstab", NULL, 10, 20);
  assert(strlen(p) == SHA1_DIGEST_STRING_LENGTH - 1);

  printf("test8: '%s'\n", p);

  free(p);
}

void test9() {
  uint8_t entropy[] = { 0x11, 0x22, 0x33, 0x44, 0x55, 0x66 };
  char digest[SHA1_DIGEST_STRING_LENGTH];

  char *p = SHA1Data(entropy, __arraycount(entropy), digest);
  assert(p == digest);

  printf("test9: '%s'\n", p);
}

void test10() {
  uint8_t entropy[] = { 0x11, 0x22, 0x33, 0x44, 0x55, 0x66 };

  char *p = SHA1Data(entropy, __arraycount(entropy), NULL);
  assert(strlen(p) == SHA1_DIGEST_STRING_LENGTH - 1);

  printf("test10: '%s'\n", p);

  free(p);
}

int main(void) {
  printf("SHA1\n");

  test1();
  test2();
  test3();
  test4();
  test5();
  test6();
  test7();
  test8();
  test9();
  test10();

  // CHECK: SHA1
  // CHECK: test1: '57d1b759bf3d1811135748cb0328c73b51fa6f57'
  // CHECK: test2: '57d1b759bf3d1811135748cb0328c73b51fa6f57'
  // CHECK: test3: '57d1b759bf3d1811135748cb0328c73b51fa6f57'
  // CHECK: test4: '57d1b759bf3d1811135748cb0328c73b51fa6f57'
  // CHECK: test5: '{{.*}}'
  // CHECK: test6: '{{.*}}'
  // CHECK: test7: '{{.*}}'
  // CHECK: test8: '{{.*}}'
  // CHECK: test9: '57d1b759bf3d1811135748cb0328c73b51fa6f57'
  // CHECK: test10: '57d1b759bf3d1811135748cb0328c73b51fa6f57'

  return 0;
}
