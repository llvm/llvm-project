// RUN: %clangxx -O0 -g %s -DSHASIZE=224 -o %t && %run %t 2>&1 | FileCheck %s -check-prefix=CHECK-224
// RUN: %clangxx -O0 -g %s -DSHASIZE=256 -o %t && %run %t 2>&1 | FileCheck %s -check-prefix=CHECK-256
// RUN: %clangxx -O0 -g %s -DSHASIZE=384 -o %t && %run %t 2>&1 | FileCheck %s -check-prefix=CHECK-384
// RUN: %clangxx -O0 -g %s -DSHASIZE=512 -o %t && %run %t 2>&1 | FileCheck %s -check-prefix=CHECK-512

#include <sys/param.h>

#include <assert.h>
#include <endian.h>
#include <sha2.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef SHASIZE
#error SHASIZE must be defined
#endif

#define _SHA_CTX(x) SHA##x##_CTX
#define SHA_CTX(x) _SHA_CTX(x)

#define _SHA_DIGEST_LENGTH(x) SHA##x##_DIGEST_LENGTH
#define SHA_DIGEST_LENGTH(x) _SHA_DIGEST_LENGTH(x)

#define _SHA_DIGEST_STRING_LENGTH(x) SHA##x##_DIGEST_STRING_LENGTH
#define SHA_DIGEST_STRING_LENGTH(x) _SHA_DIGEST_STRING_LENGTH(x)

#define _SHA_Init(x) SHA##x##_Init
#define SHA_Init(x) _SHA_Init(x)

#define _SHA_Update(x) SHA##x##_Update
#define SHA_Update(x) _SHA_Update(x)

#define _SHA_Final(x) SHA##x##_Final
#define SHA_Final(x) _SHA_Final(x)

#define _SHA_End(x) SHA##x##_End
#define SHA_End(x) _SHA_End(x)

#define _SHA_File(x) SHA##x##_File
#define SHA_File(x) _SHA_File(x)

#define _SHA_FileChunk(x) SHA##x##_FileChunk
#define SHA_FileChunk(x) _SHA_FileChunk(x)

#define _SHA_Data(x) SHA##x##_Data
#define SHA_Data(x) _SHA_Data(x)

void test1() {
  SHA_CTX(SHASIZE) ctx;
  uint8_t entropy[] = {0x11, 0x22, 0x33, 0x44, 0x55, 0x66};
  uint8_t digest[SHA_DIGEST_LENGTH(SHASIZE)];

  SHA_Init(SHASIZE)(&ctx);
  SHA_Update(SHASIZE)(&ctx, entropy, __arraycount(entropy));
  SHA_Final(SHASIZE)(digest, &ctx);

  printf("test1: '");
  for (size_t i = 0; i < __arraycount(digest); i++)
    printf("%02x", digest[i]);
  printf("'\n");
}

void test2() {
  SHA_CTX(SHASIZE) ctx;
  uint8_t entropy[] = {0x11, 0x22, 0x33, 0x44, 0x55, 0x66};
  char digest[SHA_DIGEST_STRING_LENGTH(SHASIZE)];

  SHA_Init(SHASIZE)(&ctx);
  SHA_Update(SHASIZE)(&ctx, entropy, __arraycount(entropy));
  char *p = SHA_End(SHASIZE)(&ctx, digest);
  assert(p == digest);

  printf("test2: '%s'\n", digest);
}

void test3() {
  SHA_CTX(SHASIZE) ctx;
  uint8_t entropy[] = {0x11, 0x22, 0x33, 0x44, 0x55, 0x66};

  SHA_Init(SHASIZE)(&ctx);
  SHA_Update(SHASIZE)(&ctx, entropy, __arraycount(entropy));
  char *p = SHA_End(SHASIZE)(&ctx, NULL);
  assert(strlen(p) == SHA_DIGEST_STRING_LENGTH(SHASIZE) - 1);

  printf("test3: '%s'\n", p);

  free(p);
}

void test4() {
  char digest[SHA_DIGEST_STRING_LENGTH(SHASIZE)];

  char *p = SHA_File(SHASIZE)("/etc/fstab", digest);
  assert(p == digest);

  printf("test4: '%s'\n", p);
}

void test5() {
  char *p = SHA_File(SHASIZE)("/etc/fstab", NULL);
  assert(strlen(p) == SHA_DIGEST_STRING_LENGTH(SHASIZE) - 1);

  printf("test5: '%s'\n", p);

  free(p);
}

void test6() {
  char digest[SHA_DIGEST_STRING_LENGTH(SHASIZE)];

  char *p = SHA_FileChunk(SHASIZE)("/etc/fstab", digest, 10, 20);
  assert(p == digest);

  printf("test6: '%s'\n", p);
}

void test7() {
  char *p = SHA_FileChunk(SHASIZE)("/etc/fstab", NULL, 10, 20);
  assert(strlen(p) == SHA_DIGEST_STRING_LENGTH(SHASIZE) - 1);

  printf("test7: '%s'\n", p);

  free(p);
}

void test8() {
  uint8_t entropy[] = {0x11, 0x22, 0x33, 0x44, 0x55, 0x66};
  char digest[SHA_DIGEST_STRING_LENGTH(SHASIZE)];

  char *p = SHA_Data(SHASIZE)(entropy, __arraycount(entropy), digest);
  assert(p == digest);

  printf("test8: '%s'\n", p);
}

void test9() {
  uint8_t entropy[] = {0x11, 0x22, 0x33, 0x44, 0x55, 0x66};

  char *p = SHA_Data(SHASIZE)(entropy, __arraycount(entropy), NULL);
  assert(strlen(p) == SHA_DIGEST_STRING_LENGTH(SHASIZE) - 1);

  printf("test9: '%s'\n", p);

  free(p);
}

int main(void) {
  printf("SHA" ___STRING(SHASIZE) "\n");

  test1();
  test2();
  test3();
  test4();
  test5();
  test6();
  test7();
  test8();
  test9();

  // CHECK-224: SHA224
  // CHECK-224: test1: '760dfb93100a6bf5996c90f678e529dc945bb2f74a211eedcf0f3a48'
  // CHECK-224: test2: '760dfb93100a6bf5996c90f678e529dc945bb2f74a211eedcf0f3a48'
  // CHECK-224: test3: '760dfb93100a6bf5996c90f678e529dc945bb2f74a211eedcf0f3a48'
  // CHECK-224: test4: '{{.*}}'
  // CHECK-224: test5: '{{.*}}'
  // CHECK-224: test6: '{{.*}}'
  // CHECK-224: test7: '{{.*}}'
  // CHECK-224: test8: '760dfb93100a6bf5996c90f678e529dc945bb2f74a211eedcf0f3a48'
  // CHECK-224: test9: '760dfb93100a6bf5996c90f678e529dc945bb2f74a211eedcf0f3a48'

  // CHECK-256: SHA256
  // CHECK-256: test1: 'bb000ddd92a0a2a346f0b531f278af06e370f86932ccafccc892d68d350f80f8'
  // CHECK-256: test2: 'bb000ddd92a0a2a346f0b531f278af06e370f86932ccafccc892d68d350f80f8'
  // CHECK-256: test3: 'bb000ddd92a0a2a346f0b531f278af06e370f86932ccafccc892d68d350f80f8'
  // CHECK-256: test4: '{{.*}}'
  // CHECK-256: test5: '{{.*}}'
  // CHECK-256: test6: '{{.*}}'
  // CHECK-256: test7: '{{.*}}'
  // CHECK-256: test8: 'bb000ddd92a0a2a346f0b531f278af06e370f86932ccafccc892d68d350f80f8'
  // CHECK-256: test9: 'bb000ddd92a0a2a346f0b531f278af06e370f86932ccafccc892d68d350f80f8'

  // CHECK-384: SHA384
  // CHECK-384: test1: 'f450c023b168ebd56ff916ca9b1f1f0010b8c592d28205cc91fa3056f629eed108e8bac864f01ca37a3edee596739e12'
  // CHECK-384: test2: 'f450c023b168ebd56ff916ca9b1f1f0010b8c592d28205cc91fa3056f629eed108e8bac864f01ca37a3edee596739e12'
  // CHECK-384: test3: 'f450c023b168ebd56ff916ca9b1f1f0010b8c592d28205cc91fa3056f629eed108e8bac864f01ca37a3edee596739e12'
  // CHECK-384: test4: '{{.*}}'
  // CHECK-384: test5: '{{.*}}'
  // CHECK-384: test6: '{{.*}}'
  // CHECK-384: test7: '{{.*}}'
  // CHECK-384: test8: 'f450c023b168ebd56ff916ca9b1f1f0010b8c592d28205cc91fa3056f629eed108e8bac864f01ca37a3edee596739e12'
  // CHECK-384: test9: 'f450c023b168ebd56ff916ca9b1f1f0010b8c592d28205cc91fa3056f629eed108e8bac864f01ca37a3edee596739e12'

  // CHECK-512: SHA512
  // CHECK-512: test1: '0e3f68731c0e2a6a4eab5d713c9a80dc78086b5fa7d2b5ab127277958e68d1b1dee1882b083b0106cd4319de42c0c8f452871364f5baa8a6379690612c6b844e'
  // CHECK-512: test2: '0e3f68731c0e2a6a4eab5d713c9a80dc78086b5fa7d2b5ab127277958e68d1b1dee1882b083b0106cd4319de42c0c8f452871364f5baa8a6379690612c6b844e'
  // CHECK-512: test3: '0e3f68731c0e2a6a4eab5d713c9a80dc78086b5fa7d2b5ab127277958e68d1b1dee1882b083b0106cd4319de42c0c8f452871364f5baa8a6379690612c6b844e'
  // CHECK-512: test4: '{{.*}}'
  // CHECK-512: test5: '{{.*}}'
  // CHECK-512: test6: '{{.*}}'
  // CHECK-512: test7: '{{.*}}'
  // CHECK-512: test8: '0e3f68731c0e2a6a4eab5d713c9a80dc78086b5fa7d2b5ab127277958e68d1b1dee1882b083b0106cd4319de42c0c8f452871364f5baa8a6379690612c6b844e'
  // CHECK-512: test9: '0e3f68731c0e2a6a4eab5d713c9a80dc78086b5fa7d2b5ab127277958e68d1b1dee1882b083b0106cd4319de42c0c8f452871364f5baa8a6379690612c6b844e'

  return 0;
}
