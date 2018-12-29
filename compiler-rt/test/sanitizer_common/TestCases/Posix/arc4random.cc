// RUN: %clangxx -O0 -g %s -o %t && %run %t 2>&1 | FileCheck %s
//
// UNSUPPORTED: linux, darwin, solaris

#include <cstdlib>
#include <ctime>
#include <cstdio>
#include <inttypes.h>

void print_buf(unsigned char *buf, size_t buflen) {
  printf("buf '");
  for (auto i = 0; i < buflen; i ++)
    printf("%" PRIx8, buf[i]);
  printf("'\n");
}

void test_seed() {
  time_t now = ::time(nullptr);
  arc4random_addrandom((unsigned char *)&now, sizeof(now));
}

void test_arc4random() {
  printf("test_arc4random\n");
  auto i = arc4random();
  print_buf((unsigned char *)&i, sizeof(i));
}

void test_arc4random_uniform() {
  printf("test_arc4random_uniform\n");
  auto i = arc4random_uniform(1024);
  print_buf((unsigned char *)&i, sizeof(i));
}

void test_arc4random_buf10() {
  printf("test_arc4random_buf10\n");
  char buf[10];
  arc4random_stir();
  arc4random_buf(buf, sizeof(buf));
  print_buf((unsigned char *)buf, sizeof(buf));
}

void test_arc4random_buf256() {
  printf("test_arc4random_buf256\n");
  char buf[256];
  arc4random_stir();
  arc4random_buf(buf, sizeof(buf));
  print_buf((unsigned char *)buf, sizeof(buf));
}

int main(void) {
  test_seed();
  test_arc4random();
  test_arc4random_uniform();
  test_arc4random_buf10();
  test_arc4random_buf256();
  return 0;
  // CHECK: test_arc4random
  // CHECK: buf '{{.*}}'
  // CHECK: test_arc4random_uniform
  // CHECK: buf '{{.*}}'
  // CHECK: test_arc4random_buf10
  // CHECK: buf '{{.*}}'
  // CHECK: test_arc4random_buf256
  // CHECK: buf '{{.*}}'
}
