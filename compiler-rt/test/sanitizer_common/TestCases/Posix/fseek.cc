// RUN: %clangxx -O0 -g %s -o %t && %run %t 2>&1 | FileCheck %s
//
// UNSUPPORTED: linux, darwin, solaris

#include <assert.h>
#include <inttypes.h>
#include <stdio.h>

int main(void) {
  printf("fseek\n");

  FILE *fp = fopen("/etc/fstab", "r");
  assert(fp);

  int rv = fseek(fp, 10, SEEK_SET);
  assert(!rv);

  printf("position: %ld\n", ftell(fp));

  rewind(fp);

  printf("position: %ld\n", ftell(fp));

  rv = fseeko(fp, 15, SEEK_SET);
  assert(!rv);

  printf("position: %" PRIuMAX "\n", (uintmax_t)ftello(fp));

  fpos_t pos;
  rv = fgetpos(fp, &pos);
  assert(!rv);

  rewind(fp);

  printf("position: %" PRIuMAX "\n", (uintmax_t)ftello(fp));

  rv = fsetpos(fp, &pos);
  assert(!rv);

  printf("position: %" PRIuMAX "\n", (uintmax_t)ftello(fp));

  rv = fclose(fp);
  assert(!rv);

  // CHECK: fseek
  // CHECK: position: 10
  // CHECK: position: 0
  // CHECK: position: 15
  // CHECK: position: 0
  // CHECK: position: 15

  return 0;
}
