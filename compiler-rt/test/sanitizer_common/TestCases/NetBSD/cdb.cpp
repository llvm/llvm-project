// RUN: %clangxx -O0 -g %s -o %t && %run %t 2>&1 | FileCheck %s

#include <sys/param.h>

#include <sys/types.h>

#include <sys/mman.h>
#include <sys/stat.h>

#include <assert.h>
#include <cdbr.h>
#include <cdbw.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static char *name;

const char data1[] = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07};
const char data2[] = {0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17};
const char key1[] = {0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27};
const char key2[] = {0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37};

void test_cdbw() {
  uint32_t idx;

  struct cdbw *cdbw = cdbw_open();
  assert(cdbw);

  int rv = cdbw_put_data(cdbw, data1, __arraycount(data1), &idx);
  assert(!rv);

  rv = cdbw_put_key(cdbw, key1, __arraycount(key1), idx);
  assert(!rv);

  rv = cdbw_put(cdbw, key2, __arraycount(key2), data2, __arraycount(data2));
  assert(!rv);

  name = strdup("/tmp/temp.XXXXXX");
  assert(name);

  name = mktemp(name);
  assert(name);

  int fd = open(name, O_RDWR | O_CREAT, 0644);
  assert(fd != -1);

  cdbw_output(cdbw, fd, "TEST1", cdbw_stable_seeder);

  cdbw_close(cdbw);

  rv = close(fd);
  assert(rv != -1);
}

void test_cdbr1() {
  struct cdbr *cdbr = cdbr_open(name, CDBR_DEFAULT);
  assert(cdbr);

  uint32_t idx = cdbr_entries(cdbr);
  assert(idx > 0);
  printf("entries: %" PRIu32 "\n", idx);

  const void *data;
  size_t data_len;
  int rv = cdbr_get(cdbr, idx - 1, &data, &data_len);
  assert(rv == 0);

  printf("data: ");
  for (size_t i = 0; i < data_len; i++)
    printf("%02" PRIx8, ((uint8_t *)data)[i]);
  printf("\n");

  rv = cdbr_find(cdbr, key1, __arraycount(key1), &data, &data_len);

  printf("data: ");
  for (size_t i = 0; i < data_len; i++)
    printf("%02" PRIx8, ((uint8_t *)data)[i]);
  printf("\n");

  cdbr_close(cdbr);
}

#define COOKIE ((void *)1)

static void cdbr_unmap(void *cookie, void *base, size_t sz) {
  assert(cookie == COOKIE);
  int rv = munmap(base, sz);
  assert(rv != -1);
}

void test_cdbr2() {
  struct stat sb;

  int fd = open(name, O_RDONLY);
  assert(fd != -1);

  int rv = fstat(fd, &sb);
  assert(rv != -1);

  size_t sz = sb.st_size;
  assert(sz < SSIZE_MAX);

  void *base = mmap(NULL, sz, PROT_READ, MAP_FILE | MAP_SHARED, fd, 0);
  assert(base != MAP_FAILED);

  rv = close(fd);
  assert(rv != -1);

  struct cdbr *cdbr = cdbr_open_mem(base, sz, CDBR_DEFAULT, cdbr_unmap, COOKIE);
  assert(cdbr);

  printf("entries: %" PRIu32 "\n", cdbr_entries(cdbr));

  cdbr_close(cdbr);
}

int main(void) {
  printf("cdb\n");

  test_cdbw();
  test_cdbr1();
  test_cdbr2();

  // CHECK: cdb
  // CHECK: entries: 2
  // CHECK: data: 1011121314151617
  // CHECK: data: 0001020304050607
  // CHECK: entries: 2

  return 0;
}
