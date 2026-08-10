// Test whether mmap'ing profile counters onto an open file is feasible. As
// this involves some platform-specific logic, this test is designed to be a
// minimum viable proof-of-concept: it may be useful when porting the mmap()
// mode to a new platform, but is not in and of itself a test of the profiling
// runtime.

// REQUIRES: darwin, target={{arm64.*}}

// Align counters and data to the maximum expected page size (16K).
// RUN: %clang -g -o %t %s \
// RUN:   -Wl,-sectalign,__DATA,__pcnts,0x4000 \
// RUN:   -Wl,-sectalign,__DATA,__pdata,0x4000

// Create a 'profile' using mmap() and validate it.
// RUN: %run %t create %t.tmpfile
// RUN: %run %t validate %t.tmpfile

#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

__attribute__((section("__DATA,__pcnts"))) int counters[] = {0xbad};
extern int cnts_start __asm("section$start$__DATA$__pcnts");
const size_t cnts_len = 0x4000;

__attribute__((section("__DATA,__pdata"))) int data[] = {1, 2, 3};
extern int data_start __asm("section$start$__DATA$__pdata");
const size_t data_len = sizeof(int) * 3;

int create_tmpfile(char *path) {
  // Create a temp file.
  int fd = open(path, O_RDWR | O_TRUNC | O_CREAT, 0666);
  if (fd == -1) {
    perror("open");
    return EXIT_FAILURE;
  }

  // Grow the file to hold data and counters.
  if (0 != ftruncate(fd, cnts_len + data_len)) {
    perror("ftruncate");
    return EXIT_FAILURE;
  }

  // Write the data first (at offset 0x1000, after the counters).
  if (data_len != pwrite(fd, &data, data_len, cnts_len)) {
    perror("write");
    return EXIT_FAILURE;
  }

  // Map the counters into the file, before the data.
  //
  // Requirements (on Darwin):
  // - &cnts_start must be page-aligned.
  // - The length and offset-into-fd must be page-aligned.
  int *counter_map = (int *)mmap(&cnts_start, cnts_len, PROT_READ | PROT_WRITE,
                                 MAP_FIXED | MAP_SHARED, fd, 0);
  if (counter_map != &cnts_start) {
    perror("mmap");
    return EXIT_FAILURE;
  }

  // Update counters 1..9. These updates should be visible in the file.
  // Expect counter 0 (0xbad), which is not updated, to be zero in the file.
  for (int i = 1; i < 10; ++i)
    counter_map[i] = i;

  // Intentionally do not msync(), munmap(), or close().
  return EXIT_SUCCESS;
}

int validate_tmpfile(char *path) {
  int fd = open(path, O_RDONLY);
  if (fd == -1) {
    perror("open");
    return EXIT_FAILURE;
  }

  // Verify that the file length is: sizeof(counters) + sizeof(data).
  const size_t num_bytes = cnts_len + data_len;
  int buf[num_bytes];
  if (num_bytes != read(fd, &buf, num_bytes)) {
    perror("read");
    return EXIT_FAILURE;
  }

  // Verify the values of counters 1..9 (i.e. that the mmap() worked).
  for (int i = 0; i < 10; ++i) {
    if (buf[i] != i) {
      fprintf(stderr,
          "validate_tmpfile: Expected '%d' at pos=%d, but got '%d' instead.\n",
          i, i, buf[i]);
      return EXIT_FAILURE;
    }
  }

  // Verify that the rest of the counters (after counter 9) are 0.
  const int num_cnts = cnts_len / sizeof(int);
  for (int i = 10; i < num_cnts; ++i) {
    if (buf[i] != 0) {
      fprintf(stderr,
          "validate_tmpfile: Expected '%d' at pos=%d, but got '%d' instead.\n",
          0, i, buf[i]);
      return EXIT_FAILURE;
    }
  }

  // Verify that the data written after the counters is equal to the "data[]"
  // array (i.e. {1, 2, 3}).
  for (int i = num_cnts; i < num_cnts + 3; ++i) {
    if (buf[i] != (i - num_cnts + 1)) {
      fprintf(stderr,
          "validate_tmpfile: Expected '%d' at pos=%d, but got '%d' instead.\n",
          i - num_cnts + 1, i, buf[i]);
      return EXIT_FAILURE;
    }
  }

  // Intentionally do not close().
  return EXIT_SUCCESS;
}

int main(int argc, char **argv) {
  intptr_t cnts_start_int = (intptr_t)&cnts_start;
  intptr_t data_start_int = (intptr_t)&data_start;
  int pagesz = getpagesize();

  if (cnts_start_int % pagesz != 0) {
    fprintf(stderr, "__pcnts is not page-aligned: 0x%lx.\n", cnts_start_int);
    return EXIT_FAILURE;
  }
  if (data_start_int % 0x4000 != 0) {
    fprintf(stderr, "__pdata is not correctly aligned: 0x%lx.\n",
            data_start_int);
    return EXIT_FAILURE;
  }
  if (cnts_start_int + 0x4000 != data_start_int) {
    fprintf(stderr, "__pdata not ordered after __pcnts.\n");
    return EXIT_FAILURE;
  }

  char *action = argv[1];
  char *path = argv[2];
  if (0 == strcmp(action, "create"))
    return create_tmpfile(path);
  else if (0 == strcmp(action, "validate"))
    return validate_tmpfile(path);
  else
    return EXIT_FAILURE;
}
