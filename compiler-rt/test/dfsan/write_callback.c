// RUN: %clang_dfsan %s -o %t && %run %t | FileCheck %s

// Tests that the custom implementation of write() does writes with or without
// a callback set using dfsan_set_write_callback().
// REQUIRES: stable-runtime

#include <sanitizer/dfsan_interface.h>

#include <assert.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

// Check write callback arguments by having the callback store them in
// the following variables:
static int last_callback_arg_fd;
static const void *last_callback_arg_buf;
static size_t last_callback_arg_count;

// Allow tests to check the number of callbacks made by incrementing
// this count.  When callbacks are verified, the count is reset.
static int count_unverified_callbacks = 0;

// This callbact will be installed using dfsan_set_write_callback()
// in tests below.
static void write_callback(int fd, const void *buf, size_t count) {
  // Do not do anything in this function that might call write().
  count_unverified_callbacks++;

  last_callback_arg_fd = fd;
  last_callback_arg_buf = buf;
  last_callback_arg_count = count;
}

static void write_string_to_stdout(char *string) {
  char *cur = string;
  int bytes_left = strlen(string);
  while (bytes_left > 0) {
    int res = write(fileno(stdout), cur, bytes_left);
    assert (res >= 0);
    cur += res;
    bytes_left -= res;
  }
}

static void test_can_write_without_callback() {
  dfsan_set_write_callback(NULL);
  count_unverified_callbacks = 0;

  char aString[] = "Test that writes work without callback.\n";
  // CHECK: Test that writes work without callback.
  write_string_to_stdout(aString);

  assert(count_unverified_callbacks == 0);
}

static void test_can_write_with_callback() {
  dfsan_set_write_callback(write_callback);

  count_unverified_callbacks = 0;

  char stringWithCallback[] = "Test that writes work with callback.\n";
  // CHECK: Test that writes work with callback.
  write_string_to_stdout(stringWithCallback);

  // Data was written, so at least one call to write() was made.
  // Because a write may not process all the bytes it is passed, there
  // may have been several calls to write().
  assert(count_unverified_callbacks > 0);
  count_unverified_callbacks = 0;

  dfsan_set_write_callback(NULL);

  char stringWithoutCallback[] = "Writes work after the callback is removed.\n";
  // CHECK: Writes work after the callback is removed.
  write_string_to_stdout(stringWithoutCallback);
  assert(count_unverified_callbacks == 0);
}

static void test_failing_write_runs_callback() {
  // Open /dev/null in read-only mode.  Calling write() on fd will fail.
  int fd = open("/dev/null", O_RDONLY);
  assert(fd != -1);

  // Install a callback.
  dfsan_set_write_callback(write_callback);

  // Write to the read-only file handle.  The write will fail, but the callback
  // should still be invoked.
  char aString[] = "This text will fail to be written.\n";
  int len = strlen(aString);
  int write_result = write(fd, aString, len);
  assert(write_result == -1);

  assert(count_unverified_callbacks == 1);
  count_unverified_callbacks = 0;

  assert(fd == last_callback_arg_fd);
  assert(aString == last_callback_arg_buf);
  assert(len == last_callback_arg_count);

  close(fd);
}

int main(int argc, char* argv[]) {
  test_can_write_without_callback();
  test_can_write_with_callback();
  test_failing_write_runs_callback();
}
