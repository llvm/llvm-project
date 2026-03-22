// RUN: %check_clang_tidy %s bugprone-unsafe-api-functions-calls %t

#include <stdlib.h>
#include <stdio.h>

// === setvbuf tests ===

// Test 1: Stack buffer - should warn.
void setvbuf_stack_buffer(void) {
  char buf[BUFSIZ];
  setvbuf(stdin, buf, _IOFBF, BUFSIZ);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: passing stack-allocated buffer to 'setvbuf'
}

// Test 2: NULL buffer (unbuffered) - no warning.
void setvbuf_null_buffer(void) {
  setvbuf(stdin, (void *)0, _IONBF, 0);
  setvbuf(stdin, NULL, _IONBF, 0);
  setvbuf(stdin, 0, _IONBF, 0);
}

// Test 3: malloc'd buffer - no warning.
void setvbuf_malloc_buffer(void) {
  char *buf = (char *)malloc(BUFSIZ);
  setvbuf(stdin, buf, _IOFBF, BUFSIZ);
}

// Test 4: calloc'd buffer - no warning.
void setvbuf_calloc_buffer(void) {
  char *buf = (char *)calloc(1, BUFSIZ);
  setvbuf(stdin, buf, _IOFBF, BUFSIZ);
}

// Test 5: Static buffer - no warning.
void setvbuf_static_buffer(void) {
  static char buf[BUFSIZ];
  setvbuf(stdin, buf, _IOFBF, BUFSIZ);
}

// Test 6: Global buffer - no warning.
char global_buf[BUFSIZ];
void setvbuf_global_buffer(void) {
  setvbuf(stdin, global_buf, _IOFBF, BUFSIZ);
}

// Test 7: Stack buffer via &arr[0] - should warn.
void setvbuf_stack_buffer_addr(void) {
  char buf[BUFSIZ];
  setvbuf(stdin, &buf[0], _IOFBF, BUFSIZ);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: passing stack-allocated buffer to 'setvbuf'
}

// === setbuf tests ===

// Test 8: Stack buffer with setbuf - should warn.
void setbuf_stack_buffer(void) {
  char buf[BUFSIZ];
  setbuf(stdin, buf);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: passing stack-allocated buffer to 'setbuf'
}

// Test 9: NULL buffer with setbuf (disable buffering) - no warning.
void setbuf_null_buffer(void) {
  setbuf(stdin, (void *)0);
  setbuf(stdin, 0);
  setbuf(stdin, NULL);
}

// Test 10: malloc'd buffer with setbuf - no warning.
void setbuf_malloc_buffer(void) {
  char *buf = (char*)malloc(BUFSIZ);
  setbuf(stdin, buf);
}

// Test 11: Static buffer with setbuf - no warning.
void setbuf_static_buffer(void) {
  static char buf[BUFSIZ];
  setbuf(stdin, buf);
}

// Test 12: Global buffer with setbuf - no warning.
void setbuf_global_buffer(void) {
  setbuf(stdin, global_buf);
}

// Test 13: Stack buffer via &arr[0] with setbuf - should warn.
void setbuf_stack_buffer_addr(void) {
  char buf[BUFSIZ];
  setbuf(stdin, &buf[0]);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: passing stack-allocated buffer to 'setbuf'
}
