// RUN: %check_clang_tidy %s bugprone-setvbuf-stack-buffer %t

typedef unsigned long size_t;
typedef struct FILE FILE;
extern FILE *stdin;

int setvbuf(FILE *stream, char *buf, int mode, size_t size);
void *malloc(size_t size);
void *calloc(size_t count, size_t size);

#define _IOFBF 0
#define _IOLBF 1
#define _IONBF 2
#define BUFSIZ 1024

// Test 1: Stack buffer — should warn.
void stack_buffer(void) {
  char buf[BUFSIZ];
  setvbuf(stdin, buf, _IOFBF, BUFSIZ);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: passing stack-allocated buffer to 'setvbuf'
}

// Test 2: NULL buffer (unbuffered) — no warning.
void null_buffer(void) {
  setvbuf(stdin, (void *)0, _IONBF, 0);
}

// Test 3: malloc'd buffer — no warning.
void malloc_buffer(void) {
  char *buf = (char *)malloc(BUFSIZ);
  setvbuf(stdin, buf, _IOFBF, BUFSIZ);
}

// Test 4: calloc'd buffer — no warning.
void calloc_buffer(void) {
  char *buf = (char *)calloc(1, BUFSIZ);
  setvbuf(stdin, buf, _IOFBF, BUFSIZ);
}

// Test 5: Static buffer — no warning.
void static_buffer(void) {
  static char buf[BUFSIZ];
  setvbuf(stdin, buf, _IOFBF, BUFSIZ);
}

// Test 6: Global buffer — no warning.
char global_buf[BUFSIZ];
void global_buffer(void) {
  setvbuf(stdin, global_buf, _IOFBF, BUFSIZ);
}

// Test 7: Stack buffer via &arr[0] — should warn.
void stack_buffer_addr(void) {
  char buf[BUFSIZ];
  setvbuf(stdin, &buf[0], _IOFBF, BUFSIZ);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: passing stack-allocated buffer to 'setvbuf'
}
