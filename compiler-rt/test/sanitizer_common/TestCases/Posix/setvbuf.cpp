// RUN: %clangxx -O0 -g %s -o %t && %run %t 2>&1 | FileCheck %s

// UNSUPPORTED: solaris

#include <stdio.h>

void print_something() {
  for (size_t i = 0; i < 10 * BUFSIZ; i++)
    printf("Hello world %zu\n", i);
}

void print_one_byte(char *buf) {
  printf("First byte is %c\n", buf[0]);
}

void test_setbuf() {
  char buf[BUFSIZ];

  setbuf(stdout, NULL);

  print_something();

  setbuf(stdout, buf);

  print_something();

  print_one_byte(buf);

  setbuf(stdout, NULL);
}

void test_setbuffer() {
  char buf[BUFSIZ];

  setbuffer(stdout, NULL, 0);

  print_something();

  // Ensure that interceptor reads correct size
  // (not BUFSIZ as by default, hence BUFSIZ/2).
  setbuffer(stdout, buf, BUFSIZ / 2);

  print_something();

  print_one_byte(buf);

  setbuffer(stdout, NULL, 0);
}

void test_setlinebuf() {
  setlinebuf(stdout);

  print_something();
}

void test_setvbuf() {
  char buf[BUFSIZ];

  setvbuf(stdout, NULL, _IONBF, 0);

  print_something();

  setvbuf(stdout, buf, _IOLBF, BUFSIZ);

  print_something();

  print_one_byte(buf);

  setvbuf(stdout, buf, _IOFBF, BUFSIZ);

  print_something();

  print_one_byte(buf);

  setvbuf(stdout, NULL, _IONBF, 0);
}

int main(void) {
  printf("setvbuf\n");

  test_setbuf();
  test_setbuffer();
  test_setlinebuf();
  test_setvbuf();

  // CHECK: setvbuf

  return 0;
}
