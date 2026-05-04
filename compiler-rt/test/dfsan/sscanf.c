// RUN: %clang_dfsan %s -o %t && %run %t

#include <assert.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
  {
    char buf[256] = "10000000000-100000000000 rw-p 00000000 00:00 0";
    long rss = 0;
    // This test exposes a bug in DFSan's sscanf, that leads to flakiness
    // in release_shadow_space.c (see
    // https://github.com/llvm/llvm-project/issues/91287)
    int r = sscanf(buf, "Garbage text before, %ld, Garbage text after", &rss);
    assert(r == 0);
  }

  // Testing other variations of sscanf behavior.
  {
    int a = 0;
    int b = 0;
    int r = sscanf("abc42 cat 99", "abc%d cat %d", &a, &b);
    assert(a == 42);
    assert(b == 99);
    assert(r == 2);
  }

  {
    int a = 0;
    int b = 0;
    int r = sscanf("abc42 cat 99", "abc%d dog %d", &a, &b);
    assert(a == 42);
    assert(r == 1);
  }

  {
    int a = 0;
    int b = 0;
    int r = sscanf("abx42 dog 99", "abc%d dog %d", &a, &b);
    assert(r == 0);
  }

  {
    int r = sscanf("abx", "abc");
    assert(r == 0);
  }

  {
    int r = sscanf("abc", "abc");
    assert(r == 0);
  }

  {
    int n = 0;
    int r = sscanf("abc", "abc%n", &n);
    assert(n == 3);
    assert(r == 0);
  }

  {
    int n = 1234;
    int r = sscanf("abxy", "abcd%n", &n);
    assert(n == 1234);
    assert(r == 0);
  }

  {
    int a = 0;
    int n = 1234;
    int r = sscanf("abcd99", "abcd%d%n", &a, &n);
    assert(a == 99);
    assert(n == 6);
    assert(r == 1);
  }

  {
    int n = 1234;
    int r = sscanf("abcdsuffix", "abcd%n", &n);
    assert(n == 4);
    assert(r == 0);
  }

  {
    int n = 1234;
    int r = sscanf("abxxsuffix", "abcd%n", &n);
    assert(n == 1234);
    assert(r == 0);
  }

  {
    int a = 0;
    int b = 0;
    int n = 1234;
    int r = sscanf("abcd99 xy100", "abcd%d xy%d%n", &a, &b, &n);
    assert(a == 99);
    assert(b == 100);
    assert(n == 12);
    assert(r == 2);
  }

  {
    int a = 0;
    int b = 0;
    int n = 1234;
    int r = sscanf("abcd99 xy100", "abcd%d zz%d%n", &a, &b, &n);
    assert(a == 99);
    assert(b == 0);
    assert(n == 1234);
    assert(r == 1);
  }

  return 0;
}
