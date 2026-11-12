// RUN: %clang_cc1 -triple=x86_64-pc-linux-gnu -Wover-aligned -verify %s
// expected-no-diagnostics

// This test verifies that we don't warn when the global operator new is
// overridden. That's why we can't merge this with the other test file.

void* operator new(unsigned long);
void* operator new[](unsigned long);

struct Test {
  template <typename T>
  struct SeparateCacheLines {
    T data;
  } __attribute__((aligned(256)));

  SeparateCacheLines<int> high_contention_data[10];
};

void helper() {
  Test t;
  new Test;
  new Test[10];
}
