// RUN: %clang_cc1 -triple=x86_64-pc-linux-gnu -Wover-aligned -verify %s

namespace test1 {
struct Test {
  template <typename T>
  struct SeparateCacheLines {
    T data;
  } __attribute__((aligned(256)));

  SeparateCacheLines<int> high_contention_data[10];
};

void helper() {
  Test t;
  new Test;  // expected-warning {{type 'Test' requires 256 bytes of alignment and the default allocator only guarantees}}
  new Test[10];  // expected-warning {{type 'Test' requires 256 bytes of alignment and the default allocator only guarantees}}
}
}

namespace test2 {
class Test {
  typedef int __attribute__((aligned(256))) aligned_int;
  aligned_int high_contention_data[10];
};

void helper() {
  Test t;
  new Test;  // expected-warning {{type 'Test' requires 256 bytes of alignment and the default allocator only guarantees}}
  new Test[10];  // expected-warning {{type 'Test' requires 256 bytes of alignment and the default allocator only guarantees}}
}
}

namespace test3 {
struct Test {
  template <typename T>
  struct SeparateCacheLines {
    T data;
  } __attribute__((aligned(256)));

  void* operator new(unsigned long) {
    return 0; // expected-warning {{'operator new' should not return a null pointer unless it is declared 'throw()'}}
  }

  SeparateCacheLines<int> high_contention_data[10];
};

void helper() {
  Test t;
  new Test;
  new Test[10];  // expected-warning {{type 'Test' requires 256 bytes of alignment and the default allocator only guarantees}}
}
}

namespace test4 {
struct Test {
  template <typename T>
  struct SeparateCacheLines {
    T data;
  } __attribute__((aligned(256)));

  void* operator new[](unsigned long) {
    return 0; // expected-warning {{'operator new[]' should not return a null pointer unless it is declared 'throw()'}}
  }

  SeparateCacheLines<int> high_contention_data[10];
};

void helper() {
  Test t;
  new Test;  // expected-warning {{type 'Test' requires 256 bytes of alignment and the default allocator only guarantees}}
  new Test[10];
}
}
