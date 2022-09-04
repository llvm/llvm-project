// RUN: %clang_cc1 -triple=x86_64-pc-linux-gnu -Wover-aligned -verify=precxx17 %std_cxx98-14 %s
// RUN: %clang_cc1 -triple=x86_64-pc-linux-gnu -Wover-aligned -verify=cxx17 %std_cxx17- %s

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
  new Test;  // precxx17-warning {{type 'Test' requires 256 bytes of alignment and the default allocator only guarantees}}
  new Test[10];  // precxx17-warning {{type 'Test' requires 256 bytes of alignment and the default allocator only guarantees}}
}
}

namespace test2 {
class Test {
  typedef int __attribute__((aligned(256))) aligned_int;
  aligned_int high_contention_data[10];
};

void helper() {
  Test t;
  new Test;  // precxx17-warning {{type 'Test' requires 256 bytes of alignment and the default allocator only guarantees}}
  new Test[10];  // precxx17-warning {{type 'Test' requires 256 bytes of alignment and the default allocator only guarantees}}
}
}

namespace test3 {
struct Test {
  template <typename T>
  struct SeparateCacheLines {
    T data;
  } __attribute__((aligned(256)));

  void* operator new(unsigned long) {
    return 0; // precxx17-warning {{'operator new' should not return a null pointer unless it is declared 'throw()'}} \
                 cxx17-warning {{'operator new' should not return a null pointer unless it is declared 'throw()' or 'noexcept'}}
  }

  SeparateCacheLines<int> high_contention_data[10];
};

void helper() {
  Test t;
  new Test;
  new Test[10];  // precxx17-warning {{type 'Test' requires 256 bytes of alignment and the default allocator only guarantees}}
}
}

namespace test4 {
struct Test {
  template <typename T>
  struct SeparateCacheLines {
    T data;
  } __attribute__((aligned(256)));

  void* operator new[](unsigned long) {
    return 0; // precxx17-warning {{'operator new[]' should not return a null pointer unless it is declared 'throw()'}} \
                 cxx17-warning {{'operator new[]' should not return a null pointer unless it is declared 'throw()' or 'noexcept'}}
  }

  SeparateCacheLines<int> high_contention_data[10];
};

void helper() {
  Test t;
  new Test;  // precxx17-warning {{type 'Test' requires 256 bytes of alignment and the default allocator only guarantees}}
  new Test[10];
}
}
