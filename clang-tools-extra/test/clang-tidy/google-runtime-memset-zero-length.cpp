// RUN: %check_clang_tidy %s google-runtime-memset %t

void *memset(void *, int, __SIZE_TYPE__);

namespace std {
  using ::memset;
}

template <int i, typename T>
void memtmpl() {
  memset(0, sizeof(int), i);
  memset(0, sizeof(T), sizeof(T));
  memset(0, sizeof(T), 0);
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: memset of size zero, potentially swapped argument
// CHECK-FIXES: memset(0, 0, sizeof(T));
  memset(0, sizeof(int), 0);
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: memset of size zero, potentially swapped argument
// CHECK-FIXES: memset(0, 0, sizeof(int));
}

void foo(void *a, int xsize, int ysize) {
  memset(a, sizeof(int), 0);
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: memset of size zero, potentially swapped argument
// CHECK-FIXES: memset(a, 0, sizeof(int));
#define M memset(a, sizeof(int), 0);
  M
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: memset of size zero, potentially swapped argument
// CHECK-FIXES: #define M memset(a, sizeof(int), 0);
  ::memset(a, xsize *
           ysize, 0);
// CHECK-MESSAGES: :[[@LINE-2]]:3: warning: memset of size zero, potentially swapped argument
// CHECK-FIXES: ::memset(a, 0, xsize *
// CHECK-FIXES-NEXT: ysize);
  std::memset(a, sizeof(int), 0x00);
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: memset of size zero, potentially swapped argument
// CHECK-FIXES: std::memset(a, 0x00, sizeof(int));

  const int v = 0;
  memset(a, sizeof(int), v);
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: memset of size zero, potentially swapped argument
// CHECK-FIXES: memset(a, v, sizeof(int));

  memset(a, sizeof(int), v + v);
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: memset of size zero, potentially swapped argument
// CHECK-FIXES: memset(a, v + v, sizeof(int));

  memset(a, sizeof(int), v + 1);

  memset(a, -1, sizeof(int));
  memset(a, 0xcd, 1);

  // Don't warn when the fill char and the length are both known to be
  // zero.  No bug is possible.
  memset(a, 0, v);
  memset(a, v, 0);

  // -1 is clearly not a length by virtue of being negative, so no warning
  // despite v == 0.
  memset(a, -1, v);

  memtmpl<0, int>();
}
