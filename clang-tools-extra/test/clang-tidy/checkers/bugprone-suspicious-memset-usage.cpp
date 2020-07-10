// RUN: %check_clang_tidy %s bugprone-suspicious-memset-usage %t

void *memset(void *, int, __SIZE_TYPE__);

namespace std {
  using ::memset;
}

template <typename T>
void mtempl(int *ptr) {
  memset(ptr, '0', sizeof(T));
// CHECK-MESSAGES: :[[@LINE-1]]:15: warning: memset fill value is char '0', potentially mistaken for int 0 [bugprone-suspicious-memset-usage]
// CHECK-FIXES: memset(ptr, 0, sizeof(T));
  memset(ptr, 256, sizeof(T));
// CHECK-MESSAGES: :[[@LINE-1]]:15: warning: memset fill value is out of unsigned character range, gets truncated [bugprone-suspicious-memset-usage]
  memset(0, sizeof(T), 0);
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: memset of size zero, potentially swapped arguments [bugprone-suspicious-memset-usage]
// CHECK-FIXES: memset(0, 0, sizeof(T));
  memset(0, sizeof(int), 0);
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: memset of size zero, potentially swapped arguments [bugprone-suspicious-memset-usage]
// CHECK-FIXES: memset(0, 0, sizeof(int));
}

void foo(int xsize, int ysize) {
  int i[5] = {1, 2, 3, 4, 5};
  char ca[3] = {'a', 'b', 'c'};
  int *p = i;
  int l = 5;
  char z = '1';
  char *c = &z;
  int v = 0;

  memset(p, '0', l);
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: memset fill value is char '0', potentially mistaken for int 0 [bugprone-suspicious-memset-usage]
// CHECK-FIXES: memset(p, 0, l);

  memset(p, 0xabcd, l);
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: memset fill value is out of unsigned character range, gets truncated [bugprone-suspicious-memset-usage]

  memset(p, sizeof(int), 0);
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: memset of size zero, potentially swapped arguments [bugprone-suspicious-memset-usage]
// CHECK-FIXES: memset(p, 0, sizeof(int));
  std::memset(p, sizeof(int), 0x00);
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: memset of size zero, potentially swapped arguments [bugprone-suspicious-memset-usage]
// CHECK-FIXES: std::memset(p, 0x00, sizeof(int));

#define M_CHAR_ZERO memset(p, '0', l);
  M_CHAR_ZERO
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: memset fill value is char '0', potentially mistaken for int 0 [bugprone-suspicious-memset-usage]

#define M_OUTSIDE_RANGE memset(p, 0xabcd, l);
  M_OUTSIDE_RANGE
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: memset fill value is out of unsigned character range, gets truncated [bugprone-suspicious-memset-usage]

#define M_ZERO_LENGTH memset(p, sizeof(int), 0);
  M_ZERO_LENGTH
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: memset of size zero, potentially swapped arguments [bugprone-suspicious-memset-usage]

  memset(p, '2', l);
  memset(p, 0, l);
  memset(c, '0', 1);
  memset(ca, '0', sizeof(ca));

  memset(p, 0x00, l);
  mtempl<int>(p);

  memset(p, sizeof(int), v + 1);
  memset(p, 0xcd, 1);

  // Don't warn when the fill char and the length are both known to be
  // zero.  No bug is possible.
  memset(p, 0, v);

  // -1 is clearly not a length by virtue of being negative, so no warning
  // despite v == 0.
  memset(p, -1, v);
}

void *memset(int);
void NoCrash() {
  memset(1);
}
