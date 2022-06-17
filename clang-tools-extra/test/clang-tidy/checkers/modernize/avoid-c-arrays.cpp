// RUN: %check_clang_tidy %s modernize-avoid-c-arrays %t

int a[] = {1, 2};
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: do not declare C-style arrays, use std::array<> instead

int b[1];
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: do not declare C-style arrays, use std::array<> instead

void foo() {
  int c[b[0]];
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C VLA arrays, use std::vector<> instead

  using d = decltype(c);
  d e;
  // Semi-FIXME: we do not diagnose these last two lines separately,
  // because we point at typeLoc.getBeginLoc(), which is the decl before that
  // (int c[b[0]];), which is already diagnosed.
}

template <typename T, int Size>
class array {
  T d[Size];
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use std::array<> instead

  int e[1];
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use std::array<> instead
};

array<int[4], 2> d;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: do not declare C-style arrays, use std::array<> instead

using k = int[4];
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: do not declare C-style arrays, use std::array<> instead

array<k, 2> dk;

template <typename T>
class unique_ptr {
  T *d;

  int e[1];
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use std::array<> instead
};

unique_ptr<int[]> d2;
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: do not declare C-style arrays, use std::array<> instead

using k2 = int[];
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: do not declare C-style arrays, use std::array<> instead

unique_ptr<k2> dk2;

// Some header
extern "C" {

int f[] = {1, 2};

int j[1];

inline void bar() {
  {
    int j[j[0]];
  }
}

extern "C++" {
int f3[] = {1, 2};
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: do not declare C-style arrays, use std::array<> instead

int j3[1];
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: do not declare C-style arrays, use std::array<> instead

struct Foo {
  int f3[3] = {1, 2};
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use std::array<> instead

  int j3[1];
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use std::array<> instead
};
}

struct Bar {

  int f[3] = {1, 2};

  int j[1];
};
}
