// RUN: %check_clang_tidy %s modernize-loop-convert %t -- -- -I %S/Inputs/loop-convert

#include "structures.h"

void inclusiveArrays() {
  static int ImplicitSize[] = {1, 0};
  for (int I = 0; I <= 1; ++I) {
    --ImplicitSize[I];
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead [modernize-loop-convert]
  // CHECK-FIXES: for (int & I : ImplicitSize) {
  // CHECK-FIXES-NEXT: --I;

  const int N = 6;
  int Arr[N];
  for (int I = 0; I <= N - 1; ++I) {
    ++Arr[I];
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int & I : Arr) {
  // CHECK-FIXES-NEXT: ++I;

  const int Last = N - 1;
  for (int I = 0; Last >= I; ++I) {
    ++Arr[I];
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int & I : Arr) {
  // CHECK-FIXES-NEXT: ++I;

  for (unsigned I = 0; I <= sizeof(Arr) / sizeof(Arr[0]) - 1; ++I) {
    ++Arr[I];
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int & I : Arr) {
  // CHECK-FIXES-NEXT: ++I;

  for (unsigned I = 0; 5ULL >= I; ++I) {
    ++Arr[I];
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int & I : Arr) {
  // CHECK-FIXES-NEXT: ++I;

  int Single[1];
  for (int I = 0; I <= 0; ++I) {
    ++Single[I];
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int & I : Single) {
  // CHECK-FIXES-NEXT: ++I;

  // The increment after the final iteration still fits in the index type.
  int Small[__SCHAR_MAX__];
  for (signed char I = 0; I <= __SCHAR_MAX__ - 1; ++I) {
    ++Small[I];
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int & I : Small) {
  // CHECK-FIXES-NEXT: ++I;

  for (unsigned char I = 0; I <= __SCHAR_MAX__ - 1; ++I) {
    ++Small[I];
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int & I : Small) {
  // CHECK-FIXES-NEXT: ++I;
}

void invalidBounds(int Last) {
  int Arr[6];
  for (int I = 0; I <= 6; ++I)
    ++Arr[I];
  // CHECK-FIXES: for (int I = 0; I <= 6; ++I)

  for (int I = 0; 6 >= I; ++I)
    ++Arr[I];
  // CHECK-FIXES: for (int I = 0; 6 >= I; ++I)

  for (int I = 0; I <= 4; ++I)
    ++Arr[I];
  // CHECK-FIXES: for (int I = 0; I <= 4; ++I)

  for (int I = 0; 4 >= I; ++I)
    ++Arr[I];
  // CHECK-FIXES: for (int I = 0; 4 >= I; ++I)

  for (int I = 0; I <= -1; ++I)
    ++Arr[I];
  // CHECK-FIXES: for (int I = 0; I <= -1; ++I)

  for (unsigned I = 0; I <= ~0U; ++I)
    ++Arr[I];
  // CHECK-FIXES: for (unsigned I = 0; I <= ~0U; ++I)

  for (int I = 0; I <= Last; ++I)
    ++Arr[I];
  // CHECK-FIXES: for (int I = 0; I <= Last; ++I)

  for (int I = 1; I <= 5; ++I)
    ++Arr[I];
  // CHECK-FIXES: for (int I = 1; I <= 5; ++I)

  for (int I = 0; I <= 5; ++I)
    Arr[I] += I;
  // CHECK-FIXES: for (int I = 0; I <= 5; ++I)

  // Incrementing the final index would overflow or wrap.
  int Signed[__SCHAR_MAX__ + 1];
  for (signed char I = 0; I <= __SCHAR_MAX__; ++I)
    ++Signed[I];
  // CHECK-FIXES: for (signed char I = 0; I <= __SCHAR_MAX__; ++I)

  int Unsigned[__SCHAR_MAX__ * 2 + 2];
  for (unsigned char I = 0; I <= __SCHAR_MAX__ * 2 + 1; ++I)
    ++Unsigned[I];
  // CHECK-FIXES: for (unsigned char I = 0; I <= __SCHAR_MAX__ * 2 + 1; ++I)

  int Empty[0];
  for (unsigned long long I = 0; I <= ~0ULL; ++I)
    ++Empty[I];
  // CHECK-FIXES: for (unsigned long long I = 0; I <= ~0ULL; ++I)
}

void inclusiveContainer() {
  dependent<int> Ss;
  for (int I = 0; I <= Ss.size(); ++I)
    ++Ss[I];
  // CHECK-FIXES: for (int I = 0; I <= Ss.size(); ++I)
}

template <int N>
void dependentBound() {
  int Arr[N];
  for (int I = 0; I <= N - 1; ++I)
    ++Arr[I];
  // CHECK-FIXES: for (int I = 0; I <= N - 1; ++I)
}
