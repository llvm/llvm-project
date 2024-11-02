// RUN: %check_clang_tidy %s modernize-loop-convert %t -- \
// RUN:   -config="{CheckOptions: [{key: modernize-loop-convert.NamingStyle, value: 'UPPER_CASE'}]}" \
// RUN:   -- -I %S/Inputs/loop-convert

#include "structures.h"

const int N = 10;
int ARR[N];
int NUMS[N];
int NUMS_[N];

void naming() {
  for (int I = 0; I < N; ++I) {
    printf("%d\n", ARR[I]);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead [modernize-loop-convert]
  // CHECK-FIXES: for (int I : ARR)
  // CHECK-FIXES-NEXT: printf("%d\n", I);

  for (int I = 0; I < N; ++I) {
    printf("%d\n", NUMS[I]);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int NUM : NUMS)
  // CHECK-FIXES-NEXT: printf("%d\n", NUM);

  for (int I = 0; I < N; ++I) {
    printf("%d\n", NUMS_[I]);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int NUM : NUMS_)
  // CHECK-FIXES-NEXT: printf("%d\n", NUM);

  int NUM = 0;
  for (int I = 0; I < N; ++I) {
    printf("%d\n", NUMS[I] + NUM);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int I : NUMS)
  // CHECK-FIXES-NEXT: printf("%d\n", I + NUM);
}
