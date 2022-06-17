// RUN: %check_clang_tidy %s modernize-loop-convert %t -- \
// RUN:   -config="{CheckOptions: [{key: modernize-loop-convert.NamingStyle, value: 'camelBack'}]}" \
// RUN:   -- -I %S/Inputs/loop-convert

#include "structures.h"

const int n = 10;
int arr[n];
int nums[n];

void naming() {
  for (int i = 0; i < n; ++i) {
    printf("%d\n", arr[i]);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead [modernize-loop-convert]
  // CHECK-FIXES: for (int i : arr)
  // CHECK-FIXES-NEXT: printf("%d\n", i);

  for (int i = 0; i < n; ++i) {
    printf("%d\n", nums[i]);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int num : nums)
  // CHECK-FIXES-NEXT: printf("%d\n", num);

  int num = 0;
  for (int i = 0; i < n; ++i) {
    printf("%d\n", nums[i] + num);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int i : nums)
  // CHECK-FIXES-NEXT: printf("%d\n", i + num);
}
