// RUN: %check_clang_tidy %s bugprone-sizeof-expression %t -- \
// RUN:   -config="{CheckOptions: { \
// RUN:     bugprone-sizeof-expression.WarnOnSizeOfConstant: false, \
// RUN:     bugprone-sizeof-expression.WarnOnSizeOfThis: false, \
// RUN:     bugprone-sizeof-expression.WarnOnSizeOfCompareToConstant: false, \
// RUN:     bugprone-sizeof-expression.WarnOnSizeOfInLoopTermination: false \
// RUN:   }}"

#define LEN 8

class C {
  int size() { return sizeof(this); }
};

int Test() {
  int A = sizeof(LEN);
  int B = sizeof(LEN + 1);
  int C = sizeof(1);

  if (sizeof(A) < 0x100000)
    return 0;
  if (sizeof(A) <= 0)
    return 0;

  int arr[10];
  for (int i = 0; sizeof(arr) < i; ++i) {}
  while (sizeof(arr) < 10) {}

  int x = sizeof(A, 1);
  // CHECK-MESSAGES: [[@LINE-1]]:19: warning: suspicious usage of 'sizeof(..., ...)' [bugprone-sizeof-expression]

  return A + B + C + x;
}
