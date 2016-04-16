// RUN: %check_clang_tidy %s misc-pointer-and-integral-operation %t

bool* pb;
char* pc;
int* pi;

int Test() {
  if (pi < 0) return 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious comparison of pointer with zero [misc-pointer-and-integral-operation]
  if (pi <= 0) return 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious comparison of pointer with zero

  if (nullptr <= pb) return 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: suspicious comparison of pointer with null
  if (pc < nullptr) return 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious comparison of pointer with null
  if (pi > nullptr) return 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious comparison of pointer with null

  return 1;
}

int Valid() {
  *pb = false;
  *pc = '\0';

  pi += (pi != nullptr);
  pi -= (pi == nullptr);
  pc += (pb != nullptr);
}
