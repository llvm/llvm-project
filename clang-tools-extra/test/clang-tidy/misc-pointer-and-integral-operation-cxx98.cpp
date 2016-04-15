// RUN: %check_clang_tidy %s misc-pointer-and-integral-operation %t -- -- -std=c++98

bool* pb;
char* pc;
int* pi;

int Test() {
  pb = false;
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: suspicious assignment from bool to pointer [misc-pointer-and-integral-operation]
  pc = '\0';
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: suspicious assignment from char to pointer

  pb = (false?false:false);
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: suspicious assignment from bool to pointer
  pb = (4 != 5?false:false);
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: suspicious assignment from bool to pointer

  if (pb < false) return 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious operation between pointer and bool literal
  if (pb != false) return 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious operation between pointer and bool literal
  if (pc < '\0') return 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious operation between pointer and character literal
  if (pc != '\0') return 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious operation between pointer and character literal
  if (pi < '\0') return 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious operation between pointer and character literal
  if (pi != '\0') return 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious operation between pointer and character literal

  return 1;
}

int Valid() {
  *pb = false;
  *pc = '\0';

  pb += 0;
  pc += 0;
  pi += 0;

  pb += (pb != 0);
  pc += (pc != 0);
  pi += (pi != 0);
}
