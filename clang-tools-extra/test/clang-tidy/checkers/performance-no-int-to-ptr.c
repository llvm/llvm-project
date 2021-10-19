// RUN: %check_clang_tidy %s performance-no-int-to-ptr %t

void *t0(char x) {
  return x;
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: integer to pointer cast pessimizes optimization opportunities [performance-no-int-to-ptr]
}
void *t1(signed char x) {
  return x;
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: integer to pointer cast pessimizes optimization opportunities [performance-no-int-to-ptr]
}
void *t2(unsigned char x) {
  return x;
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: integer to pointer cast pessimizes optimization opportunities [performance-no-int-to-ptr]
}

void *t3(short x) {
  return x;
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: integer to pointer cast pessimizes optimization opportunities [performance-no-int-to-ptr]
}
void *t4(unsigned short x) {
  return x;
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: integer to pointer cast pessimizes optimization opportunities [performance-no-int-to-ptr]
}
void *t5(signed short x) {
  return x;
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: integer to pointer cast pessimizes optimization opportunities [performance-no-int-to-ptr]
}

void *t6(int x) {
  return x;
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: integer to pointer cast pessimizes optimization opportunities [performance-no-int-to-ptr]
}
void *t7(unsigned int x) {
  return x;
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: integer to pointer cast pessimizes optimization opportunities [performance-no-int-to-ptr]
}
void *t8(signed int x) {
  return x;
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: integer to pointer cast pessimizes optimization opportunities [performance-no-int-to-ptr]
}

void *t9(long x) {
  return x;
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: integer to pointer cast pessimizes optimization opportunities [performance-no-int-to-ptr]
}
void *t10(unsigned long x) {
  return x;
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: integer to pointer cast pessimizes optimization opportunities [performance-no-int-to-ptr]
}
void *t11(signed long x) {
  return x;
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: integer to pointer cast pessimizes optimization opportunities [performance-no-int-to-ptr]
}

void *t12(long long x) {
  return x;
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: integer to pointer cast pessimizes optimization opportunities [performance-no-int-to-ptr]
}
void *t13(unsigned long long x) {
  return x;
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: integer to pointer cast pessimizes optimization opportunities [performance-no-int-to-ptr]
}
void *t14(signed long long x) {
  return x;
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: integer to pointer cast pessimizes optimization opportunities [performance-no-int-to-ptr]
}
