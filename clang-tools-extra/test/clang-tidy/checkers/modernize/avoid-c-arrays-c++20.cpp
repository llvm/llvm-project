// RUN: %check_clang_tidy -std=c++20 %s modernize-avoid-c-arrays %t

int f1(int data[], int size) {
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: do not declare C-style arrays, use 'std::span' instead
  int f4[] = {1, 2};
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use 'std::array' instead
}

int f2(int data[100]) {
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: do not declare C-style arrays, use 'std::array' instead
}
