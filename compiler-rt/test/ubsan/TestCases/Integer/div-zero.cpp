// RUN: %clangxx -fsanitize=integer-divide-by-zero -DDIVIDEND=0 %s -o %t && %run %t 2>&1 | FileCheck %s
// RUN: %clangxx -fsanitize=integer-divide-by-zero -DDIVIDEND=1U %s -o %t && %run %t 2>&1 | FileCheck %s
// RUN: %clangxx -fsanitize=integer-divide-by-zero -DDIVIDEND='intmax(123)' %s -o %t && %run %t 2>&1 | FileCheck %s

#if defined(__SIZEOF_INT128__) && !defined(_WIN32)
typedef __int128 intmax;
#else
typedef long long intmax;
#endif

int main() {
  // CHECK: div-zero.cpp:[[@LINE+1]]:12: runtime error: division by zero
  DIVIDEND / 0;
}
