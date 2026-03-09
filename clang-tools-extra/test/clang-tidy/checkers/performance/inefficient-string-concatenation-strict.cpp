// RUN: %check_clang_tidy %s performance-inefficient-string-concatenation %t -- \
// RUN:   -config="{CheckOptions: {performance-inefficient-string-concatenation.StrictMode: true}}" -- \
// RUN:   -isystem %clang_tidy_headers

#include <string>

void f(std::string) {}

int main() {
  std::string mystr1, mystr2;
  mystr1 = mystr1 + mystr2;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: string concatenation results in allocation of unnecessary temporary strings; consider using 'operator+=' or 'string::append()' instead [performance-inefficient-string-concatenation]

  f(mystr1 + mystr2 + mystr1);
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: string concatenation results in allocation of unnecessary temporary strings; consider using 'operator+=' or 'string::append()' instead

  return 0;
}
