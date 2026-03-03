// RUN: %check_clang_tidy %s performance-inefficient-string-concatenation %t -- -- -isystem %clang_tidy_headers
#include <string>

void f(std::string) {}
std::string g(std::string);

int main() {
  std::string mystr1, mystr2;
  std::wstring mywstr1, mywstr2;
  auto myautostr1 = mystr1;
  auto myautostr2 = mystr2;

  for (int i = 0; i < 10; ++i) {
    f(mystr1 + mystr2 + mystr1);
    // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: string concatenation results in allocation of unnecessary temporary strings; consider using 'operator+=' or 'string::append()' instead
    mystr1 = mystr1 + mystr2;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: string concatenation
    mystr1 = mystr2 + mystr2 + mystr2;
    // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: string concatenation
    mystr1 = mystr2 + mystr1;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: string concatenation
    mywstr1 = mywstr2 + mywstr1;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: string concatenation
    mywstr1 = mywstr2 + mywstr2 + mywstr2;
    // CHECK-MESSAGES: :[[@LINE-1]]:33: warning: string concatenation
    myautostr1 = myautostr1 + myautostr2;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: string concatenation

    mywstr1 = mywstr2 + mywstr2;
    mystr1 = mystr2 + mystr2;
    mystr1 += mystr2;
    f(mystr2 + mystr1);
    mystr1 = g(mystr1);
  }
  return 0;
}
