// RUN: %check_clang_tidy %s performance-inefficient-string-concatenation %t

namespace std {
template <typename T>
class basic_string {
public:
  basic_string() {}
  ~basic_string() {}
  basic_string<T> *operator+=(const basic_string<T> &) {}
  friend basic_string<T> operator+(const basic_string<T> &, const basic_string<T> &) {}
};
typedef basic_string<char> string;
typedef basic_string<wchar_t> wstring;
}

void f(std::string) {}
std::string g(std::string) {}

int main() {
  std::string mystr1, mystr2;
  std::wstring mywstr1, mywstr2;

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

    mywstr1 = mywstr2 + mywstr2;
    mystr1 = mystr2 + mystr2;
    mystr1 += mystr2;
    f(mystr2 + mystr1);
    mystr1 = g(mystr1);
  }
  return 0;
}
