// RUN: %check_clang_tidy -std=c++11-or-later %s misc-function-pointer-equality %t -- \
// RUN:   -config='{CheckOptions: { misc-function-pointer-equality.KeyedContainerTypes: "::std::MyCustomContainer" }}'

#include <set>

void f();

namespace std
{
template<typename T>
class MyCustomContainer {
public:
  MyCustomContainer(T myVal): val(myVal) {};
  T val;
};
}

void test_custom_container() {
  std::MyCustomContainer<void (*)()> a(&f);
  // CHECK-MESSAGES: :[[@LINE-1]]:38: warning: associative or unordered container 'a' with a function pointer key type is unsound with '--icf=all' enabled [misc-function-pointer-equality]
  
  std::set<void (*)()> s;
}

