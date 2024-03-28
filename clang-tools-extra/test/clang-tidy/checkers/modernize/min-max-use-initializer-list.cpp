// RUN: %check_clang_tidy %s modernize-min-max-use-initializer-list %t

// CHECK-FIXES: #include <algorithm>
namespace std {
template< class T >
const T& max( const T& a, const T& b ) {
  return (a < b) ? b : a;
};

template< class T, class Compare >
const T& max( const T& a, const T& b, Compare comp ) {
  return (comp(a, b)) ? b : a;
};

template< class T >
const T& min( const T& a, const T& b ) {
  return (b < a) ? b : a;
};

template< class T, class Compare >
const T& min( const T& a, const T& b, Compare comp ) {
  return (comp(b, a)) ? b : a;
};
}

using namespace std;

namespace {
bool fless_than(int a, int b) {
return a < b;
}

bool fgreater_than(int a, int b) {
return a > b;
}
auto less_than = [](int a, int b) { return a < b; };
auto greater_than = [](int a, int b) { return a > b; };

int max1 = std::max(1, std::max(2, 3));
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: do not use nested std::max calls, use std::max({1, 2, 3}) instead [modernize-min-max-use-initializer-list]
// CHECK-FIXES: int max1 = std::max({1, 2, 3});

int min1 = std::min(1, std::min(2, 3));
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: do not use nested std::min calls, use std::min({1, 2, 3}) instead [modernize-min-max-use-initializer-list]
// CHECK-FIXES: int min1 = std::min({1, 2, 3});

int max2 = std::max(1, std::max(2, std::max(3, 4)));
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: do not use nested std::max calls, use std::max({1, 2, 3, 4}) instead [modernize-min-max-use-initializer-list]
// CHECK-FIXES: int max2 = std::max({1, 2, 3, 4});

int min2 = std::min(1, std::min(2, std::min(3, 4)));
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: do not use nested std::min calls, use std::min({1, 2, 3, 4}) instead [modernize-min-max-use-initializer-list]
// CHECK-FIXES: int min2 = std::min({1, 2, 3, 4});

int max3 = std::max(std::max(4, 5), std::min(2, std::min(3, 1)));
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: do not use nested std::max calls, use std::max({4, 5, std::min({2, 3, 1})}) instead [modernize-min-max-use-initializer-list]
// CHECK-MESSAGES: :[[@LINE-2]]:37: warning: do not use nested std::min calls, use std::min({2, 3, 1}) instead [modernize-min-max-use-initializer-list]
// CHECK-FIXES: int max3 = std::max({4, 5, std::min({2, 3, 1})});

int min3 = std::min(std::min(4, 5), std::max(2, std::max(3, 1)));
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: do not use nested std::min calls, use std::min({4, 5, std::max({2, 3, 1})}) instead [modernize-min-max-use-initializer-list]
// CHECK-MESSAGES: :[[@LINE-2]]:37: warning: do not use nested std::max calls, use std::max({2, 3, 1}) instead [modernize-min-max-use-initializer-list]
// CHECK-FIXES: int min3 = std::min({4, 5, std::max({2, 3, 1})});

int max4 = std::max(1,std::max(2,3, greater_than), less_than);
// CHECK-FIXES: int max4 = std::max(1,std::max(2,3, greater_than), less_than);

int min4 = std::min(1,std::min(2,3, greater_than), less_than);
// CHECK-FIXES: int min4 = std::min(1,std::min(2,3, greater_than), less_than);

int max5 = std::max(1,std::max(2,3), less_than);
// CHECK-FIXES: int max5 = std::max(1,std::max(2,3), less_than);

int min5 = std::min(1,std::min(2,3), less_than);
// CHECK-FIXES: int min5 = std::min(1,std::min(2,3), less_than);

int max6 = std::max(1,std::max(2,3, greater_than), greater_than);
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: do not use nested std::max calls, use std::max({1, 2, 3}, greater_than) instead [modernize-min-max-use-initializer-list]
// CHECK-FIXES: int max6 = std::max({1, 2, 3}, greater_than);

int min6 = std::min(1,std::min(2,3, greater_than), greater_than);
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: do not use nested std::min calls, use std::min({1, 2, 3}, greater_than) instead [modernize-min-max-use-initializer-list]
// CHECK-FIXES: int min6 = std::min({1, 2, 3}, greater_than);

int max7 = std::max(1,std::max(2,3, fless_than), fgreater_than);
// CHECK-FIXES: int max7 = std::max(1,std::max(2,3, fless_than), fgreater_than);

int min7 = std::min(1,std::min(2,3, fless_than), fgreater_than);
// CHECK-FIXES: int min7 = std::min(1,std::min(2,3, fless_than), fgreater_than);

int max8 = std::max(1,std::max(2,3, fless_than), less_than);
// CHECK-FIXES: int max8 = std::max(1,std::max(2,3, fless_than), less_than)

int min8 = std::min(1,std::min(2,3, fless_than), less_than);
// CHECK-FIXES: int min8 = std::min(1,std::min(2,3, fless_than), less_than);

int max9 = std::max(1,std::max(2,3, fless_than), fless_than);
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: do not use nested std::max calls, use std::max({1, 2, 3}, fless_than) instead [modernize-min-max-use-initializer-list]
// CHECK-FIXES: int max9 = std::max({1, 2, 3}, fless_than);

int min9 = std::min(1,std::min(2,3, fless_than), fless_than);
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: do not use nested std::min calls, use std::min({1, 2, 3}, fless_than) instead [modernize-min-max-use-initializer-list]
// CHECK-FIXES: int min9 = std::min({1, 2, 3}, fless_than);

}