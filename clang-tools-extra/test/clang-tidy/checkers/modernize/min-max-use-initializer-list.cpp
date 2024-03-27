// RUN: %check_clang_tidy %s modernize-min-max-use-initializer-list %t

// CHECK-FIXES: #include <algorithm>
namespace utils {
template <typename T>
T max(T a, T b) {
  return (a < b) ? b : a;
}
} // namespace utils

namespace std {
template< class T >
struct initializer_list {
  initializer_list()=default;
  initializer_list(T*,int){}
  const T* begin() const {return nullptr;}
  const T* end() const {return nullptr;}
};

template<class ForwardIt>
ForwardIt min_element(ForwardIt first, ForwardIt last)
{
    if (first == last)
        return last;

    ForwardIt smallest = first;

    while (++first != last)
        if (*first < *smallest)
            smallest = first;

    return smallest;
}

template<class ForwardIt>
ForwardIt max_element(ForwardIt first, ForwardIt last)
{
    if (first == last)
        return last;

    ForwardIt largest = first;

    while (++first != last)
        if (*largest < *first)
            largest = first;

    return largest;
}

template< class T >
const T& max( const T& a, const T& b ) {
  return (a < b) ? b : a;
};

template< class T >
T max(std::initializer_list<T> ilist)
{
    return *std::max_element(ilist.begin(), ilist.end());
}

template< class T, class Compare >
const T& max( const T& a, const T& b, Compare comp ) {
  return (comp(a, b)) ? b : a;
};

template< class T, class Compare >
T max(std::initializer_list<T> ilist, Compare comp) {
    return *std::max_element(ilist.begin(), ilist.end(), comp);
};

template< class T >
const T& min( const T& a, const T& b ) {
  return (b < a) ? b : a;
};

template< class T >
T min(std::initializer_list<T> ilist)
{
    return *std::min_element(ilist.begin(), ilist.end());
}


template< class T, class Compare >
const T& min( const T& a, const T& b, Compare comp ) {
  return (comp(b, a)) ? b : a;
};

template< class T, class Compare >
T min(std::initializer_list<T> ilist, Compare comp) {
    return *std::min_element(ilist.begin(), ilist.end(), comp);
};

} // namespace std

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
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: do not use nested 'std::max' calls, use 'std::max({1, 2, 3})' instead [modernize-min-max-use-initializer-list]
// CHECK-FIXES: int max1 = std::max({1, 2, 3});

int min1 = std::min(1, std::min(2, 3));
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: do not use nested 'std::min' calls, use 'std::min({1, 2, 3})' instead [modernize-min-max-use-initializer-list]
// CHECK-FIXES: int min1 = std::min({1, 2, 3});

int max2 = std::max(1, std::max(2, std::max(3, 4)));
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: do not use nested 'std::max' calls, use 'std::max({1, 2, 3, 4})' instead [modernize-min-max-use-initializer-list]
// CHECK-FIXES: int max2 = std::max({1, 2, 3, 4});

int max2b = std::max(std::max(std::max(1, 2), std::max(3, 4)), std::max(std::max(5, 6), std::max(7, 8)));
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: do not use nested 'std::max' calls, use 'std::max({1, 2, 3, 4, 5, 6, 7, 8})' instead [modernize-min-max-use-initializer-list]
// CHECK-FIXES: int max2b = std::max({1, 2, 3, 4, 5, 6, 7, 8});

int max2c = std::max(std::max(1, std::max(2, 3)), 4);
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: do not use nested 'std::max' calls, use 'std::max({1, 2, 3, 4})' instead [modernize-min-max-use-initializer-list]
// CHECK-FIXES: int max2c = std::max({1, 2, 3, 4});

int max2d = std::max(std::max({1, 2, 3}), 4);
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: do not use nested 'std::max' calls, use 'std::max({1, 2, 3, 4})' instead [modernize-min-max-use-initializer-list]
// CHECK-FIXES: int max2d = std::max({1, 2, 3, 4});

int max2e = std::max(1, max(2, max(3, 4)));
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: do not use nested 'std::max' calls, use 'std::max({1, 2, 3, 4})' instead [modernize-min-max-use-initializer-list]
// CHECK-FIXES: int max2e = std::max({1, 2, 3, 4});

int min2 = std::min(1, std::min(2, std::min(3, 4)));
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: do not use nested 'std::min' calls, use 'std::min({1, 2, 3, 4})' instead [modernize-min-max-use-initializer-list]
// CHECK-FIXES: int min2 = std::min({1, 2, 3, 4});

int max3 = std::max(std::max(4, 5), std::min(2, std::min(3, 1)));
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: do not use nested 'std::max' calls, use 'std::max({4, 5, std::min({2, 3, 1})})' instead [modernize-min-max-use-initializer-list]
// CHECK-MESSAGES: :[[@LINE-2]]:37: warning: do not use nested 'std::min' calls, use 'std::min({2, 3, 1})' instead [modernize-min-max-use-initializer-list]
// CHECK-FIXES: int max3 = std::max({4, 5, std::min({2, 3, 1})});

int min3 = std::min(std::min(4, 5), std::max(2, std::max(3, 1)));
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: do not use nested 'std::min' calls, use 'std::min({4, 5, std::max({2, 3, 1})})' instead [modernize-min-max-use-initializer-list]
// CHECK-MESSAGES: :[[@LINE-2]]:37: warning: do not use nested 'std::max' calls, use 'std::max({2, 3, 1})' instead [modernize-min-max-use-initializer-list]
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
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: do not use nested 'std::max' calls, use 'std::max({1, 2, 3}, greater_than)' instead [modernize-min-max-use-initializer-list]
// CHECK-FIXES: int max6 = std::max({1, 2, 3}, greater_than);

int min6 = std::min(1,std::min(2,3, greater_than), greater_than);
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: do not use nested 'std::min' calls, use 'std::min({1, 2, 3}, greater_than)' instead [modernize-min-max-use-initializer-list]
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
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: do not use nested 'std::max' calls, use 'std::max({1, 2, 3}, fless_than)' instead [modernize-min-max-use-initializer-list]
// CHECK-FIXES: int max9 = std::max({1, 2, 3}, fless_than);

int min9 = std::min(1,std::min(2,3, fless_than), fless_than);
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: do not use nested 'std::min' calls, use 'std::min({1, 2, 3}, fless_than)' instead [modernize-min-max-use-initializer-list]
// CHECK-FIXES: int min9 = std::min({1, 2, 3}, fless_than);

int min10 = std::min(std::min(4, 5), std::max(2, utils::max(3, 1)));
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: do not use nested 'std::min' calls, use 'std::min({4, 5, std::max(2, utils::max(3, 1))})' instead [modernize-min-max-use-initializer-list]
// CHECK-FIXES: int min10 = std::min({4, 5, std::max(2, utils::max(3, 1))});

int typecastTest = std::max(std::max<int>(0U, 0.0f), 0);
// CHECK-MESSAGES: :[[@LINE-1]]:20: warning: do not use nested 'std::max' calls, use 'std::max({static_cast<int>(0U), static_cast<int>(0.0f), 0})' instead [modernize-min-max-use-initializer-list]
// CHECK-FIXES: int typecastTest = std::max({static_cast<int>(0U), static_cast<int>(0.0f), 0});

int typecastTest1 = std::max(std::max<long>(0U, 0.0f), 0L);
// CHECK-MESSAGES: :[[@LINE-1]]:21: warning: do not use nested 'std::max' calls, use 'std::max({static_cast<long>(0U), static_cast<long>(0.0f), 0L})' instead [modernize-min-max-use-initializer-list]
// CHECK-FIXES: int typecastTest1 = std::max({static_cast<long>(0U), static_cast<long>(0.0f), 0L});

int typecastTest2 = std::max(std::max<int>(10U, 20.0f), 30);
// CHECK-MESSAGES: :[[@LINE-1]]:21: warning: do not use nested 'std::max' calls, use 'std::max({static_cast<int>(10U), static_cast<int>(20.0f), 30})' instead [modernize-min-max-use-initializer-list]
// CHECK-FIXES: int typecastTest2 = std::max({static_cast<int>(10U), static_cast<int>(20.0f), 30});

int typecastTest3 = std::max(std::max<int>(0U, std::max<int>(0.0f, 1.0f)), 0);
// CHECK-MESSAGES: :[[@LINE-1]]:21: warning: do not use nested 'std::max' calls, use 'std::max({static_cast<int>(0U), static_cast<int>(0.0f), static_cast<int>(1.0f), 0})' instead [modernize-min-max-use-initializer-list]
// CHECK-FIXES: int typecastTest3 = std::max({static_cast<int>(0U), static_cast<int>(0.0f), static_cast<int>(1.0f), 0});

}
