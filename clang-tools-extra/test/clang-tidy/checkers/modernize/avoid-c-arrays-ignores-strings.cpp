// RUN: %check_clang_tidy -std=c++17 %s modernize-avoid-c-arrays %t -- \
// RUN:  -config='{CheckOptions: { modernize-avoid-c-arrays.AllowStringArrays: true }}'

const char name[] = "name";
const char array[] = {'n', 'a', 'm', 'e', '\0'};
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: do not declare C-style arrays, use 'std::array' instead [modernize-avoid-c-arrays]

void takeCharArray(const char name[]);
// CHECK-MESSAGES: :[[@LINE-1]]:26: warning: do not declare C-style arrays, use 'std::array' or 'std::vector' instead [modernize-avoid-c-arrays]

template <typename T = const char[10], typename U = char[10], char[10]>
// CHECK-MESSAGES: :[[@LINE-1]]:30: warning: do not declare C-style arrays, use 'std::array' instead [modernize-avoid-c-arrays]
// CHECK-MESSAGES: :[[@LINE-2]]:53: warning: do not declare C-style arrays, use 'std::array' instead [modernize-avoid-c-arrays]
// CHECK-MESSAGES: :[[@LINE-3]]:63: warning: do not declare C-style arrays, use 'std::array' instead [modernize-avoid-c-arrays]
void func() {}

template <typename T = const char[], typename U = char[], char[]>
// CHECK-MESSAGES: :[[@LINE-1]]:30: warning: do not declare C-style arrays, use 'std::array' instead [modernize-avoid-c-arrays]
// CHECK-MESSAGES: :[[@LINE-2]]:51: warning: do not declare C-style arrays, use 'std::array' instead [modernize-avoid-c-arrays]
// CHECK-MESSAGES: :[[@LINE-3]]:59: warning: do not declare C-style arrays, use 'std::array' instead [modernize-avoid-c-arrays]
void fun() {}
