// RUN: %check_clang_tidy -std=c++17 %s modernize-avoid-c-arrays %t -- \
// RUN:  -config='{CheckOptions: { modernize-avoid-c-arrays.AllowStringArrays: true }}'

const char name[] = "name";
const char array[] = {'n', 'a', 'm', 'e', '\0'};
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: do not declare C-style arrays, use 'std::array' instead [modernize-avoid-c-arrays]

void takeCharArray(const char name[]);
// CHECK-MESSAGES: :[[@LINE-1]]:26: warning: do not declare C-style arrays, use 'std::array' or 'std::vector' instead [modernize-avoid-c-arrays]
