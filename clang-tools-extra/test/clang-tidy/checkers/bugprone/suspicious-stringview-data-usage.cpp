// RUN: %check_clang_tidy -std=c++17-or-later %s bugprone-suspicious-stringview-data-usage %t -- -- -isystem %clang_tidy_headers
#include <string>

struct View {
   const char* str;
};

struct Pair {
   const char* begin;
   const char* end;
};

struct ViewWithSize {
   const char* str;
   std::string_view::size_type size;
};

void something(const char*);
void something(const char*, unsigned);
void something(const char*, unsigned, const char*);
void something_str(std::string, unsigned);

void invalid(std::string_view sv, std::string_view sv2) {
  std::string s(sv.data());
// CHECK-MESSAGES: :[[@LINE-1]]:20: warning: result of a `data()` call may not be null terminated, provide size information to the callee to prevent potential issues
  std::string si{sv.data()};
// CHECK-MESSAGES: :[[@LINE-1]]:21: warning: result of a `data()` call may not be null terminated, provide size information to the callee to prevent potential issues
  std::string_view s2(sv.data());
// CHECK-MESSAGES: :[[@LINE-1]]:26: warning: result of a `data()` call may not be null terminated, provide size information to the callee to prevent potential issues
  something(sv.data());
// CHECK-MESSAGES: :[[@LINE-1]]:16: warning: result of a `data()` call may not be null terminated, provide size information to the callee to prevent potential issues
  something(sv.data(), sv.size(), sv2.data());
// CHECK-MESSAGES: :[[@LINE-1]]:39: warning: result of a `data()` call may not be null terminated, provide size information to the callee to prevent potential issues
  something_str(sv.data(), sv.size());
// CHECK-MESSAGES: :[[@LINE-1]]:20: warning: result of a `data()` call may not be null terminated, provide size information to the callee to prevent potential issues
  View view{sv.data()};
// CHECK-MESSAGES: :[[@LINE-1]]:16: warning: result of a `data()` call may not be null terminated, provide size information to the callee to prevent potential issues
}

void valid(std::string_view sv) {
  std::string s1(sv.data(), sv.data() + sv.size());
  std::string s2(sv.data(), sv.data() + sv.length());
  std::string s3(sv.data(), sv.size() + sv.data());
  std::string s4(sv.data(), sv.length() + sv.data());
  std::string s5(sv.data(), sv.size());
  std::string s6(sv.data(), sv.length());
  something(sv.data(), sv.size());
  something(sv.data(), sv.length());
  ViewWithSize view1{sv.data(), sv.size()};
  ViewWithSize view2{sv.data(), sv.length()};
  Pair view3{sv.data(), sv.data() + sv.size()};
  Pair view4{sv.data(), sv.data() + sv.length()};
  Pair view5{sv.data(), sv.size() + sv.data()};
  Pair view6{sv.data(), sv.length() + sv.data()};
  const char* str{sv.data()};
}
