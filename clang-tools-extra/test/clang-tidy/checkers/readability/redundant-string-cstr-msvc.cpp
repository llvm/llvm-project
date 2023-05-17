// RUN: %check_clang_tidy %s readability-redundant-string-cstr %t

namespace std {
template <typename T>
class allocator {};
template <typename T>
class char_traits {};
template <typename C, typename T, typename A>
struct basic_string {
  basic_string();
  // MSVC headers define two constructors instead of using optional arguments.
  basic_string(const C *p);
  basic_string(const C *p, const A &a);
  const C *c_str() const;
  const C *data() const;
};
typedef basic_string<char, std::char_traits<char>, std::allocator<char>> string;
}
namespace llvm {
struct StringRef {
  StringRef(const char *p);
  StringRef(const std::string &);
};
}

void f1(const std::string &s) {
  f1(s.c_str());
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: redundant call to 'c_str' [readability-redundant-string-cstr]
  // CHECK-FIXES: {{^  }}f1(s);{{$}}
  f1(s.data());
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: redundant call to 'data' [readability-redundant-string-cstr]
  // CHECK-FIXES: {{^  }}f1(s);{{$}}
}
void f2(const llvm::StringRef r) {
  std::string s;
  f2(s.c_str());
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: redundant call {{.*}}
  // CHECK-FIXES: {{^  }}std::string s;{{$}}
  // CHECK-FIXES-NEXT: {{^  }}f2(s);{{$}}
}
void f3(const llvm::StringRef &r) {
  std::string s;
  f3(s.c_str());
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: redundant call {{.*}}
  // CHECK-FIXES: {{^  }}std::string s;{{$}}
  // CHECK-FIXES-NEXT: {{^  }}f3(s);{{$}}
}
