// RUN: %check_clang_tidy %s abseil-str-cat-append %t

typedef unsigned __INT16_TYPE__ char16;
typedef unsigned __INT32_TYPE__ char32;
typedef __SIZE_TYPE__ size;

namespace std {
template <typename T>
class allocator {};
template <typename T>
class char_traits {};
template <typename C, typename T, typename A>
struct basic_string {
  typedef basic_string<C, T, A> _Type;
  basic_string();
  basic_string(const C* p, const A& a = A());

  const C* c_str() const;
  const C* data() const;

  _Type& append(const C* s);
  _Type& append(const C* s, size n);
  _Type& assign(const C* s);
  _Type& assign(const C* s, size n);

  int compare(const _Type&) const;
  int compare(const C* s) const;
  int compare(size pos, size len, const _Type&) const;
  int compare(size pos, size len, const C* s) const;

  size find(const _Type& str, size pos = 0) const;
  size find(const C* s, size pos = 0) const;
  size find(const C* s, size pos, size n) const;

  _Type& insert(size pos, const _Type& str);
  _Type& insert(size pos, const C* s);
  _Type& insert(size pos, const C* s, size n);

  _Type& operator+=(const _Type& str);
  _Type& operator+=(const C* s);
  _Type& operator=(const _Type& str);
  _Type& operator=(const C* s);
};

typedef basic_string<char, std::char_traits<char>, std::allocator<char>> string;
typedef basic_string<wchar_t, std::char_traits<wchar_t>,
                     std::allocator<wchar_t>>
    wstring;
typedef basic_string<char16, std::char_traits<char16>, std::allocator<char16>>
    u16string;
typedef basic_string<char32, std::char_traits<char32>, std::allocator<char32>>
    u32string;
}  // namespace std

std::string operator+(const std::string&, const std::string&);
std::string operator+(const std::string&, const char*);
std::string operator+(const char*, const std::string&);

bool operator==(const std::string&, const std::string&);
bool operator==(const std::string&, const char*);
bool operator==(const char*, const std::string&);

namespace llvm {
struct StringRef {
  StringRef(const char* p);
  StringRef(const std::string&);
};
}  // namespace llvm

namespace absl {

struct AlphaNum {
  AlphaNum(int i);
  AlphaNum(double f);
  AlphaNum(const char* c_str);
  AlphaNum(const std::string& str);

 private:
  AlphaNum(const AlphaNum&);
  AlphaNum& operator=(const AlphaNum&);
};

std::string StrCat(const AlphaNum& A);
std::string StrCat(const AlphaNum& A, const AlphaNum& B);

template <typename A>
void Foo(A& a) {
  a = StrCat(a);
}

void Bar() {
  std::string A, B;
  Foo<std::string>(A);

  std::string C = StrCat(A);
  A = StrCat(A);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: call to 'absl::StrCat' has no effect
  A = StrCat(A, B);
// CHECK-MESSAGES: [[@LINE-1]]:3: warning: call 'absl::StrAppend' instead of 'absl::StrCat' when appending to a string to avoid a performance penalty
// CHECK-FIXES: {{^}}  absl::StrAppend(&A, B);
  B = StrCat(A, B);

#define M(X) X = StrCat(X, A)
  M(B);
// CHECK-MESSAGES: [[@LINE-1]]:5: warning: call 'absl::StrAppend' instead of 'absl::StrCat' when appending to a string to avoid a performance penalty
// CHECK-FIXES: #define M(X) X = StrCat(X, A)
}

void Regression_SelfAppend() {
  std::string A;
  A = StrCat(A, A);
}

}  // namespace absl

void OutsideAbsl() {
  std::string A, B;
  A = absl::StrCat(A, B);
// CHECK-MESSAGES: [[@LINE-1]]:3: warning: call 'absl::StrAppend' instead of 'absl::StrCat' when appending to a string to avoid a performance penalty
// CHECK-FIXES: {{^}}  absl::StrAppend(&A, B);
}

void OutsideUsingAbsl() {
  std::string A, B;
  using absl::StrCat;
  A = StrCat(A, B);
// CHECK-MESSAGES: [[@LINE-1]]:3: warning: call 'absl::StrAppend' instead of 'absl::StrCat' when appending to a string to avoid a performance penalty
// CHECK-FIXES: {{^}}  absl::StrAppend(&A, B);
}
