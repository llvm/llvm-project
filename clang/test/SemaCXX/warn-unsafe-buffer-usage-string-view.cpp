// RUN: %clang_cc1 -std=c++20 -Wunsafe-buffer-usage-in-container -verify %s

namespace std {
  typedef __SIZE_TYPE__ size_t;
  template <typename T> class basic_string_view {
  public:
    basic_string_view(const T *, size_t);
    template <typename It> basic_string_view(It, It);
  };
  typedef basic_string_view<char> string_view;
  typedef basic_string_view<wchar_t> wstring_view;
  template <typename T> class vector { public: T* begin(); T* end(); };
}

typedef std::size_t size_t;

void test_final_coverage() {
  std::vector<char> v1, v2;
  
  // 1. Iterator Pairs
  std::string_view it_ok(v1.begin(), v1.end()); // no-warning
  // expected-warning@+1 {{the two-parameter std::string_view construction is unsafe}}
  std::string_view it_bad(v1.begin(), v2.end()); 

  // 2. Character Types
  std::string_view s1("hi", 2); // no-warning
  // expected-warning@+1 {{the two-parameter std::string_view construction is unsafe}}
  std::string_view s2("hi", 3); 
  
  std::wstring_view w1(L"hi", 2); // no-warning
  // expected-warning@+1 {{the two-parameter std::string_view construction is unsafe}}
  std::wstring_view w2(L"hi", 3); 

  // 3. Arrays
  char arr[5];
  std::string_view a1(arr, 5); // no-warning
  // expected-warning@+1 {{the two-parameter std::string_view construction is unsafe}}
  std::string_view a2(arr, 6); 

  // 4. Dynamic/Unknown
  extern size_t get_size();
  // expected-warning@+1 {{the two-parameter std::string_view construction is unsafe}}
  std::string_view d1("hi", get_size()); 
}
