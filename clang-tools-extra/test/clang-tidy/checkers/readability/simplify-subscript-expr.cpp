// RUN: %check_clang_tidy %s readability-simplify-subscript-expr %t \
// RUN: -config="{CheckOptions: \
// RUN: [{key: readability-simplify-subscript-expr.Types, \
// RUN:   value: '::std::basic_string;::std::basic_string_view;MyVector'}]}" --

namespace std {

template <class T>
class basic_string {
 public:
   using size_type = unsigned;
   using value_type = T;
   using reference = value_type&;
   using const_reference = const value_type&;

   reference operator[](size_type);
   const_reference operator[](size_type) const;
   T* data();
   const T* data() const;
};

using string = basic_string<char>;

template <class T>
class basic_string_view {
 public:
  using size_type = unsigned;
  using const_reference = const T&;
  using const_pointer = const T*;

  constexpr const_reference operator[](size_type) const;
  constexpr const_pointer data() const noexcept;
};

using string_view = basic_string_view<char>;

}

template <class T>
class MyVector {
 public:
  using size_type = unsigned;
  using const_reference = const T&;
  using const_pointer = const T*;

  const_reference operator[](size_type) const;
  const T* data() const noexcept;
};

#define DO(x) do { x; } while (false)
#define ACCESS(x) (x)
#define GET(x, i) (x).data()[i]

template <class T>
class Foo {
 public:
  char bar(int i) {
    return x.data()[i];
  }
 private:
  T x;
};

void f(int i) {
  MyVector<int> v;
  int x = v.data()[i];
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: accessing an element of the container does not require a call to 'data()'; did you mean to use 'operator[]'? [readability-simplify-subscript-expr]
  // CHECK-FIXES: int x = v[i];

  std::string s;
  char c1 = s.data()[i];
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: accessing an element
  // CHECK-FIXES: char c1 = s[i];

  std::string_view sv;
  char c2 = sv.data()[i];
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: accessing an element
  // CHECK-FIXES: char c2 = sv[i];

  std::string* ps = &s;
  char c3 = ps->data()[i];
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: accessing an element
  // CHECK-FIXES: char c3 = (*ps)[i];

  char c4 = (*ps).data()[i];
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: accessing an element
  // CHECK-FIXES: char c4 = (*ps)[i];

  DO(char c5 = s.data()[i]);
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: accessing an element
  // CHECK-FIXES: DO(char c5 = s[i]);

  char c6 = ACCESS(s).data()[i];
  // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: accessing an element
  // CHECK-FIXES: char c6 = ACCESS(s)[i];

  char c7 = ACCESS(s.data())[i];
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: accessing an element
  // CHECK-FIXES: char c7 = ACCESS(s)[i];

  char c8 = ACCESS(s.data()[i]);
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: accessing an element
  // CHECK-FIXES: char c8 = ACCESS(s[i]);

  char c9 = GET(s, i);

  char c10 = Foo<std::string>{}.bar(i);
}
