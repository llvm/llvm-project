// RUN: %clang_cc1 -std=c++11 %s -Wunused -verify
// expected-no-diagnostics

template<typename T, typename U>
struct is_same {
  static const bool value = false;
};

template<typename T>
struct is_same<T, T> {
  static const bool value = true;
};

void f3() {
  float x, &r = x;
  int i;
  int &ir = i;
  const int &irc = i;

  [=,&irc,&ir] {
    static_assert(is_same<decltype(((r))), float const&>::value, 
                  "should be const float&");
    static_assert(is_same<decltype(x), float>::value, "should be float");
    static_assert(is_same<decltype((x)), const float&>::value, 
                  "should be const float&");
    static_assert(is_same<decltype(r), float&>::value, "should be float&");
    static_assert(is_same<decltype(ir), int&>::value, "should be int&");
    static_assert(is_same<decltype((ir)), int&>::value, "should be int&");
    static_assert(is_same<decltype(irc), const int&>::value, 
                  "should be const int&");
    static_assert(is_same<decltype((irc)), const int&>::value, 
                  "should be const int&");
  }();

  [=] {
    [=] () mutable {
      static_assert(is_same<decltype(x), float>::value, "should be float");
      static_assert(is_same<decltype((x)), float&>::value, 
                    "should be float&");
    }();
  }();

  [&i] {
    static_assert(is_same<decltype((i)), int&>::value, "should be int&");
  }();
}
