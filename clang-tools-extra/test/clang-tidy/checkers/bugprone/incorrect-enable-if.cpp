// RUN: %check_clang_tidy -std=c++11 %s bugprone-incorrect-enable-if %t
// RUN: %check_clang_tidy -check-suffix=CXX20 -std=c++20 %s bugprone-incorrect-enable-if %t

// NOLINTBEGIN
namespace std {
template <bool B, class T = void> struct enable_if { };

template <class T> struct enable_if<true, T> { typedef T type; };

template <bool B, class T = void>
using enable_if_t = typename enable_if<B, T>::type;

} // namespace std
// NOLINTEND

template <typename T, typename = typename std::enable_if<T::some_value>::type>
void valid_function1() {}

template <typename T, typename std::enable_if<T::some_value>::type = nullptr>
void valid_function2() {}

template <typename T, typename std::enable_if<T::some_value>::type = nullptr>
struct ValidClass1 {};

template <typename T, typename = std::enable_if<T::some_value>>
void invalid() {}
// CHECK-MESSAGES: [[@LINE-2]]:23: warning: incorrect std::enable_if usage detected; use 'typename std::enable_if<...>::type' [bugprone-incorrect-enable-if]
// CHECK-FIXES: template <typename T, typename = typename std::enable_if<T::some_value>::type>
// CHECK-FIXES-CXX20: template <typename T, typename = std::enable_if<T::some_value>::type>

template <typename T, typename = std::enable_if<T::some_value> >
void invalid_extra_whitespace() {}
// CHECK-MESSAGES: [[@LINE-2]]:23: warning: incorrect std::enable_if usage detected; use 'typename std::enable_if<...>::type' [bugprone-incorrect-enable-if]
// CHECK-FIXES: template <typename T, typename = typename std::enable_if<T::some_value>::type >
// CHECK-FIXES-CXX20: template <typename T, typename = std::enable_if<T::some_value>::type >

template <typename T, typename=std::enable_if<T::some_value>>
void invalid_extra_no_whitespace() {}
// CHECK-MESSAGES: [[@LINE-2]]:23: warning: incorrect std::enable_if usage detected; use 'typename std::enable_if<...>::type' [bugprone-incorrect-enable-if]
// CHECK-FIXES: template <typename T, typename=typename std::enable_if<T::some_value>::type>
// CHECK-FIXES-CXX20: template <typename T, typename=std::enable_if<T::some_value>::type>

template <typename T, typename /*comment1*/ = /*comment2*/std::enable_if<T::some_value>/*comment3*/>
void invalid_extra_comment() {}
// CHECK-MESSAGES: [[@LINE-2]]:23: warning: incorrect std::enable_if usage detected; use 'typename std::enable_if<...>::type' [bugprone-incorrect-enable-if]
// CHECK-FIXES: template <typename T, typename /*comment1*/ = /*comment2*/typename std::enable_if<T::some_value>::type/*comment3*/>

template <typename T, typename = std::enable_if<T::some_value>, typename = std::enable_if<T::other_value>>
void invalid_multiple() {}
// CHECK-MESSAGES: [[@LINE-2]]:23: warning: incorrect std::enable_if usage detected; use 'typename std::enable_if<...>::type' [bugprone-incorrect-enable-if]
// CHECK-MESSAGES: [[@LINE-3]]:65: warning: incorrect std::enable_if usage detected; use 'typename std::enable_if<...>::type' [bugprone-incorrect-enable-if]
// CHECK-FIXES: template <typename T, typename = typename std::enable_if<T::some_value>::type, typename = typename std::enable_if<T::other_value>::type>
// CHECK-FIXES-CXX20: template <typename T, typename = std::enable_if<T::some_value>::type, typename = std::enable_if<T::other_value>::type>

template <typename T, typename = std::enable_if<T::some_value>>
struct InvalidClass {};
// CHECK-MESSAGES: [[@LINE-2]]:23: warning: incorrect std::enable_if usage detected; use 'typename std::enable_if<...>::type' [bugprone-incorrect-enable-if]
// CHECK-FIXES: template <typename T, typename = typename std::enable_if<T::some_value>::type>
// CHECK-FIXES-CXX20: template <typename T, typename = std::enable_if<T::some_value>::type>
