// RUN: %check_clang_tidy -std=c++14 %s modernize-type-traits %t -check-suffixes=',MACRO'
// RUN: %check_clang_tidy -std=c++14 %s modernize-type-traits %t -- \
// RUN:   -config='{CheckOptions: {modernize-type-traits.IgnoreMacros: true}}'
// RUN: %check_clang_tidy -std=c++17 %s modernize-type-traits %t -check-suffixes=',CXX17,MACRO,CXX17MACRO'

namespace std {
  template <typename>
  struct is_const {
    static constexpr bool value = true;
  };

  template <typename, typename>
  struct is_same {
    static constexpr bool value = true;
  };

  template<bool, typename T = void>
  struct enable_if {
    using type = T;
  };

inline namespace __std_lib_version1 {
  template<typename T>
  struct add_const {
    using type = T;
  };
} // namespace __std_lib_version1

namespace ext {
  template<typename T>
  struct add_const {
    using type = T;
  };
} // namespace ext

} // namespace std

bool NoTemplate = std::is_const<bool>::value;
// CHECK-MESSAGES-CXX17: :[[@LINE-1]]:19: warning: use c++17 style variable templates
// CHECK-FIXES-CXX17: bool NoTemplate = std::is_const_v<bool>

template<typename T>
constexpr bool InTemplate = std::is_const<T>::value;
// CHECK-MESSAGES-CXX17: :[[@LINE-1]]:29: warning: use c++17 style variable templates
// CHECK-FIXES-CXX17: constexpr bool InTemplate = std::is_const_v<T>;

template<typename U, typename V>
constexpr bool Template2Params = std::is_same<U,V>::value;
// CHECK-MESSAGES-CXX17: :[[@LINE-1]]:34: warning: use c++17 style variable templates
// CHECK-FIXES-CXX17: constexpr bool Template2Params = std::is_same_v<U,V>;

template<bool b>
typename std::enable_if<b>::type inTemplate();
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use c++14 style type templates
// CHECK-FIXES: std::enable_if_t<b>inTemplate();

typename std::enable_if<true>::type noTemplate();
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use c++14 style type templates
// CHECK-FIXES: std::enable_if_t<true>noTemplate();

std::enable_if<true>::type noTemplateOrTypename();
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use c++14 style type templates
// CHECK-FIXES: std::enable_if_t<true>noTemplateOrTypename();

using UsingNoTypename = std::enable_if<true>::type;
// CHECK-MESSAGES: :[[@LINE-1]]:25: warning: use c++14 style type templates
// CHECK-FIXES: using UsingNoTypename = std::enable_if_t<true>;

using UsingSpace = std::enable_if <true>::type;
// CHECK-MESSAGES: :[[@LINE-1]]:20: warning: use c++14 style type templates
// CHECK-FIXES: using UsingSpace = std::enable_if_t <true>;

template<bool b>
using UsingSpaceTemplate = typename std::enable_if <b>::type;
// CHECK-MESSAGES: :[[@LINE-1]]:37: warning: use c++14 style type templates
// CHECK-FIXES: using UsingSpaceTemplate = std::enable_if_t <b>;

bool NoTemplateSpace = std::is_const <bool> ::value;
// CHECK-MESSAGES-CXX17: :[[@LINE-1]]:24: warning: use c++17 style variable templates
// CHECK-FIXES-CXX17: bool NoTemplateSpace = std::is_const_v <bool> ;

template<typename T>
constexpr bool InTemplateSpace = std::is_const  <T> ::value;
// CHECK-MESSAGES-CXX17: :[[@LINE-1]]:34: warning: use c++17 style variable templates
// CHECK-FIXES-CXX17: constexpr bool InTemplateSpace = std::is_const_v  <T> ;

// For macros, no diagnostics if IgnoreMacros is set,
// No fixes emitted even if IgnoreMacros is unset.

#define VALUE_MACRO std::is_same<int, int>::value
bool MacroValue = VALUE_MACRO;
// CHECK-MESSAGES-CXX17MACRO: :[[@LINE-1]]:19: warning: use c++17 style variable templates
// CHECK-FIXES-CXX17MACRO: #define VALUE_MACRO std::is_same<int, int>::value

#define TYPE_MACRO typename std::enable_if<true>::type
using MacroType = TYPE_MACRO;
// CHECK-MESSAGES-MACRO: :[[@LINE-1]]:19: warning: use c++14 style type templates
// CHECK-FIXES-MACRO: #define TYPE_MACRO typename std::enable_if<true>::type


// Names defined and accessed inside an inline namespace should be converted.
// Whether or not the inline namespace is specified

using InlineUnspecified = std::add_const<bool>::type;
// CHECK-MESSAGES: :[[@LINE-1]]:27: warning: use c++14 style type templates
// CHECK-FIXES: using InlineUnspecified = std::add_const_t<bool>;

using Inline = std::__std_lib_version1::add_const<bool>::type;
// CHECK-MESSAGES: :[[@LINE-1]]:16: warning: use c++14 style type templates
// CHECK-FIXES: using Inline = std::__std_lib_version1::add_const_t<bool>;

// Don't try to offer any fix if the name is an extension to the standard library
using Ext = std::ext::add_const<bool>::type;

namespace my_std = std;

using Alias = my_std::add_const<bool>::type;
// CHECK-MESSAGES: :[[@LINE-1]]:15: warning: use c++14 style type templates
// CHECK-FIXES: using Alias = my_std::add_const_t<bool>;
