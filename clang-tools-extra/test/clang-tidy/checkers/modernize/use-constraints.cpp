// RUN: %check_clang_tidy -std=c++20 %s modernize-use-constraints %t -- -- -fno-delayed-template-parsing

// NOLINTBEGIN
namespace std {
template <bool B, class T = void> struct enable_if { };

template <class T> struct enable_if<true, T> { typedef T type; };

template <bool B, class T = void>
using enable_if_t = typename enable_if<B, T>::type;

} // namespace std
// NOLINTEND

template <typename...>
struct ConsumeVariadic;

struct Obj {
};

namespace enable_if_in_return_type {

////////////////////////////////
// Section 1: enable_if in return type of function
////////////////////////////////

////////////////////////////////
// General tests
////////////////////////////////

template <typename T>
typename std::enable_if<T::some_value, Obj>::type basic() {
  return Obj{};
}
// CHECK-MESSAGES: :[[@LINE-3]]:1: warning: use C++20 requires constraints instead of enable_if [modernize-use-constraints]
// CHECK-FIXES: {{^}}Obj basic() requires T::some_value {{{$}}

template <typename T>
std::enable_if_t<T::some_value, Obj> basic_t() {
  return Obj{};
}
// CHECK-MESSAGES: :[[@LINE-3]]:1: warning: use C++20 requires constraints instead of enable_if [modernize-use-constraints]
// CHECK-FIXES: {{^}}Obj basic_t() requires T::some_value {{{$}}

template <typename T>
auto basic_trailing() -> typename std::enable_if<T::some_value, Obj>::type {
  return Obj{};
}
// CHECK-MESSAGES: :[[@LINE-3]]:26: warning: use C++20 requires constraints instead of enable_if [modernize-use-constraints]
// CHECK-FIXES: {{^}}auto basic_trailing() -> Obj requires T::some_value {{{$}}

template <typename T>
typename std::enable_if<T::some_value, Obj>::type existing_constraint() requires (T::another_value) {
  return Obj{};
}
// CHECK-MESSAGES: :[[@LINE-3]]:1: warning: use C++20 requires constraints instead of enable_if [modernize-use-constraints]
// CHECK-FIXES: {{^}}typename std::enable_if<T::some_value, Obj>::type existing_constraint() requires (T::another_value) {{{$}}

template <typename U>
typename std::enable_if<U::some_value, Obj>::type decl_without_def();

template <typename U>
typename std::enable_if<U::some_value, Obj>::type decl_with_separate_def();

template <typename U>
typename std::enable_if<U::some_value, Obj>::type decl_with_separate_def() {
  return Obj{};
}
// FIXME - Support definitions with separate decls

template <typename U>
std::enable_if_t<true, Obj> no_dependent_type(U) {
  return Obj{};
}
// FIXME - Support non-dependent enable_ifs. Low priority though...

template <typename T>
typename std::enable_if<T::some_value, int>::type* pointer_of_enable_if() {
  return nullptr;
}
// CHECK-MESSAGES: :[[@LINE-3]]:1: warning: use C++20 requires constraints instead of enable_if [modernize-use-constraints]
// CHECK-FIXES: {{^}}template <typename T>{{$}}
// CHECK-FIXES-NEXT: {{^}}int* pointer_of_enable_if() requires T::some_value {{{$}}

template <typename T>
std::enable_if_t<T::some_value, int>* pointer_of_enable_if_t() {
  return nullptr;
}
// CHECK-MESSAGES: :[[@LINE-3]]:1: warning: use C++20 requires constraints instead of enable_if [modernize-use-constraints]
// CHECK-FIXES: {{^}}template <typename T>{{$}}
// CHECK-FIXES-NEXT: {{^}}int* pointer_of_enable_if_t() requires T::some_value {{{$}}

template <typename T>
const std::enable_if_t<T::some_value, int>* const_pointer_of_enable_if_t() {
  return nullptr;
}
// CHECK-MESSAGES: :[[@LINE-3]]:7: warning: use C++20 requires constraints instead of enable_if [modernize-use-constraints]
// CHECK-FIXES: {{^}}template <typename T>{{$}}
// CHECK-FIXES-NEXT: {{^}}const int* const_pointer_of_enable_if_t() requires T::some_value {{{$}}

template <typename T>
std::enable_if_t<T::some_value, int> const * const_pointer_of_enable_if_t2() {
  return nullptr;
}
// CHECK-MESSAGES: :[[@LINE-3]]:1: warning: use C++20 requires constraints instead of enable_if [modernize-use-constraints]
// CHECK-FIXES: {{^}}template <typename T>{{$}}
// CHECK-FIXES-NEXT: {{^}}int const * const_pointer_of_enable_if_t2() requires T::some_value {{{$}}


template <typename T>
std::enable_if_t<T::some_value, int>& reference_of_enable_if_t() {
  static int x; return x;
}
// CHECK-MESSAGES: :[[@LINE-3]]:1: warning: use C++20 requires constraints instead of enable_if [modernize-use-constraints]
// CHECK-FIXES: {{^}}template <typename T>{{$}}
// CHECK-FIXES-NEXT: {{^}}int& reference_of_enable_if_t() requires T::some_value {{{$}}

template <typename T>
const std::enable_if_t<T::some_value, int>& const_reference_of_enable_if_t() {
  static int x; return x;
}
// CHECK-MESSAGES: :[[@LINE-3]]:7: warning: use C++20 requires constraints instead of enable_if [modernize-use-constraints]
// CHECK-FIXES: {{^}}template <typename T>{{$}}
// CHECK-FIXES-NEXT: {{^}}const int& const_reference_of_enable_if_t() requires T::some_value {{{$}}

template <typename T>
typename std::enable_if<T::some_value>::type enable_if_default_void() {
}
// CHECK-MESSAGES: :[[@LINE-2]]:1: warning: use C++20 requires constraints instead of enable_if [modernize-use-constraints]
// CHECK-FIXES: {{^}}void enable_if_default_void() requires T::some_value {{{$}}

template <typename T>
std::enable_if_t<T::some_value> enable_if_t_default_void() {
}
// CHECK-MESSAGES: :[[@LINE-2]]:1: warning: use C++20 requires constraints instead of enable_if [modernize-use-constraints]
// CHECK-FIXES: {{^}}void enable_if_t_default_void() requires T::some_value {{{$}}

template <typename T>
std::enable_if_t<T::some_value>* enable_if_t_default_void_pointer() {
}
// CHECK-MESSAGES: :[[@LINE-2]]:1: warning: use C++20 requires constraints instead of enable_if [modernize-use-constraints]
// CHECK-FIXES: {{^}}void* enable_if_t_default_void_pointer() requires T::some_value {{{$}}

namespace using_namespace_std {

using namespace std;

template <typename T>
typename enable_if<T::some_value>::type with_typename() {
}
// CHECK-MESSAGES: :[[@LINE-2]]:1: warning: use C++20 requires constraints instead of enable_if [modernize-use-constraints]
// CHECK-FIXES: {{^}}void with_typename() requires T::some_value {{{$}}

template <typename T>
enable_if_t<T::some_value> with_t() {
}
// CHECK-MESSAGES: :[[@LINE-2]]:1: warning: use C++20 requires constraints instead of enable_if [modernize-use-constraints]
// CHECK-FIXES: {{^}}void with_t() requires T::some_value {{{$}}

template <typename T>
typename enable_if<T::some_value, int>::type with_typename_and_type() {
}
// CHECK-MESSAGES: :[[@LINE-2]]:1: warning: use C++20 requires constraints instead of enable_if [modernize-use-constraints]
// CHECK-FIXES: {{^}}int with_typename_and_type() requires T::some_value {{{$}}

template <typename T>
enable_if_t<T::some_value, int> with_t_and_type() {
}
// CHECK-MESSAGES: :[[@LINE-2]]:1: warning: use C++20 requires constraints instead of enable_if [modernize-use-constraints]
// CHECK-FIXES: {{^}}int with_t_and_type() requires T::some_value {{{$}}

} // namespace using_namespace_std


////////////////////////////////
// Negative tests - incorrect uses of enable_if
////////////////////////////////
template <typename U>
std::enable_if<U::some_value, Obj> not_enable_if() {
  return {};
}
template <typename U>
typename std::enable_if<U::some_value, Obj>::type123 not_enable_if_wrong_type() {
  return {};
}
template <typename U>
typename std::enable_if_t<U::some_value, Obj>::type not_enable_if_t() {
  return {};
}
template <typename U>
typename std::enable_if_t<U::some_value, Obj>::type123 not_enable_if_t_again() {
  return {};
}
template <typename U>
std::enable_if<U::some_value, int>* not_pointer_of_enable_if() {
  return nullptr;
}
template <typename U>
typename std::enable_if<U::some_value, int>::type123 * not_pointer_of_enable_if_t() {
  return nullptr;
}


namespace primary_expression_tests {

////////////////////////////////
// Primary/non-primary expression tests
////////////////////////////////

template <typename T> struct Traits;

template <typename T>
std::enable_if_t<Traits<T>::value> type_trait_value() {
}
// CHECK-MESSAGES: :[[@LINE-2]]:1: warning: use C++20 requires constraints instead of enable_if [modernize-use-constraints]
// CHECK-FIXES: {{^}}void type_trait_value() requires Traits<T>::value {{{$}}

template <typename T>
std::enable_if_t<Traits<T>::member()> type_trait_member_call() {
}
// CHECK-MESSAGES: :[[@LINE-2]]:1: warning: use C++20 requires constraints instead of enable_if [modernize-use-constraints]
// CHECK-FIXES: {{^}}void type_trait_member_call() requires (Traits<T>::member()) {{{$}}

template <typename T>
std::enable_if_t<!Traits<T>::value> negate() {
}
// CHECK-MESSAGES: :[[@LINE-2]]:1: warning: use C++20 requires constraints instead of enable_if [modernize-use-constraints]
// CHECK-FIXES: {{^}}void negate() requires (!Traits<T>::value) {{{$}}

template <typename T>
std::enable_if_t<Traits<T>::value1 && Traits<T>::value2> conjunction() {
}
// CHECK-MESSAGES: :[[@LINE-2]]:1: warning: use C++20 requires constraints instead of enable_if [modernize-use-constraints]
// CHECK-FIXES: {{^}}void conjunction() requires (Traits<T>::value1 && Traits<T>::value2) {{{$}}

template <typename T>
std::enable_if_t<Traits<T>::value1 || Traits<T>::value2> disjunction() {
}
// CHECK-MESSAGES: :[[@LINE-2]]:1: warning: use C++20 requires constraints instead of enable_if [modernize-use-constraints]
// CHECK-FIXES: {{^}}void disjunction() requires (Traits<T>::value1 || Traits<T>::value2) {{{$}}

template <typename T>
std::enable_if_t<Traits<T>::value1 && !Traits<T>::value2> conjunction_with_negate() {
}
// CHECK-MESSAGES: :[[@LINE-2]]:1: warning: use C++20 requires constraints instead of enable_if [modernize-use-constraints]
// CHECK-FIXES: {{^}}void conjunction_with_negate() requires (Traits<T>::value1 && !Traits<T>::value2) {{{$}}

template <typename T>
std::enable_if_t<Traits<T>::value1 == (Traits<T>::value2 + 5)> complex_operators() {
}
// CHECK-MESSAGES: :[[@LINE-2]]:1: warning: use C++20 requires constraints instead of enable_if [modernize-use-constraints]
// CHECK-FIXES: {{^}}void complex_operators() requires (Traits<T>::value1 == (Traits<T>::value2 + 5)) {{{$}}

} // namespace primary_expression_tests


////////////////////////////////
// Functions with specifier
////////////////////////////////

template <typename T>
constexpr typename std::enable_if<T::some_value, int>::type constexpr_decl() {
  return 10;
}
// CHECK-MESSAGES: :[[@LINE-3]]:11: warning: use C++20 requires constraints instead of enable_if [modernize-use-constraints]
// CHECK-FIXES: {{^}}constexpr int constexpr_decl() requires T::some_value {{{$}}

template <typename T>
static inline constexpr typename std::enable_if<T::some_value, int>::type static_inline_constexpr_decl() {
  return 10;
}
// CHECK-MESSAGES: :[[@LINE-3]]:25: warning: use C++20 requires constraints instead of enable_if [modernize-use-constraints]
// CHECK-FIXES: {{^}}static inline constexpr int static_inline_constexpr_decl() requires T::some_value {{{$}}

template <typename T>
static
typename std::enable_if<T::some_value, int>::type
static_decl() {
  return 10;
}
// CHECK-MESSAGES: :[[@LINE-4]]:1: warning: use C++20 requires constraints instead of enable_if [modernize-use-constraints]
// CHECK-FIXES: {{^}}static{{$}}
// CHECK-FIXES-NEXT: {{^}}int{{$}}
// CHECK-FIXES-NEXT: {{^}}static_decl() requires T::some_value {{{$}}

template <typename T>
constexpr /* comment */ typename std::enable_if<T::some_value, int>::type constexpr_comment_decl() {
  return 10;
}
// CHECK-MESSAGES: :[[@LINE-3]]:25: warning: use C++20 requires constraints instead of enable_if [modernize-use-constraints]
// CHECK-FIXES: {{^}}constexpr /* comment */ int constexpr_comment_decl() requires T::some_value {{{$}}


////////////////////////////////
// Class definition tests
////////////////////////////////

struct AClass {

  template <typename T>
  static typename std::enable_if<T::some_value, Obj>::type static_method() {
    return Obj{};
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:10: warning: use C++20 requires constraints instead of enable_if [modernize-use-constraints]
  // CHECK-FIXES: {{^}}  static Obj static_method() requires T::some_value {{{$}}

  template <typename T>
  typename std::enable_if<T::some_value, Obj>::type member() {
    return Obj{};
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use C++20 requires constraints instead of enable_if [modernize-use-constraints]
  // CHECK-FIXES: {{^}}  Obj member() requires T::some_value {{{$}}

  template <typename T>
  typename std::enable_if<T::some_value, Obj>::type const_qualifier() const {
    return Obj{};
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use C++20 requires constraints instead of enable_if [modernize-use-constraints]
  // CHECK-FIXES: {{^}}  Obj const_qualifier() const requires T::some_value {{{$}}

  template <typename T>
  typename std::enable_if<T::some_value, Obj>::type rvalue_ref_qualifier() && {
    return Obj{};
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use C++20 requires constraints instead of enable_if [modernize-use-constraints]
  // CHECK-FIXES: {{^}}  Obj rvalue_ref_qualifier() && requires T::some_value {{{$}}

  template <typename T>
  typename std::enable_if<T::some_value, Obj>::type rvalue_ref_qualifier_comment() /* c1 */ && /* c2 */ {
    return Obj{};
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use C++20 requires constraints instead of enable_if [modernize-use-constraints]
  // CHECK-FIXES: {{^}}  Obj rvalue_ref_qualifier_comment() /* c1 */ && /* c2 */ requires T::some_value {{{$}}

  template <typename T>
  std::enable_if_t<T::some_value, AClass&> operator=(T&&) = delete;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use C++20 requires constraints instead of enable_if [modernize-use-constraints]
  // CHECK-FIXES: {{^}}  AClass& operator=(T&&) requires T::some_value = delete;

  template<typename T>
  std::enable_if_t<T::some_value, AClass&> operator=(ConsumeVariadic<T>) noexcept(requires (T t) { t = 4; }) = delete;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use C++20 requires constraints instead of enable_if [modernize-use-constraints]
  // CHECK-FIXES: {{^}}  AClass& operator=(ConsumeVariadic<T>) noexcept(requires (T t) { t = 4; }) requires T::some_value = delete;

};


////////////////////////////////
// Comments and whitespace tests
////////////////////////////////

template <typename T>
typename std::enable_if</* check1 */ T::some_value, Obj>::type leading_comment() {
  return Obj{};
}
// CHECK-MESSAGES: :[[@LINE-3]]:1: warning: use C++20 requires constraints instead of enable_if [modernize-use-constraints]
// CHECK-FIXES: {{^}}Obj leading_comment() requires /* check1 */ T::some_value {{{$}}

template <typename T>
typename std::enable_if<T::some_value, Obj>::type body_on_next_line()
{
  return Obj{};
}
// CHECK-MESSAGES: :[[@LINE-4]]:1: warning: use C++20 requires constraints instead of enable_if [modernize-use-constraints]
// CHECK-FIXES: {{^}}Obj body_on_next_line(){{$}}
// CHECK-FIXES-NEXT: {{^}}requires T::some_value {{{$}}

template <typename T>
typename std::enable_if<  /* check1 */ T::some_value, Obj>::type leading_comment_whitespace() {
  return Obj{};
}
// CHECK-MESSAGES: :[[@LINE-3]]:1: warning: use C++20 requires constraints instead of enable_if [modernize-use-constraints]
// CHECK-FIXES: {{^}}Obj leading_comment_whitespace() requires /* check1 */ T::some_value {{{$}}

template <typename T>
typename std::enable_if</* check1 */ T::some_value /* check2 */, Obj>::type leading_and_trailing_comment() {
  return Obj{};
}
// CHECK-MESSAGES: :[[@LINE-3]]:1: warning: use C++20 requires constraints instead of enable_if [modernize-use-constraints]
// CHECK-FIXES: {{^}}Obj leading_and_trailing_comment() requires /* check1 */ T::some_value /* check2 */ {{{$}}

template <typename T, typename U>
typename std::enable_if<T::some_value &&
                        U::another_value, Obj>::type condition_on_two_lines() {
  return Obj{};
}
// CHECK-MESSAGES: :[[@LINE-4]]:1: warning: use C++20 requires constraints instead of enable_if [modernize-use-constraints]
// CHECK-FIXES: {{^}}Obj condition_on_two_lines() requires (T::some_value &&{{$}}
// CHECK-FIXES-NEXT: U::another_value) {{{$}}

template <typename T>
typename std::enable_if<T::some_value, int> :: type* pointer_of_enable_if_t_with_spaces() {
}
// CHECK-MESSAGES: :[[@LINE-2]]:1: warning: use C++20 requires constraints instead of enable_if [modernize-use-constraints]
// CHECK-FIXES: {{^}}template <typename T>{{$}}
// CHECK-FIXES-NEXT: {{^}}int* pointer_of_enable_if_t_with_spaces() requires T::some_value {{{$}}

template <typename T>
typename std::enable_if<T::some_value, int> :: /*c*/ type* pointer_of_enable_if_t_with_comment() {
}
// CHECK-MESSAGES: :[[@LINE-2]]:1: warning: use C++20 requires constraints instead of enable_if [modernize-use-constraints]
// CHECK-FIXES: {{^}}template <typename T>{{$}}
// CHECK-FIXES-NEXT: {{^}}int* pointer_of_enable_if_t_with_comment() requires T::some_value {{{$}}

template <typename T>
std::enable_if_t<T::some_value // comment
              > trailing_slash_slash_comment() {
}
// CHECK-MESSAGES: :[[@LINE-3]]:1: warning: use C++20 requires constraints instead of enable_if [modernize-use-constraints]
// CHECK-FIXES: {{^}}template <typename T>{{$}}
// CHECK-FIXES-NEXT: {{^}}void trailing_slash_slash_comment() requires T::some_value // comment{{$}}
// CHECK-FIXES-NEXT: {{^}}               {{{$}}

} // namespace enable_if_in_return_type


namespace enable_if_trailing_non_type_parameter {

////////////////////////////////
// Section 2: enable_if as final template non-type parameter
////////////////////////////////

template <typename T, typename std::enable_if<T::some_value, int>::type = 0>
void basic() {
}
// CHECK-MESSAGES: :[[@LINE-3]]:23: warning: use C++20 requires constraints instead of enable_if [modernize-use-constraints]
// CHECK-FIXES: {{^}}template <typename T>{{$}}
// CHECK-FIXES-NEXT: {{^}}void basic() requires T::some_value {{{$}}

template <typename T, std::enable_if_t<T::some_value, int> = 0>
void basic_t() {
}
// CHECK-MESSAGES: :[[@LINE-3]]:23: warning: use C++20 requires constraints instead of enable_if [modernize-use-constraints]
// CHECK-FIXES: {{^}}template <typename T>{{$}}
// CHECK-FIXES-NEXT: {{^}}void basic_t() requires T::some_value {{{$}}

template <typename T, template <typename> class U, class V, std::enable_if_t<T::some_value, int> = 0>
void basic_many_template_params() {
}
// CHECK-MESSAGES: :[[@LINE-3]]:61: warning: use C++20 requires constraints instead of enable_if [modernize-use-constraints]
// CHECK-FIXES: {{^}}template <typename T, template <typename> class U, class V>{{$}}
// CHECK-FIXES-NEXT: {{^}}void basic_many_template_params() requires T::some_value {{{$}}

template <std::enable_if_t<true, int> = 0>
void no_dependent_type() {
}
// FIXME - Support non-dependent enable_ifs. Low priority though...

struct ABaseClass {
  ABaseClass();
  ABaseClass(int);
};

template <typename T>
struct AClass : ABaseClass {
  template <std::enable_if_t<T::some_value, int> = 0>
  void no_other_template_params() {
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:13: warning: use C++20 requires constraints instead of enable_if [modernize-use-constraints]
  // CHECK-FIXES: {{^}}  {{$}}
  // CHECK-FIXES-NEXT: {{^}}  void no_other_template_params() requires T::some_value {{{$}}

  template <typename U, std::enable_if_t<U::some_value, int> = 0>
  AClass() {}
  // CHECK-MESSAGES: :[[@LINE-2]]:25: warning: use C++20 requires constraints instead of enable_if [modernize-use-constraints]
  // CHECK-FIXES: {{^}}  template <typename U>{{$}}
  // CHECK-FIXES-NEXT: {{^}}  AClass() requires U::some_value {}{{$}}

  template <typename U, std::enable_if_t<U::some_value, int> = 0>
  AClass(int) : data(0) {}
  // CHECK-MESSAGES: :[[@LINE-2]]:25: warning: use C++20 requires constraints instead of enable_if [modernize-use-constraints]
  // CHECK-FIXES: {{^}}  template <typename U>{{$}}
  // CHECK-FIXES-NEXT: {{^}}  AClass(int) requires U::some_value : data(0) {}{{$}}

  template <typename U, std::enable_if_t<U::some_value, int> = 0>
  AClass(int, int) : AClass(0) {}
  // CHECK-MESSAGES: :[[@LINE-2]]:25: warning: use C++20 requires constraints instead of enable_if [modernize-use-constraints]
  // CHECK-FIXES: {{^}}  template <typename U>{{$}}
  // CHECK-FIXES-NEXT: {{^}}  AClass(int, int) requires U::some_value : AClass(0) {}{{$}}

  template <typename U, std::enable_if_t<U::some_value, int> = 0>
  AClass(int, int, int) : ABaseClass(0) {}
  // CHECK-MESSAGES: :[[@LINE-2]]:25: warning: use C++20 requires constraints instead of enable_if [modernize-use-constraints]
  // CHECK-FIXES: {{^}}  template <typename U>{{$}}
  // CHECK-FIXES-NEXT: {{^}}  AClass(int, int, int) requires U::some_value : ABaseClass(0) {}{{$}}

  template <typename U, std::enable_if_t<U::some_value, int> = 0>
  AClass(int, int, int, int) : data2(), data() {}
  // CHECK-MESSAGES: :[[@LINE-2]]:25: warning: use C++20 requires constraints instead of enable_if [modernize-use-constraints]
  // CHECK-FIXES: {{^}}  template <typename U>{{$}}
  // CHECK-FIXES-NEXT: {{^}}  AClass(int, int, int, int) requires U::some_value : data2(), data() {}{{$}}

  int data;
  int data2;
};

template <typename T>
struct AClass2 : ABaseClass {

  template <typename U, std::enable_if_t<U::some_value, int> = 0>
  AClass2() {}
  // CHECK-MESSAGES: :[[@LINE-2]]:25: warning: use C++20 requires constraints instead of enable_if [modernize-use-constraints]
  // CHECK-FIXES: {{^}}  template <typename U>{{$}}
  // CHECK-FIXES-NEXT: {{^}}  AClass2() requires U::some_value {}{{$}}

  template <typename U, std::enable_if_t<U::some_value, int> = 0>
  AClass2(int) : data2(0) {}
  // CHECK-MESSAGES: :[[@LINE-2]]:25: warning: use C++20 requires constraints instead of enable_if [modernize-use-constraints]
  // CHECK-FIXES: {{^}}  template <typename U>{{$}}
  // CHECK-FIXES-NEXT: {{^}}  AClass2(int) requires U::some_value : data2(0) {}{{$}}

  int data = 10;
  int data2;
  int data3;
};

template <typename T, std::enable_if_t<T::some_value, T>* = 0>
void pointer_type() {
}
// CHECK-MESSAGES: :[[@LINE-3]]:23: warning: use C++20 requires constraints instead of enable_if [modernize-use-constraints]
// CHECK-FIXES: {{^}}template <typename T>{{$}}
// CHECK-FIXES-NEXT: {{^}}void pointer_type() requires T::some_value {{{$}}

template <typename T,
          std::enable_if_t<T::some_value, T>* = nullptr>
void param_on_newline() {
}
// CHECK-MESSAGES: :[[@LINE-3]]:11: warning: use C++20 requires constraints instead of enable_if [modernize-use-constraints]
// CHECK-FIXES: {{^}}template <typename T>{{$}}
// CHECK-FIXES-NEXT: {{^}}void param_on_newline() requires T::some_value {{{$}}

template <typename T,
          typename U,
          std::enable_if_t<
            ConsumeVariadic<T,
                            U>::value, T>* = nullptr>
void param_split_on_two_lines() {
}
// CHECK-MESSAGES: :[[@LINE-5]]:11: warning: use C++20 requires constraints instead of enable_if [modernize-use-constraints]
// CHECK-FIXES: {{^}}template <typename T,{{$}}
// CHECK-FIXES-NEXT: {{^}}          typename U>{{$}}
// CHECK-FIXES-NEXT: {{^}}void param_split_on_two_lines() requires ConsumeVariadic<T,{{$}}
// CHECK-FIXES-NEXT: {{^}}                            U>::value {{{$}}

template <typename T, std::enable_if_t<T::some_value // comment
         >* = nullptr>
void trailing_slash_slash_comment() {
}
// CHECK-MESSAGES: :[[@LINE-4]]:23: warning: use C++20 requires constraints instead of enable_if [modernize-use-constraints]
// CHECK-FIXES: {{^}}template <typename T>{{$}}
// CHECK-FIXES-NEXT: {{^}}void trailing_slash_slash_comment() requires T::some_value // comment{{$}}
// CHECK-FIXES-NEXT: {{^}}          {{{$}}

template <typename T, std::enable_if_t<T::some_value>* = nullptr, std::enable_if_t<T::another_value>* = nullptr>
void two_enable_ifs() {
}
// CHECK-MESSAGES: :[[@LINE-3]]:67: warning: use C++20 requires constraints instead of enable_if [modernize-use-constraints]
// CHECK-FIXES: {{^}}template <typename T, std::enable_if_t<T::some_value>* = nullptr>{{$}}
// CHECK-FIXES-NEXT: {{^}}void two_enable_ifs() requires T::another_value {{{$}}

////////////////////////////////
// Negative tests
////////////////////////////////

template <typename U, std::enable_if_t<U::some_value, int> V = 0>
void non_type_param_has_name() {
}
template <typename U, std::enable_if_t<U::some_value, int>>
void non_type_param_has_no_default() {
}
template <typename U, std::enable_if_t<U::some_value, int> V>
void non_type_param_has_name_and_no_default() {
}
template <typename U, std::enable_if_t<U::some_value, int>...>
void non_type_variadic() {
}
template <typename U, std::enable_if_t<U::some_value, int> = 0, int = 0>
void non_type_not_last() {
}

#define TEMPLATE_REQUIRES(U, IF) template <typename U, std::enable_if_t<IF, int> = 0>
TEMPLATE_REQUIRES(U, U::some_value)
void macro_entire_enable_if() {
}
// CHECK-MESSAGES: :[[@LINE-3]]:1: warning: use C++20 requires constraints instead of enable_if [modernize-use-constraints]
// CHECK-MESSAGES: :[[@LINE-5]]:56: note: expanded from macro 'TEMPLATE_REQUIRES'
// CHECK-FIXES: {{^}}TEMPLATE_REQUIRES(U, U::some_value)
// CHECK-FIXES-NEXT: {{^}}void macro_entire_enable_if() {{{$}}

#define CONDITION U::some_value
template <typename U, std::enable_if_t<CONDITION, int> = 0>
void macro_condition() {
}
// CHECK-MESSAGES: :[[@LINE-3]]:23: warning: use C++20 requires constraints instead of enable_if [modernize-use-constraints]
// CHECK-FIXES: {{^}}template <typename U>{{$}}
// CHECK-FIXES-NEXT: {{^}}void macro_condition() requires CONDITION {{{$}}

#undef CONDITION
#define CONDITION !U::some_value
template <typename U, std::enable_if_t<CONDITION, int> = 0>
void macro_condition_not_primary() {
}
// CHECK-MESSAGES: :[[@LINE-3]]:23: warning: use C++20 requires constraints instead of enable_if [modernize-use-constraints]
// CHECK-FIXES: {{^}}template <typename U>{{$}}
// CHECK-FIXES-NEXT: {{^}}void macro_condition_not_primary() requires (CONDITION) {{{$}}

} // namespace enable_if_trailing_non_type_parameter


namespace enable_if_trailing_type_parameter {

////////////////////////////////
// Section 3: enable_if as final template nameless defaulted type parameter
////////////////////////////////

template <typename T, typename = std::enable_if<T::some_value>::type>
void basic() {
}
// CHECK-MESSAGES: :[[@LINE-3]]:23: warning: use C++20 requires constraints instead of enable_if [modernize-use-constraints]
// CHECK-FIXES: {{^}}template <typename T>{{$}}
// CHECK-FIXES-NEXT: {{^}}void basic() requires T::some_value {{{$}}

template <typename T, typename = std::enable_if_t<T::some_value>>
void basic_t() {
}
// CHECK-MESSAGES: :[[@LINE-3]]:23: warning: use C++20 requires constraints instead of enable_if [modernize-use-constraints]
// CHECK-FIXES: {{^}}template <typename T>{{$}}
// CHECK-FIXES-NEXT: {{^}}void basic_t() requires T::some_value {{{$}}

template <typename T, template <typename> class U, class V, typename = std::enable_if_t<T::some_value>>
void basic_many_template_params() {
}
// CHECK-MESSAGES: :[[@LINE-3]]:61: warning: use C++20 requires constraints instead of enable_if [modernize-use-constraints]
// CHECK-FIXES: {{^}}template <typename T, template <typename> class U, class V>{{$}}
// CHECK-FIXES-NEXT: {{^}}void basic_many_template_params() requires T::some_value {{{$}}

struct ABaseClass {
  ABaseClass();
  ABaseClass(int);
};

template <typename T>
struct AClass : ABaseClass {
  template <typename = std::enable_if_t<T::some_value>>
  void no_other_template_params() {
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:13: warning: use C++20 requires constraints instead of enable_if [modernize-use-constraints]
  // CHECK-FIXES: {{^}}  {{$}}
  // CHECK-FIXES-NEXT: {{^}}  void no_other_template_params() requires T::some_value {{{$}}

  template <typename U, typename = std::enable_if_t<U::some_value>>
  AClass() {}
  // CHECK-MESSAGES: :[[@LINE-2]]:25: warning: use C++20 requires constraints instead of enable_if [modernize-use-constraints]
  // CHECK-FIXES: {{^}}  template <typename U>{{$}}
  // CHECK-FIXES-NEXT: {{^}}  AClass() requires U::some_value {}{{$}}

  template <typename U, typename = std::enable_if_t<U::some_value>>
  AClass(int) : data(0) {}
  // CHECK-MESSAGES: :[[@LINE-2]]:25: warning: use C++20 requires constraints instead of enable_if [modernize-use-constraints]
  // CHECK-FIXES: {{^}}  template <typename U>{{$}}
  // CHECK-FIXES-NEXT: {{^}}  AClass(int) requires U::some_value : data(0) {}{{$}}

  template <typename U, typename = std::enable_if_t<U::some_value>>
  AClass(int, int) : AClass(0) {}
  // CHECK-MESSAGES: :[[@LINE-2]]:25: warning: use C++20 requires constraints instead of enable_if [modernize-use-constraints]
  // CHECK-FIXES: {{^}}  template <typename U>{{$}}
  // CHECK-FIXES-NEXT: {{^}}  AClass(int, int) requires U::some_value : AClass(0) {}{{$}}

  template <typename U, typename = std::enable_if_t<U::some_value>>
  AClass(int, int, int) : ABaseClass(0) {}
  // CHECK-MESSAGES: :[[@LINE-2]]:25: warning: use C++20 requires constraints instead of enable_if [modernize-use-constraints]
  // CHECK-FIXES: {{^}}  template <typename U>{{$}}
  // CHECK-FIXES-NEXT: {{^}}  AClass(int, int, int) requires U::some_value : ABaseClass(0) {}{{$}}

  int data;
};

template <typename T, typename = std::enable_if_t<T::some_value>*>
void pointer_type() {
}
// CHECK-MESSAGES: :[[@LINE-3]]:23: warning: use C++20 requires constraints instead of enable_if [modernize-use-constraints]
// CHECK-FIXES: {{^}}template <typename T>{{$}}
// CHECK-FIXES-NEXT: {{^}}void pointer_type() requires T::some_value {{{$}}

template <typename T, typename = std::enable_if_t<T::some_value>&>
void reference_type() {
}
// CHECK-MESSAGES: :[[@LINE-3]]:23: warning: use C++20 requires constraints instead of enable_if [modernize-use-constraints]
// CHECK-FIXES: {{^}}template <typename T>{{$}}
// CHECK-FIXES-NEXT: {{^}}void reference_type() requires T::some_value {{{$}}

template <typename T,
          typename = std::enable_if_t<T::some_value>*>
void param_on_newline() {
}
// CHECK-MESSAGES: :[[@LINE-3]]:11: warning: use C++20 requires constraints instead of enable_if [modernize-use-constraints]
// CHECK-FIXES: {{^}}template <typename T>{{$}}
// CHECK-FIXES-NEXT: {{^}}void param_on_newline() requires T::some_value {{{$}}

template <typename T,
          typename U,
          typename = std::enable_if_t<
            ConsumeVariadic<T,
                            U>::value>>
void param_split_on_two_lines() {
}
// CHECK-MESSAGES: :[[@LINE-5]]:11: warning: use C++20 requires constraints instead of enable_if [modernize-use-constraints]
// CHECK-FIXES: {{^}}template <typename T,{{$}}
// CHECK-FIXES-NEXT: {{^}}          typename U>{{$}}
// CHECK-FIXES-NEXT: {{^}}void param_split_on_two_lines() requires ConsumeVariadic<T,{{$}}
// CHECK-FIXES-NEXT: {{^}}                            U>::value {{{$}}


////////////////////////////////
// Negative tests
////////////////////////////////

template <typename U, typename Named = std::enable_if_t<U::some_value>>
void param_has_name() {
}

template <typename U, typename = std::enable_if_t<U::some_value>, typename = int>
void not_last_param() {
}

} // namespace enable_if_trailing_type_parameter


// Issue fixes:

namespace PR91872 {

enum expression_template_option { value1, value2 };

template <typename T> struct number_category {
  static const int value = 0;
};

constexpr int number_kind_complex = 1;

template <typename T, expression_template_option ExpressionTemplates>
struct number {
  using type = T;
};

template <typename T> struct component_type {
  using type = T;
};

template <class T, expression_template_option ExpressionTemplates>
inline typename std::enable_if<
    number_category<T>::value == number_kind_complex,
    component_type<number<T, ExpressionTemplates>>>::type::type
abs(const number<T, ExpressionTemplates> &v) {
  return {};
}

}
