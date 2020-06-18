// RUN: %clang_cc1 -std=c++11 -fsyntax-only -Wdocumentation -Wdocumentation-pedantic -verify %s
// RUN: %clang_cc1 -std=c++14 -fsyntax-only -Wdocumentation -Wdocumentation-pedantic -verify %s

// This file contains lots of corner cases, so ensure that XML we generate is not invalid.
// RUN: c-index-test -test-load-source all -comments-xml-schema=%S/../../bindings/xml/comment-xml-schema.rng %s | FileCheck %s -check-prefix=WRONG
// WRONG-NOT: CommentXMLInvalid

// expected-warning@+2 {{HTML tag 'a' requires an end tag}}
// expected-warning@+1 {{expected quoted string after equals sign}}
/// <a href=>
int test_html1(int);

// expected-warning@+2 {{HTML tag 'a' requires an end tag}}
// expected-warning@+1 {{expected quoted string after equals sign}}
/// <a href==>
int test_html2(int);

// expected-warning@+3 {{HTML tag 'a' requires an end tag}}
// expected-warning@+2 {{expected quoted string after equals sign}}
// expected-warning@+1 {{HTML start tag prematurely ended, expected attribute name or '>'}}
/// <a href= blah
int test_html3(int);

// expected-warning@+2 {{HTML tag 'a' requires an end tag}}
// expected-warning@+1 {{HTML start tag prematurely ended, expected attribute name or '>'}}
/// <a =>
int test_html4(int);

// expected-warning@+2 {{HTML tag 'a' requires an end tag}}
// expected-warning@+1 {{HTML start tag prematurely ended, expected attribute name or '>'}}
/// <a "aaa">
int test_html5(int);

// expected-warning@+2 {{HTML tag 'a' requires an end tag}}
// expected-warning@+1 {{HTML start tag prematurely ended, expected attribute name or '>'}}
/// <a a="b" =>
int test_html6(int);

// expected-warning@+2 {{HTML tag 'a' requires an end tag}}
// expected-warning@+1 {{HTML start tag prematurely ended, expected attribute name or '>'}}
/// <a a="b" "aaa">
int test_html7(int);

// expected-warning@+2 {{HTML tag 'a' requires an end tag}}
// expected-warning@+1 {{HTML start tag prematurely ended, expected attribute name or '>'}}
/// <a a="b" =
int test_html8(int);

// expected-warning@+2 {{HTML start tag prematurely ended, expected attribute name or '>'}} expected-note@+1 {{HTML tag started here}}
/** Aaa bbb<img ddd eee
 * fff ggg.
 */
int test_html9(int);

// expected-warning@+1 {{HTML start tag prematurely ended, expected attribute name or '>'}}
/** Aaa bbb<img ddd eee 42%
 * fff ggg.
 */
int test_html10(int);

// expected-warning@+1 {{HTML end tag 'br' is forbidden}}
/// <br></br>
int test_html11(int);

/// <blockquote>Meow</blockquote>
int test_html_nesting1(int);

/// <b><i>Meow</i></b>
int test_html_nesting2(int);

/// <p>Aaa<br>
/// Bbb</p>
int test_html_nesting3(int);

/// <p>Aaa<br />
/// Bbb</p>
int test_html_nesting4(int);

// expected-warning@+3 {{HTML tag 'b' requires an end tag}}
// expected-warning@+2 {{HTML tag 'i' requires an end tag}}
// expected-warning@+1 {{HTML end tag does not match any start tag}}
/// <b><i>Meow</a>
int test_html_nesting5(int);

// expected-warning@+2 {{HTML start tag 'i' closed by 'b'}}
// expected-warning@+1 {{HTML end tag does not match any start tag}}
/// <b><i>Meow</b></b>
int test_html_nesting6(int);

// expected-warning@+2 {{HTML start tag 'i' closed by 'b'}}
// expected-warning@+1 {{HTML end tag does not match any start tag}}
/// <b><i>Meow</b></i>
int test_html_nesting7(int);

// expected-warning@+1 {{HTML tag 'b' requires an end tag}}
/// <b>Meow
int test_html_nesting8(int);

// expected-warning@+1 {{empty paragraph passed to '\brief' command}}
/// \brief\returns Aaa
int test_block_command1(int);

// expected-warning@+1 {{empty paragraph passed to '\brief' command}}
/// \brief \returns Aaa
int test_block_command2(int);

// expected-warning@+1 {{empty paragraph passed to '\brief' command}}
/// \brief
/// \returns Aaa
int test_block_command3(int);

// expected-warning@+1 {{empty paragraph passed to '\brief' command}}
/// \brief
///
/// \returns Aaa
int test_block_command4(int);

// There is trailing whitespace on one of the following lines, don't remove it!
// expected-warning@+1 {{empty paragraph passed to '\brief' command}}
/// \brief
/// 
/// \returns Aaa
int test_block_command5(int);

/// \brief \c Aaa
int test_block_command6(int);

// expected-warning@+5 {{duplicated command '\brief'}} expected-note@+1 {{previous command '\brief' here}}
/// \brief Aaa
///
/// Bbb
///
/// \brief Ccc
int test_duplicate_brief1(int);

// expected-warning@+5 {{duplicated command '\short'}} expected-note@+1 {{previous command '\short' here}}
/// \short Aaa
///
/// Bbb
///
/// \short Ccc
int test_duplicate_brief2(int);

// expected-warning@+5 {{duplicated command '\brief'}} expected-note@+1 {{previous command '\short' (an alias of '\brief') here}}
/// \short Aaa
///
/// Bbb
///
/// \brief Ccc
int test_duplicate_brief3(int);


/// \return Aaa
///
/// Bbb
///
/// \return Ccc
int test_multiple_returns1(int);

/// \returns Aaa
///
/// Bbb
///
/// \returns Ccc
int test_multiple_returns2(int);

/// \result Aaa
///
/// Bbb
///
/// \result Ccc
int test_multiple_returns3(int);

/// \returns Aaa
///
/// Bbb
///
/// \return Ccc
int test_multiple_returns4(int);


// expected-warning@+1 {{'\param' command used in a comment that is not attached to a function declaration}}
/// \param a Blah blah.
int test_param1_backslash;

// rdar://13066276
// Check that the diagnostic uses the same command marker as the comment.
// expected-warning@+1 {{'@param' command used in a comment that is not attached to a function declaration}}
/// @param a Blah blah.
int test_param1_at;

// expected-warning@+1 {{empty paragraph passed to '\param' command}}
/// \param
/// \param a Blah blah.
int test_param2(int a);

// expected-warning@+1 {{empty paragraph passed to '\param' command}}
/// \param a
int test_param3(int a);

/// \param a Blah blah.
int test_param4(int a);

/// \param [in] a Blah blah.
int test_param5(int a);

/// \param [out] a Blah blah.
int test_param6(int a);

/// \param [in,out] a Blah blah.
int test_param7(int a);

// expected-warning@+1 {{whitespace is not allowed in parameter passing direction}}
/// \param [ in ] a Blah blah.
int test_param8(int a);

// expected-warning@+1 {{whitespace is not allowed in parameter passing direction}}
/// \param [in, out] a Blah blah.
int test_param9(int a);

// expected-warning@+1 {{unrecognized parameter passing direction, valid directions are '[in]', '[out]' and '[in,out]'}}
/// \param [ junk] a Blah blah.
int test_param10(int a);

// expected-warning@+1 {{parameter 'a' not found in the function declaration}}
/// \param a Blah blah.
int test_param11();

// expected-warning@+1 {{parameter 'A' not found in the function declaration}} expected-note@+1 {{did you mean 'a'?}}
/// \param A Blah blah.
int test_param12(int a);

// expected-warning@+1 {{parameter 'aab' not found in the function declaration}} expected-note@+1 {{did you mean 'aaa'?}}
/// \param aab Blah blah.
int test_param13(int aaa, int bbb);

// expected-warning@+2 {{parameter 'aab' not found in the function declaration}} expected-note@+2 {{did you mean 'bbb'?}}
/// \param aaa Blah blah.
/// \param aab Blah blah.
int test_param14(int aaa, int bbb);

// expected-warning@+1 {{parameter 'aab' not found in the function declaration}}
/// \param aab Blah blah.
int test_param15(int bbb, int ccc);

// expected-warning@+1 {{parameter 'aab' not found in the function declaration}}
/// \param aab Ccc.
/// \param aaa Aaa.
/// \param bbb Bbb.
int test_param16(int aaa, int bbb);

// expected-warning@+2 {{parameter 'aab' not found in the function declaration}}
/// \param aaa Aaa.
/// \param aab Ccc.
/// \param bbb Bbb.
int test_param17(int aaa, int bbb);

// expected-warning@+3 {{parameter 'aab' not found in the function declaration}}
/// \param aaa Aaa.
/// \param bbb Bbb.
/// \param aab Ccc.
int test_param18(int aaa, int bbb);

class C {
  // expected-warning@+1 {{parameter 'aaa' not found in the function declaration}}
  /// \param aaa Blah blah.
  C(int bbb, int ccc);

  // expected-warning@+1 {{parameter 'aaa' not found in the function declaration}}
  /// \param aaa Blah blah.
 int test_param19(int bbb, int ccc);
};

// expected-warning@+1 {{parameter 'aab' not found in the function declaration}}
/// \param aab Blah blah.
template<typename T>
void test_param20(int bbb, int ccc);

// expected-warning@+3 {{parameter 'a' is already documented}}
// expected-note@+1 {{previous documentation}}
/// \param a Aaa.
/// \param a Aaa.
int test_param21(int a);

// expected-warning@+4 {{parameter 'x2' is already documented}}
// expected-note@+2 {{previous documentation}}
/// \param x1 Aaa.
/// \param x2 Bbb.
/// \param x2 Ccc.
int test_param22(int x1, int x2, int x3);

// expected-warning@+1 {{empty paragraph passed to '\param' command}}
/// \param a
/// \retval 0 Blah blah.
int test_param23(int a);

/// \param a \ref test_param23 has an empty paragraph, this doesn't.
int test_param24(int a);

//===---
// Test that we treat typedefs to some non-function types as functions for the
// purposes of documentation comment parsing.
//===---

namespace foo {
  inline namespace bar {
    template<typename>
    struct function_wrapper {};

    template<unsigned>
    struct not_a_function_wrapper {};
  }
};

// expected-warning@+2 {{parameter 'bbb' not found in the function declaration}} expected-note@+2 {{did you mean 'ccc'?}}
/// \param aaa Meow.
/// \param bbb Bbb.
/// \returns aaa.
typedef int test_function_like_typedef1(int aaa, int ccc);

// expected-warning@+2 {{parameter 'bbb' not found in the function declaration}} expected-note@+2 {{did you mean 'ccc'?}}
/// \param aaa Meow.
/// \param bbb Bbb.
/// \returns aaa.
typedef int (*test_function_like_typedef2)(int aaa, int ccc);

// expected-warning@+2 {{parameter 'bbb' not found in the function declaration}} expected-note@+2 {{did you mean 'ccc'?}}
/// \param aaa Meow.
/// \param bbb Bbb.
/// \returns aaa.
typedef int (* const test_function_like_typedef3)(int aaa, int ccc);

// expected-warning@+2 {{parameter 'bbb' not found in the function declaration}} expected-note@+2 {{did you mean 'ccc'?}}
/// \param aaa Meow.
/// \param bbb Bbb.
/// \returns aaa.
typedef int (C::*test_function_like_typedef4)(int aaa, int ccc);

// expected-warning@+2 {{parameter 'bbb' not found in the function declaration}} expected-note@+2 {{did you mean 'ccc'?}}
/// \param aaa Meow.
/// \param bbb Bbb.
/// \returns aaa.
typedef foo::function_wrapper<int (int aaa, int ccc)> test_function_like_typedef5;

// expected-warning@+2 {{parameter 'bbb' not found in the function declaration}} expected-note@+2 {{did you mean 'ccc'?}}
/// \param aaa Meow.
/// \param bbb Bbb.
/// \returns aaa.
typedef foo::function_wrapper<int (int aaa, int ccc)> *test_function_like_typedef6;

// expected-warning@+2 {{parameter 'bbb' not found in the function declaration}} expected-note@+2 {{did you mean 'ccc'?}}
/// \param aaa Meow.
/// \param bbb Bbb.
/// \returns aaa.
typedef foo::function_wrapper<int (int aaa, int ccc)> &test_function_like_typedef7;

// expected-warning@+2 {{parameter 'bbb' not found in the function declaration}} expected-note@+2 {{did you mean 'ccc'?}}
/// \param aaa Meow.
/// \param bbb Bbb.
/// \returns aaa.
typedef foo::function_wrapper<int (int aaa, int ccc)> &&test_function_like_typedef8;


typedef int (*test_not_function_like_typedef1)(int aaa);

// expected-warning@+1 {{'\param' command used in a comment that is not attached to a function declaration}}
/// \param aaa Meow.
typedef test_not_function_like_typedef1 test_not_function_like_typedef2;

// rdar://13066276
// Check that the diagnostic uses the same command marker as the comment.
// expected-warning@+1 {{'@param' command used in a comment that is not attached to a function declaration}}
/// @param aaa Meow.
typedef unsigned int test_not_function_like_typedef3;

// expected-warning@+1 {{'\param' command used in a comment that is not attached to a function declaration}}
/// \param aaa Meow.
typedef foo::not_a_function_wrapper<1> test_not_function_like_typedef4;

// expected-warning@+2 {{parameter 'bbb' not found in the function declaration}} expected-note@+2 {{did you mean 'ccc'?}}
/// \param aaa Meow.
/// \param bbb Bbb.
/// \returns aaa.
using test_function_like_using1 = int (int aaa, int ccc);

// expected-warning@+2 {{parameter 'bbb' not found in the function declaration}} expected-note@+2 {{did you mean 'ccc'?}}
/// \param aaa Meow.
/// \param bbb Bbb.
/// \returns aaa.
using test_function_like_using2 = int (*)(int aaa, int ccc);

// expected-warning@+2 {{parameter 'bbb' not found in the function declaration}} expected-note@+2 {{did you mean 'ccc'?}}
/// \param aaa Meow.
/// \param bbb Bbb.
/// \returns aaa.
using test_function_like_using3 = int (* const)(int aaa, int ccc);

// expected-warning@+2 {{parameter 'bbb' not found in the function declaration}} expected-note@+2 {{did you mean 'ccc'?}}
/// \param aaa Meow.
/// \param bbb Bbb.
/// \returns aaa.
using test_function_like_using4 = int (C::*)(int aaa, int ccc);

// expected-warning@+2 {{parameter 'bbb' not found in the function declaration}} expected-note@+2 {{did you mean 'ccc'?}}
/// \param aaa Meow.
/// \param bbb Bbb.
/// \returns aaa.
using test_function_like_using5 = foo::function_wrapper<int (int aaa, int ccc)>;

// expected-warning@+2 {{parameter 'bbb' not found in the function declaration}} expected-note@+2 {{did you mean 'ccc'?}}
/// \param aaa Meow.
/// \param bbb Bbb.
/// \returns aaa.
using test_function_like_using6 = foo::function_wrapper<int (int aaa, int ccc)> *;

// expected-warning@+2 {{parameter 'bbb' not found in the function declaration}} expected-note@+2 {{did you mean 'ccc'?}}
/// \param aaa Meow.
/// \param bbb Bbb.
/// \returns aaa.
using test_function_like_using7 = foo::function_wrapper<int (int aaa, int ccc)> &;

// expected-warning@+2 {{parameter 'bbb' not found in the function declaration}} expected-note@+2 {{did you mean 'ccc'?}}
/// \param aaa Meow.
/// \param bbb Bbb.
/// \returns aaa.
using test_function_like_using8 = foo::function_wrapper<int (int aaa, int ccc)> &&;

// expected-warning@+4 {{template parameter 'U' not found in the template declaration}} expected-note@+4 {{did you mean 'T'?}}
// expected-warning@+2 {{parameter 'bbb' not found in the function declaration}} expected-note@+2 {{did you mean 'ccc'?}}
/// \param aaa Meow.
/// \param bbb Bbb.
/// \tparam U Uuu.
template<typename T>
using test_function_like_using9 = int(T aaa, int ccc);

// expected-warning@+4 {{template parameter 'U' not found in the template declaration}} expected-note@+4 {{did you mean 'T'?}}
// expected-warning@+2 {{parameter 'bbb' not found in the function declaration}} expected-note@+2 {{did you mean 'ccc'?}}
/// \param aaa Meow.
/// \param bbb Bbb.
/// \tparam U Uuu.
template<typename T>
using test_function_like_using10 = int (*)(T aaa, int ccc);

// expected-warning@+4 {{template parameter 'U' not found in the template declaration}} expected-note@+4 {{did you mean 'T'?}}
// expected-warning@+2 {{parameter 'bbb' not found in the function declaration}} expected-note@+2 {{did you mean 'ccc'?}}
/// \param aaa Meow.
/// \param bbb Bbb.
/// \tparam U Uuu.
template<typename T>
using test_function_like_using11 = foo::function_wrapper<int (T aaa, int ccc)>;

// expected-warning@+4 {{template parameter 'U' not found in the template declaration}} expected-note@+4 {{did you mean 'T'?}}
// expected-warning@+2 {{parameter 'bbb' not found in the function declaration}} expected-note@+2 {{did you mean 'ccc'?}}
/// \param aaa Meow.
/// \param bbb Bbb.
/// \tparam U Uuu.
template<typename T>
using test_function_like_using12 = foo::function_wrapper<int (T aaa, int ccc)> *;

using test_not_function_like_using1 = int (*)(int aaa);

// expected-warning@+1 {{'\param' command used in a comment that is not attached to a function declaration}}
/// \param aaa Meow.
using test_not_function_like_using2 = test_not_function_like_using1;

// Check that the diagnostic uses the same command marker as the comment.
// expected-warning@+1 {{'@param' command used in a comment that is not attached to a function declaration}}
/// @param aaa Meow.
using test_not_function_like_using3 = unsigned int;

// expected-warning@+1 {{'\param' command used in a comment that is not attached to a function declaration}}
/// \param aaa Meow.
using test_not_function_like_using4 = foo::not_a_function_wrapper<1>;

/// \param aaa Aaa
/// \param ... Vararg
int test_vararg_param1(int aaa, ...);

/// \param ... Vararg
int test_vararg_param2(...);

// expected-warning@+1 {{parameter '...' not found in the function declaration}} expected-note@+1 {{did you mean 'aaa'?}}
/// \param ... Vararg
int test_vararg_param3(int aaa);

// expected-warning@+1 {{parameter '...' not found in the function declaration}}
/// \param ... Vararg
int test_vararg_param4();


/// \param aaa Aaa
/// \param ... Vararg
template<typename T>
int test_template_vararg_param1(int aaa, ...);

/// \param ... Vararg
template<typename T>
int test_template_vararg_param2(...);

// expected-warning@+1 {{parameter '...' not found in the function declaration}} expected-note@+1 {{did you mean 'aaa'?}}
/// \param ... Vararg
template<typename T>
int test_template_vararg_param3(int aaa);

// expected-warning@+1 {{parameter '...' not found in the function declaration}}
/// \param ... Vararg
template<typename T>
int test_template_vararg_param4();


// expected-warning@+1 {{'\tparam' command used in a comment that is not attached to a template declaration}}
/// \tparam T Aaa
int test_tparam1;

// expected-warning@+1 {{'\tparam' command used in a comment that is not attached to a template declaration}}
/// \tparam T Aaa
void test_tparam2(int aaa);

// expected-warning@+1 {{empty paragraph passed to '\tparam' command}}
/// \tparam
/// \param aaa Blah blah
template<typename T>
void test_tparam3(T aaa);

// expected-warning@+1 {{template parameter 'T' not found in the template declaration}} expected-note@+1 {{did you mean 'TT'?}}
/// \tparam T Aaa
template<typename TT>
void test_tparam4(TT aaa);

// expected-warning@+1 {{template parameter 'T' not found in the template declaration}} expected-note@+1 {{did you mean 'TT'?}}
/// \tparam T Aaa
template<typename TT>
class test_tparam5 {
  // expected-warning@+1 {{template parameter 'T' not found in the template declaration}} expected-note@+1 {{did you mean 'TTT'?}}
  /// \tparam T Aaa
  template<typename TTT>
  void test_tparam6(TTT aaa);
};

/// \tparam T1 Aaa
/// \tparam T2 Bbb
template<typename T1, typename T2>
void test_tparam7(T1 aaa, T2 bbb);

// expected-warning@+1 {{template parameter 'SomTy' not found in the template declaration}} expected-note@+1 {{did you mean 'SomeTy'?}}
/// \tparam SomTy Aaa
/// \tparam OtherTy Bbb
template<typename SomeTy, typename OtherTy>
void test_tparam8(SomeTy aaa, OtherTy bbb);

// expected-warning@+2 {{template parameter 'T1' is already documented}} expected-note@+1 {{previous documentation}}
/// \tparam T1 Aaa
/// \tparam T1 Bbb
template<typename T1, typename T2>
void test_tparam9(T1 aaa, T2 bbb);

/// \tparam T Aaa
/// \tparam TT Bbb
template<template<typename T> class TT>
void test_tparam10(TT<int> aaa);

/// \tparam T Aaa
/// \tparam TT Bbb
/// \tparam TTT Ccc
template<template<template<typename T> class TT, class C> class TTT>
void test_tparam11();

/// \tparam I Aaa
template<int I>
void test_tparam12();

template<typename T, typename U>
class test_tparam13 { };

/// \tparam T Aaa
template<typename T>
using test_tparam14 = test_tparam13<T, int>;

// expected-warning@+1 {{template parameter 'U' not found in the template declaration}} expected-note@+1 {{did you mean 'T'?}}
/// \tparam U Aaa
template<typename T>
using test_tparam15 = test_tparam13<T, int>;

// ----

/// \tparam T Aaa
template<typename T>
class test_tparam16 { };

typedef test_tparam16<int> test_tparam17;
typedef test_tparam16<double> test_tparam18;

// ----

template<typename T>
class test_tparam19;

typedef test_tparam19<int> test_tparam20;
typedef test_tparam19<double> test_tparam21;

/// \tparam T Aaa
template<typename T>
class test_tparam19 { };

// ----

// expected-warning@+1 {{'@tparam' command used in a comment that is not attached to a template declaration}}
/// @tparam T Aaa
int test_tparam22;

// ----


/// Aaa
/// \deprecated Bbb
void test_deprecated_1(int a) __attribute__((deprecated));

#if __cplusplus >= 201402L
/// Aaa
/// \deprecated Bbb
[[deprecated]] void test_deprecated_no_warning_std14(int a);
#endif

// We don't want \deprecated to warn about empty paragraph.  It is fine to use
// \deprecated by itself without explanations.

/// Aaa
/// \deprecated
void test_deprecated_2(int a) __attribute__((deprecated));

/// Aaa
/// \deprecated
void test_deprecated_3(int a) __attribute__((availability(macosx,introduced=10.4)));

/// Aaa
/// \deprecated
void test_deprecated_4(int a) __attribute__((unavailable));

// expected-warning@+2 {{declaration is marked with '\deprecated' command but does not have a deprecation attribute}} expected-note@+3 {{add a deprecation attribute to the declaration to silence this warning}}
/// Aaa
/// \deprecated
void test_deprecated_5(int a);

// expected-warning@+2 {{declaration is marked with '@deprecated' command but does not have a deprecation attribute}} expected-note@+3 {{add a deprecation attribute to the declaration to silence this warning}}
/// Aaa
/// @deprecated
void test_deprecated_6(int a) {
}

// expected-warning@+2 {{declaration is marked with '\deprecated' command but does not have a deprecation attribute}}
/// Aaa
/// \deprecated
template<typename T>
void test_deprecated_7(T aaa);

class PR43753 {
  // expected-warning@+2 {{declaration is marked with '\deprecated' command but does not have a deprecation attribute}}
  // expected-note@+2 {{add a deprecation attribute to the declaration to silence this warning}}
  /// \deprecated
  static void test_deprecated_static();

  // expected-warning@+2 {{declaration is marked with '\deprecated' command but does not have a deprecation attribute}}
  // expected-note@+2 {{add a deprecation attribute to the declaration to silence this warning}}
  /// \deprecated
  static auto test_deprecated_static_trailing_return() -> int;

#if __cplusplus >= 201402L
  // expected-warning@+2 {{declaration is marked with '\deprecated' command but does not have a deprecation attribute}}
  // expected-note@+2 {{add a deprecation attribute to the declaration to silence this warning}}
  /// \deprecated
  static decltype(auto) test_deprecated_static_decltype_auto() { return 1; }
#endif

  // expected-warning@+2 {{declaration is marked with '\deprecated' command but does not have a deprecation attribute}}
  // expected-note@+2 {{add a deprecation attribute to the declaration to silence this warning}}
  /// \deprecated
  void test_deprecated_const() const;

  // expected-warning@+2 {{declaration is marked with '\deprecated' command but does not have a deprecation attribute}}
  // expected-note@+2 {{add a deprecation attribute to the declaration to silence this warning}}
  /// \deprecated
  auto test_deprecated_trailing_return() -> int;

#if __cplusplus >= 201402L
  // expected-warning@+2 {{declaration is marked with '\deprecated' command but does not have a deprecation attribute}}
  // expected-note@+2 {{add a deprecation attribute to the declaration to silence this warning}}
  /// \deprecated
  decltype(auto) test_deprecated_decltype_auto() const { return a; }

private:
  int a{0};
#endif
};

// rdar://12397511
// expected-note@+2 {{previous command '\headerfile' here}}
// expected-warning@+2 {{duplicated command '\headerfile'}}
/// \headerfile ""
/// \headerfile foo.h
int test__headerfile_1(int a);


/// \invariant aaa
void test_invariant_1(int a);

// expected-warning@+1 {{empty paragraph passed to '\invariant' command}}
/// \invariant
void test_invariant_2(int a);


// no-warning
/// \returns Aaa
int test_returns_right_decl_1(int);

class test_returns_right_decl_2 {
  // no-warning
  /// \returns Aaa
  int test_returns_right_decl_3(int);
};

// no-warning
/// \returns Aaa
template<typename T>
int test_returns_right_decl_4(T aaa);

// no-warning
/// \returns Aaa
template<>
int test_returns_right_decl_4(int aaa);

/// \returns Aaa
template<typename T>
T test_returns_right_decl_5(T aaa);

// expected-warning@+1 {{'\returns' command used in a comment that is not attached to a function or method declaration}}
/// \returns Aaa
int test_returns_wrong_decl_1_backslash;

// rdar://13066276
// Check that the diagnostic uses the same command marker as the comment.
// expected-warning@+1 {{'@returns' command used in a comment that is not attached to a function or method declaration}}
/// @returns Aaa
int test_returns_wrong_decl_1_at;

// expected-warning@+1 {{'\return' command used in a comment that is not attached to a function or method declaration}}
/// \return Aaa
int test_returns_wrong_decl_2;

// expected-warning@+1 {{'\result' command used in a comment that is not attached to a function or method declaration}}
/// \result Aaa
int test_returns_wrong_decl_3;

// expected-warning@+1 {{'\returns' command used in a comment that is attached to a function returning void}}
/// \returns Aaa
void test_returns_wrong_decl_4(int);

// expected-warning@+1 {{'\returns' command used in a comment that is attached to a function returning void}}
/// \returns Aaa
template<typename T>
void test_returns_wrong_decl_5(T aaa);

// expected-warning@+1 {{'\returns' command used in a comment that is attached to a function returning void}}
/// \returns Aaa
template<>
void test_returns_wrong_decl_5(int aaa);

// expected-warning@+1 {{'\returns' command used in a comment that is not attached to a function or method declaration}}
/// \returns Aaa
struct test_returns_wrong_decl_6 { };

// expected-warning@+1 {{'\returns' command used in a comment that is not attached to a function or method declaration}}
/// \returns Aaa
class test_returns_wrong_decl_7 {
  // expected-warning@+1 {{'\returns' command used in a comment that is attached to a constructor}}
  /// \returns Aaa
  test_returns_wrong_decl_7();

  // expected-warning@+1 {{'\returns' command used in a comment that is attached to a destructor}}
  /// \returns Aaa
  ~test_returns_wrong_decl_7();
};

// expected-warning@+1 {{'\returns' command used in a comment that is not attached to a function or method declaration}}
/// \returns Aaa
enum test_returns_wrong_decl_8 {
  // expected-warning@+1 {{'\returns' command used in a comment that is not attached to a function or method declaration}}
  /// \returns Aaa
  test_returns_wrong_decl_9
};

// expected-warning@+1 {{'\returns' command used in a comment that is not attached to a function or method declaration}}
/// \returns Aaa
namespace test_returns_wrong_decl_10 { };

// rdar://13094352
// expected-warning@+1 {{'@function' command should be used in a comment attached to a function declaration}}
/*!	@function test_function
*/
typedef unsigned int Base64Flags;
unsigned test_function(Base64Flags inFlags);

// expected-warning@+1 {{'@callback' command should be used in a comment attached to a pointer to function declaration}}
/*! @callback test_callback
*/
typedef unsigned int BaseFlags;
unsigned (*test_callback)(BaseFlags inFlags);

// expected-warning@+1 {{'\endverbatim' command does not terminate a verbatim text block}}
/// \endverbatim
int test_verbatim_1();

// expected-warning@+1 {{'\endcode' command does not terminate a verbatim text block}}
/// \endcode
int test_verbatim_2();

// FIXME: we give a bad diagnostic here because we throw away non-documentation
// comments early.
//
// expected-warning@+3 {{'\endcode' command does not terminate a verbatim text block}}
/// \code
//  foo
/// \endcode
int test_verbatim_3();


// expected-warning@+1 {{empty paragraph passed to '\brief' command}}
int test1; ///< \brief\author Aaa

// expected-warning@+2 {{empty paragraph passed to '\brief' command}}
// expected-warning@+2 {{empty paragraph passed to '\brief' command}}
int test2, ///< \brief\author Aaa
    test3; ///< \brief\author Aaa

// expected-warning@+1 {{empty paragraph passed to '\brief' command}}
int test4; ///< \brief
           ///< \author Aaa


class TestRelates {};

/// \relates TestRelates
/// \brief Aaa
void test_relates_1();

/// \related TestRelates
/// \brief Aaa
void test_relates_2();

/// \relatesalso TestRelates
/// \brief Aaa
void test_relates_3();

/// \relatedalso TestRelates
/// \brief Aaa
void test_relates_4();


// Check that we attach the comment to the declaration during parsing in the
// following cases.  The test is based on the fact that we don't parse
// documentation comments that are not attached to anything.

// expected-warning@+1 {{empty paragraph passed to '\brief' command}}
/// \brief\author Aaa
int test_attach1;

// expected-warning@+1 {{empty paragraph passed to '\brief' command}}
/// \brief\author Aaa
int test_attach2(int);

// expected-warning@+1 {{empty paragraph passed to '\brief' command}}
/// \brief\author Aaa
struct test_attach3 {
  // expected-warning@+1 {{empty paragraph passed to '\brief' command}}
  /// \brief\author Aaa
  int test_attach4;

  // expected-warning@+1 {{empty paragraph passed to '\brief' command}}
  int test_attach5; ///< \brief\author Aaa

  // expected-warning@+1 {{empty paragraph passed to '\brief' command}}
  /// \brief\author Aaa
  int test_attach6(int);
};

// expected-warning@+1 {{empty paragraph passed to '\brief' command}}
/// \brief\author Aaa
class test_attach7 {
  // expected-warning@+1 {{empty paragraph passed to '\brief' command}}
  /// \brief\author Aaa
  int test_attach8;

  // expected-warning@+1 {{empty paragraph passed to '\brief' command}}
  int test_attach9; ///< \brief\author Aaa

  // expected-warning@+1 {{empty paragraph passed to '\brief' command}}
  /// \brief\author Aaa
  int test_attach10(int);
};

// expected-warning@+1 {{empty paragraph passed to '\brief' command}}
/// \brief\author Aaa
enum test_attach9 {
  // expected-warning@+1 {{empty paragraph passed to '\brief' command}}
  /// \brief\author Aaa
  test_attach10,

  // expected-warning@+1 {{empty paragraph passed to '\brief' command}}
  test_attach11 ///< \brief\author Aaa
};

// expected-warning@+1 {{empty paragraph passed to '\brief' command}}
/// \brief\author Aaa
struct test_noattach12 *test_attach13;

// expected-warning@+1 {{empty paragraph passed to '\brief' command}}
/// \brief\author Aaa
typedef struct test_noattach14 *test_attach15;

// expected-warning@+1 + {{empty paragraph passed to '\brief' command}}
/// \brief\author Aaa
typedef struct test_attach16 { int a; } test_attach17;

struct S { int a; };

// expected-warning@+1 {{empty paragraph passed to '\brief' command}}
/// \brief\author Aaa
struct S *test_attach18;

// expected-warning@+1 {{empty paragraph passed to '\brief' command}}
/// \brief\author Aaa
typedef struct S *test_attach19;

// expected-warning@+1 {{empty paragraph passed to '\brief' command}}
/// \brief\author Aaa
struct test_attach20;

// expected-warning@+1 + {{empty paragraph passed to '\brief' command}}
/// \brief\author Aaa
typedef struct test_attach21 {
  // expected-warning@+1 {{empty paragraph passed to '\brief' command}}
  /// \brief\author Aaa
  int test_attach22;
} test_attach23;

// expected-warning@+1 {{empty paragraph passed to '\brief' command}}
/// \brief\author Aaa
namespace test_attach24 {
  // expected-warning@+1 {{empty paragraph passed to '\brief' command}}
  /// \brief\author Aaa
  namespace test_attach25 {
  }
}

// expected-warning@+1 {{empty paragraph passed to '\brief' command}}
/// \brief\author Aaa
/// \tparam T Aaa
template<typename T>
void test_attach26(T aaa);

// expected-warning@+1 {{empty paragraph passed to '\brief' command}}
/// \brief\author Aaa
/// \tparam T Aaa
template<typename T, typename U>
void test_attach27(T aaa, U bbb);

// expected-warning@+2 {{empty paragraph passed to '\brief' command}}
// expected-warning@+2 {{template parameter 'T' not found in the template declaration}}
/// \brief\author Aaa
/// \tparam T Aaa
template<>
void test_attach27(int aaa, int bbb);

// expected-warning@+1 {{empty paragraph passed to '\brief' command}}
/// \brief\author Aaa
/// \tparam T Aaa
template<typename T>
class test_attach28 {
  T aaa;
};

// expected-warning@+1 {{empty paragraph passed to '\brief' command}}
/// \brief\author Aaa
using test_attach29 = test_attach28<int>;

// expected-warning@+1 {{empty paragraph passed to '\brief' command}}
/// \brief\author Aaa
/// \tparam T Aaa
template<typename T, typename U>
class test_attach30 { };

// expected-warning@+1 {{empty paragraph passed to '\brief' command}}
/// \brief\author Aaa
/// \tparam T Aaa
template<typename T>
class test_attach30<T, int> { };

// expected-warning@+1 {{empty paragraph passed to '\brief' command}}
/// \brief\author Aaa
template<>
class test_attach30<int, int> { };

// expected-warning@+1 {{empty paragraph passed to '\brief' command}}
/// \brief\author Aaa
template<typename T>
using test_attach31 = test_attach30<T, int>;

// expected-warning@+1 {{empty paragraph passed to '\brief' command}}
/// \brief\author Aaa
/// \tparam T Aaa
template<typename T, typename U, typename V>
class test_attach32 { };

// expected-warning@+1 {{empty paragraph passed to '\brief' command}}
/// \brief\author Aaa
/// \tparam T Aaa
template<typename T, typename U>
class test_attach32<T, U, int> { };

// expected-warning@+1 {{empty paragraph passed to '\brief' command}}
/// \brief\author Aaa
/// \tparam T Aaa
template<typename T>
class test_attach32<T, int, int> { };

// expected-warning@+2 {{empty paragraph passed to '\brief' command}}
// expected-warning@+2 {{template parameter 'T' not found in the template declaration}}
/// \brief\author Aaa
/// \tparam T Aaa
template<>
class test_attach32<int, int, int> { };

// expected-warning@+1 {{empty paragraph passed to '\brief' command}}
/// \brief\author Aaa
class test_attach33 {
  // expected-warning@+1 {{empty paragraph passed to '\brief' command}}
  /// \brief\author Aaa
  /// \tparam T Aaa
  template<typename T, typename U>
  void test_attach34(T aaa, U bbb);
};

template<typename T>
class test_attach35 {
  // expected-warning@+2 {{empty paragraph passed to '\brief' command}}
  // expected-warning@+2 {{template parameter 'T' not found in the template declaration}}
  /// \brief\author Aaa
  /// \tparam T Aaa
  template<typename TT, typename UU>
  void test_attach36(TT aaa, UU bbb);
};

// expected-warning@+2 {{empty paragraph passed to '\brief' command}}
// expected-warning@+2 {{template parameter 'T' not found in the template declaration}}
/// \brief\author Aaa
/// \tparam T Aaa
template<> template<>
void test_attach35<int>::test_attach36(int aaa, int bbb) {}

template<typename T>
class test_attach37 {
  // expected-warning@+2 {{empty paragraph passed to '\brief' command}}
  // expected-warning@+2 {{'\tparam' command used in a comment that is not attached to a template declaration}}
  /// \brief\author Aaa
  /// \tparam T Aaa
  void test_attach38(int aaa, int bbb);

  void test_attach39(int aaa, int bbb);
};

// expected-warning@+2 {{empty paragraph passed to '\brief' command}}
// expected-warning@+2 {{template parameter 'T' not found in the template declaration}}
/// \brief\author Aaa
/// \tparam T Aaa
template<>
void test_attach37<int>::test_attach38(int aaa, int bbb) {}

// expected-warning@+1 {{empty paragraph passed to '\brief' command}}
/// \brief\author Aaa
/// \tparam T Aaa
template<typename T>
void test_attach37<T>::test_attach39(int aaa, int bbb) {}

// We used to emit warning that parameter 'a' is not found because we parsed
// the comment in context of the redeclaration which does not have parameter
// names.
template <typename T>
struct test_attach38 {
  /*!
    \param a  First param
    \param b  Second param
  */
  template <typename B>
  void test_attach39(T a, B b);
};

template <>
template <typename B>
void test_attach38<int>::test_attach39(int, B);

// The inline comments expect a string after the command.
// expected-warning@+1 {{'\a' command does not have a valid word argument}}
/// \a
int test_inline_no_argument_a_bad(int);

/// \a A
int test_inline_no_argument_a_good(int);

// expected-warning@+1 {{'\anchor' command does not have a valid word argument}}
/// \anchor
int test_inline_no_argument_anchor_bad(int);

/// \anchor A
int test_inline_no_argument_anchor_good(int);

// expected-warning@+1 {{'@b' command does not have a valid word argument}}
/// @b
int test_inline_no_argument_b_bad(int);

/// @b A
int test_inline_no_argument_b_good(int);

// expected-warning@+1 {{'\c' command does not have a valid word argument}}
/// \c
int test_inline_no_argument_c_bad(int);

/// \c A
int test_inline_no_argument_c_good(int);

// expected-warning@+1 {{'\e' command does not have a valid word argument}}
/// \e
int test_inline_no_argument_e_bad(int);

/// \e A
int test_inline_no_argument_e_good(int);

// expected-warning@+1 {{'\em' command does not have a valid word argument}}
/// \em
int test_inline_no_argument_em_bad(int);

/// \em A
int test_inline_no_argument_em_good(int);

// expected-warning@+1 {{'\p' command does not have a valid word argument}}
/// \p
int test_inline_no_argument_p_bad(int);

/// \p A
int test_inline_no_argument_p_good(int);

// PR13411, reduced.  We used to crash on this.
/**
 * @code Aaa.
 */
void test_nocrash1(int);

// We used to crash on this.
// expected-warning@+2 {{empty paragraph passed to '\param' command}}
// expected-warning@+1 {{empty paragraph passed to '\brief' command}}
/// \param\brief
void test_nocrash2(int);

// PR13593, example 1 and 2

/**
* Bla.
*/
template <typename>
void test_nocrash3();

/// Foo
template <typename, typename>
void test_nocrash4() { }

template <typename>
void test_nocrash3()
{
}

// PR13593, example 3

/**
 * aaa
 */
template <typename T>
inline T test_nocrash5(T a1)
{
    return a1;
}

///
//,

inline void test_nocrash6()
{
    test_nocrash5(1);
}

// We used to crash on this.

/*!
  Blah.
*/
typedef const struct test_nocrash7 * test_nocrash8;

// We used to crash on this.

// expected-warning@+1 {{unknown command tag name}}
/// aaa \unknown aaa \unknown aaa
int test_nocrash9;

// We used to crash on this.  PR15068

// expected-warning@+2 {{empty paragraph passed to '@param' command}}
// expected-warning@+2 {{empty paragraph passed to '@param' command}}
///@param x
///@param y
int test_nocrash10(int x, int y);

// expected-warning@+2 {{empty paragraph passed to '@param' command}} expected-warning@+2 {{parameter 'x' not found in the function declaration}}
// expected-warning@+2 {{empty paragraph passed to '@param' command}} expected-warning@+2 {{parameter 'y' not found in the function declaration}}
///@param x
///@param y
int test_nocrash11();

// expected-warning@+3 {{empty paragraph passed to '@param' command}} expected-warning@+3 {{parameter 'x' not found in the function declaration}}
// expected-warning@+3 {{empty paragraph passed to '@param' command}} expected-warning@+3 {{parameter 'y' not found in the function declaration}}
/**
@param x
@param y
**/
int test_nocrash12();

// expected-warning@+2 {{empty paragraph passed to '@param' command}}
// expected-warning@+1 {{empty paragraph passed to '@param' command}}
///@param x@param y
int test_nocrash13(int x, int y);

/**
 * \verbatim
 * Aaa
 **/
int test_nocrash14();

// rdar://12379114
// expected-warning@+2 {{'@union' command should not be used in a comment attached to a non-union declaration}}
/*!
   @union U This is new 
*/
struct U { int iS; };

/*!
  @union U1
*/
union U1 {int i; };

// expected-warning@+2 {{'@struct' command should not be used in a comment attached to a non-struct declaration}}
/*!
 @struct S2
*/
union S2 {};

/*!
  @class C1
*/
class C1;

/*!
  @struct S3;
*/
class S3;

// rdar://14124702
//----------------------------------------------------------------------
/// @class Predicate Predicate.h "lldb/Host/Predicate.h"
/// @brief A C++ wrapper class for providing threaded access to a value
/// of type T.
///
/// A templatized class.
/// specified values.
//----------------------------------------------------------------------
template <class T, class T1>
class Predicate
{
};

//----------------------------------------------------------------------
/// @class Predicate<int, char> Predicate.h "lldb/Host/Predicate.h"
/// @brief A C++ wrapper class for providing threaded access to a value
/// of type T.
///
/// A template specialization class.
//----------------------------------------------------------------------
template<> class Predicate<int, char>
{
};

//----------------------------------------------------------------------
/// @class Predicate<T, int> Predicate.h "lldb/Host/Predicate.h"
/// @brief A C++ wrapper class for providing threaded access to a value
/// of type T.
///
/// A partial specialization template class.
//----------------------------------------------------------------------
template<class T> class Predicate<T, int>
{
};

/*!     @function test_function
*/
template <class T> T test_function (T arg);

/*!     @function test_function<int>
*/
template <> int test_function<int> (int arg);

namespace AllowParamAndReturnsOnFunctionPointerVars {

/**
 * functionPointerVariable
 *
 * @param i is integer.
 * @returns integer.
 */
int (*functionPointerVariable)(int i);

struct HasFields {
  /**
   * functionPointerField
   *
   * @param i is integer.
   * @returns integer.
   */
  int (*functionPointerField)(int i);
};

// expected-warning@+5 {{'\returns' command used in a comment that is attached to a function returning void}}
/**
 * functionPointerVariable
 *
 * \param p not here.
 * \returns integer.
 */
void (*functionPointerVariableThatLeadsNowhere)();

// Still warn about param/returns commands for variables that don't specify
// the type directly:

/**
 * FunctionPointerTypedef
 *
 * \param i is integer.
 * \returns integer.
 */
typedef int (*FunctionPointerTypedef)(int i);

/**
 * FunctionPointerTypealias
 *
 * \param i is integer.
 * \returns integer.
 */
using FunctionPointerTypealias = int (*)(int i);

// expected-warning@+5 {{'@param' command used in a comment that is not attached to a function declaration}}
// expected-warning@+5 {{'@returns' command used in a comment that is not attached to a function or method declaration}}
/**
 * functionPointerVariable
 *
 * @param i is integer.
 * @returns integer.
 */
FunctionPointerTypedef functionPointerTypedefVariable;

struct HasMoreFields {
  // expected-warning@+5 {{'\param' command used in a comment that is not attached to a function declaration}}
  // expected-warning@+5 {{'\returns' command used in a comment that is not attached to a function or method declaration}}
  /**
   * functionPointerTypealiasField
   *
   * \param i is integer.
   * \returns integer.
   */
  FunctionPointerTypealias functionPointerTypealiasField;
};

}

/*!
 * Function pointer typedef with variadic params.
 *
 * @param a
 * works
 *
 * @param ...
 * now should work too.
 */
typedef void (*VariadicFnType)(int a, ...);

/*!
 * Function pointer type alias with variadic params.
 *
 * @param a
 * works
 *
 * @param ...
 * now should work too.
 */
using VariadicFnType2 = void (*)(int a, ...);

// expected-warning@+2 {{empty paragraph passed to '@note' command}}
/**
@note
\f$\texttt{mu}_{00}=\texttt{m}_{00}\f$, \f$\texttt{nu}_{00}=1\f$
\f$\texttt{nu}_{10}=\texttt{mu}_{10}=\texttt{mu}_{01}=\texttt{mu}_{10}=0\f$
 */
class EmptyNoteNoCrash {
};

namespace PR42844 { // Assertion failures when using typedefed function pointers
typedef void (*AA)();
typedef AA A();
A *a; ///< \return none
// expected-warning@-1 {{'\return' command used in a comment that is not attached to a function or method declaration}}

typedef void B();
B *b; ///< \return none
// expected-warning@-1 {{'\return' command used in a comment that is not attached to a function or method declaration}}

void CC();
typedef void C();
C &c = CC; ///< \return none
// expected-warning@-1 {{'\return' command used in a comment that is not attached to a function or method declaration}}

using DD = void(*)();
using D = DD();
D *d; ///< \return none
// expected-warning@-1 {{'\return' command used in a comment that is not attached to a function or method declaration}}

using E = void();
E *e; ///< \return none
// expected-warning@-1 {{'\return' command used in a comment that is not attached to a function or method declaration}}

void FF();
using F = void();
F &f = FF; ///< \return none
// expected-warning@-1 {{'\return' command used in a comment that is not attached to a function or method declaration}}

} // namespace PR42844
