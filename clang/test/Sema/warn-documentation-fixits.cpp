// RUN: %clang_cc1 -fsyntax-only -Wdocumentation -Wdocumentation-pedantic -fcomment-block-commands=foobar -verify %s
// RUN: %clang_cc1 -std=c++11 -fsyntax-only -Wdocumentation -Wdocumentation-pedantic -fcomment-block-commands=foobar -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck -DATTRIBUTE="__attribute__((deprecated))" %s
// RUN: %clang_cc1 -std=c++14 -fsyntax-only -Wdocumentation -Wdocumentation-pedantic -fcomment-block-commands=foobar -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck --check-prefixes=CHECK,CHECK14 -DATTRIBUTE="[[deprecated]]" %s

// expected-warning@+1 {{parameter 'ZZZZZZZZZZ' not found in the function declaration}} expected-note@+1 {{did you mean 'a'?}}
/// \param ZZZZZZZZZZ Blah blah.
int test1(int a);

// expected-warning@+1 {{parameter 'aab' not found in the function declaration}} expected-note@+1 {{did you mean 'aaa'?}}
/// \param aab Blah blah.
int test2(int aaa, int bbb);

// expected-warning@+1 {{template parameter 'ZZZZZZZZZZ' not found in the template declaration}} expected-note@+1 {{did you mean 'T'?}}
/// \tparam ZZZZZZZZZZ Aaa
template<typename T>
void test3(T aaa);

// expected-warning@+1 {{template parameter 'SomTy' not found in the template declaration}} expected-note@+1 {{did you mean 'SomeTy'?}}
/// \tparam SomTy Aaa
/// \tparam OtherTy Bbb
template<typename SomeTy, typename OtherTy>
void test4(SomeTy aaa, OtherTy bbb);

// expected-warning@+1 {{declaration is marked with '\deprecated' command but does not have a deprecation attribute}} expected-note@+2 {{add a deprecation attribute to the declaration to silence this warning}}
/// \deprecated
void test_deprecated_1();

// expected-warning@+1 {{declaration is marked with '\deprecated' command but does not have a deprecation attribute}} expected-note@+2 {{add a deprecation attribute to the declaration to silence this warning}}
/// \deprecated
void test_deprecated_2(int a);

struct test_deprecated_3 {
  // expected-warning@+1 {{declaration is marked with '\deprecated' command but does not have a deprecation attribute}} expected-note@+2 {{add a deprecation attribute to the declaration to silence this warning}}
  /// \deprecated
  void test_deprecated_4();

  // expected-warning@+1 {{declaration is marked with '\deprecated' command but does not have a deprecation attribute}} expected-note@+2 {{add a deprecation attribute to the declaration to silence this warning}}
  /// \deprecated
  void test_deprecated_5() {
  }
};

template<typename T>
struct test_deprecated_6 {
  // expected-warning@+1 {{declaration is marked with '\deprecated' command but does not have a deprecation attribute}} expected-note@+2 {{add a deprecation attribute to the declaration to silence this warning}}
  /// \deprecated
  void test_deprecated_7();

  // expected-warning@+1 {{declaration is marked with '\deprecated' command but does not have a deprecation attribute}} expected-note@+2 {{add a deprecation attribute to the declaration to silence this warning}}
  /// \deprecated
  void test_deprecated_8() {
  }
};

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
#define MY_ATTR_DEPRECATED __attribute__((deprecated))

// expected-warning@+1 {{declaration is marked with '\deprecated' command but does not have a deprecation attribute}} expected-note@+2 {{add a deprecation attribute to the declaration to silence this warning}}
/// \deprecated
void test_deprecated_9(int a);

#if __cplusplus >= 201402L
#define ATTRIBUTE_DEPRECATED [[deprecated]]

// expected-warning@+1 {{declaration is marked with '\deprecated' command but does not have a deprecation attribute}} expected-note@+2 {{add a deprecation attribute to the declaration to silence this warning}}
/// \deprecated
void test_deprecated_10(int a);
#endif

// expected-warning@+2  {{unknown command tag name 'retur'; did you mean 'return'?}}
/// \brief testing fixit
/// \retur int in FooBar
int FooBar();

// expected-warning@+1  {{unknown command tag name 'fooba'; did you mean 'foobar'?}}
/// \fooba bbb IS_DOXYGEN_END
int gorf();

// expected-warning@+1 {{unknown command tag name}}
/// \t bbb IS_DOXYGEN_END
int Bar();

// expected-warning@+2  {{unknown command tag name 'encode'; did you mean 'endcode'?}}
// expected-warning@+1  {{'\endcode' command does not terminate a verbatim text block}}
/// \encode PR18051
int PR18051();

// CHECK: fix-it:"{{.*}}":{6:12-6:22}:"a"
// CHECK: fix-it:"{{.*}}":{10:12-10:15}:"aaa"
// CHECK: fix-it:"{{.*}}":{14:13-14:23}:"T"
// CHECK: fix-it:"{{.*}}":{19:13-19:18}:"SomeTy"
// CHECK: fix-it:"{{.*}}":{26:1-26:1}:"[[ATTRIBUTE]] "
// CHECK: fix-it:"{{.*}}":{30:1-30:1}:"[[ATTRIBUTE]] "
// CHECK: fix-it:"{{.*}}":{35:3-35:3}:"[[ATTRIBUTE]] "
// CHECK: fix-it:"{{.*}}":{39:3-39:3}:"[[ATTRIBUTE]] "
// CHECK: fix-it:"{{.*}}":{47:3-47:3}:"[[ATTRIBUTE]] "
// CHECK: fix-it:"{{.*}}":{51:3-51:3}:"[[ATTRIBUTE]] "
// CHECK: fix-it:"{{.*}}":{76:3-76:3}:"[[ATTRIBUTE]] "
// CHECK: fix-it:"{{.*}}":{81:3-81:3}:"[[ATTRIBUTE]] "
// CHECK14: fix-it:"{{.*}}":{87:3-87:3}:"[[ATTRIBUTE]] "
// CHECK: fix-it:"{{.*}}":{97:1-97:1}:"MY_ATTR_DEPRECATED "
// CHECK14: fix-it:"{{.*}}":{104:1-104:1}:"ATTRIBUTE_DEPRECATED "
// CHECK: fix-it:"{{.*}}":{109:6-109:11}:"return"
// CHECK: fix-it:"{{.*}}":{113:6-113:11}:"foobar"
// CHECK: fix-it:"{{.*}}":{122:6-122:12}:"endcode"
