// RUN: %clang_cc1 -fsyntax-only -Wdocumentation -fcomment-block-commands=foobar -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wdocumentation -fcomment-block-commands=foobar -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s

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

#define MY_ATTR_DEPRECATED __attribute__((deprecated))

// expected-warning@+1 {{declaration is marked with '\deprecated' command but does not have a deprecation attribute}} expected-note@+2 {{add a deprecation attribute to the declaration to silence this warning}}
/// \deprecated
void test_deprecated_9(int a);

// rdar://12381408
// expected-warning@+2  {{unknown command tag name 'retur'; did you mean 'return'?}}
/// \brief testing fixit
/// \retur int in FooBar
int FooBar();

// expected-warning@+1  {{unknown command tag name 'fooba'; did you mean 'foobar'?}}
/// \fooba bbb IS_DOXYGEN_END
int gorf();

/// \t bbb IS_DOXYGEN_END
int Bar();

// expected-warning@+2  {{unknown command tag name 'encode'; did you mean 'endcode'?}}
// expected-warning@+1  {{'\endcode' command does not terminate a verbatim text block}}
/// \encode PR18051
int PR18051();

// CHECK: fix-it:"{{.*}}":{5:12-5:22}:"a"
// CHECK: fix-it:"{{.*}}":{9:12-9:15}:"aaa"
// CHECK: fix-it:"{{.*}}":{13:13-13:23}:"T"
// CHECK: fix-it:"{{.*}}":{18:13-18:18}:"SomeTy"
// CHECK: fix-it:"{{.*}}":{25:25-25:25}:" __attribute__((deprecated))"
// CHECK: fix-it:"{{.*}}":{29:30-29:30}:" __attribute__((deprecated))"
// CHECK: fix-it:"{{.*}}":{34:27-34:27}:" __attribute__((deprecated))"
// CHECK: fix-it:"{{.*}}":{38:27-38:27}:" __attribute__((deprecated))"
// CHECK: fix-it:"{{.*}}":{46:27-46:27}:" __attribute__((deprecated))"
// CHECK: fix-it:"{{.*}}":{50:27-50:27}:" __attribute__((deprecated))"
// CHECK: fix-it:"{{.*}}":{58:30-58:30}:" MY_ATTR_DEPRECATED"
// CHECK: fix-it:"{{.*}}":{63:6-63:11}:"return"
// CHECK: fix-it:"{{.*}}":{67:6-67:11}:"foobar"
// CHECK: fix-it:"{{.*}}":{75:6-75:12}:"endcode"
