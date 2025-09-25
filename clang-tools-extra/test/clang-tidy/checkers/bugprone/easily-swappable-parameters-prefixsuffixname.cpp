// RUN: %check_clang_tidy %s bugprone-easily-swappable-parameters %t \
// RUN:   -config='{CheckOptions: { \
// RUN:     bugprone-easily-swappable-parameters.MinimumLength: 2, \
// RUN:     bugprone-easily-swappable-parameters.IgnoredParameterNames: "", \
// RUN:     bugprone-easily-swappable-parameters.IgnoredParameterTypeSuffixes: "", \
// RUN:     bugprone-easily-swappable-parameters.QualifiersMix: 0, \
// RUN:     bugprone-easily-swappable-parameters.ModelImplicitConversions: 0, \
// RUN:     bugprone-easily-swappable-parameters.SuppressParametersUsedTogether: 0, \
// RUN:     bugprone-easily-swappable-parameters.NamePrefixSuffixSilenceDissimilarityThreshold: 1 \
// RUN:  }}' --

namespace std {
struct string {};
} // namespace std
class Matrix {};

void test1(int Foo, int Bar) {}
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: 2 adjacent parameters of 'test1' of similar type ('int') are easily swapped by mistake [bugprone-easily-swappable-parameters]
// CHECK-MESSAGES: :[[@LINE-2]]:16: note: the first parameter in the range is 'Foo'
// CHECK-MESSAGES: :[[@LINE-3]]:25: note: the last parameter in the range is 'Bar'

void test2(int A, int B) {}
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: 2 adjacent parameters of 'test2' of similar type ('int')
// CHECK-MESSAGES: :[[@LINE-2]]:16: note: the first parameter in the range is 'A'
// CHECK-MESSAGES: :[[@LINE-3]]:23: note: the last parameter in the range is 'B'

void test3(int Val1, int Val2) {} // NO-WARN.

void test4(int ValA, int Valb) {} // NO-WARN.

void test5(int Val1, int ValZ) {} // NO-WARN.

void test6(int PValue, int QValue) {} // NO-WARN.

void test7(std::string Astr, std::string Bstr) {} // NO-WARN.

void test8(int Aladdin, int Alabaster) {}
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: 2 adjacent parameters of 'test8' of similar type ('int')
// CHECK-MESSAGES: :[[@LINE-2]]:16: note: the first parameter in the range is 'Aladdin'
// CHECK-MESSAGES: :[[@LINE-3]]:29: note: the last parameter in the range is 'Alabaster'

void test9(Matrix Qmat, Matrix Rmat, Matrix Tmat) {} // NO-WARN.

void test10(int Something, int Other, int Foo, int Bar1, int Bar2, int Baz, int Qux) {}
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: 4 adjacent parameters of 'test10' of similar type ('int') are
// CHECK-MESSAGES: :[[@LINE-2]]:17: note: the first parameter in the range is 'Something'
// CHECK-MESSAGES: :[[@LINE-3]]:52: note: the last parameter in the range is 'Bar1'
//
// CHECK-MESSAGES: :[[@LINE-5]]:58: warning: 3 adjacent parameters of 'test10' of similar type ('int') are
// CHECK-MESSAGES: :[[@LINE-6]]:62: note: the first parameter in the range is 'Bar2'
// CHECK-MESSAGES: :[[@LINE-7]]:81: note: the last parameter in the range is 'Qux'

void test11(int Foobar, int Foo) {}
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: 2 adjacent parameters of 'test11' of similar type ('int')
// CHECK-MESSAGES: :[[@LINE-2]]:17: note: the first parameter in the range is 'Foobar'
// CHECK-MESSAGES: :[[@LINE-3]]:29: note: the last parameter in the range is 'Foo'
