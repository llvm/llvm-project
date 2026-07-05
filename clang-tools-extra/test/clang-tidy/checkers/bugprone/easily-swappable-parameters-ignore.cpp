// RUN: %check_clang_tidy %s bugprone-easily-swappable-parameters %t \
// RUN:   -config='{CheckOptions: { \
// RUN:     bugprone-easily-swappable-parameters.MinimumLength: 2, \
// RUN:     bugprone-easily-swappable-parameters.IgnoredParameterNames: "\"\";Foo;Bar", \
// RUN:     bugprone-easily-swappable-parameters.IgnoredParameterTypeSuffixes: "T", \
// RUN:     bugprone-easily-swappable-parameters.QualifiersMix: 0, \
// RUN:     bugprone-easily-swappable-parameters.ModelImplicitConversions: 0, \
// RUN:     bugprone-easily-swappable-parameters.SuppressParametersUsedTogether: 0, \
// RUN:     bugprone-easily-swappable-parameters.NamePrefixSuffixSilenceDissimilarityThreshold: 0 \
// RUN:  }}' --

void ignoredUnnamed(int I, int, int) {} // NO-WARN: No >= 2 length of non-unnamed.

void nothingIgnored(int I, int J) {}
// CHECK-MESSAGES: :[[@LINE-1]]:21: warning: 2 adjacent parameters of 'nothingIgnored' of similar type ('int') are easily swapped by mistake [bugprone-easily-swappable-parameters]
// CHECK-MESSAGES: :[[@LINE-2]]:25: note: the first parameter in the range is 'I'
// CHECK-MESSAGES: :[[@LINE-3]]:32: note: the last parameter in the range is 'J'

void ignoredParameter(int Foo, int I, int J) {}
// CHECK-MESSAGES: :[[@LINE-1]]:32: warning: 2 adjacent parameters of 'ignoredParameter' of similar type ('int')
// CHECK-MESSAGES: :[[@LINE-2]]:36: note: the first parameter in the range is 'I'
// CHECK-MESSAGES: :[[@LINE-3]]:43: note: the last parameter in the range is 'J'

void ignoredParameterBoth(int Foo, int Bar) {} // NO-WARN.

struct S {};
struct T {};
struct MyT {};

void notIgnoredType(S S1, S S2) {}
// CHECK-MESSAGES: :[[@LINE-1]]:21: warning: 2 adjacent parameters of 'notIgnoredType' of similar type ('S')
// CHECK-MESSAGES: :[[@LINE-2]]:23: note: the first parameter in the range is 'S1'
// CHECK-MESSAGES: :[[@LINE-3]]:29: note: the last parameter in the range is 'S2'

void ignoredTypeExact(T T1, T T2) {} // NO-WARN.

void ignoredTypeSuffix(MyT M1, MyT M2) {} // NO-WARN.
