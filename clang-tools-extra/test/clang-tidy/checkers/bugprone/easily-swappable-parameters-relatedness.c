// RUN: %check_clang_tidy -std=c99,c11,c17 -check-suffixes=,BEFORE-C23 %s bugprone-easily-swappable-parameters %t \
// RUN:   -config='{CheckOptions: { \
// RUN:     bugprone-easily-swappable-parameters.MinimumLength: 2, \
// RUN:     bugprone-easily-swappable-parameters.IgnoredParameterNames: "", \
// RUN:     bugprone-easily-swappable-parameters.IgnoredParameterTypeSuffixes: "", \
// RUN:     bugprone-easily-swappable-parameters.QualifiersMix: 0, \
// RUN:     bugprone-easily-swappable-parameters.ModelImplicitConversions: 0, \
// RUN:     bugprone-easily-swappable-parameters.SuppressParametersUsedTogether: 1, \
// RUN:     bugprone-easily-swappable-parameters.NamePrefixSuffixSilenceDissimilarityThreshold: 0 \
// RUN:  }}' -- -Wno-strict-prototypes -x c
//
// RUN: %check_clang_tidy -std=c23-or-later %s bugprone-easily-swappable-parameters %t \
// RUN:   -config='{CheckOptions: { \
// RUN:     bugprone-easily-swappable-parameters.MinimumLength: 2, \
// RUN:     bugprone-easily-swappable-parameters.IgnoredParameterNames: "", \
// RUN:     bugprone-easily-swappable-parameters.IgnoredParameterTypeSuffixes: "", \
// RUN:     bugprone-easily-swappable-parameters.QualifiersMix: 0, \
// RUN:     bugprone-easily-swappable-parameters.ModelImplicitConversions: 0, \
// RUN:     bugprone-easily-swappable-parameters.SuppressParametersUsedTogether: 1, \
// RUN:     bugprone-easily-swappable-parameters.NamePrefixSuffixSilenceDissimilarityThreshold: 0 \
// RUN:  }}' -- -Wno-strict-prototypes -x c

int add(int X, int Y);

void notRelated(int A, int B) {}
// CHECK-MESSAGES: :[[@LINE-1]]:17: warning: 2 adjacent parameters of 'notRelated' of similar type ('int')
// CHECK-MESSAGES: :[[@LINE-2]]:21: note: the first parameter in the range is 'A'
// CHECK-MESSAGES: :[[@LINE-3]]:28: note: the last parameter in the range is 'B'

int addedTogether(int A, int B) { return add(A, B); } // NO-WARN: Passed to same function.

#if __STDC_VERSION__ < 202311L
int myprint();
void passedToSameKNRFunction(int A, int B) {
  myprint("foo", A);
  myprint("bar", B);
}
// CHECK-MESSAGES-BEFORE-C23: :[[@LINE-4]]:30: warning: 2 adjacent parameters of 'passedToSameKNRFunction' of similar type ('int')
// CHECK-MESSAGES-BEFORE-C23: :[[@LINE-5]]:34: note: the first parameter in the range is 'A'
// CHECK-MESSAGES-BEFORE-C23: :[[@LINE-6]]:41: note: the last parameter in the range is 'B'
// FIXME: This is actually a false positive: the "passed to same function" heuristic
// can't map the parameter index 1 to A and B because myprint() has no
// parameters.
// If you fix this, you should be able to combine the `%check_clang_tidy` invocations
// in this file into one and remove the `-std` and `-check-suffixes` arguments.
#endif
