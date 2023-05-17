// RUN: %check_clang_tidy %s bugprone-easily-swappable-parameters %t \
// RUN:   -config='{CheckOptions: [ \
// RUN:     {key: bugprone-easily-swappable-parameters.MinimumLength, value: 2}, \
// RUN:     {key: bugprone-easily-swappable-parameters.IgnoredParameterNames, value: ""}, \
// RUN:     {key: bugprone-easily-swappable-parameters.IgnoredParameterTypeSuffixes, value: ""}, \
// RUN:     {key: bugprone-easily-swappable-parameters.QualifiersMix, value: 0}, \
// RUN:     {key: bugprone-easily-swappable-parameters.ModelImplicitConversions, value: 0}, \
// RUN:     {key: bugprone-easily-swappable-parameters.SuppressParametersUsedTogether, value: 1}, \
// RUN:     {key: bugprone-easily-swappable-parameters.NamePrefixSuffixSilenceDissimilarityTreshold, value: 0} \
// RUN:  ]}' -- -Wno-strict-prototypes -x c

int myprint();
int add(int X, int Y);

void notRelated(int A, int B) {}
// CHECK-MESSAGES: :[[@LINE-1]]:17: warning: 2 adjacent parameters of 'notRelated' of similar type ('int')
// CHECK-MESSAGES: :[[@LINE-2]]:21: note: the first parameter in the range is 'A'
// CHECK-MESSAGES: :[[@LINE-3]]:28: note: the last parameter in the range is 'B'

int addedTogether(int A, int B) { return add(A, B); } // NO-WARN: Passed to same function.

void passedToSameKNRFunction(int A, int B) {
  myprint("foo", A);
  myprint("bar", B);
}
// CHECK-MESSAGES: :[[@LINE-4]]:30: warning: 2 adjacent parameters of 'passedToSameKNRFunction' of similar type ('int')
// CHECK-MESSAGES: :[[@LINE-5]]:34: note: the first parameter in the range is 'A'
// CHECK-MESSAGES: :[[@LINE-6]]:41: note: the last parameter in the range is 'B'
// This is actually a false positive: the "passed to same function" heuristic
// can't map the parameter index 1 to A and B because myprint() has no
// parameters.
