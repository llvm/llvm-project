// RUN: %check_clang_tidy %s bugprone-easily-swappable-parameters %t \
// RUN:   -config='{CheckOptions: { \
// RUN:     bugprone-easily-swappable-parameters.MinimumLength: 3, \
// RUN:     bugprone-easily-swappable-parameters.IgnoredParameterNames: "", \
// RUN:     bugprone-easily-swappable-parameters.IgnoredParameterTypeSuffixes: "", \
// RUN:     bugprone-easily-swappable-parameters.QualifiersMix: 0, \
// RUN:     bugprone-easily-swappable-parameters.ModelImplicitConversions: 0, \
// RUN:     bugprone-easily-swappable-parameters.SuppressParametersUsedTogether: 0, \
// RUN:     bugprone-easily-swappable-parameters.NamePrefixSuffixSilenceDissimilarityTreshold: 0 \
// RUN:  }}' --

int add(int Left, int Right) { return Left + Right; } // NO-WARN: Only 2 parameters.

int magic(int Left, int Right, int X, int Y) { return 0; }
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: 4 adjacent parameters of 'magic' of similar type ('int') are easily swapped by mistake [bugprone-easily-swappable-parameters]
// CHECK-MESSAGES: :[[@LINE-2]]:15: note: the first parameter in the range is 'Left'
// CHECK-MESSAGES: :[[@LINE-3]]:43: note: the last parameter in the range is 'Y'

void multipleDistinctTypes(int I, int J, int K,
                           long L, long M,
                           double D, double E, double F) {}
// CHECK-MESSAGES: :[[@LINE-3]]:28: warning: 3 adjacent parameters of 'multipleDistinctTypes' of similar type ('int')
// CHECK-MESSAGES: :[[@LINE-4]]:32: note: the first parameter in the range is 'I'
// CHECK-MESSAGES: :[[@LINE-5]]:46: note: the last parameter in the range is 'K'
// NO-WARN: The [long, long] range is length of 2.
// CHECK-MESSAGES: :[[@LINE-5]]:28: warning: 3 adjacent parameters of 'multipleDistinctTypes' of similar type ('double')
// CHECK-MESSAGES: :[[@LINE-6]]:35: note: the first parameter in the range is 'D'
// CHECK-MESSAGES: :[[@LINE-7]]:55: note: the last parameter in the range is 'F'
