// RUN: %check_clang_tidy %s bugprone-easily-swappable-parameters %t \
// RUN:   -config='{CheckOptions: { \
// RUN:     bugprone-easily-swappable-parameters.MinimumLength: 2, \
// RUN:     bugprone-easily-swappable-parameters.IgnoredParameterNames: "", \
// RUN:     bugprone-easily-swappable-parameters.IgnoredParameterTypeSuffixes: "", \
// RUN:     bugprone-easily-swappable-parameters.QualifiersMix: 1, \
// RUN:     bugprone-easily-swappable-parameters.ModelImplicitConversions: 1, \
// RUN:     bugprone-easily-swappable-parameters.SuppressParametersUsedTogether: 0, \
// RUN:     bugprone-easily-swappable-parameters.NamePrefixSuffixSilenceDissimilarityTreshold: 0 \
// RUN:  }}' --

void numericAndQualifierConversion(int I, const double CD) { numericAndQualifierConversion(CD, I); }
// CHECK-MESSAGES: :[[@LINE-1]]:36: warning: 2 adjacent parameters of 'numericAndQualifierConversion' of convertible types are easily swapped by mistake [bugprone-easily-swappable-parameters]
// CHECK-MESSAGES: :[[@LINE-2]]:40: note: the first parameter in the range is 'I'
// CHECK-MESSAGES: :[[@LINE-3]]:56: note: the last parameter in the range is 'CD'
// CHECK-MESSAGES: :[[@LINE-4]]:43: note: 'int' and 'const double' parameters accept and bind the same kind of values
// CHECK-MESSAGES: :[[@LINE-5]]:43: note: 'int' and 'const double' may be implicitly converted: 'int' -> 'const double' (as 'double'), 'const double' (as 'double') -> 'int'
