// RUN: %check_clang_tidy %s bugprone-easily-swappable-parameters %t \
// RUN:   -config='{CheckOptions: { \
// RUN:     bugprone-easily-swappable-parameters.MinimumLength: 2, \
// RUN:     bugprone-easily-swappable-parameters.IgnoredParameterNames: "", \
// RUN:     bugprone-easily-swappable-parameters.IgnoredParameterTypeSuffixes: "", \
// RUN:     bugprone-easily-swappable-parameters.QualifiersMix: 0, \
// RUN:     bugprone-easily-swappable-parameters.ModelImplicitConversions: 1, \
// RUN:     bugprone-easily-swappable-parameters.SuppressParametersUsedTogether: 0, \
// RUN:     bugprone-easily-swappable-parameters.NamePrefixSuffixSilenceDissimilarityThreshold: 0 \
// RUN:  }}' --

void implicitDoesntBreakOtherStuff(int A, int B) {}
// CHECK-MESSAGES: :[[@LINE-1]]:36: warning: 2 adjacent parameters of 'implicitDoesntBreakOtherStuff' of similar type ('int') are easily swapped by mistake [bugprone-easily-swappable-parameters]
// CHECK-MESSAGES: :[[@LINE-2]]:40: note: the first parameter in the range is 'A'
// CHECK-MESSAGES: :[[@LINE-3]]:47: note: the last parameter in the range is 'B'

void arrayAndPtr1(int *IP, int IA[]) { arrayAndPtr1(IA, IP); }
// CHECK-MESSAGES: :[[@LINE-1]]:19: warning: 2 adjacent parameters of 'arrayAndPtr1' of similar type ('int *')
// CHECK-MESSAGES: :[[@LINE-2]]:24: note: the first parameter in the range is 'IP'
// CHECK-MESSAGES: :[[@LINE-3]]:32: note: the last parameter in the range is 'IA'

void arrayAndPtr2(int *IP, int IA[8]) { arrayAndPtr2(IA, IP); }
// CHECK-MESSAGES: :[[@LINE-1]]:19: warning: 2 adjacent parameters of 'arrayAndPtr2' of similar type ('int *')
// CHECK-MESSAGES: :[[@LINE-2]]:24: note: the first parameter in the range is 'IP'
// CHECK-MESSAGES: :[[@LINE-3]]:32: note: the last parameter in the range is 'IA'

void arrayAndElement(int I, int IA[]) {} // NO-WARN.

void numericConversion1(int I, double D) { numericConversion1(D, I); }
// CHECK-MESSAGES: :[[@LINE-1]]:25: warning: 2 adjacent parameters of 'numericConversion1' of convertible types are easily swapped by mistake [bugprone-easily-swappable-parameters]
// CHECK-MESSAGES: :[[@LINE-2]]:29: note: the first parameter in the range is 'I'
// CHECK-MESSAGES: :[[@LINE-3]]:39: note: the last parameter in the range is 'D'
// CHECK-MESSAGES: :[[@LINE-4]]:32: note: 'int' and 'double' may be implicitly converted{{$}}

void numericConversion2(int I, short S) { numericConversion2(S, I); }
// CHECK-MESSAGES: :[[@LINE-1]]:25: warning: 2 adjacent parameters of 'numericConversion2' of convertible types
// CHECK-MESSAGES: :[[@LINE-2]]:29: note: the first parameter in the range is 'I'
// CHECK-MESSAGES: :[[@LINE-3]]:38: note: the last parameter in the range is 'S'
// CHECK-MESSAGES: :[[@LINE-4]]:32: note: 'int' and 'short' may be implicitly converted{{$}}

void numericConversion3(float F, unsigned long UL) { numericConversion3(UL, F); }
// CHECK-MESSAGES: :[[@LINE-1]]:25: warning: 2 adjacent parameters of 'numericConversion3' of convertible types
// CHECK-MESSAGES: :[[@LINE-2]]:31: note: the first parameter in the range is 'F'
// CHECK-MESSAGES: :[[@LINE-3]]:48: note: the last parameter in the range is 'UL'
// CHECK-MESSAGES: :[[@LINE-4]]:34: note: 'float' and 'unsigned long' may be implicitly converted{{$}}

enum Unscoped { U_A,
                U_B };
enum UnscopedFixed : char { UF_A,
                            UF_B };

void numericConversion4(int I, enum Unscoped U) { numericConversion4(U, I); }
// CHECK-MESSAGES: :[[@LINE-1]]:25: warning: 2 adjacent parameters of 'numericConversion4' of convertible types
// CHECK-MESSAGES: :[[@LINE-2]]:29: note: the first parameter in the range is 'I'
// CHECK-MESSAGES: :[[@LINE-3]]:46: note: the last parameter in the range is 'U'
// CHECK-MESSAGES: :[[@LINE-4]]:32: note: 'int' and 'enum Unscoped' may be implicitly converted{{$}}

void numericConversion5(int I, enum UnscopedFixed UF) { numericConversion5(UF, I); }
// CHECK-MESSAGES: :[[@LINE-1]]:25: warning: 2 adjacent parameters of 'numericConversion5' of convertible types
// CHECK-MESSAGES: :[[@LINE-2]]:29: note: the first parameter in the range is 'I'
// CHECK-MESSAGES: :[[@LINE-3]]:51: note: the last parameter in the range is 'UF'
// CHECK-MESSAGES: :[[@LINE-4]]:32: note: 'int' and 'enum UnscopedFixed' may be implicitly converted{{$}}

void numericConversion7(double D, enum Unscoped U) { numericConversion7(U, D); }
// CHECK-MESSAGES: :[[@LINE-1]]:25: warning: 2 adjacent parameters of 'numericConversion7' of convertible types
// CHECK-MESSAGES: :[[@LINE-2]]:32: note: the first parameter in the range is 'D'
// CHECK-MESSAGES: :[[@LINE-3]]:49: note: the last parameter in the range is 'U'
// CHECK-MESSAGES: :[[@LINE-4]]:35: note: 'double' and 'enum Unscoped' may be implicitly converted{{$}}

void numericConversion8(double D, enum UnscopedFixed UF) { numericConversion8(UF, D); }
// CHECK-MESSAGES: :[[@LINE-1]]:25: warning: 2 adjacent parameters of 'numericConversion8' of convertible types
// CHECK-MESSAGES: :[[@LINE-2]]:32: note: the first parameter in the range is 'D'
// CHECK-MESSAGES: :[[@LINE-3]]:54: note: the last parameter in the range is 'UF'
// CHECK-MESSAGES: :[[@LINE-4]]:35: note: 'double' and 'enum UnscopedFixed' may be implicitly converted{{$}}

void pointeeConversion(int *IP, double *DP) { pointeeConversion(DP, IP); }
// NO-WARN: Even though this is possible in C, a swap is diagnosed by the compiler.
