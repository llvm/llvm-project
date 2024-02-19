// RUN: %check_clang_tidy %s bugprone-easily-swappable-parameters %t \
// RUN:   -config='{CheckOptions: { \
// RUN:     bugprone-easily-swappable-parameters.MinimumLength: 2, \
// RUN:     bugprone-easily-swappable-parameters.IgnoredParameterNames: "", \
// RUN:     bugprone-easily-swappable-parameters.IgnoredParameterTypeSuffixes: "bool;MyBool;struct U;MAKE_LOGICAL_TYPE(int)", \
// RUN:     bugprone-easily-swappable-parameters.QualifiersMix: 0, \
// RUN:     bugprone-easily-swappable-parameters.ModelImplicitConversions: 0, \
// RUN:     bugprone-easily-swappable-parameters.SuppressParametersUsedTogether: 0, \
// RUN:     bugprone-easily-swappable-parameters.NamePrefixSuffixSilenceDissimilarityTreshold: 0 \
// RUN:  }}' -- -Wno-strict-prototypes -x c

#define bool _Bool
#define true 1
#define false 0

typedef bool MyBool;

#define TheLogicalType bool

void declVoid(void);         // NO-WARN: Declaration only.
void decl();                 // NO-WARN: Declaration only.
void oneParam(int I) {}      // NO-WARN: 1 parameter.
void variadic(int I, ...) {} // NO-WARN: 1 visible parameter.

void trivial(int I, int J) {}
// CHECK-MESSAGES: :[[@LINE-1]]:14: warning: 2 adjacent parameters of 'trivial' of similar type ('int') are easily swapped by mistake [bugprone-easily-swappable-parameters]
// CHECK-MESSAGES: :[[@LINE-2]]:18: note: the first parameter in the range is 'I'
// CHECK-MESSAGES: :[[@LINE-3]]:25: note: the last parameter in the range is 'J'

void qualifier(int I, const int CI) {} // NO-WARN: Distinct types.

void restrictQualifier(char *restrict CPR1, char *restrict CPR2) {}
// CHECK-MESSAGES: :[[@LINE-1]]:24: warning: 2 adjacent parameters of 'restrictQualifier' of similar type ('char *restrict')
// CHECK-MESSAGES: :[[@LINE-2]]:39: note: the first parameter in the range is 'CPR1'
// CHECK-MESSAGES: :[[@LINE-3]]:60: note: the last parameter in the range is 'CPR2'

void pointer1(int *IP1, int *IP2) {}
// CHECK-MESSAGES: :[[@LINE-1]]:15: warning: 2 adjacent parameters of 'pointer1' of similar type ('int *')
// CHECK-MESSAGES: :[[@LINE-2]]:20: note: the first parameter in the range is 'IP1'
// CHECK-MESSAGES: :[[@LINE-3]]:30: note: the last parameter in the range is 'IP2'

void pointerConversion(int *IP, long *LP) {}
// NO-WARN: Even though C can convert any T* to U* back and forth, compiler
// warnings already exist for this.

void testVariadicsCall() {
  int IVal = 1;
  decl(IVal); // NO-WARN: Particular calls to "variadics" are like template
              // instantiations, and we do not model them.

  variadic(IVal);          // NO-WARN.
  variadic(IVal, 2, 3, 4); // NO-WARN.
}

struct S {};
struct T {};

void taggedTypes1(struct S SVar, struct T TVar) {} // NO-WARN: Distinct types.

void taggedTypes2(struct S SVar1, struct S SVar2) {}
// CHECK-MESSAGES: :[[@LINE-1]]:19: warning: 2 adjacent parameters of 'taggedTypes2' of similar type ('struct S')
// CHECK-MESSAGES: :[[@LINE-2]]:28: note: the first parameter in the range is 'SVar1'
// CHECK-MESSAGES: :[[@LINE-3]]:44: note: the last parameter in the range is 'SVar2'

void wrappers(struct { int I; } I1, struct { int I; } I2) {} // NO-WARN: Distinct anonymous types.

void knr(I, J)
  int I;
  int J;
{}
// CHECK-MESSAGES: :[[@LINE-3]]:3: warning: 2 adjacent parameters of 'knr' of similar type ('int')
// CHECK-MESSAGES: :[[@LINE-4]]:7: note: the first parameter in the range is 'I'
// CHECK-MESSAGES: :[[@LINE-4]]:7: note: the last parameter in the range is 'J'

void boolAsWritten(bool B1, bool B2) {} // NO-WARN: The type name is ignored.
// Note that "bool" is a macro that expands to "_Bool" internally, but it is
// only "bool" that is ignored from the two.

void underscoreBoolAsWritten(_Bool B1, _Bool B2) {}
// Even though it is "_Bool" that is written in the code, the diagnostic message
// respects the printing policy as defined by the compilation commands. Clang's
// default in C mode seems to say that the type itself is "bool", not "_Bool".
// CHECK-MESSAGES: :[[@LINE-4]]:30: warning: 2 adjacent parameters of 'underscoreBoolAsWritten' of similar type ('bool')
// CHECK-MESSAGES: :[[@LINE-5]]:36: note: the first parameter in the range is 'B1'
// CHECK-MESSAGES: :[[@LINE-6]]:46: note: the last parameter in the range is 'B2'

void typedefdBoolAsWritten(MyBool MB1, MyBool MB2) {} // NO-WARN: "MyBool" as written type name ignored.

void otherBoolMacroAsWritten(TheLogicalType TLT1, TheLogicalType TLT2) {}
// CHECK-MESSAGES: :[[@LINE-1]]:30: warning: 2 adjacent parameters of 'otherBoolMacroAsWritten' of similar type ('bool')
// CHECK-MESSAGES: :[[@LINE-2]]:45: note: the first parameter in the range is 'TLT1'
// CHECK-MESSAGES: :[[@LINE-3]]:66: note: the last parameter in the range is 'TLT2'

struct U {};
typedef struct U U;

void typedefStruct(U X, U Y) {}
// CHECK-MESSAGES: :[[@LINE-1]]:20: warning: 2 adjacent parameters of 'typedefStruct' of similar type ('U')
// CHECK-MESSAGES: :[[@LINE-2]]:22: note: the first parameter in the range is 'X'
// CHECK-MESSAGES: :[[@LINE-3]]:27: note: the last parameter in the range is 'Y'

void ignoredStructU(struct U X, struct U Y) {} // NO-WARN: "struct U" ignored.

#define TYPE_TAG_TO_USE struct // We are in C!
#define MAKE_TYPE_NAME(T) TYPE_TAG_TO_USE T

void macroMagic1(TYPE_TAG_TO_USE T X, TYPE_TAG_TO_USE T Y) {}
// CHECK-MESSAGES: :[[@LINE-1]]:18: warning: 2 adjacent parameters of 'macroMagic1' of similar type ('struct T')
// CHECK-MESSAGES: :[[@LINE-5]]:25: note: expanded from macro 'TYPE_TAG_TO_USE'
// CHECK-MESSAGES: :[[@LINE-3]]:36: note: the first parameter in the range is 'X'
// CHECK-MESSAGES: :[[@LINE-4]]:57: note: the last parameter in the range is 'Y'

void macroMagic2(TYPE_TAG_TO_USE U X, TYPE_TAG_TO_USE U Y) {}
// "struct U" is ignored, but that is not what is written here!
// CHECK-MESSAGES: :[[@LINE-2]]:18: warning: 2 adjacent parameters of 'macroMagic2' of similar type ('struct U')
// CHECK-MESSAGES: :[[@LINE-12]]:25: note: expanded from macro 'TYPE_TAG_TO_USE'
// CHECK-MESSAGES: :[[@LINE-4]]:36: note: the first parameter in the range is 'X'
// CHECK-MESSAGES: :[[@LINE-5]]:57: note: the last parameter in the range is 'Y'

void evenMoreMacroMagic1(MAKE_TYPE_NAME(T) X, MAKE_TYPE_NAME(T) Y) {}
// CHECK-MESSAGES: :[[@LINE-1]]:26: warning: 2 adjacent parameters of 'evenMoreMacroMagic1' of similar type ('struct T')
// CHECK-MESSAGES: :[[@LINE-17]]:27: note: expanded from macro 'MAKE_TYPE_NAME'
// CHECK-MESSAGES: :[[@LINE-19]]:25: note: expanded from macro 'TYPE_TAG_TO_USE'
// CHECK-MESSAGES: :[[@LINE-4]]:44: note: the first parameter in the range is 'X'
// CHECK-MESSAGES: :[[@LINE-5]]:65: note: the last parameter in the range is 'Y'

void evenMoreMacroMagic2(MAKE_TYPE_NAME(U) X, MAKE_TYPE_NAME(U) Y) {}
// "struct U" is ignored, but that is not what is written here!
// CHECK-MESSAGES: :[[@LINE-2]]:26: warning: 2 adjacent parameters of 'evenMoreMacroMagic2' of similar type ('struct U')
// CHECK-MESSAGES: :[[@LINE-25]]:27: note: expanded from macro 'MAKE_TYPE_NAME'
// CHECK-MESSAGES: :[[@LINE-27]]:25: note: expanded from macro 'TYPE_TAG_TO_USE'
// CHECK-MESSAGES: :[[@LINE-5]]:44: note: the first parameter in the range is 'X'
// CHECK-MESSAGES: :[[@LINE-6]]:65: note: the last parameter in the range is 'Y'

#define MAKE_PRIMITIVE_WRAPPER(WRAPPED_TYPE) \
  MAKE_TYPE_NAME() {                         \
    WRAPPED_TYPE Member;                     \
  }

void thisIsGettingRidiculous(MAKE_PRIMITIVE_WRAPPER(int) I1,
                             MAKE_PRIMITIVE_WRAPPER(int) I2) {} // NO-WARN: Distinct anonymous types.

#define MAKE_LOGICAL_TYPE(X) bool

void macroMagic3(MAKE_LOGICAL_TYPE(char) B1, MAKE_LOGICAL_TYPE(long) B2) {}
// CHECK-MESSAGES: :[[@LINE-1]]:18: warning: 2 adjacent parameters of 'macroMagic3' of similar type ('bool')
// CHECK-MESSAGES: :[[@LINE-4]]:30: note: expanded from macro 'MAKE_LOGICAL_TYPE'
// CHECK-MESSAGES: :[[@LINE-136]]:14: note: expanded from macro 'bool'
// CHECK-MESSAGES: :[[@LINE-4]]:42: note: the first parameter in the range is 'B1'
// CHECK-MESSAGES: :[[@LINE-5]]:70: note: the last parameter in the range is 'B2'

void macroMagic4(MAKE_LOGICAL_TYPE(int) B1, MAKE_LOGICAL_TYPE(int) B2) {} // NO-WARN: "Type name" ignored.
