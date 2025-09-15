// RUN: %check_clang_tidy -std=c++17-or-later %s bugprone-easily-swappable-parameters %t \
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

typedef int Point2D[2];
typedef int Point3D[3];

void arrays1(Point2D P2D, Point3D P3D) {} // In reality this is (int*, int*).
// CHECK-MESSAGES: :[[@LINE-1]]:14: warning: 2 adjacent parameters of 'arrays1' of similar type ('int *') are
// CHECK-MESSAGES: :[[@LINE-2]]:22: note: the first parameter in the range is 'P2D'
// CHECK-MESSAGES: :[[@LINE-3]]:35: note: the last parameter in the range is 'P3D'

void crefToArrayTypedef1(int I, const Point2D &P) {}
// NO-WARN.

void crefToArrayTypedef2(int *IA, const Point2D &P) {}
// NO-WARN.

void crefToArrayTypedef3(int P1[2], const Point2D &P) {}
// NO-WARN.

void crefToArrayTypedefBoth1(const Point2D &VecDescartes, const Point3D &VecThreeD) {}
// NO-WARN: Distinct types.

template <int N, int M>
void templatedArrayRef(int (&Array1)[N], int (&Array2)[M]) {}
// NO-WARN: Distinct template types in the primary template.

void templatedArrayRefTest() {
  int Foo[12], Bar[12];
  templatedArrayRef(Foo, Bar);

  int Baz[12], Quux[42];
  templatedArrayRef(Baz, Quux);

  // NO-WARN: Implicit instantiations are not checked.
}

template <>
void templatedArrayRef(int (&Array1)[8], int (&Array2)[8]) { templatedArrayRef(Array2, Array1); }
// CHECK-MESSAGES: :[[@LINE-1]]:24: warning: 2 adjacent parameters of 'templatedArrayRef<8, 8>' of similar type ('int (&)[8]') are
// CHECK-MESSAGES: :[[@LINE-2]]:30: note: the first parameter in the range is 'Array1'
// CHECK-MESSAGES: :[[@LINE-3]]:48: note: the last parameter in the range is 'Array2'

template <>
void templatedArrayRef(int (&Array1)[16], int (&Array2)[24]) {}
// NO-WARN: Not the same type.

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

void numericConversion3(float F, unsigned long long ULL) { numericConversion3(ULL, F); }
// CHECK-MESSAGES: :[[@LINE-1]]:25: warning: 2 adjacent parameters of 'numericConversion3' of convertible types
// CHECK-MESSAGES: :[[@LINE-2]]:31: note: the first parameter in the range is 'F'
// CHECK-MESSAGES: :[[@LINE-3]]:53: note: the last parameter in the range is 'ULL'
// CHECK-MESSAGES: :[[@LINE-4]]:34: note: 'float' and 'unsigned long long' may be implicitly converted{{$}}

enum Unscoped { U_A,
                U_B };
enum UnscopedFixed : char { UF_A,
                            UF_B };
enum struct Scoped { A,
                     B };

void numericConversion4(int I, Unscoped U) {} // NO-WARN.

void numericConversion5(int I, UnscopedFixed UF) {} // NO-WARN.

void numericConversion6(int I, Scoped S) {} // NO-WARN.

void numericConversion7(double D, Unscoped U) {} // NO-WARN.

void numericConversion8(double D, UnscopedFixed UF) {} // NO-WARN.

void numericConversion9(double D, Scoped S) {} // NO-WARN.

void numericConversionMultiUnique(int I, double D1, double D2) {}
// CHECK-MESSAGES: :[[@LINE-1]]:35: warning: 3 adjacent parameters of 'numericConversionMultiUnique' of convertible types
// CHECK-MESSAGES: :[[@LINE-2]]:39: note: the first parameter in the range is 'I'
// CHECK-MESSAGES: :[[@LINE-3]]:60: note: the last parameter in the range is 'D2'
// CHECK-MESSAGES: :[[@LINE-4]]:42: note: 'int' and 'double' may be implicitly converted{{$}}
// (Note: int<->double conversion for I<->D2 not diagnosed again.)

typedef int MyInt;
using MyDouble = double;

void numericConversion10(MyInt MI, MyDouble MD) { numericConversion10(MD, MI); }
// CHECK-MESSAGES: :[[@LINE-1]]:26: warning: 2 adjacent parameters of 'numericConversion10' of convertible types
// CHECK-MESSAGES: :[[@LINE-2]]:32: note: the first parameter in the range is 'MI'
// CHECK-MESSAGES: :[[@LINE-3]]:45: note: the last parameter in the range is 'MD'
// CHECK-MESSAGES: :[[@LINE-4]]:36: note: 'MyInt' and 'MyDouble' may be implicitly converted: 'MyInt' (as 'int') -> 'MyDouble' (as 'double'), 'MyDouble' (as 'double') -> 'MyInt' (as 'int')

void numericAndQualifierConversion(int I, const double CD) { numericAndQualifierConversion(CD, I); }
// NO-WARN: Qualifier mixing is handled by a different check option.

struct FromInt {
  FromInt(int);
};

void oneWayConversion1(int I, FromInt FI) {} // NO-WARN: One-way.

struct AmbiguousConvCtor {
  AmbiguousConvCtor(int);
  AmbiguousConvCtor(double);
};

void ambiguous1(long L, AmbiguousConvCtor ACC) {} // NO-WARN: Ambiguous, one-way.

struct ToInt {
  operator int() const;
};

void oneWayConversion2(ToInt TI, int I) {} // NO-WARN: One-way.

struct AmbiguousConvOp {
  operator int() const;
  operator double() const;
};

void ambiguous2(AmbiguousConvOp ACO, long L) {} // NO-WARN: Ambiguous, one-way.

struct AmbiguousEverything1;
struct AmbiguousEverything2;
struct AmbiguousEverything1 {
  AmbiguousEverything1();
  AmbiguousEverything1(AmbiguousEverything2);
  operator AmbiguousEverything2() const;
};
struct AmbiguousEverything2 {
  AmbiguousEverything2();
  AmbiguousEverything2(AmbiguousEverything1);
  operator AmbiguousEverything1() const;
};

void ambiguous3(AmbiguousEverything1 AE1, AmbiguousEverything2 AE2) {} // NO-WARN: Ambiguous.

struct Integer {
  Integer(int);
  operator int() const;
};

void userDefinedConversion1(int I1, Integer I2) { userDefinedConversion1(I2, I1); }
// CHECK-MESSAGES: :[[@LINE-1]]:29: warning: 2 adjacent parameters of 'userDefinedConversion1' of convertible types
// CHECK-MESSAGES: :[[@LINE-2]]:33: note: the first parameter in the range is 'I1'
// CHECK-MESSAGES: :[[@LINE-3]]:45: note: the last parameter in the range is 'I2'
// CHECK-MESSAGES: :[[@LINE-4]]:37: note: 'int' and 'Integer' may be implicitly converted{{$}}
// CHECK-MESSAGES: :[[@LINE-9]]:3: note: the implicit conversion involves the converting constructor declared here
// CHECK-MESSAGES: :[[@LINE-9]]:3: note: the implicit conversion involves the conversion operator declared here

struct Ambiguous {
  Ambiguous(int);
  Ambiguous(double);
  operator long() const;
  operator float() const;
};

void ambiguous3(char C, Ambiguous A) {} // NO-WARN: Ambiguous.

struct CDouble {
  CDouble(const double &);
  operator const double &() const;
};

void userDefinedConversion2(double D, CDouble CD) { userDefinedConversion2(CD, D); }
// CHECK-MESSAGES: :[[@LINE-1]]:29: warning: 2 adjacent parameters of 'userDefinedConversion2' of convertible types
// CHECK-MESSAGES: :[[@LINE-2]]:36: note: the first parameter in the range is 'D'
// CHECK-MESSAGES: :[[@LINE-3]]:47: note: the last parameter in the range is 'CD'
// CHECK-MESSAGES: :[[@LINE-4]]:39: note: 'double' and 'CDouble' may be implicitly converted: 'double' -> 'const double &' -> 'CDouble', 'CDouble' -> 'const double &' -> 'double'
// CHECK-MESSAGES: :[[@LINE-9]]:3: note: the implicit conversion involves the converting constructor declared here
// CHECK-MESSAGES: :[[@LINE-9]]:3: note: the implicit conversion involves the conversion operator declared here

void userDefinedConversion3(int I, CDouble CD) { userDefinedConversion3(CD, I); }
// CHECK-MESSAGES: :[[@LINE-1]]:29: warning: 2 adjacent parameters of 'userDefinedConversion3' of convertible types
// CHECK-MESSAGES: :[[@LINE-2]]:33: note: the first parameter in the range is 'I'
// CHECK-MESSAGES: :[[@LINE-3]]:44: note: the last parameter in the range is 'CD'
// CHECK-MESSAGES: :[[@LINE-4]]:36: note: 'int' and 'CDouble' may be implicitly converted: 'int' -> 'double' -> 'const double &' -> 'CDouble', 'CDouble' -> 'const double &' -> 'int'
// CHECK-MESSAGES: :[[@LINE-17]]:3: note: the implicit conversion involves the converting constructor declared here
// CHECK-MESSAGES: :[[@LINE-17]]:3: note: the implicit conversion involves the conversion operator declared here

struct TDInt {
  TDInt(const MyInt &);
  operator MyInt() const;
};

void userDefinedConversion4(int I, TDInt TDI) { userDefinedConversion4(TDI, I); }
// CHECK-MESSAGES: :[[@LINE-1]]:29: warning: 2 adjacent parameters of 'userDefinedConversion4' of convertible types
// CHECK-MESSAGES: :[[@LINE-2]]:33: note: the first parameter in the range is 'I'
// CHECK-MESSAGES: :[[@LINE-3]]:42: note: the last parameter in the range is 'TDI'
// CHECK-MESSAGES: :[[@LINE-4]]:36: note: 'int' and 'TDInt' may be implicitly converted: 'int' -> 'const MyInt &' -> 'TDInt', 'TDInt' -> 'MyInt' -> 'int'
// CHECK-MESSAGES: :[[@LINE-9]]:3: note: the implicit conversion involves the converting constructor declared here
// CHECK-MESSAGES: :[[@LINE-9]]:3: note: the implicit conversion involves the conversion operator declared here

struct TDIntDouble {
  TDIntDouble(const MyInt &);
  TDIntDouble(const MyDouble &);
  operator MyInt() const;
  operator MyDouble() const;
};

void userDefinedConversion5(int I, TDIntDouble TDID) { userDefinedConversion5(TDID, I); }
// CHECK-MESSAGES: :[[@LINE-1]]:29: warning: 2 adjacent parameters of 'userDefinedConversion5' of convertible types
// CHECK-MESSAGES: :[[@LINE-2]]:33: note: the first parameter in the range is 'I'
// CHECK-MESSAGES: :[[@LINE-3]]:48: note: the last parameter in the range is 'TDID'
// CHECK-MESSAGES: :[[@LINE-4]]:36: note: 'int' and 'TDIntDouble' may be implicitly converted: 'int' -> 'const MyInt &' -> 'TDIntDouble', 'TDIntDouble' -> 'MyInt' -> 'int'
// CHECK-MESSAGES: :[[@LINE-11]]:3: note: the implicit conversion involves the converting constructor declared here
// CHECK-MESSAGES: :[[@LINE-10]]:3: note: the implicit conversion involves the conversion operator declared here

void userDefinedConversion6(double D, TDIntDouble TDID) { userDefinedConversion6(TDID, D); }
// CHECK-MESSAGES: :[[@LINE-1]]:29: warning: 2 adjacent parameters of 'userDefinedConversion6' of convertible types
// CHECK-MESSAGES: :[[@LINE-2]]:36: note: the first parameter in the range is 'D'
// CHECK-MESSAGES: :[[@LINE-3]]:51: note: the last parameter in the range is 'TDID'
// CHECK-MESSAGES: :[[@LINE-4]]:39: note: 'double' and 'TDIntDouble' may be implicitly converted: 'double' -> 'const MyDouble &' -> 'TDIntDouble', 'TDIntDouble' -> 'MyDouble' -> 'double'
// CHECK-MESSAGES: :[[@LINE-18]]:3: note: the implicit conversion involves the converting constructor declared here
// CHECK-MESSAGES: :[[@LINE-17]]:3: note: the implicit conversion involves the conversion operator declared here

void userDefinedConversion7(char C, TDIntDouble TDID) {} // NO-WARN: Ambiguous.

struct Forward1;
struct Forward2;

void incomplete(Forward1 *F1, Forward2 *F2) {} // NO-WARN: Do not compare incomplete types.

void pointeeConverison(int *IP, double *DP) {} // NO-WARN.

void pointerConversion1(void *VP, int *IP) {} // NO-WARN: One-way.

struct PointerBox {
  PointerBox(void *);
  operator int *() const;
};

void pointerConversion2(PointerBox PB, int *IP) { pointerConversion2(IP, PB); }
// CHECK-MESSAGES: :[[@LINE-1]]:25: warning: 2 adjacent parameters of 'pointerConversion2' of convertible types
// CHECK-MESSAGES: :[[@LINE-2]]:36: note: the first parameter in the range is 'PB'
// CHECK-MESSAGES: :[[@LINE-3]]:45: note: the last parameter in the range is 'IP'
// CHECK-MESSAGES: :[[@LINE-4]]:40: note: 'PointerBox' and 'int *' may be implicitly converted: 'PointerBox' -> 'int *', 'int *' -> 'void *' -> 'PointerBox'
// CHECK-MESSAGES: :[[@LINE-8]]:3: note: the implicit conversion involves the conversion operator declared here
// CHECK-MESSAGES: :[[@LINE-10]]:3: note: the implicit conversion involves the converting constructor declared here

void pointerConversion3(PointerBox PB, double *DP) {} // NO-WARN: Not convertible.

struct Base {};
struct Derived : Base {};

void pointerConversion4(Base *BP, Derived *DP) {} // NO-WARN: One-way.

struct BaseAndDerivedInverter {
  BaseAndDerivedInverter(Base); // Takes a Base
  operator Derived() const;     // and becomes a Derived.
};

void pointerConversion5(BaseAndDerivedInverter BADI, Derived D) { pointerConversion5(D, BADI); }
// CHECK-MESSAGES: :[[@LINE-1]]:25: warning: 2 adjacent parameters of 'pointerConversion5' of convertible types
// CHECK-MESSAGES: :[[@LINE-2]]:48: note: the first parameter in the range is 'BADI'
// CHECK-MESSAGES: :[[@LINE-3]]:62: note: the last parameter in the range is 'D'
// CHECK-MESSAGES: :[[@LINE-4]]:54: note: 'BaseAndDerivedInverter' and 'Derived' may be implicitly converted: 'BaseAndDerivedInverter' -> 'Derived', 'Derived' -> 'Base' -> 'BaseAndDerivedInverter'
// CHECK-MESSAGES: :[[@LINE-8]]:3: note: the implicit conversion involves the conversion operator declared here
// CHECK-MESSAGES: :[[@LINE-10]]:3: note: the implicit conversion involves the converting constructor declared here

void pointerConversion6(void (*NTF)() noexcept, void (*TF)()) {}
// NO-WARN: This call cannot be swapped, even if "getCanonicalType()" believes otherwise.

using NonThrowingFunction = void (*)() noexcept;

struct NoexceptMaker {
  NoexceptMaker(void (*ThrowingFunction)());
  // Need to use a typedef here because
  // "conversion function cannot convert to a function type".
  // operator (void (*)() noexcept) () const;
  operator NonThrowingFunction() const;
};

void pointerConversion7(void (*NTF)() noexcept, NoexceptMaker NM) { pointerConversion7(NM, NTF); }
// CHECK-MESSAGES: :[[@LINE-1]]:25: warning: 2 adjacent parameters of 'pointerConversion7' of convertible types
// CHECK-MESSAGES: :[[@LINE-2]]:32: note: the first parameter in the range is 'NTF'
// CHECK-MESSAGES: :[[@LINE-3]]:63: note: the last parameter in the range is 'NM'
// CHECK-MESSAGES: :[[@LINE-4]]:49: note: 'void (*)() noexcept' and 'NoexceptMaker' may be implicitly converted: 'void (*)() noexcept' -> 'void (*)()' -> 'NoexceptMaker', 'NoexceptMaker' -> 'NonThrowingFunction' -> 'void (*)() noexcept'
// CHECK-MESSAGES: :[[@LINE-12]]:3: note: the implicit conversion involves the converting constructor declared here
// CHECK-MESSAGES: :[[@LINE-9]]:3: note: the implicit conversion involves the conversion operator declared here

struct ToType;
struct MiddleStep1 {
  operator ToType() const;
};
struct FromType {
  operator MiddleStep1() const;
};
struct MiddleStep2 {
  operator FromType() const;
};
struct ToType {
  operator MiddleStep2() const;
};

void f(FromType F, ToType T) { // NO-WARN: The path takes two steps.
  MiddleStep2 MS2 = T;
  FromType F2 = MS2;

  MiddleStep1 MS1 = F;
  ToType T2 = MS1;

  f(F2, T2);
}

// Synthesised example from OpenCV.
template <typename T>
struct TemplateConversion {
  template <typename T2>
  operator TemplateConversion<T2>() const;
};
using IntConverter = TemplateConversion<int>;
using FloatConverter = TemplateConversion<float>;

void templateConversion(IntConverter IC, FloatConverter FC) { templateConversion(FC, IC); }
// Note: even though this swap is possible, we do not model things when it comes to "template magic".
// But at least the check should not crash!
