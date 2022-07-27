// RUN: %check_clang_tidy %s bugprone-easily-swappable-parameters %t \
// RUN:   -config='{CheckOptions: [ \
// RUN:     {key: bugprone-easily-swappable-parameters.MinimumLength, value: 2}, \
// RUN:     {key: bugprone-easily-swappable-parameters.IgnoredParameterNames, value: ""}, \
// RUN:     {key: bugprone-easily-swappable-parameters.IgnoredParameterTypeSuffixes, value: ""}, \
// RUN:     {key: bugprone-easily-swappable-parameters.QualifiersMix, value: 0}, \
// RUN:     {key: bugprone-easily-swappable-parameters.ModelImplicitConversions, value: 0}, \
// RUN:     {key: bugprone-easily-swappable-parameters.SuppressParametersUsedTogether, value: 0}, \
// RUN:     {key: bugprone-easily-swappable-parameters.NamePrefixSuffixSilenceDissimilarityTreshold, value: 0} \
// RUN:  ]}' --

namespace std {
using size_t = decltype(sizeof(int));
} // namespace std

#define assert(X) ((void)(X))

void declaration(int Param, int Other); // NO-WARN: No chance to change this function.

struct S {};

S *allocate() { return nullptr; }                           // NO-WARN: 0 parameters.
void allocate(S **Out) {}                                   // NO-WARN: 1 parameter.
bool operator<(const S &LHS, const S &RHS) { return true; } // NO-WARN: Binary operator.

struct MyComparator {
  bool operator()(const S &LHS, const S &RHS) { return true; } // NO-WARN: Binary operator.
};

struct MyFactory {
  S operator()() { return {}; }             // NO-WARN: 0 parameters, overloaded operator.
  S operator()(int I) { return {}; }        // NO-WARN: 1 parameter, overloaded operator.
  S operator()(int I, int J) { return {}; } // NO-WARN: Binary operator.

  S operator()(int I, int J, int K) { return {}; }
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: 3 adjacent parameters of 'operator()' of similar type ('int') are easily swapped by mistake [bugprone-easily-swappable-parameters]
  // CHECK-MESSAGES: :[[@LINE-2]]:20: note: the first parameter in the range is 'I'
  // CHECK-MESSAGES: :[[@LINE-3]]:34: note: the last parameter in the range is 'K'
};

// Variadic functions are not checked because the types are not seen from the
// *definition*. It would require analysing the call sites to do something
// for these.
int printf(const char *Format, ...) { return 0; } // NO-WARN: Variadic function not checked.
int sum(...) { return 0; }                        // NO-WARN: Variadic function not checked.

void *operator new(std::size_t Count, S &Manager, S &Janitor) noexcept { return nullptr; }
// CHECK-MESSAGES: :[[@LINE-1]]:39: warning: 2 adjacent parameters of 'operator new' of similar type ('S &')
// CHECK-MESSAGES: :[[@LINE-2]]:42: note: the first parameter in the range is 'Manager'
// CHECK-MESSAGES: :[[@LINE-3]]:54: note: the last parameter in the range is 'Janitor'

void redeclChain(int, int, int);
void redeclChain(int I, int, int);
void redeclChain(int, int J, int);
void redeclChain(int I, int J, int K) {}
// CHECK-MESSAGES: :[[@LINE-1]]:18: warning: 3 adjacent parameters of 'redeclChain' of similar type ('int')
// CHECK-MESSAGES: :[[@LINE-2]]:22: note: the first parameter in the range is 'I'
// CHECK-MESSAGES: :[[@LINE-3]]:36: note: the last parameter in the range is 'K'

void copyMany(S *Src, S *Dst, unsigned Num) {}
// CHECK-MESSAGES: :[[@LINE-1]]:15: warning: 2 adjacent parameters of 'copyMany' of similar type ('S *')
// CHECK-MESSAGES: :[[@LINE-2]]:18: note: the first parameter in the range is 'Src'
// CHECK-MESSAGES: :[[@LINE-3]]:26: note: the last parameter in the range is 'Dst'

template <typename T, typename U>
bool binaryPredicate(T L, U R) { return false; } // NO-WARN: Distinct types in template.

template <> // Explicit specialisation.
bool binaryPredicate(S *L, S *R) { return true; }
// CHECK-MESSAGES: :[[@LINE-1]]:22: warning: 2 adjacent parameters of 'binaryPredicate<S *, S *>' of similar type ('S *')
// CHECK-MESSAGES: :[[@LINE-2]]:25: note: the first parameter in the range is 'L'
// CHECK-MESSAGES: :[[@LINE-3]]:31: note: the last parameter in the range is 'R'

template <typename T>
T algebraicOperation(T L, T R) { return L; }
// CHECK-MESSAGES: :[[@LINE-1]]:22: warning: 2 adjacent parameters of 'algebraicOperation' of similar type ('T')
// CHECK-MESSAGES: :[[@LINE-2]]:24: note: the first parameter in the range is 'L'
// CHECK-MESSAGES: :[[@LINE-3]]:29: note: the last parameter in the range is 'R'

void applyBinaryToS(S SInstance) { // NO-WARN: 1 parameter.
  assert(binaryPredicate(SInstance, SInstance) !=
         binaryPredicate(&SInstance, &SInstance));
  // NO-WARN: binaryPredicate(S, S) is instantiated, but it's not written
  // by the user.
}

void unnamedParameter(int I, int, int K, int) {}
// CHECK-MESSAGES: :[[@LINE-1]]:23: warning: 4 adjacent parameters of 'unnamedParameter' of similar type ('int')
// CHECK-MESSAGES: :[[@LINE-2]]:27: note: the first parameter in the range is 'I'
// CHECK-MESSAGES: :[[@LINE-3]]:45: note: the last parameter in the range is '<unnamed>'

void fullyUnnamed(int, int) {}
// CHECK-MESSAGES: :[[@LINE-1]]:19: warning: 2 adjacent parameters of 'fullyUnnamed' of similar type ('int')
// CHECK-MESSAGES: :[[@LINE-2]]:22: note: the first parameter in the range is '<unnamed>'
// CHECK-MESSAGES: :[[@LINE-3]]:27: note: the last parameter in the range is '<unnamed>'

void multipleDistinctTypes(int I, int J, long L, long M) {}
// CHECK-MESSAGES: :[[@LINE-1]]:28: warning: 2 adjacent parameters of 'multipleDistinctTypes' of similar type ('int')
// CHECK-MESSAGES: :[[@LINE-2]]:32: note: the first parameter in the range is 'I'
// CHECK-MESSAGES: :[[@LINE-3]]:39: note: the last parameter in the range is 'J'
// CHECK-MESSAGES: :[[@LINE-4]]:42: warning: 2 adjacent parameters of 'multipleDistinctTypes' of similar type ('long')
// CHECK-MESSAGES: :[[@LINE-5]]:47: note: the first parameter in the range is 'L'
// CHECK-MESSAGES: :[[@LINE-6]]:55: note: the last parameter in the range is 'M'

void variableAndPtr(int I, int *IP) {} // NO-WARN: Not the same type.

void differentPtrs(int *IP, long *LP) {} // NO-WARN: Not the same type.

typedef int MyInt1;
using MyInt2 = int;
typedef MyInt2 MyInt2b;

using CInt = const int;
using CMyInt1 = const MyInt1;
using CMyInt2 = const MyInt2;

typedef long MyLong1;
using MyLong2 = long;

void typedefAndTypedef1(MyInt1 I1, MyInt1 I2) {}
// CHECK-MESSAGES: :[[@LINE-1]]:25: warning: 2 adjacent parameters of 'typedefAndTypedef1' of similar type ('MyInt1')
// CHECK-MESSAGES: :[[@LINE-2]]:32: note: the first parameter in the range is 'I1'
// CHECK-MESSAGES: :[[@LINE-3]]:43: note: the last parameter in the range is 'I2'

void typedefAndTypedef2(MyInt2 I1, MyInt2 I2) {}
// CHECK-MESSAGES: :[[@LINE-1]]:25: warning: 2 adjacent parameters of 'typedefAndTypedef2' of similar type ('MyInt2')
// CHECK-MESSAGES: :[[@LINE-2]]:32: note: the first parameter in the range is 'I1'
// CHECK-MESSAGES: :[[@LINE-3]]:43: note: the last parameter in the range is 'I2'

void typedefMultiple(MyInt1 I1, MyInt2 I2x, MyInt2 I2y) {}
// CHECK-MESSAGES: :[[@LINE-1]]:22: warning: 3 adjacent parameters of 'typedefMultiple' of similar type are
// CHECK-MESSAGES: :[[@LINE-2]]:29: note: the first parameter in the range is 'I1'
// CHECK-MESSAGES: :[[@LINE-3]]:52: note: the last parameter in the range is 'I2y'
// CHECK-MESSAGES: :[[@LINE-4]]:22: note: after resolving type aliases, the common type of 'MyInt1' and 'MyInt2' is 'int'

void throughTypedef1(int I, MyInt1 J) {}
// CHECK-MESSAGES: :[[@LINE-1]]:22: warning: 2 adjacent parameters of 'throughTypedef1' of similar type are
// CHECK-MESSAGES: :[[@LINE-2]]:26: note: the first parameter in the range is 'I'
// CHECK-MESSAGES: :[[@LINE-3]]:36: note: the last parameter in the range is 'J'
// CHECK-MESSAGES: :[[@LINE-4]]:22: note: after resolving type aliases, 'int' and 'MyInt1' are the same

void betweenTypedef2(MyInt1 I, MyInt2 J) {}
// CHECK-MESSAGES: :[[@LINE-1]]:22: warning: 2 adjacent parameters of 'betweenTypedef2' of similar type are
// CHECK-MESSAGES: :[[@LINE-2]]:29: note: the first parameter in the range is 'I'
// CHECK-MESSAGES: :[[@LINE-3]]:39: note: the last parameter in the range is 'J'
// CHECK-MESSAGES: :[[@LINE-4]]:22: note: after resolving type aliases, the common type of 'MyInt1' and 'MyInt2' is 'int'

void typedefChain(int I, MyInt1 MI1, MyInt2 MI2, MyInt2b MI2b) {}
// CHECK-MESSAGES: :[[@LINE-1]]:19: warning: 4 adjacent parameters of 'typedefChain' of similar type are
// CHECK-MESSAGES: :[[@LINE-2]]:23: note: the first parameter in the range is 'I'
// CHECK-MESSAGES: :[[@LINE-3]]:58: note: the last parameter in the range is 'MI2b'
// CHECK-MESSAGES: :[[@LINE-4]]:19: note: after resolving type aliases, 'int' and 'MyInt1' are the same
// CHECK-MESSAGES: :[[@LINE-5]]:19: note: after resolving type aliases, 'int' and 'MyInt2' are the same
// CHECK-MESSAGES: :[[@LINE-6]]:19: note: after resolving type aliases, 'int' and 'MyInt2b' are the same

void throughTypedefToOtherType(MyInt1 I, MyLong1 J) {} // NO-WARN: int and long.

void qualified1(int I, const int CI) {} // NO-WARN: Different qualifiers.

void qualified2(int I, volatile int VI) {} // NO-WARN: Different qualifiers.

void qualified3(int *IP, const int *CIP) {} // NO-WARN: Different qualifiers.

void qualified4(const int CI, const long CL) {} // NO-WARN: Not the same type.

void qualifiedPtr1(int *IP, int *const IPC) {} // NO-WARN: Different qualifiers.

void qualifiedTypeAndQualifiedPtr1(const int *CIP, int *const volatile IPCV) {} // NO-WARN: Not the same type.

void qualifiedThroughTypedef1(int I, CInt CI) {} // NO-WARN: Different qualifiers.

void qualifiedThroughTypedef2(CInt CI1, const int CI2) {}
// CHECK-MESSAGES: :[[@LINE-1]]:31: warning: 2 adjacent parameters of 'qualifiedThroughTypedef2' of similar type are
// CHECK-MESSAGES: :[[@LINE-2]]:36: note: the first parameter in the range is 'CI1'
// CHECK-MESSAGES: :[[@LINE-3]]:51: note: the last parameter in the range is 'CI2'
// CHECK-MESSAGES: :[[@LINE-4]]:31: note: after resolving type aliases, 'CInt' and 'const int' are the same

void qualifiedThroughTypedef3(CInt CI1, const MyInt1 CI2, const int CI3) {}
// CHECK-MESSAGES: :[[@LINE-1]]:31: warning: 3 adjacent parameters of 'qualifiedThroughTypedef3' of similar type are
// CHECK-MESSAGES: :[[@LINE-2]]:36: note: the first parameter in the range is 'CI1'
// CHECK-MESSAGES: :[[@LINE-3]]:69: note: the last parameter in the range is 'CI3'
// CHECK-MESSAGES: :[[@LINE-4]]:31: note: after resolving type aliases, the common type of 'CInt' and 'const MyInt1' is 'const int'
// CHECK-MESSAGES: :[[@LINE-5]]:31: note: after resolving type aliases, 'CInt' and 'const int' are the same
// CHECK-MESSAGES: :[[@LINE-6]]:41: note: after resolving type aliases, 'const MyInt1' and 'const int' are the same

void qualifiedThroughTypedef4(CInt CI1, const MyInt1 CI2, const MyInt2 CI3) {}
// CHECK-MESSAGES: :[[@LINE-1]]:31: warning: 3 adjacent parameters of 'qualifiedThroughTypedef4' of similar type are
// CHECK-MESSAGES: :[[@LINE-2]]:36: note: the first parameter in the range is 'CI1'
// CHECK-MESSAGES: :[[@LINE-3]]:72: note: the last parameter in the range is 'CI3'
// CHECK-MESSAGES: :[[@LINE-4]]:31: note: after resolving type aliases, the common type of 'CInt' and 'const MyInt1' is 'const int'
// CHECK-MESSAGES: :[[@LINE-5]]:31: note: after resolving type aliases, the common type of 'CInt' and 'const MyInt2' is 'const int'
// CHECK-MESSAGES: :[[@LINE-6]]:41: note: after resolving type aliases, the common type of 'const MyInt1' and 'const MyInt2' is 'const int'

void qualifiedThroughTypedef5(CMyInt1 CMI1, CMyInt2 CMI2) {}
// CHECK-MESSAGES: :[[@LINE-1]]:31: warning: 2 adjacent parameters of 'qualifiedThroughTypedef5' of similar type are
// CHECK-MESSAGES: :[[@LINE-2]]:39: note: the first parameter in the range is 'CMI1'
// CHECK-MESSAGES: :[[@LINE-3]]:53: note: the last parameter in the range is 'CMI2'
// CHECK-MESSAGES: :[[@LINE-4]]:31: note: after resolving type aliases, the common type of 'CMyInt1' and 'CMyInt2' is 'const int'

void qualifiedThroughTypedef6(CMyInt1 CMI1, int I) {} // NO-WARN: Different qualifiers.

template <typename T>
void copy(const T *Dest, T *Source) {} // NO-WARN: Different qualifiers.

void reference1(int I, int &IR) {} // NO-WARN: Distinct semantics when called.

void reference2(int I, const int &CIR) {}
// CHECK-MESSAGES: :[[@LINE-1]]:17: warning: 2 adjacent parameters of 'reference2' of similar type are
// CHECK-MESSAGES: :[[@LINE-2]]:21: note: the first parameter in the range is 'I'
// CHECK-MESSAGES: :[[@LINE-3]]:35: note: the last parameter in the range is 'CIR'
// CHECK-MESSAGES: :[[@LINE-4]]:24: note: 'int' and 'const int &' parameters accept and bind the same kind of values

void reference3(int I, int &&IRR) {} // NO-WARN: Distinct semantics when called.

void reference4(int I, const int &&CIRR) {} // NO-WARN: Distinct semantics when called.

void reference5(const int CI, const int &CIR) {}
// CHECK-MESSAGES: :[[@LINE-1]]:17: warning: 2 adjacent parameters of 'reference5' of similar type are
// CHECK-MESSAGES: :[[@LINE-2]]:27: note: the first parameter in the range is 'CI'
// CHECK-MESSAGES: :[[@LINE-3]]:42: note: the last parameter in the range is 'CIR'
// CHECK-MESSAGES: :[[@LINE-4]]:31: note: 'const int' and 'const int &' parameters accept and bind the same kind of values

void reference6(int I, const int &CIR, int J, const int &CJR) {}
// CHECK-MESSAGES: :[[@LINE-1]]:17: warning: 4 adjacent parameters of 'reference6' of similar type are
// CHECK-MESSAGES: :[[@LINE-2]]:21: note: the first parameter in the range is 'I'
// CHECK-MESSAGES: :[[@LINE-3]]:58: note: the last parameter in the range is 'CJR'
// CHECK-MESSAGES: :[[@LINE-4]]:24: note: 'int' and 'const int &' parameters accept and bind the same kind of values

using ICRTy = const int &;
using MyIntCRTy = const MyInt1 &;

void referenceToTypedef1(CInt &CIR, int I) {}
// CHECK-MESSAGES: :[[@LINE-1]]:26: warning: 2 adjacent parameters of 'referenceToTypedef1' of similar type are
// CHECK-MESSAGES: :[[@LINE-2]]:32: note: the first parameter in the range is 'CIR'
// CHECK-MESSAGES: :[[@LINE-3]]:41: note: the last parameter in the range is 'I'
// CHECK-MESSAGES: :[[@LINE-4]]:37: note: 'CInt &' and 'int' parameters accept and bind the same kind of values

void referenceThroughTypedef(int I, ICRTy Builtin, MyIntCRTy MyInt) {}
// CHECK-MESSAGES: :[[@LINE-1]]:30: warning: 3 adjacent parameters of 'referenceThroughTypedef' of similar type are
// CHECK-MESSAGES: :[[@LINE-2]]:34: note: the first parameter in the range is 'I'
// CHECK-MESSAGES: :[[@LINE-3]]:62: note: the last parameter in the range is 'MyInt'
// CHECK-MESSAGES: :[[@LINE-4]]:37: note: 'int' and 'ICRTy' parameters accept and bind the same kind of values
// CHECK-MESSAGES: :[[@LINE-5]]:30: note: after resolving type aliases, 'int' and 'MyIntCRTy' are the same
// CHECK-MESSAGES: :[[@LINE-6]]:52: note: 'int' and 'MyIntCRTy' parameters accept and bind the same kind of values
// CHECK-MESSAGES: :[[@LINE-7]]:37: note: after resolving type aliases, the common type of 'ICRTy' and 'MyIntCRTy' is 'const int &'

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
// NO-WARN: Distinct types and no conversion because of &.

short const typedef int unsigned Eldritch;
typedef const unsigned short Holy;

void collapse(Eldritch Cursed, Holy Blessed) {}
// CHECK-MESSAGES: :[[@LINE-1]]:15: warning: 2 adjacent parameters of 'collapse' of similar type are
// CHECK-MESSAGES: :[[@LINE-2]]:24: note: the first parameter in the range is 'Cursed'
// CHECK-MESSAGES: :[[@LINE-3]]:37: note: the last parameter in the range is 'Blessed'
// CHECK-MESSAGES: :[[@LINE-4]]:15: note: after resolving type aliases, the common type of 'Eldritch' and 'Holy' is 'const unsigned short'

void collapseAndTypedef(Eldritch Cursed, const Holy &Blessed) {}
// CHECK-MESSAGES: :[[@LINE-1]]:25: warning: 2 adjacent parameters of 'collapseAndTypedef' of similar type are
// CHECK-MESSAGES: :[[@LINE-2]]:34: note: the first parameter in the range is 'Cursed'
// CHECK-MESSAGES: :[[@LINE-3]]:54: note: the last parameter in the range is 'Blessed'
// CHECK-MESSAGES: :[[@LINE-4]]:25: note: after resolving type aliases, the common type of 'Eldritch' and 'const Holy &' is 'const unsigned short'
// CHECK-MESSAGES: :[[@LINE-5]]:42: note: 'Eldritch' and 'const Holy &' parameters accept and bind the same kind of values

template <typename T1, typename T2>
struct Pair {};

void templateParam1(Pair<int, int> P1, Pair<int, int> P2) {}
// CHECK-MESSAGES: :[[@LINE-1]]:21: warning: 2 adjacent parameters of 'templateParam1' of similar type ('Pair<int, int>')
// CHECK-MESSAGES: :[[@LINE-2]]:36: note: the first parameter in the range is 'P1'
// CHECK-MESSAGES: :[[@LINE-3]]:55: note: the last parameter in the range is 'P2'

void templateParam2(Pair<int, long> P1, Pair<int, long> P2) {}
// CHECK-MESSAGES: :[[@LINE-1]]:21: warning: 2 adjacent parameters of 'templateParam2' of similar type ('Pair<int, long>')
// CHECK-MESSAGES: :[[@LINE-2]]:37: note: the first parameter in the range is 'P1'
// CHECK-MESSAGES: :[[@LINE-3]]:57: note: the last parameter in the range is 'P2'

void templateParam3(Pair<int, int> P1, Pair<int, long> P2) {} // NO-WARN: Not the same type.

template <typename X, typename Y>
struct Coord {};

void templateAndOtherTemplate1(Pair<int, int> P, Coord<int, int> C) {} // NO-WARN: Not the same type.

template <typename Ts>
void templateVariadic1(Ts TVars...) {} // NO-WARN: Requires instantiation to check.

template <typename T, typename... Us>
void templateVariadic2(T TVar, Us... UVars) {} // NO-WARN: Distinct types in primary template.

template <>
void templateVariadic2(int TVar, int UVars1, int UVars2) {}
// CHECK-MESSAGES: :[[@LINE-1]]:24: warning: 3 adjacent parameters of 'templateVariadic2<int, int, int>' of similar type ('int')
// CHECK-MESSAGES: :[[@LINE-2]]:28: note: the first parameter in the range is 'TVar'
// CHECK-MESSAGES: :[[@LINE-3]]:50: note: the last parameter in the range is 'UVars2'

template <typename T>
using TwoOf = Pair<T, T>;

void templateAndAliasTemplate(Pair<int, int> P, TwoOf<int> I) {}
// CHECK-MESSAGES: :[[@LINE-1]]:31: warning: 2 adjacent parameters of 'templateAndAliasTemplate' of similar type ('Pair<int, int>')
// CHECK-MESSAGES: :[[@LINE-2]]:46: note: the first parameter in the range is 'P'
// CHECK-MESSAGES: :[[@LINE-3]]:60: note: the last parameter in the range is 'I'

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

template <typename T>
struct Vector {
  typedef T element_type;
  typedef T &reference_type;
  typedef const T const_element_type;
  typedef const T &const_reference_type;
};

void memberTypedef(int I, Vector<int>::element_type E) {}
// CHECK-MESSAGES: :[[@LINE-1]]:20: warning: 2 adjacent parameters of 'memberTypedef' of similar type are
// CHECK-MESSAGES: :[[@LINE-2]]:24: note: the first parameter in the range is 'I'
// CHECK-MESSAGES: :[[@LINE-3]]:53: note: the last parameter in the range is 'E'
// CHECK-MESSAGES: :[[@LINE-4]]:20: note: after resolving type aliases, 'int' and 'Vector<int>::element_type' are the same

template <typename T>
void memberTypedefDependent1(T T1, typename Vector<T>::element_type T2) {} // NO-WARN: Dependent name is not instantiated and resolved against other type.

template <typename T>
void memberTypedefDependent2(typename Vector<T>::element_type E1,
                             typename Vector<T>::element_type E2) {}
// CHECK-MESSAGES: :[[@LINE-2]]:30: warning: 2 adjacent parameters of 'memberTypedefDependent2' of similar type ('typename Vector<T>::element_type')
// CHECK-MESSAGES: :[[@LINE-3]]:63: note: the first parameter in the range is 'E1'
// CHECK-MESSAGES: :[[@LINE-3]]:63: note: the last parameter in the range is 'E2'

template <typename T>
void memberTypedefDependentReference1(
    typename Vector<T>::element_type E,
    typename Vector<T>::const_element_type &R) {} // NO-WARN: Not instantiated.

template <typename T>
void memberTypedefDependentReference2(
    typename Vector<T>::element_type E,
    typename Vector<T>::const_reference_type R) {} // NO-WARN: Not instantiated.

template <typename T>
void memberTypedefDependentReference3(
    typename Vector<T>::element_type E,
    const typename Vector<T>::element_type &R) {}
// CHECK-MESSAGES: :[[@LINE-2]]:5: warning: 2 adjacent parameters of 'memberTypedefDependentReference3' of similar type are
// CHECK-MESSAGES: :[[@LINE-3]]:38: note: the first parameter in the range is 'E'
// CHECK-MESSAGES: :[[@LINE-3]]:45: note: the last parameter in the range is 'R'
// CHECK-MESSAGES: :[[@LINE-4]]:5: note: 'typename Vector<T>::element_type' and 'const typename Vector<T>::element_type &' parameters accept and bind the same kind of values

void functionPrototypeLosesNoexcept(void (*NonThrowing)() noexcept, void (*Throwing)()) {}
// NO-WARN: This call cannot be swapped, even if "getCanonicalType()" believes otherwise.

void attributedParam1(const __attribute__((address_space(256))) int *One,
                      const __attribute__((address_space(256))) int *Two) {}
// CHECK-MESSAGES: :[[@LINE-2]]:23: warning: 2 adjacent parameters of 'attributedParam1' of similar type ('const __attribute__((address_space(256))) int *') are
// CHECK-MESSAGES: :[[@LINE-3]]:70: note: the first parameter in the range is 'One'
// CHECK-MESSAGES: :[[@LINE-3]]:70: note: the last parameter in the range is 'Two'

void attributedParam1Typedef(const __attribute__((address_space(256))) int *One,
                             const __attribute__((address_space(256))) MyInt1 *Two) {}
// CHECK-MESSAGES: :[[@LINE-2]]:30: warning: 2 adjacent parameters of 'attributedParam1Typedef' of similar type are
// CHECK-MESSAGES: :[[@LINE-3]]:77: note: the first parameter in the range is 'One'
// CHECK-MESSAGES: :[[@LINE-3]]:80: note: the last parameter in the range is 'Two'
// CHECK-MESSAGES: :[[@LINE-5]]:30: note: after resolving type aliases, 'const __attribute__((address_space(256))) int *' and 'const __attribute__((address_space(256))) MyInt1 *' are the same

void attributedParam1TypedefRef(
    const __attribute__((address_space(256))) int &OneR,
    __attribute__((address_space(256))) MyInt1 &TwoR) {}
// NO-WARN: One is CVR-qualified, the other is not.

void attributedParam1TypedefCRef(
    const __attribute__((address_space(256))) int &OneR,
    const __attribute__((address_space(256))) MyInt1 &TwoR) {}
// CHECK-MESSAGES: :[[@LINE-2]]:5: warning: 2 adjacent parameters of 'attributedParam1TypedefCRef' of similar type are
// CHECK-MESSAGES: :[[@LINE-3]]:52: note: the first parameter in the range is 'OneR'
// CHECK-MESSAGES: :[[@LINE-3]]:55: note: the last parameter in the range is 'TwoR'
// CHECK-MESSAGES: :[[@LINE-5]]:5: note: after resolving type aliases, 'const __attribute__((address_space(256))) int &' and 'const __attribute__((address_space(256))) MyInt1 &' are the same

void attributedParam2(__attribute__((address_space(256))) int *One,
                      const __attribute__((address_space(256))) MyInt1 *Two) {}
// NO-WARN: One is CVR-qualified, the other is not.

void attributedParam3(const int *One,
                      const __attribute__((address_space(256))) MyInt1 *Two) {}
// NO-WARN: One is attributed, the other is not.

void attributedParam4(const __attribute__((address_space(512))) int *One,
                      const __attribute__((address_space(256))) MyInt1 *Two) {}
// NO-WARN: Different value of the attribute.
