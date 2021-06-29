// RUN: %check_clang_tidy %s bugprone-easily-swappable-parameters %t \
// RUN:   -config='{CheckOptions: [ \
// RUN:     {key: bugprone-easily-swappable-parameters.MinimumLength, value: 2}, \
// RUN:     {key: bugprone-easily-swappable-parameters.IgnoredParameterNames, value: ""}, \
// RUN:     {key: bugprone-easily-swappable-parameters.IgnoredParameterTypeSuffixes, value: ""} \
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

void typedefAndTypedef1(MyInt1 I1, MyInt1 I2) {}
// CHECK-MESSAGES: :[[@LINE-1]]:25: warning: 2 adjacent parameters of 'typedefAndTypedef1' of similar type ('MyInt1')
// CHECK-MESSAGES: :[[@LINE-2]]:32: note: the first parameter in the range is 'I1'
// CHECK-MESSAGES: :[[@LINE-3]]:43: note: the last parameter in the range is 'I2'

void typedefAndTypedef2(MyInt2 I1, MyInt2 I2) {}
// CHECK-MESSAGES: :[[@LINE-1]]:25: warning: 2 adjacent parameters of 'typedefAndTypedef2' of similar type ('MyInt2')
// CHECK-MESSAGES: :[[@LINE-2]]:32: note: the first parameter in the range is 'I1'
// CHECK-MESSAGES: :[[@LINE-3]]:43: note: the last parameter in the range is 'I2'

void throughTypedef(int I, MyInt1 J) {}
// CHECK-MESSAGES: :[[@LINE-1]]:21: warning: 2 adjacent parameters of 'throughTypedef' of similar type ('int')
// CHECK-MESSAGES: :[[@LINE-2]]:25: note: the first parameter in the range is 'I'
// CHECK-MESSAGES: :[[@LINE-3]]:35: note: the last parameter in the range is 'J'

void betweenTypedef(MyInt1 I, MyInt2 J) {}
// CHECK-MESSAGES: :[[@LINE-1]]:21: warning: 2 adjacent parameters of 'betweenTypedef' of similar type ('MyInt1')
// CHECK-MESSAGES: :[[@LINE-2]]:28: note: the first parameter in the range is 'I'
// CHECK-MESSAGES: :[[@LINE-3]]:38: note: the last parameter in the range is 'J'

typedef long MyLong1;
using MyLong2 = long;

void throughTypedefToOtherType(MyInt1 I, MyLong1 J) {} // NO-WARN: Not the same type.

void qualified1(int I, const int CI) {} // NO-WARN: Not the same type.

void qualified2(int I, volatile int VI) {} // NO-WARN: Not the same type.

void qualified3(int *IP, const int *CIP) {} // NO-WARN: Not the same type.

void qualified4(const int CI, const long CL) {} // NO-WARN: Not the same type.

using CInt = const int;

void qualifiedThroughTypedef1(int I, CInt CI) {} // NO-WARN: Not the same type.

void qualifiedThroughTypedef2(CInt CI1, const int CI2) {} // NO-WARN: Not the same type.
// CHECK-MESSAGES: :[[@LINE-1]]:31: warning: 2 adjacent parameters of 'qualifiedThroughTypedef2' of similar type ('CInt')
// CHECK-MESSAGES: :[[@LINE-2]]:36: note: the first parameter in the range is 'CI1'
// CHECK-MESSAGES: :[[@LINE-3]]:51: note: the last parameter in the range is 'CI2'

void reference1(int I, int &IR) {} // NO-WARN: Not the same type.

void reference2(int I, const int &CIR) {} // NO-WARN: Not the same type.

void reference3(int I, int &&IRR) {} // NO-WARN: Not the same type.

void reference4(int I, const int &&CIRR) {} // NO-WARN: Not the same type.

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
