// RUN: %check_clang_tidy %s bugprone-easily-swappable-parameters %t \
// RUN:   -config='{CheckOptions: [ \
// RUN:     {key: bugprone-easily-swappable-parameters.MinimumLength, value: 2}, \
// RUN:     {key: bugprone-easily-swappable-parameters.IgnoredParameterNames, value: ""}, \
// RUN:     {key: bugprone-easily-swappable-parameters.IgnoredParameterTypeSuffixes, value: ""}, \
// RUN:     {key: bugprone-easily-swappable-parameters.QualifiersMix, value: 1}, \
// RUN:     {key: bugprone-easily-swappable-parameters.ModelImplicitConversions, value: 0}, \
// RUN:     {key: bugprone-easily-swappable-parameters.SuppressParametersUsedTogether, value: 0}, \
// RUN:     {key: bugprone-easily-swappable-parameters.NamePrefixSuffixSilenceDissimilarityTreshold, value: 0} \
// RUN:  ]}' --

typedef int MyInt1;
typedef int MyInt2;
using CInt = const int;
using CMyInt1 = const MyInt1;
using CMyInt2 = const MyInt2;

void qualified1(int I, const int CI) {}
// CHECK-MESSAGES: :[[@LINE-1]]:17: warning: 2 adjacent parameters of 'qualified1' of similar type are easily swapped by mistake [bugprone-easily-swappable-parameters]
// CHECK-MESSAGES: :[[@LINE-2]]:21: note: the first parameter in the range is 'I'
// CHECK-MESSAGES: :[[@LINE-3]]:34: note: the last parameter in the range is 'CI'
// CHECK-MESSAGES: :[[@LINE-4]]:24: note: 'int' and 'const int' parameters accept and bind the same kind of values

void qualified2(int I, volatile int VI) {}
// CHECK-MESSAGES: :[[@LINE-1]]:17: warning: 2 adjacent parameters of 'qualified2' of similar type are
// CHECK-MESSAGES: :[[@LINE-2]]:21: note: the first parameter in the range is 'I'
// CHECK-MESSAGES: :[[@LINE-3]]:37: note: the last parameter in the range is 'VI'
// CHECK-MESSAGES: :[[@LINE-4]]:24: note: 'int' and 'volatile int' parameters accept and bind the same kind of values

void qualified3(int I, const volatile int CVI) {}
// CHECK-MESSAGES: :[[@LINE-1]]:17: warning: 2 adjacent parameters of 'qualified3' of similar type are
// CHECK-MESSAGES: :[[@LINE-2]]:21: note: the first parameter in the range is 'I'
// CHECK-MESSAGES: :[[@LINE-3]]:43: note: the last parameter in the range is 'CVI'
// CHECK-MESSAGES: :[[@LINE-4]]:24: note: 'int' and 'const volatile int' parameters accept and bind the same kind of values

void qualified4(int *IP, const int *CIP) {}
// CHECK-MESSAGES: :[[@LINE-1]]:17: warning: 2 adjacent parameters of 'qualified4' of similar type are
// CHECK-MESSAGES: :[[@LINE-2]]:22: note: the first parameter in the range is 'IP'
// CHECK-MESSAGES: :[[@LINE-3]]:37: note: the last parameter in the range is 'CIP'
// CHECK-MESSAGES: :[[@LINE-4]]:26: note: 'int *' and 'const int *' parameters accept and bind the same kind of values

void qualified5(const int CI, const long CL) {} // NO-WARN: Not the same type

void qualifiedPtr1(int *IP, int *const IPC) {}
// CHECK-MESSAGES: :[[@LINE-1]]:20: warning: 2 adjacent parameters of 'qualifiedPtr1' of similar type are
// CHECK-MESSAGES: :[[@LINE-2]]:25: note: the first parameter in the range is 'IP'
// CHECK-MESSAGES: :[[@LINE-3]]:40: note: the last parameter in the range is 'IPC'
// CHECK-MESSAGES: :[[@LINE-4]]:29: note: 'int *' and 'int *const' parameters accept and bind the same kind of values

void qualifiedPtr2(int *IP, int *volatile IPV) {}
// CHECK-MESSAGES: :[[@LINE-1]]:20: warning: 2 adjacent parameters of 'qualifiedPtr2' of similar type are
// CHECK-MESSAGES: :[[@LINE-2]]:25: note: the first parameter in the range is 'IP'
// CHECK-MESSAGES: :[[@LINE-3]]:43: note: the last parameter in the range is 'IPV'
// CHECK-MESSAGES: :[[@LINE-4]]:29: note: 'int *' and 'int *volatile' parameters accept and bind the same kind of values

void qualifiedTypeAndQualifiedPtr1(const int *CIP, int *const volatile IPCV) {}
// CHECK-MESSAGES: :[[@LINE-1]]:36: warning: 2 adjacent parameters of 'qualifiedTypeAndQualifiedPtr1' of similar type are
// CHECK-MESSAGES: :[[@LINE-2]]:47: note: the first parameter in the range is 'CIP'
// CHECK-MESSAGES: :[[@LINE-3]]:72: note: the last parameter in the range is 'IPCV'
// CHECK-MESSAGES: :[[@LINE-4]]:52: note: 'const int *' and 'int *const volatile' parameters accept and bind the same kind of values

void qualifiedThroughTypedef1(int I, CInt CI) {}
// CHECK-MESSAGES: :[[@LINE-1]]:31: warning: 2 adjacent parameters of 'qualifiedThroughTypedef1' of similar type are
// CHECK-MESSAGES: :[[@LINE-2]]:35: note: the first parameter in the range is 'I'
// CHECK-MESSAGES: :[[@LINE-3]]:43: note: the last parameter in the range is 'CI'
// CHECK-MESSAGES: :[[@LINE-4]]:31: note: after resolving type aliases, 'int' and 'CInt' share a common type
// CHECK-MESSAGES: :[[@LINE-5]]:38: note: 'int' and 'CInt' parameters accept and bind the same kind of values

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

void qualifiedThroughTypedef6(CMyInt1 CMI1, int I) {}
// CHECK-MESSAGES: :[[@LINE-1]]:31: warning: 2 adjacent parameters of 'qualifiedThroughTypedef6' of similar type are
// CHECK-MESSAGES: :[[@LINE-2]]:39: note: the first parameter in the range is 'CMI1'
// CHECK-MESSAGES: :[[@LINE-3]]:49: note: the last parameter in the range is 'I'
// CHECK-MESSAGES: :[[@LINE-4]]:31: note: after resolving type aliases, 'CMyInt1' and 'int' share a common type
// CHECK-MESSAGES: :[[@LINE-5]]:45: note: 'CMyInt1' and 'int' parameters accept and bind the same kind of values

void referenceToTypedef1(CInt &CIR, int I) {}
// CHECK-MESSAGES: :[[@LINE-1]]:26: warning: 2 adjacent parameters of 'referenceToTypedef1' of similar type are
// CHECK-MESSAGES: :[[@LINE-2]]:32: note: the first parameter in the range is 'CIR'
// CHECK-MESSAGES: :[[@LINE-3]]:41: note: the last parameter in the range is 'I'
// CHECK-MESSAGES: :[[@LINE-4]]:37: note: 'CInt &' and 'int' parameters accept and bind the same kind of values

template <typename T>
void copy(const T *Dest, T *Source) {}
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: 2 adjacent parameters of 'copy' of similar type are
// CHECK-MESSAGES: :[[@LINE-2]]:20: note: the first parameter in the range is 'Dest'
// CHECK-MESSAGES: :[[@LINE-3]]:29: note: the last parameter in the range is 'Source'
// CHECK-MESSAGES: :[[@LINE-4]]:26: note: 'const T *' and 'T *' parameters accept and bind the same kind of values

void attributedParam1TypedefRef(
    const __attribute__((address_space(256))) int &OneR,
    __attribute__((address_space(256))) MyInt1 &TwoR) {}
// CHECK-MESSAGES: :[[@LINE-2]]:5: warning: 2 adjacent parameters of 'attributedParam1TypedefRef' of similar type are
// CHECK-MESSAGES: :[[@LINE-3]]:52: note: the first parameter in the range is 'OneR'
// CHECK-MESSAGES: :[[@LINE-3]]:49: note: the last parameter in the range is 'TwoR'
// CHECK-MESSAGES: :[[@LINE-5]]:5: note: after resolving type aliases, the common type of 'const __attribute__((address_space(256))) int &' and '__attribute__((address_space(256))) MyInt1 &' is '__attribute__((address_space(256))) int &'
// CHECK-MESSAGES: :[[@LINE-5]]:5: note: 'const __attribute__((address_space(256))) int &' and '__attribute__((address_space(256))) MyInt1 &' parameters accept and bind the same kind of values

void attributedParam2(__attribute__((address_space(256))) int *One,
                      const __attribute__((address_space(256))) MyInt1 *Two) {}
// CHECK-MESSAGES: :[[@LINE-2]]:23: warning: 2 adjacent parameters of 'attributedParam2' of similar type are
// CHECK-MESSAGES: :[[@LINE-3]]:64: note: the first parameter in the range is 'One'
// CHECK-MESSAGES: :[[@LINE-3]]:73: note: the last parameter in the range is 'Two'
// CHECK-MESSAGES: :[[@LINE-5]]:23: note: after resolving type aliases, '__attribute__((address_space(256))) int *' and 'const __attribute__((address_space(256))) MyInt1 *' share a common type
// CHECK-MESSAGES: :[[@LINE-5]]:23: note: '__attribute__((address_space(256))) int *' and 'const __attribute__((address_space(256))) MyInt1 *' parameters accept and bind the same kind of values
