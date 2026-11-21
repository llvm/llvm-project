// RUN: %clang++ -std=gnu++11 -O2 -g %s -o %t
// RUN: %dexter -w \
// RUN:     --binary %t %dexter_lldb_args -v -- %s | FileCheck %s
// RUN: %clang++ -std=gnu++11 -O0 -g %s -o %t
// RUN: %dexter -w \
// RUN:     --binary %t %dexter_lldb_args -- %s | FileCheck %s

// REQUIRES: lldb
// Currently getting intermittent failures on darwin.
// UNSUPPORTED: system-windows, system-darwin

//// Check that the debugging experience with __attribute__((optnone)) at O2
//// matches O0. Test simple template functions performing simple arithmetic
//// vector operations and trivial loops.

typedef int int4 __attribute__((ext_vector_type(4)));
template<typename T> struct TypeTraits {};

template<>
struct TypeTraits<int4> {
  static const unsigned NumElements = 4;
  static const unsigned UnusedField = 0xDEADBEEFU;
  static unsigned MysteryNumber;
};
unsigned TypeTraits<int4>::MysteryNumber = 3U;

template<typename T>
__attribute__((optnone))
T test1(T x, T y) {
  T tmp = x + y; // !dex_label break_0
  T tmp2 = tmp + y;
  return tmp; // !dex_label break_1
}

template<typename T>
__attribute__((optnone))
T test2(T x, T y) {
  T tmp = x;
  int break_2 = 0; // !dex_label break_2
  for (unsigned i = 0; i != TypeTraits<T>::NumElements; ++i) {
    tmp <<= 1; // !dex_label break_3
    tmp |= y;
  }

  tmp[0] >>= TypeTraits<T>::MysteryNumber;
  return tmp; // !dex_label break_5
}

template<typename T>
__attribute__((optnone))
T test3(T InVec) {
  T result;
  for (unsigned i=0; i != TypeTraits<T>::NumElements; ++i)
    result[i] = InVec[i]; // !dex_label break_6
  return result;          // !dex_label break_7
}

template<typename T>
__attribute__((optnone))
T test4(T x, T y) {
  for (unsigned i=0; i != TypeTraits<T>::NumElements; ++i)
    x[i] = (x[i] > y[i])? x[i] : y[i] + TypeTraits<T>::MysteryNumber; // !dex_label break_11
  return x; // !dex_label break_12
}

int main() {
  int4 a = (int4){1,2,3,4};
  int4 b = (int4){5,6,7,8};

  int4 tmp = test1(a,b);
  tmp = test2(tmp,b);
  tmp = test3(tmp);
  tmp += test4(a,b);
  return tmp[0];
}

// CHECK-DAG: seen_values: 64
// CHECK-DAG: correct_step_coverage: 100.0%

/*
---
!where {lines: !label break_0}:
  ## FIXME: gdb can print this but lldb cannot. Perhaps PR42920?
  # !value 'TypeTraits<int __attribute__((ext_vector_type(4)))>::NumElements': 4
  # !value 'TypeTraits<int __attribute__((ext_vector_type(4)))>::UnusedField': 0xdeadbeef
  !value 'x[0]': 1
  !value 'x[1]': 2
  !value 'x[2]': 3
  !value 'x[3]': 4
  !value 'y[0]': 5
  !value 'y[1]': 6
  !value 'y[2]': 7
  !value 'y[3]': 8
!where {lines: !label break_1}:
  !value 'tmp[0]': 6
  !value 'tmp[1]': 8
  !value 'tmp[2]': 10
  !value 'tmp[3]': 12
  !value 'tmp2[0]': 11
  !value 'tmp2[1]': 14
  !value 'tmp2[2]': 17
  !value 'tmp2[3]': 20
!where {lines: !label break_2}:
  !value 'x[0]': 6
  !value 'x[1]': 8
  !value 'x[2]': 10
  !value 'x[3]': 12
  !value 'y[0]': 5
  !value 'y[1]': 6
  !value 'y[2]': 7
  !value 'y[3]': 8
  !value 'tmp[0]': 6
  !value 'tmp[1]': 8
  !value 'tmp[2]': 10
  !value 'tmp[3]': 12
!where {lines: !label break_3, conditions: "i == 3"}:
  !value 'tmp[0]': 63
  !value 'tmp[1]': 94
  !value 'tmp[2]': 95
  !value 'tmp[3]': 120
!where {lines: !label break_5}:
  !value 'tmp[0]': 15
!where {lines: !label break_6, conditions: "i == 3"}:
  !value 'InVec[0]': 15
  !value 'InVec[1]': 190
  !value 'InVec[2]': 191
  !value 'InVec[3]': 248
  !value 'result[0]': 15
  !value 'result[1]': 190
  !value 'result[2]': 191
!where {lines: !label break_7}:
  !value 'InVec[0]': 15
  !value 'InVec[1]': 190
  !value 'InVec[2]': 191
  !value 'InVec[3]': 248
  !value 'result[0]': 15
  !value 'result[1]': 190
  !value 'result[2]': 191
  !value 'result[3]': 248
!where {lines: !range [!label break_11, !label break_12]}:
  ## FIXME: lldb won't print this but gdb unexpectedly says it's optimized out, even at O0.
  # !value 'TypeTraits<int __attribute__((ext_vector_type(4)))>::MysteryNumber': 3
  !and {lines: !label break_11}:
    !value 'i': [0, 1, 2, 3]
  !value 'x[0]': [1, 8]
  !value 'x[1]': [2, 9]
  !value 'x[2]': [3, 10]
  !value 'x[3]': [4, 11]
  !value 'y[0]': 5
  !value 'y[1]': 6
  !value 'y[2]': 7
  !value 'y[3]': 8
...
*/
