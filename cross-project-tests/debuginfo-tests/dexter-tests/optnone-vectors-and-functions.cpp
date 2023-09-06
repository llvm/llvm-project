// RUN: %clang++ -std=gnu++11 -O2 -g %s -o %t
// RUN: %dexter --fail-lt 1.0 -w \
// RUN:     --binary %t --debugger 'lldb' -v -- %s
// RUN: %clang++ -std=gnu++11 -O0 -g %s -o %t
// RUN: %dexter --fail-lt 1.0 -w \
// RUN:     --binary %t --debugger 'lldb' -- %s

// REQUIRES: lldb
// UNSUPPORTED: system-windows

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
  T tmp = x + y; // DexLabel('break_0')
  T tmp2 = tmp + y;
  return tmp; // DexLabel('break_1')
}
// DexLimitSteps('1', '1', from_line=ref('break_0'), to_line=ref('break_1'))
//// FIXME: gdb can print this but lldb cannot. Perhaps PR42920?
//     \DexExpectWatchValue('TypeTraits<int __attribute__((ext_vector_type(4)))>::NumElements', 4, on_line=ref('break_0'))
//     \DexExpectWatchValue('TypeTraits<int __attribute__((ext_vector_type(4)))>::UnusedField', 0xdeadbeef, on_line=ref('break_0'))
//   DexExpectWatchValue('x[0]', 1, on_line=ref('break_0'))
//   DexExpectWatchValue('x[1]', 2, on_line=ref('break_0'))
//   DexExpectWatchValue('x[2]', 3, on_line=ref('break_0'))
//   DexExpectWatchValue('x[3]', 4, on_line=ref('break_0'))
//   DexExpectWatchValue('y[0]', 5, on_line=ref('break_0'))
//   DexExpectWatchValue('y[1]', 6, on_line=ref('break_0'))
//   DexExpectWatchValue('y[2]', 7, on_line=ref('break_0'))
//   DexExpectWatchValue('y[3]', 8, on_line=ref('break_0'))
//   DexExpectWatchValue('tmp[0]', 6, on_line=ref('break_1'))
//   DexExpectWatchValue('tmp[1]', 8, on_line=ref('break_1'))
//   DexExpectWatchValue('tmp[2]', 10, on_line=ref('break_1'))
//   DexExpectWatchValue('tmp[3]', 12, on_line=ref('break_1'))
//   DexExpectWatchValue('tmp2[0]', 11, on_line=ref('break_1'))
//   DexExpectWatchValue('tmp2[1]', 14, on_line=ref('break_1'))
//   DexExpectWatchValue('tmp2[2]', 17, on_line=ref('break_1'))
//   DexExpectWatchValue('tmp2[3]', 20, on_line=ref('break_1'))

template<typename T>
__attribute__((optnone))
T test2(T x, T y) {
  T tmp = x;
  int break_2 = 0; // DexLabel('break_2')
  for (unsigned i = 0; i != TypeTraits<T>::NumElements; ++i) {
    tmp <<= 1; // DexLabel('break_3')
    tmp |= y;
  }

  tmp[0] >>= TypeTraits<T>::MysteryNumber;
  return tmp; // DexLabel('break_5')
}
// DexLimitSteps('1', '1', on_line=ref('break_2'))
//   DexExpectWatchValue('x[0]', 6, on_line=ref('break_2'))
//   DexExpectWatchValue('x[1]', 8, on_line=ref('break_2'))
//   DexExpectWatchValue('x[2]', 10, on_line=ref('break_2'))
//   DexExpectWatchValue('x[3]', 12, on_line=ref('break_2'))
//   DexExpectWatchValue('y[0]', 5, on_line=ref('break_2'))
//   DexExpectWatchValue('y[1]', 6, on_line=ref('break_2'))
//   DexExpectWatchValue('y[2]', 7, on_line=ref('break_2'))
//   DexExpectWatchValue('y[3]', 8, on_line=ref('break_2'))
//   DexExpectWatchValue('tmp[0]', 6, on_line=ref('break_2'))
//   DexExpectWatchValue('tmp[1]', 8, on_line=ref('break_2'))
//   DexExpectWatchValue('tmp[2]', 10, on_line=ref('break_2'))
//   DexExpectWatchValue('tmp[3]', 12, on_line=ref('break_2'))
// DexLimitSteps('i', 3, from_line=ref('break_3'), to_line=ref('break_5'))
//   DexExpectWatchValue('tmp[0]', 63, on_line=ref('break_3'))
//   DexExpectWatchValue('tmp[1]', 94, on_line=ref('break_3'))
//   DexExpectWatchValue('tmp[2]', 95, on_line=ref('break_3'))
//   DexExpectWatchValue('tmp[3]', 120, on_line=ref('break_3'))
//   DexExpectWatchValue('tmp[0]', 15, on_line=ref('break_5'))

template<typename T>
__attribute__((optnone))
T test3(T InVec) {
  T result;
  for (unsigned i=0; i != TypeTraits<T>::NumElements; ++i)
    result[i] = InVec[i]; // DexLabel('break_6')
  return result; // DexLabel('break_7')
}
// DexLimitSteps('i', '3', from_line=ref('break_6'), to_line=ref('break_7'))
//   DexExpectWatchValue('InVec[0]', 15, from_line=ref('break_6'), to_line=ref('break_7'))
//   DexExpectWatchValue('InVec[1]', 190, from_line=ref('break_6'), to_line=ref('break_7'))
//   DexExpectWatchValue('InVec[2]', 191, from_line=ref('break_6'), to_line=ref('break_7'))
//   DexExpectWatchValue('InVec[3]', 248, from_line=ref('break_6'), to_line=ref('break_7'))
//   DexExpectWatchValue('result[0]', 15, from_line=ref('break_6'), to_line=ref('break_7'))
//   DexExpectWatchValue('result[1]', 190, from_line=ref('break_6'), to_line=ref('break_7'))
//   DexExpectWatchValue('result[2]', 191, from_line=ref('break_6'), to_line=ref('break_7'))
//   DexExpectWatchValue('result[3]', 248, on_line=ref('break_7'))

template<typename T>
__attribute__((optnone))
T test4(T x, T y) {
  for (unsigned i=0; i != TypeTraits<T>::NumElements; ++i)
    x[i] = (x[i] > y[i])? x[i] : y[i] + TypeTraits<T>::MysteryNumber; // DexLabel('break_11')
  return x; // DexLabel('break_12')
}
// DexLimitSteps('1', '1', from_line=ref('break_11'), to_line=ref('break_12'))
//// FIXME: lldb won't print this but gdb unexpectedly says it's optimized out, even at O0.
//     \DexExpectWatchValue('TypeTraits<int __attribute__((ext_vector_type(4)))>::MysteryNumber', 3, on_line=ref('break_11'))
//   DexExpectWatchValue('i', 0, 1, 2, 3, on_line=ref('break_11'))
//   DexExpectWatchValue('x[0]', 1, 8, from_line=ref('break_11'), to_line=ref('break_12'))
//   DexExpectWatchValue('x[1]', 2, 9, from_line=ref('break_11'), to_line=ref('break_12'))
//   DexExpectWatchValue('x[2]', 3, 10, from_line=ref('break_11'), to_line=ref('break_12'))
//   DexExpectWatchValue('x[3]', 4, 11, from_line=ref('break_11'), to_line=ref('break_12'))
//   DexExpectWatchValue('y[0]', 5, from_line=ref('break_11'), to_line=ref('break_12'))
//   DexExpectWatchValue('y[1]', 6, from_line=ref('break_11'), to_line=ref('break_12'))
//   DexExpectWatchValue('y[2]', 7, from_line=ref('break_11'), to_line=ref('break_12'))
//   DexExpectWatchValue('y[3]', 8, from_line=ref('break_11'), to_line=ref('break_12'))

int main() {
  int4 a = (int4){1,2,3,4};
  int4 b = (int4){5,6,7,8};

  int4 tmp = test1(a,b);
  tmp = test2(tmp,b);
  tmp = test3(tmp);
  tmp += test4(a,b);
  return tmp[0];
}
