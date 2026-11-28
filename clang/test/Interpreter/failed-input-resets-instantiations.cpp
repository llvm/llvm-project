// REQUIRES: host-supports-jit
// UNSUPPORTED: system-aix
// RUN: cat %s | clang-repl 2>&1 | FileCheck %s

// A failed input still triggers the implicit instantiations it uses. The
// interpreter resets them, so a later input instantiates them again without
// the declarations of the failed input.

extern "C" int printf(const char *, ...);

template <class T> T twice(T x) { return x + x; }
extern "C" int twice_fail(int *p) { return twice(*p) + no_such_name; }
// CHECK-DAG: error: use of undeclared identifier 'no_such_name'
extern "C" int twice_after(int x) { return twice(x); }
printf("twice_after = %d\n", twice_after(21));
// CHECK-DAG: twice_after = 42

template <class T> struct Holder { static T value; };
template <class T> T Holder<T>::value = T(7);
extern "C" int holder_fail() { return Holder<int>::value + no_such_name; }
// CHECK-DAG: error: use of undeclared identifier 'no_such_name'
printf("Holder<int>::value = %d\n", Holder<int>::value);
// CHECK-DAG: Holder<int>::value = 7

// The instantiation of readv<S> binds value(t) to a function of the failed
// input. A later use must not keep that binding.
template <class T> int readv(T t) { return value(t); }
struct S {};
int value(S) { return no_such_name; } extern "C" int readv_fail() { return readv(S{}); }
// CHECK-DAG: error: use of undeclared identifier 'no_such_name'
extern "C" int readv_again() { return readv(S{}); }
// CHECK-DAG: error: use of undeclared identifier 'value'
int value(const S &) { return 7; }
printf("readv = %d\n", readv(S{}));
// CHECK-DAG: readv = 7

%quit
