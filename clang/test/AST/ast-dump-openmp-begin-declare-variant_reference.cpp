// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify -emit-llvm -o - %s | FileCheck %s
// expected-no-diagnostics

// Our very own std::move, copied from libcxx.
template <class _Tp> struct remove_reference { typedef _Tp type; };
template <class _Tp> struct remove_reference<_Tp &> { typedef _Tp type; };
template <class _Tp> struct remove_reference<_Tp &&> { typedef _Tp type; };

template <class _Tp>
inline typename remove_reference<_Tp>::type &&
move(_Tp &&__t) {
  typedef typename remove_reference<_Tp>::type _Up;
  return static_cast<_Up &&>(__t);
}
// ---

int Good, Bad;
int &also_before() {
  return Bad;
}
int also_before(float &&) {
  return 0;
}

#pragma omp begin declare variant match(implementation = {vendor(score(100) \
                                                                 : llvm)})
int also_after(void) {
  return 1;
}
int also_after(int &) {
  return 2;
}
// This one does overload the int(*)(double&) version!
int also_after(double &) {
  return 0;
}
int also_after(double &&) {
  return 3;
}
int also_after(short &) {
  return 5;
}
int also_after(short &&) {
  return 0;
}
#pragma omp end declare variant
#pragma omp begin declare variant match(implementation = {vendor(score(0) \
                                                                 : llvm)})
// This one does overload the int&(*)(void) version!
int &also_before() {
  return Good;
}
// This one does *not* overload the int(*)(float&&) version!
int also_before(float &) {
  return 6;
}
#pragma omp end declare variant

int also_after(void) {
  return 7;
}
int also_after(int) {
  return 8;
}
int also_after(double &) {
  return 9;
}
int also_after(short &&) {
  return 10;
}

int test1() {
  // Should return 0.
  double d;
  return also_after(d);
}

int test2() {
  // Should return 0.
  return &also_before() == &Good;
}

int test3(float &&f) {
  // Should return 0.
  return also_before(move(f));
}

int test4(short &&s) {
  // Should return 0.
  return also_after(move(s));
}

int test(float &&f, short &&s) {
  // Should return 0.
  return test1() + test2() + test3(move(f)) + test4(move(s));
}

// CHECK-LABEL: define {{.*}} @_Z5test1v
// CHECK:         call {{.*}} @"_Z34also_after$ompvariant$S4$s11$PllvmRd"

// CHECK-LABEL: define {{.*}} @_Z5test2v
// CHECK:         call {{.*}} @"_Z35also_before$ompvariant$S4$s11$Pllvmv"

// CHECK-LABEL: define {{.*}} @_Z5test3Of
// CHECK:         call {{.*}} @_Z11also_beforeOf

// CHECK-LABEL: define {{.*}} @_Z5test4Os
// CHECK:         call {{.*}} @"_Z34also_after$ompvariant$S4$s11$PllvmOs"
