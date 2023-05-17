// RUN: not %clang_cc1 -fdiagnostics-parseable-fixits -x c++ -std=c++14 %s 2>&1 | FileCheck %s

const int a; // expected-error {{default initialization of an object of const type}}
// CHECK: fix-it:"{{.*}}":{3:12-3:12}:" = 0"

template <class, class> const int b; // expected-error {{default initialization of an object of const type}}
// CHECK: fix-it:"{{.*}}":{6:36-6:36}:" = 0"

template <class T> const int b<int, T>; // expected-error {{default initialization of an object of const type}}
// CHECK: fix-it:"{{.*}}":{9:39-9:39}:" = 0"

template <> const int b<int, float>; // expected-error {{default initialization of an object of const type}}
// CHECK: fix-it:"{{.*}}":{12:36-12:36}:" = 0"

constexpr float c; // expected-error {{must be initialized by a constant expression}}
// CHECK: fix-it:"{{.*}}":{15:18-15:18}:" = 0.0"

template <class, class> constexpr float d; // expected-error {{must be initialized by a constant expression}}
// CHECK: fix-it:"{{.*}}":{18:42-18:42}:" = 0.0"

template <class T> constexpr float d<T, int>; // expected-error {{must be initialized by a constant expression}}
// CHECK: fix-it:"{{.*}}":{21:45-21:45}:" = 0.0"

template <> constexpr float d<int, float>; // expected-error {{must be initialized by a constant expression}}
// CHECK: fix-it:"{{.*}}":{24:42-24:42}:" = 0.0"

void (* const func)(int, int); // expected-error {{default initialization of an object of const type}}
// CHECK: fix-it:"{{.*}}":{27:30-27:30}:" = nullptr"
