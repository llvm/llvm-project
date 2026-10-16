// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -disable-llvm-passes -fcopyprof -fcopyprof-static-size-threshold=32 %s -o - | FileCheck %s

// Assert that move constructors and move assignment operators are never annotated by CopyProf.

struct S {
  long a[4];
  S();
  S(const S &);
  S(S &&);
  S &operator=(const S &);
  S &operator=(S &&);
  ~S();
};
static_assert(sizeof(S) == 32);

S::S() {}
S::S(const S &) {}
S::S(S &&) {}
S &S::operator=(const S &) { return *this; }
S &S::operator=(S &&) { return *this; }
S::~S() {}

// CHECK: define {{.*}} @_ZN1SC1Ev({{.*}}#[[CTOR:[0-9]+]]
// CHECK: define {{.*}} @_ZN1SC1ERKS_({{.*}}#[[COPY_CTOR:[0-9]+]]
// CHECK: define {{.*}} @_ZN1SC1EOS_({{.*}}#[[MOVE:[0-9]+]]
// CHECK: define {{.*}} @_ZN1SaSERKS_({{.*}}#[[COPY_ASSIGN:[0-9]+]]
// CHECK: define {{.*}} @_ZN1SaSEOS_({{.*}}#[[MOVE]]
// CHECK: define {{.*}} @_ZN1SD1Ev({{.*}}#[[DTOR:[0-9]+]]

// CHECK: attributes #[[CTOR]] = {{{.*}}"copyprof-ctor"="32"{{.*}}}
// CHECK: attributes #[[COPY_CTOR]] = {{{.*}}"copyprof-copy-ctor"="32"{{.*}}}
// The move constructor and the move assignment operator share this attribute
// group, and it must not carry any CopyProf annotation.
// CHECK: attributes #[[MOVE]] =
// CHECK-NOT: "copyprof-
// CHECK: attributes #[[COPY_ASSIGN]] = {{{.*}}"copyprof-copy-assign-op"="32"{{.*}}}
// CHECK: attributes #[[DTOR]] = {{{.*}}"copyprof-dtor"="32"{{.*}}}
