// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -disable-llvm-passes -fcopyprof -fcopyprof-static-size-threshold=16 %s -o - | FileCheck %s --check-prefix=T16
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -disable-llvm-passes -fcopyprof -fcopyprof-static-size-threshold=8 %s -o - | FileCheck %s --check-prefix=T8

// Asserts that special member functions are annotated depending on whether the object size is above the static size threshold.

struct Small {
  long a;
  Small() : a(0) {}
  Small(const Small &other) : a(other.a) {}
  Small &operator=(const Small &other) {
    a = other.a;
    return *this;
  }
  ~Small() {}
};
static_assert(sizeof(Small) == 8);

struct Large {
  long a[2];
  Large() : a{0, 0} {}
  Large(const Large &other) {
    a[0] = other.a[0];
    a[1] = other.a[1];
  }
  Large &operator=(const Large &other) {
    a[0] = other.a[0];
    a[1] = other.a[1];
    return *this;
  }
  ~Large() {}
};
static_assert(sizeof(Large) == 16);

void test() {
  Small s1;
  Small s2(s1);
  s1 = s2;

  Large l1;
  Large l2(l1);
  l1 = l2;
}

// Threshold 16: `Small` must not be annotated but `Large` must be.

// T16: define {{.*}} @_ZN5SmallC1Ev({{.*}}#[[SMALL:[0-9]+]]
// T16: define {{.*}} @_ZN5SmallC1ERKS_({{.*}}#[[SMALL]]
// T16: define {{.*}} @_ZN5SmallaSERKS_({{.*}}#[[SMALL]]
// T16: define {{.*}} @_ZN5LargeC1Ev({{.*}}#[[LARGE_CTOR:[0-9]+]]
// T16: define {{.*}} @_ZN5LargeC1ERKS_({{.*}}#[[LARGE_COPY_CTOR:[0-9]+]]
// T16: define {{.*}} @_ZN5LargeaSERKS_({{.*}}#[[LARGE_ASSIGN:[0-9]+]]
// T16: define {{.*}} @_ZN5LargeD1Ev({{.*}}#[[LARGE_DTOR:[0-9]+]]
// T16: define {{.*}} @_ZN5SmallD1Ev({{.*}}#[[SMALL]]

// T16: attributes #[[SMALL]] =
// T16-NOT: "copyprof-
// T16: attributes #[[LARGE_CTOR]] = {{{.*}}"copyprof-ctor"="16"{{.*}}}
// T16: attributes #[[LARGE_COPY_CTOR]] = {{{.*}}"copyprof-copy-ctor"="16"{{.*}}}
// T16: attributes #[[LARGE_ASSIGN]] = {{{.*}}"copyprof-copy-assign-op"="16"{{.*}}}
// T16: attributes #[[LARGE_DTOR]] = {{{.*}}"copyprof-dtor"="16"{{.*}}}

// Threshold 8: both structs are annotated.

// T8: define {{.*}} @_ZN5SmallC1Ev({{.*}}#[[SMALL_CTOR:[0-9]+]]
// T8: define {{.*}} @_ZN5SmallC1ERKS_({{.*}}#[[SMALL_COPY_CTOR:[0-9]+]]
// T8: define {{.*}} @_ZN5SmallaSERKS_({{.*}}#[[SMALL_ASSIGN:[0-9]+]]
// T8: define {{.*}} @_ZN5LargeC1Ev({{.*}}#[[LARGE_CTOR8:[0-9]+]]
// T8: define {{.*}} @_ZN5LargeC1ERKS_({{.*}}#[[LARGE_COPY_CTOR8:[0-9]+]]
// T8: define {{.*}} @_ZN5LargeaSERKS_({{.*}}#[[LARGE_ASSIGN8:[0-9]+]]
// T8: define {{.*}} @_ZN5LargeD1Ev({{.*}}#[[LARGE_DTOR8:[0-9]+]]
// T8: define {{.*}} @_ZN5SmallD1Ev({{.*}}#[[SMALL_DTOR:[0-9]+]]

// T8: attributes #[[SMALL_CTOR]] = {{{.*}}"copyprof-ctor"="8"{{.*}}}
// T8: attributes #[[SMALL_COPY_CTOR]] = {{{.*}}"copyprof-copy-ctor"="8"{{.*}}}
// T8: attributes #[[SMALL_ASSIGN]] = {{{.*}}"copyprof-copy-assign-op"="8"{{.*}}}
// T8: attributes #[[LARGE_CTOR8]] = {{{.*}}"copyprof-ctor"="16"{{.*}}}
// T8: attributes #[[LARGE_COPY_CTOR8]] = {{{.*}}"copyprof-copy-ctor"="16"{{.*}}}
// T8: attributes #[[LARGE_ASSIGN8]] = {{{.*}}"copyprof-copy-assign-op"="16"{{.*}}}
// T8: attributes #[[LARGE_DTOR8]] = {{{.*}}"copyprof-dtor"="16"{{.*}}}
// T8: attributes #[[SMALL_DTOR]] = {{{.*}}"copyprof-dtor"="8"{{.*}}}
