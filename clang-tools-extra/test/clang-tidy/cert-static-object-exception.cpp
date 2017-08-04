// RUN: clang-tidy %s -checks="-*,cert-err58-cpp" -- -std=c++11 -target x86_64-pc-linux-gnu \
// RUN:   | FileCheck %s -check-prefix=CHECK-EXCEPTIONS \
// RUN:   -implicit-check-not="{{warning|error}}:"
// RUN: clang-tidy %s -checks="-*,cert-err58-cpp" -- -fno-exceptions -std=c++11 -target x86_64-pc-linux-gnu \
// RUN:   | FileCheck %s -allow-empty -check-prefix=CHECK-NONEXCEPTIONS \
// RUN:   -implicit-check-not="{{warning|error}}:"

struct S {
  S() noexcept(false);
};

struct T {
  T() noexcept;
};

struct U {
  U() {}
};

struct V {
  explicit V(const char *) {} // Can throw
};

struct Cleanup {
  ~Cleanup() {}
};

struct W {
  W(Cleanup c = {}) noexcept(false);
};

struct X {
  X(S = {}) noexcept;
};

struct Y {
  S s;
};

struct Z {
  T t;
};

int f();
int g() noexcept(false);
int h() noexcept(true);

struct UserConv_Bad {
  operator int() noexcept(false);
};

struct UserConv_Good {
  operator int() noexcept;
};

UserConv_Bad some_bad_func() noexcept;
UserConv_Good some_good_func() noexcept;

S s;
// CHECK-EXCEPTIONS: :[[@LINE-1]]:3: warning: initialization of 's' with static storage duration may throw an exception that cannot be caught [cert-err58-cpp]
// CHECK-EXCEPTIONS: 9:3: note: possibly throwing constructor declared here
// CHECK-NONEXCEPTIONS-NOT: warning:
T t; // ok
U u;
// CHECK-EXCEPTIONS: :[[@LINE-1]]:3: warning: initialization of 'u' with static storage duration may throw an exception that cannot be caught
// CHECK-EXCEPTIONS: 17:3: note: possibly throwing constructor declared here
// CHECK-NONEXCEPTIONS-NOT: warning:
V v("v");
// CHECK-EXCEPTIONS: :[[@LINE-1]]:3: warning: initialization of 'v' with static storage duration may throw an exception that cannot be caught
// CHECK-EXCEPTIONS: 21:12: note: possibly throwing constructor declared here
// CHECK-NONEXCEPTIONS-NOT: warning:
W w;
// CHECK-EXCEPTIONS: :[[@LINE-1]]:3: warning: initialization of 'w' with static storage duration may throw an exception that cannot be caught
// CHECK-EXCEPTIONS: 29:3: note: possibly throwing constructor declared here
// CHECK-NONEXCEPTIONS-NOT: warning:
X x1(S{});
// CHECK-EXCEPTIONS: :[[@LINE-1]]:3: warning: initialization of 'x1' with static storage duration may throw an exception that cannot be caught
// CHECK-EXCEPTIONS: 9:3: note: possibly throwing constructor declared here
// CHECK-NONEXCEPTIONS-NOT: warning:
X x2;
// CHECK-EXCEPTIONS: :[[@LINE-1]]:3: warning: initialization of 'x2' with static storage duration may throw an exception that cannot be caught
// CHECK-EXCEPTIONS: 9:3: note: possibly throwing constructor declared here
// CHECK-NONEXCEPTIONS-NOT: warning:
Y y;
// CHECK-EXCEPTIONS: :[[@LINE-1]]:3: warning: initialization of 'y' with static storage duration may throw an exception that cannot be caught
// CHECK-EXCEPTIONS: 36:8: note: possibly throwing constructor declared here
// CHECK-NONEXCEPTIONS-NOT: warning:
Z z;

int i = f();
// CHECK-EXCEPTIONS: :[[@LINE-1]]:5: warning: initialization of 'i' with static storage duration may throw an exception that cannot be caught
// CHECK-EXCEPTIONS: 44:5: note: possibly throwing function declared here
// CHECK-NONEXCEPTIONS-NOT: warning:
int j = g();
// CHECK-EXCEPTIONS: :[[@LINE-1]]:5: warning: initialization of 'j' with static storage duration may throw an exception that cannot be caught
// CHECK-EXCEPTIONS: 45:5: note: possibly throwing function declared here
// CHECK-NONEXCEPTIONS-NOT: warning:
int k = h();
int l = some_bad_func();
// CHECK-EXCEPTIONS: :[[@LINE-1]]:5: warning: initialization of 'l' with static storage duration may throw an exception that cannot be caught
// CHECK-EXCEPTIONS: 49:3: note: possibly throwing function declared here
// CHECK-NONEXCEPTIONS-NOT: warning:
int m = some_good_func();

typedef decltype(sizeof(int)) size_t;
inline void *operator new(size_t sz, void *here) noexcept { return here; }
char n[sizeof(int)];
int *o = new (n) int();
int *p = new int();
// CHECK-EXCEPTIONS: :[[@LINE-1]]:6: warning: initialization of 'p' with static storage duration may throw an exception that cannot be caught
// CHECK-NONEXCEPTIONS-NOT: warning:

thread_local S s3;
// CHECK-EXCEPTIONS: :[[@LINE-1]]:16: warning: initialization of 's3' with thread_local storage duration may throw an exception that cannot be caught
// CHECK-NONEXCEPTIONS-NOT: warning:
thread_local T t3; // ok
thread_local U u3;
// CHECK-EXCEPTIONS: :[[@LINE-1]]:16: warning: initialization of 'u3' with thread_local storage duration may throw an exception that cannot be caught
// CHECK-NONEXCEPTIONS-NOT: warning:
thread_local V v3("v");
// CHECK-EXCEPTIONS: :[[@LINE-1]]:16: warning: initialization of 'v3' with thread_local storage duration may throw an exception that cannot be caught
// CHECK-NONEXCEPTIONS-NOT: warning:
thread_local W w3;
// CHECK-EXCEPTIONS: :[[@LINE-1]]:16: warning: initialization of 'w3' with thread_local storage duration may throw an exception that cannot be caught
// CHECK-NONEXCEPTIONS-NOT: warning:

void f(S s1, T t1, U u1, V v1, W w1) { // ok, ok, ok, ok, ok
  S s2; // ok
  T t2; // ok
  U u2; // ok
  V v2("v"); // ok
  W w2; // ok

  thread_local S s3; // ok
  thread_local T t3; // ok
  thread_local U u3; // ok
  thread_local V v3("v"); // ok
  thread_local W w3; // ok

  static S s4; // ok
  static T t4; // ok
  static U u4; // ok
  static V v4("v"); // ok
  static W w4; // ok
}

namespace {
S s;
// CHECK-EXCEPTIONS: :[[@LINE-1]]:3: warning: initialization of 's' with static storage duration may throw an exception that cannot be caught [cert-err58-cpp]
// CHECK-EXCEPTIONS: 9:3: note: possibly throwing constructor declared here
// CHECK-NONEXCEPTIONS-NOT: warning:
T t; // ok
U u;
// CHECK-EXCEPTIONS: :[[@LINE-1]]:3: warning: initialization of 'u' with static storage duration may throw an exception that cannot be caught
// CHECK-EXCEPTIONS: 17:3: note: possibly throwing constructor declared here
// CHECK-NONEXCEPTIONS-NOT: warning:
V v("v");
// CHECK-EXCEPTIONS: :[[@LINE-1]]:3: warning: initialization of 'v' with static storage duration may throw an exception that cannot be caught
// CHECK-EXCEPTIONS: 21:12: note: possibly throwing constructor declared here
// CHECK-NONEXCEPTIONS-NOT: warning:
W w;
// CHECK-EXCEPTIONS: :[[@LINE-1]]:3: warning: initialization of 'w' with static storage duration may throw an exception that cannot be caught
// CHECK-EXCEPTIONS: 29:3: note: possibly throwing constructor declared here
// CHECK-NONEXCEPTIONS-NOT: warning:

thread_local S s3;
// CHECK-EXCEPTIONS: :[[@LINE-1]]:16: warning: initialization of 's3' with thread_local storage duration may throw an exception that cannot be caught
// CHECK-NONEXCEPTIONS-NOT: warning:
thread_local T t3; // ok
thread_local U u3;
// CHECK-EXCEPTIONS: :[[@LINE-1]]:16: warning: initialization of 'u3' with thread_local storage duration may throw an exception that cannot be caught
// CHECK-NONEXCEPTIONS-NOT: warning:
thread_local V v3("v");
// CHECK-EXCEPTIONS: :[[@LINE-1]]:16: warning: initialization of 'v3' with thread_local storage duration may throw an exception that cannot be caught
// CHECK-NONEXCEPTIONS-NOT: warning:
thread_local W w3;
// CHECK-EXCEPTIONS: :[[@LINE-1]]:16: warning: initialization of 'w3' with thread_local storage duration may throw an exception that cannot be caught
// CHECK-NONEXCEPTIONS-NOT: warning:
}; // namespace

class Statics {
  static S s; // warn when initialized
  static T t; // ok
  static U u; // warn when initialized
  static V v; // warn when initialized
  static W w; // warn when initialized

  void f(S s, T t, U u, V v) {
    S s2;      // ok
    T t2;      // ok
    U u2;      // ok
    V v2("v"); // ok
    W w2;      // ok

    thread_local S s3;      // ok
    thread_local T t3;      // ok
    thread_local U u3;      // ok
    thread_local V v3("v"); // ok
    thread_local W w3;      // ok

    static S s4;      // ok
    static T t4;      // ok
    static U u4;      // ok
    static V v4("v"); // ok
    static W w4;      // ok
  }
};

S Statics::s;
// CHECK-EXCEPTIONS: :[[@LINE-1]]:12: warning: initialization of 's' with static storage duration may throw an exception that cannot be caught [cert-err58-cpp]
// CHECK-EXCEPTIONS: 9:3: note: possibly throwing constructor declared here
// CHECK-NONEXCEPTIONS-NOT: warning:
T Statics::t;
U Statics::u;
// CHECK-EXCEPTIONS: :[[@LINE-1]]:12: warning: initialization of 'u' with static storage duration may throw an exception that cannot be caught
// CHECK-EXCEPTIONS: 17:3: note: possibly throwing constructor declared here
// CHECK-NONEXCEPTIONS-NOT: warning:
V Statics::v("v");
// CHECK-EXCEPTIONS: :[[@LINE-1]]:12: warning: initialization of 'v' with static storage duration may throw an exception that cannot be caught
// CHECK-EXCEPTIONS: 21:12: note: possibly throwing constructor declared here
// CHECK-NONEXCEPTIONS-NOT: warning:
W Statics::w;
// CHECK-EXCEPTIONS: :[[@LINE-1]]:12: warning: initialization of 'w' with static storage duration may throw an exception that cannot be caught
// CHECK-EXCEPTIONS: 29:3: note: possibly throwing constructor declared here
// CHECK-NONEXCEPTIONS-NOT: warning:
