// RUN: clang-tidy %s -checks="-*,cert-err58-cpp" -- -std=c++17 -target x86_64-pc-linux-gnu \
// RUN:   | FileCheck %s -check-prefix=CHECK-EXCEPTIONS \
// RUN:   -implicit-check-not="{{warning|error}}:"
// RUN: clang-tidy %s -checks="-*,cert-err58-cpp" -- -DNONEXCEPTIONS -fno-exceptions -std=c++17 -target x86_64-pc-linux-gnu \
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

#ifndef NONEXCEPTIONS
namespace pr35457 {
constexpr int foo(int x) { if (x <= 0) throw 12; return x; }

constexpr int bar = foo(1); // OK
// CHECK-EXCEPTIONS-NOT: warning: initialization of 'bar' with static storage
int baz = foo(0); // Not OK; throws at runtime when exceptions are enabled.
// CHECK-EXCEPTIONS: :[[@LINE-1]]:5: warning: initialization of 'baz' with static storage duration may throw an exception that cannot be caught [cert-err58-cpp]
// CHECK-EXCEPTIONS: :[[@LINE-6]]:15: note: possibly throwing function declared here
} // namespace pr35457
#endif // NONEXCEPTIONS

namespace pr39777 {
struct S { S(); };
struct T { T() noexcept; };

auto Okay1 = []{ S s; };
auto Okay2 = []{ (void)new int; };
auto NotOkay1 = []{ S s; return 12; }(); // Because the lambda call is not noexcept
// CHECK-EXCEPTIONS: :[[@LINE-1]]:6: warning: initialization of 'NotOkay1' with static storage duration may throw an exception that cannot be caught [cert-err58-cpp]
// CHECK-EXCEPTIONS: :[[@LINE-7]]:12: note: possibly throwing constructor declared here
auto NotOkay2 = []() noexcept { S s; return 12; }(); // Because S::S() is not noexcept
// CHECK-EXCEPTIONS: :[[@LINE-1]]:6: warning: initialization of 'NotOkay2' with static storage duration may throw an exception that cannot be caught [cert-err58-cpp]
// CHECK-EXCEPTIONS: :[[@LINE-10]]:12: note: possibly throwing constructor declared here
auto Okay3 = []() noexcept { T t; return t; }();

struct U {
  U() noexcept;
  auto getBadLambda() const noexcept {
    return []{ S s; return s; };
  }
};
auto Okay4 = []{ U u; return u.getBadLambda(); }();
auto NotOkay3 = []() noexcept { U u; return u.getBadLambda(); }()(); // Because the lambda returned and called is not noexcept
// CHECK-EXCEPTIONS: :[[@LINE-1]]:6: warning: initialization of 'NotOkay3' with static storage duration may throw an exception that cannot be caught [cert-err58-cpp]
// CHECK-EXCEPTIONS: :[[@LINE-6]]:12: note: possibly throwing function declared here

#ifndef NONEXCEPTIONS
struct Bad {
  Bad() {
    throw 12;
  }
};

static auto NotOkay4 = [bad = Bad{}](){};
// FIXME: the above should be diagnosed because the capture init can trigger
// an exception when constructing the Bad object.
#endif // NONEXCEPTIONS
}
