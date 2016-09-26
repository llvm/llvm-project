// RUN: %check_clang_tidy %s cert-err58-cpp %t -- -- -std=c++11 -target x86_64-pc-linux-gnu

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


S s;
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: construction of 's' with static storage duration may throw an exception that cannot be caught [cert-err58-cpp]
// CHECK-MESSAGES: 4:3: note: possibly throwing constructor declared here
T t; // ok
U u;
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: construction of 'u' with static storage duration may throw an exception that cannot be caught
// CHECK-MESSAGES: 12:3: note: possibly throwing constructor declared here
V v("v");
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: construction of 'v' with static storage duration may throw an exception that cannot be caught
// CHECK-MESSAGES: 16:12: note: possibly throwing constructor declared here

thread_local S s3;
// CHECK-MESSAGES: :[[@LINE-1]]:16: warning: construction of 's3' with thread_local storage duration may throw an exception that cannot be caught
thread_local T t3; // ok
thread_local U u3;
// CHECK-MESSAGES: :[[@LINE-1]]:16: warning: construction of 'u3' with thread_local storage duration may throw an exception that cannot be caught
thread_local V v3("v");
// CHECK-MESSAGES: :[[@LINE-1]]:16: warning: construction of 'v3' with thread_local storage duration may throw an exception that cannot be caught

void f(S s1, T t1, U u1, V v1) { // ok, ok, ok, ok
  S s2; // ok
  T t2; // ok
  U u2; // ok
  V v2("v"); // ok

  thread_local S s3; // ok
  thread_local T t3; // ok
  thread_local U u3; // ok
  thread_local V v3("v"); // ok

  static S s4; // ok
  static T t4; // ok
  static U u4; // ok
  static V v4("v"); // ok
}

namespace {
S s;
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: construction of 's' with static storage duration may throw an exception that cannot be caught [cert-err58-cpp]
// CHECK-MESSAGES: 4:3: note: possibly throwing constructor declared here
T t; // ok
U u;
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: construction of 'u' with static storage duration may throw an exception that cannot be caught
// CHECK-MESSAGES: 12:3: note: possibly throwing constructor declared here
V v("v");
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: construction of 'v' with static storage duration may throw an exception that cannot be caught
// CHECK-MESSAGES: 16:12: note: possibly throwing constructor declared here

thread_local S s3;
// CHECK-MESSAGES: :[[@LINE-1]]:16: warning: construction of 's3' with thread_local storage duration may throw an exception that cannot be caught
thread_local T t3; // ok
thread_local U u3;
// CHECK-MESSAGES: :[[@LINE-1]]:16: warning: construction of 'u3' with thread_local storage duration may throw an exception that cannot be caught
thread_local V v3("v");
// CHECK-MESSAGES: :[[@LINE-1]]:16: warning: construction of 'v3' with thread_local storage duration may throw an exception that cannot be caught
};

class Statics {
  static S s;
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: construction of 's' with static storage duration may throw an exception that cannot be caught [cert-err58-cpp]
  // CHECK-MESSAGES: 4:3: note: possibly throwing constructor declared here
  static T t; // ok
  static U u;
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: construction of 'u' with static storage duration may throw an exception that cannot be caught
  // CHECK-MESSAGES: 12:3: note: possibly throwing constructor declared here
  static V v;
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: construction of 'v' with static storage duration may throw an exception that cannot be caught
  // CHECK-MESSAGES: 16:12: note: possibly throwing constructor declared here

  void f(S s, T t, U u, V v) {
    S s2;      // ok
    T t2;      // ok
    U u2;      // ok
    V v2("v"); // ok

    thread_local S s3;      // ok
    thread_local T t3;      // ok
    thread_local U u3;      // ok
    thread_local V v3("v"); // ok

    static S s4;      // ok
    static T t4;      // ok
    static U u4;      // ok
    static V v4("v"); // ok
  }
};

S Statics::s;
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: construction of 's' with static storage duration may throw an exception that cannot be caught [cert-err58-cpp]
// CHECK-MESSAGES: 4:3: note: possibly throwing constructor declared here
T Statics::t;
U Statics::u;
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: construction of 'u' with static storage duration may throw an exception that cannot be caught
// CHECK-MESSAGES: 12:3: note: possibly throwing constructor declared here
V Statics::v("v");
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: construction of 'v' with static storage duration may throw an exception that cannot be caught
// CHECK-MESSAGES: 16:12: note: possibly throwing constructor declared here
