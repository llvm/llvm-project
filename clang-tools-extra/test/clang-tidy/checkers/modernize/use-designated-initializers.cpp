// RUN: %check_clang_tidy -std=c++17 %s modernize-use-designated-initializers %t
// RUN: %check_clang_tidy -check-suffixes=,SINGLE-ELEMENT -std=c++17 %s modernize-use-designated-initializers %t \
// RUN:     -- -config="{CheckOptions: [{key: modernize-use-designated-initializers.IgnoreSingleElementAggregates, value: false}]}" \
// RUN:     --
// RUN: %check_clang_tidy -check-suffixes=POD -std=c++17 %s modernize-use-designated-initializers %t \
// RUN:     -- -config="{CheckOptions: [{key: modernize-use-designated-initializers.RestrictToPODTypes, value: true}]}" \
// RUN:     --

struct S1 {};

S1 s11{};
S1 s12 = {};
S1 s13();
S1 s14;

struct S2 { int i, j; };

S2 s21{.i=1, .j =2};

S2 s22 = {1, 2};
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use designated initializer list [modernize-use-designated-initializers]
// CHECK-MESSAGES-POD: :[[@LINE-2]]:10: warning: use designated initializer list [modernize-use-designated-initializers]
// CHECK-FIXES: S2 s22 = {.i=1, .j=2};

S2 s23{1};
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use designated initializer list [modernize-use-designated-initializers]
// CHECK-MESSAGES-POD: :[[@LINE-2]]:7: warning: use designated initializer list [modernize-use-designated-initializers]
// CHECK-FIXES: S2 s23{.i=1};

S2 s24{.i = 1};

S2 s25 = {.i=1, 2};
// CHECK-MESSAGES: :[[@LINE-1]]:17: warning: use designated init expression [modernize-use-designated-initializers]
// CHECK-MESSAGES-POD: :[[@LINE-2]]:17: warning: use designated init expression [modernize-use-designated-initializers]
// CHECK-FIXES: S2 s25 = {.i=1, .j=2};

class S3 {
  public:
    S2 s2;
    double d;
};

S3 s31 = {.s2 = 1, 2, 3.1};
// CHECK-MESSAGES: :[[@LINE-1]]:20: warning: use designated init expression [modernize-use-designated-initializers]
// CHECK-MESSAGES: :[[@LINE-2]]:23: warning: use designated init expression [modernize-use-designated-initializers]
// CHECK-MESSAGES-POD: :[[@LINE-3]]:20: warning: use designated init expression [modernize-use-designated-initializers]
// CHECK-MESSAGES-POD: :[[@LINE-4]]:23: warning: use designated init expression [modernize-use-designated-initializers]
// CHECK-FIXES: S3 s31 = {.s2 = 1, .s2.j=2, .d=3.1};

S3 s32 = {{.i = 1, 2}};
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use designated initializer list [modernize-use-designated-initializers]
// CHECK-MESSAGES: :[[@LINE-2]]:20: warning: use designated init expression [modernize-use-designated-initializers]
// CHECK-MESSAGES-POD: :[[@LINE-3]]:10: warning: use designated initializer list [modernize-use-designated-initializers]
// CHECK-MESSAGES-POD: :[[@LINE-4]]:20: warning: use designated init expression [modernize-use-designated-initializers]
// CHECK-FIXES: S3 s32 = {.s2={.i = 1, 2}};

struct S4 {
    double d;
    private: static int i;
};

S4 s41 {2.2};
// CHECK-MESSAGES-SINGLE-ELEMENT: :[[@LINE-1]]:8: warning: use designated initializer list [modernize-use-designated-initializers]
// CHECK-FIXES-SINGLE-ELEMENT: S4 s41 {.d=2.2};

S4 s42 = {{}};
// CHECK-MESSAGES-SINGLE-ELEMENT: :[[@LINE-1]]:10: warning: use designated initializer list [modernize-use-designated-initializers]
// CHECK-FIXES-SINGLE-ELEMENT: S4 s42 = {.d={}};

template<typename S> S template1() { return {10, 11}; }

S2 s26 = template1<S2>();

template<typename S> S template2() { return {}; }

S2 s27 = template2<S2>();

struct S5: S2 { int x, y; };

S5 s51 {1, 2, .x = 3, .y = 4};

struct S6 {
    int i;
    struct { int j; } s;
};

S6 s61 {1, 2};

struct S7 {
    union {
        int k;
        double d;
    } u;
};

S7 s71 {1};

struct S8: S7 { int i; };

S8 s81{1, 2};

struct S9 {
    int i, j;
    S9 &operator=(S9);
};

S9 s91{1, 2};
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use designated initializer list [modernize-use-designated-initializers]
// CHECK-FIXES: S9 s91{.i=1, .j=2};
