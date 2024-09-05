// RUN: %check_clang_tidy -std=c++20 %s modernize-use-designated-initializers %t \
// RUN:     -- \
// RUN:     -- -fno-delayed-template-parsing
// RUN: %check_clang_tidy -check-suffixes=,SINGLE-ELEMENT -std=c++20 %s modernize-use-designated-initializers %t \
// RUN:     -- -config="{CheckOptions: {modernize-use-designated-initializers.IgnoreSingleElementAggregates: false}}" \
// RUN:     -- -fno-delayed-template-parsing
// RUN: %check_clang_tidy -check-suffixes=POD -std=c++20 %s modernize-use-designated-initializers %t \
// RUN:     -- -config="{CheckOptions: {modernize-use-designated-initializers.RestrictToPODTypes: true}}" \
// RUN:     -- -fno-delayed-template-parsing
// RUN: %check_clang_tidy -check-suffixes=,MACROS -std=c++20 %s modernize-use-designated-initializers %t \
// RUN:     -- -config="{CheckOptions: {modernize-use-designated-initializers.IgnoreMacros: false}}" \
// RUN:     -- -fno-delayed-template-parsing

struct S1 {};

S1 s11{};
S1 s12 = {};
S1 s13();
S1 s14;

struct S2 { int i, j; };

S2 s21{.i=1, .j =2};

S2 s22 = {1, 2};
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use designated initializer list to initialize 'S2' [modernize-use-designated-initializers]
// CHECK-MESSAGES: :[[@LINE-6]]:1: note: aggregate type is defined here
// CHECK-MESSAGES-POD: :[[@LINE-3]]:10: warning: use designated initializer list to initialize 'S2' [modernize-use-designated-initializers]
// CHECK-MESSAGES-POD: :[[@LINE-8]]:1: note: aggregate type is defined here
// CHECK-FIXES: S2 s22 = {.i=1, .j=2};

S2 s23{1};
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use designated initializer list to initialize 'S2' [modernize-use-designated-initializers]
// CHECK-MESSAGES: :[[@LINE-13]]:1: note: aggregate type is defined here
// CHECK-MESSAGES-POD: :[[@LINE-3]]:7: warning: use designated initializer list to initialize 'S2' [modernize-use-designated-initializers]
// CHECK-MESSAGES-POD: :[[@LINE-15]]:1: note: aggregate type is defined here
// CHECK-FIXES: S2 s23{.i=1};

S2 s24{.i = 1};

S2 s25 = {.i=1, 2};
// CHECK-MESSAGES: :[[@LINE-1]]:17: warning: use designated init expression to initialize field 'j' [modernize-use-designated-initializers]
// CHECK-MESSAGES-POD: :[[@LINE-2]]:17: warning: use designated init expression to initialize field 'j' [modernize-use-designated-initializers]
// CHECK-FIXES: S2 s25 = {.i=1, .j=2};

class S3 {
  public:
    S2 s2;
    double d;
};

S3 s31 = {.s2 = 1, 2, 3.1};
// CHECK-MESSAGES: :[[@LINE-1]]:20: warning: use designated init expression to initialize field 's2.j' [modernize-use-designated-initializers]
// CHECK-MESSAGES: :[[@LINE-2]]:23: warning: use designated init expression to initialize field 'd' [modernize-use-designated-initializers]
// CHECK-MESSAGES-POD: :[[@LINE-3]]:20: warning: use designated init expression to initialize field 's2.j' [modernize-use-designated-initializers]
// CHECK-MESSAGES-POD: :[[@LINE-4]]:23: warning: use designated init expression to initialize field 'd' [modernize-use-designated-initializers]
// CHECK-FIXES: S3 s31 = {.s2 = 1, .s2.j=2, .d=3.1};

S3 s32 = {{.i = 1, 2}};
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use designated initializer list to initialize 'S3' [modernize-use-designated-initializers]
// CHECK-MESSAGES: :[[@LINE-15]]:1: note: aggregate type is defined here
// CHECK-MESSAGES: :[[@LINE-3]]:20: warning: use designated init expression to initialize field 'j' [modernize-use-designated-initializers]
// CHECK-MESSAGES-POD: :[[@LINE-4]]:10: warning: use designated initializer list to initialize 'S3' [modernize-use-designated-initializers]
// CHECK-MESSAGES-POD: :[[@LINE-18]]:1: note: aggregate type is defined here
// CHECK-MESSAGES-POD: :[[@LINE-6]]:20: warning: use designated init expression to initialize field 'j' [modernize-use-designated-initializers]
// CHECK-FIXES: S3 s32 = {.s2={.i = 1, .j=2}};

S3 s33 = {{2}, .d=3.1};
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: use designated init expression to initialize field 's2' [modernize-use-designated-initializers]
// CHECK-MESSAGES: :[[@LINE-2]]:11: warning: use designated initializer list to initialize 'S2' [modernize-use-designated-initializers]
// CHECK-MESSAGES: :[[@LINE-50]]:1: note: aggregate type is defined here
// CHECK-MESSAGES-POD: :[[@LINE-4]]:11: warning: use designated init expression to initialize field 's2' [modernize-use-designated-initializers]
// CHECK-MESSAGES-POD: :[[@LINE-5]]:11: warning: use designated initializer list to initialize 'S2' [modernize-use-designated-initializers]
// CHECK-MESSAGES-POD: :[[@LINE-53]]:1: note: aggregate type is defined here
// CHECK-FIXES: S3 s33 = {.s2={.i=2}, .d=3.1};

struct S4 {
    double d;
    private: static int i;
};

S4 s41 {2.2};
// CHECK-MESSAGES-SINGLE-ELEMENT: :[[@LINE-1]]:8: warning: use designated initializer list to initialize 'S4' [modernize-use-designated-initializers]
// CHECK-MESSAGES-SINGLE-ELEMENT: :[[@LINE-7]]:1: note: aggregate type is defined here
// CHECK-FIXES-SINGLE-ELEMENT: S4 s41 {.d=2.2};

S4 s42 = {{}};
// CHECK-MESSAGES-SINGLE-ELEMENT: :[[@LINE-1]]:10: warning: use designated initializer list to initialize 'S4' [modernize-use-designated-initializers]
// CHECK-MESSAGES-SINGLE-ELEMENT: :[[@LINE-12]]:1: note: aggregate type is defined here
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
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: use designated initializer list to initialize 'S6' [modernize-use-designated-initializers]
// CHECK-MESSAGES: :[[@LINE-7]]:1: note: aggregate type is defined here
// CHECK-MESSAGES-POD: :[[@LINE-3]]:8: warning: use designated initializer list to initialize 'S6' [modernize-use-designated-initializers]
// CHECK-MESSAGES-POD: :[[@LINE-9]]:1: note: aggregate type is defined here
// CHECK-FIXES: S6 s61 {.i=1, .s.j=2};

struct S7 {
    union {
        int k;
        double d;
    } u;
};

S7 s71 {1};
// CHECK-MESSAGES-SINGLE-ELEMENT: :[[@LINE-1]]:8: warning: use designated initializer list to initialize 'S7' [modernize-use-designated-initializers]
// CHECK-MESSAGES-SINGLE-ELEMENT: :[[@LINE-9]]:1: note: aggregate type is defined here
// CHECK-FIXES-SINGLE-ELEMENT: S7 s71 {.u.k=1};

struct S8: S7 { int i; };

S8 s81{1, 2};

struct S9 {
    int i, j;
    S9 &operator=(S9);
};

S9 s91{1, 2};
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use designated initializer list to initialize 'S9' [modernize-use-designated-initializers]
// CHECK-MESSAGES: :[[@LINE-7]]:1: note: aggregate type is defined here
// CHECK-FIXES: S9 s91{.i=1, .j=2};

struct S10 { int i = 1, j = 2; };

S10 s101 {1, .j=2};
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: use designated init expression to initialize field 'i' [modernize-use-designated-initializers]
// CHECK-FIXES: S10 s101 {.i=1, .j=2};

struct S11 { int i; S10 s10; };

S11 s111 { .i = 1 };
S11 s112 { 1 };
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use designated initializer list to initialize 'S11' [modernize-use-designated-initializers]
// CHECK-MESSAGES: :[[@LINE-5]]:1: note: aggregate type is defined here
// CHECK-FIXES: S11 s112 { .i=1 };

S11 s113 { .i=1, {}};
// CHECK-MESSAGES: :[[@LINE-1]]:18: warning: use designated init expression to initialize field 's10' [modernize-use-designated-initializers]
// CHECK-FIXES: S11 s113 { .i=1, .s10={}};

S11 s114 { .i=1, .s10={1, .j=2}};
// CHECK-MESSAGES: :[[@LINE-1]]:24: warning: use designated init expression to initialize field 'i' [modernize-use-designated-initializers]
// CHECK-FIXES: S11 s114 { .i=1, .s10={.i=1, .j=2}};

struct S12 {
    int i;
    struct { int j; };
};

S12 s121 {1, 2};
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use designated initializer list to initialize 'S12' [modernize-use-designated-initializers]
// CHECK-MESSAGES: :[[@LINE-7]]:1: note: aggregate type is defined here
// CHECK-MESSAGES-POD: :[[@LINE-3]]:10: warning: use designated initializer list to initialize 'S12' [modernize-use-designated-initializers]
// CHECK-MESSAGES-POD: :[[@LINE-9]]:1: note: aggregate type is defined here
// CHECK-FIXES: S12 s121 {.i=1, .j=2};

struct S13 {
    union {
        int k;
        double d;
    };
    int i;
};

S13 s131 {1, 2};
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use designated initializer list to initialize 'S13' [modernize-use-designated-initializers]
// CHECK-MESSAGES: :[[@LINE-10]]:1: note: aggregate type is defined here
// CHECK-MESSAGES-POD: :[[@LINE-3]]:10: warning: use designated initializer list to initialize 'S13' [modernize-use-designated-initializers]
// CHECK-MESSAGES-POD: :[[@LINE-12]]:1: note: aggregate type is defined here
// CHECK-FIXES: S13 s131 {.k=1, .i=2};

#define A (3+2)
#define B .j=1

S9 s92 {A, B};
// CHECK-MESSAGES-MACROS: :[[@LINE-1]]:9: warning: use designated init expression to initialize field 'i' [modernize-use-designated-initializers]
// CHECK-MESSAGES-MACROS: :[[@LINE-5]]:11: note: expanded from macro 'A'

#define DECLARE_S93 S9 s93 {1, 2}

DECLARE_S93;
// CHECK-MESSAGES-MACROS: :[[@LINE-1]]:1: warning: use designated initializer list to initialize 'S9' [modernize-use-designated-initializers]
// CHECK-MESSAGES-MACROS: :[[@LINE-4]]:28: note: expanded from macro 'DECLARE_S93'
// CHECK-MESSAGES-MACROS: :[[@LINE-71]]:1: note: aggregate type is defined here
