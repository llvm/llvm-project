// RUN: %check_clang_tidy %s modernize-use-designated-initializers %t

struct S1 {};

S1 s11{};
S1 s12 = {};
S1 s13();
S1 s14;

struct S2 { int i, j; };

S2 s21{.i=1, .j =2};

S2 s22 = {1, 2};
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use designated initializer list [modernize-use-designated-initializers]

S2 s23{1};
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use designated initializer list [modernize-use-designated-initializers]

S2 s24{.i = 1};

S2 s25 = {.i=1, 2};
// CHECK-MESSAGES: :[[@LINE-1]]:17: warning: use designated init expression [modernize-use-designated-initializers]

struct S3 {
    S2 s2;
    double d;
};

S3 s31 = {.s2 = 1, 2, 3.1};
// CHECK-MESSAGES: :[[@LINE-1]]:20: warning: use designated init expression [modernize-use-designated-initializers]
// CHECK-MESSAGES: :[[@LINE-2]]:23: warning: use designated init expression [modernize-use-designated-initializers]
