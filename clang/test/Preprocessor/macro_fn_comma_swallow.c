// Test the GNU comma swallowing extension.
// RUN: %clang_cc1 %s -E | FileCheck -strict-whitespace %s

// CHECK: 1: foo{A, }
#define X(Y) foo{A, Y}
1: X()


// CHECK: 2: fo2{A,}
#define X2(Y) fo2{A,##Y}
2: X2()

// should eat the comma.
// CHECK: 3: {foo}
#define X3(b, ...) {b, ## __VA_ARGS__}
3: X3(foo)



// PR3880
// CHECK: 4: AA BB
#define X4(...)  AA , ## __VA_ARGS__ BB
4: X4()

// PR7943
// CHECK: 5: 1
#define X5(x,...) x##,##__VA_ARGS__
5: X5(1)
