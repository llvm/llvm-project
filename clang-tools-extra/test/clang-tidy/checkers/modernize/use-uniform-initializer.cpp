// RUN: %check_clang_tidy -std=c++11 %s modernize-use-uniform-initializer %t

int cinit_0 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: Use uniform initializer instead of C-style initializer [modernize-use-uniform-initializer]
// CHECK-FIXES: int cinit_0 {0};

int cinit_1=0;
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: Use uniform initializer instead of C-style initializer [modernize-use-uniform-initializer]
// CHECK-FIXES: int cinit_1{0};

int callinit_0(0);
// CHECK-MESSAGES: :[[@LINE-1]]:15: warning: Use uniform initializer instead of call-style initializer [modernize-use-uniform-initializer]
// CHECK-FIXES: int callinit_0{0};

int callinit_1 ( 0 );
// CHECK-MESSAGES: :[[@LINE-1]]:16: warning: Use uniform initializer instead of call-style initializer [modernize-use-uniform-initializer]
// CHECK-FIXES: int callinit_1 {0};

int callinit_2 ( ((3 + 1 + 4) + 1 + 5) );
// CHECK-MESSAGES: :[[@LINE-1]]:16: warning: Use uniform initializer instead of call-style initializer [modernize-use-uniform-initializer]
// CHECK-FIXES: int callinit_2 {((3 + 1 + 4) + 1 + 5)};

int callinit_3((9-2)+(6+5));
// CHECK-MESSAGES: :[[@LINE-1]]:15: warning: Use uniform initializer instead of call-style initializer [modernize-use-uniform-initializer]
// CHECK-FIXES: int callinit_3{(9-2)+(6+5)};

int callinit_4 ((3 * 5 + (8 - 9) ));
// CHECK-MESSAGES: :[[@LINE-1]]:16: warning: Use uniform initializer instead of call-style initializer [modernize-use-uniform-initializer]
// CHECK-FIXES: int callinit_4 {(3 * 5 + (8 - 9) )};

int callinit_5((7 + (-9)));
// CHECK-MESSAGES: :[[@LINE-1]]:15: warning: Use uniform initializer instead of call-style initializer [modernize-use-uniform-initializer]
// CHECK-FIXES: int callinit_5{(7 + (-9))};

int mixed_0 = {0};
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: Use uniform initializer instead of C-style initializer [modernize-use-uniform-initializer]
// CHECK-FIXES: int mixed_0 {0};

int mixed_1={0};
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: Use uniform initializer instead of C-style initializer [modernize-use-uniform-initializer]
// CHECK-FIXES: int mixed_1{0};

int noinit;

int correct_0{0};

int correct_1 {0};

struct CInitList_0 {
    CInitList_0() : x(0) {}
    // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: Use uniform initializer instead of C-style initializer [modernize-use-uniform-initializer]
    // CHECK-FIXES: CInitList_0() : x{0} {}

    int x;
};

struct CInitList_1 {
    CInitList_1():x(0) {}
    // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: Use uniform initializer instead of C-style initializer [modernize-use-uniform-initializer]
    // CHECK-FIXES: CInitList_1():x{0} {}

    int x;
};

struct CInitList_2 {
    CInitList_2() : x ( 0 ) {}
    // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: Use uniform initializer instead of C-style initializer [modernize-use-uniform-initializer]
    // CHECK-FIXES: CInitList_2() : x {0} {}

    int x;
};

struct Correct_0 {
    Correct_0() : x{0} {}

    int x;
};

struct Correct_1 {
    Correct_1():x{0} {}

    int x;
};

struct InClassInitializers {
    int cinit_0 = 0;
    // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: Use uniform initializer instead of C-style initializer [modernize-use-uniform-initializer]
    // CHECK-FIXES: int cinit_0 {0};

    int cinit_1=0;
    // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: Use uniform initializer instead of C-style initializer [modernize-use-uniform-initializer]
    // CHECK-FIXES: int cinit_1{0};

    int mixed_0 = {0};
    // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: Use uniform initializer instead of C-style initializer [modernize-use-uniform-initializer]
    // CHECK-FIXES: int mixed_0 {0};

    int mixed_1={0};
    // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: Use uniform initializer instead of C-style initializer [modernize-use-uniform-initializer]
    // CHECK-FIXES: int mixed_1{0};

    int noinit;

    int correct_0 {0};

    int correct_1{0};
};

// Defaulted function arguments cannot use the uniform initializer
void f(int x = 0) {}

struct A {
    A() = default;
    A(int) {}
    A(int, int) {}
};

A a_0;
A a_1{};
A a_2(); // This is not a variable declaration it is actually a function definition (most vexing parse)
A a_3{21};
A a_4{ 21 };
A a_5(42);
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: Use uniform initializer instead of call-style initializer [modernize-use-uniform-initializer]
// CHECK-FIXES: A a_5{42};
A a_6 ( 42 );
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: Use uniform initializer instead of call-style initializer [modernize-use-uniform-initializer]
// CHECK-FIXES: A a_6 {42};
A a_7{0, 1};
A a_8{ 0, 1 };
A a_9(0, 1);
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: Use uniform initializer instead of call-style initializer [modernize-use-uniform-initializer]
// CHECK-FIXES: A a_9{0, 1};
A a_10( 1, 0 );
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: Use uniform initializer instead of call-style initializer [modernize-use-uniform-initializer]
// CHECK-FIXES: A a_10{1, 0};
A a_11(A(3));
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: Use uniform initializer instead of call-style initializer [modernize-use-uniform-initializer]
// CHECK-FIXES: A a_11{A(3)};
A a_12 ( A(3) );
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: Use uniform initializer instead of call-style initializer [modernize-use-uniform-initializer]
// CHECK-FIXES: A a_12 {A(3)};

struct Delegating {
    Delegating() : Delegating(0) {}
    // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: Use uniform initializer instead of C-style initializer [modernize-use-uniform-initializer]
    // CHECK-FIXES: Delegating() : Delegating{0} {}

    Delegating(int) {}
};

int narrow_0 = 3.14;
// CHECK-MESSAGES: :[[@LINE-1]]:14: warning: Use uniform initializer instead of C-style initializer [modernize-use-uniform-initializer]
// CHECK-FIXES: int narrow_0 {static_cast<int>(3.14)};
int narrow_1=37.73;
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: Use uniform initializer instead of C-style initializer [modernize-use-uniform-initializer]
// CHECK-FIXES: int narrow_1{static_cast<int>(37.73)};
int narrow_2(0.9);
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: Use uniform initializer instead of call-style initializer [modernize-use-uniform-initializer]
// CHECK-FIXES: int narrow_2{static_cast<int>(0.9)};
int narrow_3 ( 5.666 );
// CHECK-MESSAGES: :[[@LINE-1]]:14: warning: Use uniform initializer instead of call-style initializer [modernize-use-uniform-initializer]
// CHECK-FIXES: int narrow_3 {static_cast<int>(5.666)};

template <class T> struct Vec {
  T *begin();
  T *end();
};

void for_each() {
    Vec<int> v;
    for (int& x : v)
    {}
}
