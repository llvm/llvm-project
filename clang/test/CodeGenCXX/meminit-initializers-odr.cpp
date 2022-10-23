// RUN: %clang_cc1 -std=c++20 %s -triple x86_64-linux-gnu -emit-llvm -o - | FileCheck %s

struct ThisShouldBeCalled {
    ThisShouldBeCalled() {}
};

template <typename T>
struct ThisShouldBeCalledTPL {
    ThisShouldBeCalledTPL() {}
};

consteval int f () {
    return 42;
}

struct WithConsteval {
    WithConsteval(int x = f()) {}
};

template <typename T>
struct WithConstevalTPL {
    WithConstevalTPL(T x = f()) {}
};


struct Base {
    ThisShouldBeCalled y = {};
};

struct S : Base {
    ThisShouldBeCalledTPL<int> A =  {};
    WithConsteval B = {};
    WithConstevalTPL<double> C = {};
};
void Do(S = S{}) {}

void test() {
    Do();
}

// CHECK-LABEL: @_ZN18ThisShouldBeCalledC2Ev
// CHECK-LABEL: @_ZN21ThisShouldBeCalledTPLIiEC2Ev
// CHECK-LABEL: @_ZN13WithConstevalC2Ei
// CHECK-LABEL: @_ZN16WithConstevalTPLIdEC2Ed

namespace check_arrays {

template <typename T>
struct inner {
    inner() {}
};

struct S {
   inner<int> a {};
};

class C {
    S s[1]{};
};

int f() {
    C c;
}

// CHECK-LABEL: @_ZN12check_arrays5innerIiEC2Ev

}
