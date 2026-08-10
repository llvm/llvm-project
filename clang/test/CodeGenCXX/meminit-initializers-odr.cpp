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
    return 0;
}

// CHECK-LABEL: @_ZN12check_arrays5innerIiEC2Ev

}

namespace check_field_inits_in_base_constructors {

template <typename>
struct ShouldBeODRUsed {
  ShouldBeODRUsed() {}
};
class k {
// The private here is important,
// otherwise it would be aggregate initialized.
private:
  ShouldBeODRUsed<k> a = {};
};

struct b {
  k c{};
};
void test() { b d; }

// CHECK-LABEL: @_ZN38check_field_inits_in_base_constructors15ShouldBeODRUsedINS_1kEEC2Ev

}

namespace check_referenced_when_defined_in_default_parameter {

template <typename T>
struct Test {
    Test(auto&&) {}
};

struct Options {
    Test<bool(bool x)> identity = [](bool x) -> bool { return x; };
};

struct Wrapper {
  Wrapper(const Options& options = Options());
};

void Func() { Options options; }

// CHECK-LABEL: @_ZN50check_referenced_when_defined_in_default_parameter7OptionsC2Ev
// CHECK-LABEL: @_ZN50check_referenced_when_defined_in_default_parameter4TestIFbbEEC1INS_7Options8identityMUlbE_EEEOT_
// CHECK-LABEL: @_ZN50check_referenced_when_defined_in_default_parameter4TestIFbbEEC2INS_7Options8identityMUlbE_EEEOT_

}

namespace lambda_body {
template <typename a>
int templated_func() {
    return 0;
}
struct test_body {
  int mem = templated_func<int>();
};
struct test_capture {
  int mem = templated_func<double>();
};

struct S {
  int a = [_ = test_capture{}] { (void)test_body{}; return 0;}();
};

void test() {
    S s;
}

// CHECK-LABEL: define{{.*}} @_ZN11lambda_body14templated_funcIdEEiv
// CHECK-LABEL: define{{.*}} @_ZNK11lambda_body1S1aMUlvE_clEv
// CHECK-LABEL: define{{.*}} @_ZN11lambda_body14templated_funcIiEEiv


}
