// RUN: %clang_cc1 -triple x86_64-linux-gnu %s -emit-llvm -o - | FileCheck %s

class func {
public:
    template <typename T>
    func(T){};
    template <typename T, typename U>
    func(T, U){};
};

void GH88906(){
  class Test{
    public:
    func a{[]{ }, []{ }};
    func b{[]{ }};
    func c{[]{ }};
  } test;
}

// CHECK-LABEL: define internal void @_ZZ7GH88906vEN4TestC2Ev
// CHECK: call void @_ZN4funcC2IN7GH889064Test1aMUlvE_ENS3_UlvE0_EEET_T0_
// CHECK: call void @_ZN4funcC2IN7GH889064Test1bMUlvE_EEET_
// CHECK: call void @_ZN4funcC2IN7GH889064Test1cMUlvE_EEET_

// CHECK-LABEL: define internal void @_ZN4funcC2IN7GH889064Test1aMUlvE_ENS3_UlvE0_EEET_T0_
// CHECK-LABEL: define internal void @_ZN4funcC2IN7GH889064Test1bMUlvE_EEET_
// CHECK-LABEL: define internal void @_ZN4funcC2IN7GH889064Test1cMUlvE_EEET_
