// RUN: %clang_cc1 -std=c++11 -S -triple armv7-none-eabi -emit-llvm -o - %s | FileCheck %s

namespace reference {
  struct A {
    int i1, i2;
  };

  void single_init() {
    // No superfluous instructions allowed here, they could be
    // hiding extra temporaries.

    // CHECK: store i32 1, i32*
    // CHECK-NEXT: store i32* %{{.*}}, i32**
    const int &cri2a = 1;

    // CHECK-NEXT: store i32 1, i32*
    // CHECK-NEXT: store i32* %{{.*}}, i32**
    const int &cri1a = {1};

    // CHECK-NEXT: store i32 1, i32*
    int i = 1;
    // CHECK-NEXT: store i32* %{{.*}}, i32**
    int &ri1a = {i};

    // CHECK-NEXT: bitcast
    // CHECK-NEXT: memcpy
    A a{1, 2};
    // CHECK-NEXT: store %{{.*}}* %{{.*}}, %{{.*}}** %
    A &ra1a = {a};

    using T = A&;
    // CHECK-NEXT: store %{{.*}}* %{{.*}}, %{{.*}}** %
    A &ra1b = T{a};

    // CHECK-NEXT: ret
  }

  void reference_to_aggregate() {
    // CHECK: getelementptr {{.*}}, i32 0, i32 0
    // CHECK-NEXT: store i32 1
    // CHECK-NEXT: getelementptr {{.*}}, i32 0, i32 1
    // CHECK-NEXT: store i32 2
    // CHECK-NEXT: store %{{.*}}* %{{.*}}, %{{.*}}** %{{.*}}, align
    const A &ra1{1, 2};

    // CHECK-NEXT: getelementptr inbounds [3 x i32]* %{{.*}}, i{{32|64}} 0, i{{32|64}} 0
    // CHECK-NEXT: store i32 1
    // CHECK-NEXT: getelementptr inbounds i32* %{{.*}}, i{{32|64}} 1
    // CHECK-NEXT: store i32 2
    // CHECK-NEXT: getelementptr inbounds i32* %{{.*}}, i{{32|64}} 1
    // CHECK-NEXT: store i32 3
    // CHECK-NEXT: store [3 x i32]* %{{.*}}, [3 x i32]** %{{.*}}, align
    const int (&arrayRef)[] = {1, 2, 3};

    // CHECK-NEXT: ret
  }

  struct B {
    B();
    ~B();
  };

  void single_init_temp_cleanup()
  {
    // Ensure lifetime extension.

    // CHECK: call %"struct.reference::B"* @_ZN9reference1BC1Ev
    // CHECK-NEXT: store %{{.*}}* %{{.*}}, %{{.*}}** %
    const B &rb{ B() };
    // CHECK: call %"struct.reference::B"* @_ZN9reference1BD1Ev
  }

}
