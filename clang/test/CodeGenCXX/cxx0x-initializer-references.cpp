// RUN: %clang_cc1 -std=c++11 -S -triple armv7-none-eabi -fmerge-all-constants -emit-llvm -o - %s | FileCheck %s

// This creates and lifetime-extends a 'const char[5]' temporary.
// CHECK: @_ZGR19extended_string_ref_ = internal constant [5 x i8] c"hi\00\00\00",
// CHECK: @extended_string_ref ={{.*}} constant [5 x i8]* @_ZGR19extended_string_ref_,
const char (&extended_string_ref)[5] = {"hi"};

// This binds directly to a string literal object.
// CHECK: @nonextended_string_ref ={{.*}} constant [3 x i8]* @.str
const char (&nonextended_string_ref)[3] = {"hi"};

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

  void reference_to_aggregate(int i) {
    // CHECK: getelementptr {{.*}}, i32 0, i32 0
    // CHECK-NEXT: store i32 1
    // CHECK-NEXT: getelementptr {{.*}}, i32 0, i32 1
    // CHECK-NEXT: %[[I1:.*]] = load i32, i32*
    // CHECK-NEXT: store i32 %[[I1]]
    // CHECK-NEXT: store %{{.*}}* %{{.*}}, %{{.*}}** %{{.*}}, align
    const A &ra1{1, i};

    // CHECK-NEXT: getelementptr inbounds [3 x i32], [3 x i32]* %{{.*}}, i{{32|64}} 0, i{{32|64}} 0
    // CHECK-NEXT: store i32 1
    // CHECK-NEXT: getelementptr inbounds i32, i32* %{{.*}}, i{{32|64}} 1
    // CHECK-NEXT: store i32 2
    // CHECK-NEXT: getelementptr inbounds i32, i32* %{{.*}}, i{{32|64}} 1
    // CHECK-NEXT: %[[I2:.*]] = load i32, i32*
    // CHECK-NEXT: store i32 %[[I2]]
    // CHECK-NEXT: store [3 x i32]* %{{.*}}, [3 x i32]** %{{.*}}, align
    const int (&arrayRef)[] = {1, 2, i};

    // CHECK: store %{{.*}}* @{{.*}}, %{{.*}}** %{{.*}}, align
    const A &constra1{1, 2};

    // CHECK-NEXT: store [3 x i32]* @{{.*}}, [3 x i32]** %{{.*}}, align
    const int (&constarrayRef)[] = {1, 2, 3};

    // CHECK-NEXT: ret
  }

  struct B {
    B();
    ~B();
  };

  void single_init_temp_cleanup()
  {
    // Ensure lifetime extension.

    // CHECK: call noundef %"struct.reference::B"* @_ZN9reference1BC1Ev
    // CHECK-NEXT: store %{{.*}}* %{{.*}}, %{{.*}}** %
    const B &rb{ B() };
    // CHECK: call noundef %"struct.reference::B"* @_ZN9reference1BD1Ev
  }

}

namespace PR23165 {
struct AbstractClass {
  virtual void foo() const = 0;
};

struct ChildClass : public AbstractClass {
  virtual void foo() const {}
};

void helper(const AbstractClass &param) {
  param.foo();
}

void foo() {
// CHECK-LABEL: @_ZN7PR231653fooEv
// CHECK: call {{.*}} @_ZN7PR2316510ChildClassC1Ev
// CHECK: call void @_ZN7PR231656helperERKNS_13AbstractClassE
  helper(ChildClass());
}

struct S { struct T { int a; } t; mutable int b; };
void f() {
// CHECK-LABEL: _ZN7PR231651fEv
// CHECK: alloca
// CHECK: alloca
// CHECK: store
  const S::T &r = S().t;
}
}
