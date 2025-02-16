// REQUIRES: systemz-registered-target
// RUN: %clangxx --target=s390x-none-zos -S -emit-llvm %s -o - | FileCheck %s

// Check the variables
// CHECK: @var1 = global i32 0, align 4
// CHECK: @var2 = hidden global i32 0, align 4
// CHECK: @var3 = global i32 0, align 4
// CHECK: @var4 = hidden global i32 0, align 4
// CHECK: @var5 = global i32 0, align 4
// CHECK: @obj1 = global %class.class1 zeroinitializer, align 2
// CHECK: @obj2 = hidden global %class.class1 zeroinitializer, align 2

// Check the functions
// CHECK: define void @_Z4foo1v
// CHECK: define hidden void @_Z4foo2v
// CHECK: define void @_ZN6class13fooEv
// CHECK: define hidden void @_ZN6class23fooEv
// CHECK: define hidden void @_ZN6class33fooEv
// CHECK: define void @_ZN6class33barEv

int _Export var1;
int var2;
int _Export var3, var4, _Export var5;

void _Export foo1(){};
void foo2(){};

class _Export class1 {
public:
  void foo();
};

class class2 {
public:
  void foo();
};

void class1::foo(){};

void class2::foo(){};

class1 _Export obj1;
class1 obj2;

class class3 {
public:
  void foo();
  void _Export bar();
};

void class3::foo() {};
void class3::bar() {};
