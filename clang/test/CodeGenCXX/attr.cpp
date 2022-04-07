// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-pc-linux-gnu -emit-llvm -o - %s | FileCheck %s

// CHECK: @test2 ={{.*}} alias i32 (), i32 ()* @_Z5test1v

// CHECK: define{{.*}} i32 @_Z3foov() [[NUW:#[0-9]+]] align 1024
int foo() __attribute__((aligned(1024)));
int foo() { }

class C {
  virtual void bar1() __attribute__((aligned(1)));
  virtual void bar2() __attribute__((aligned(2)));
  virtual void bar3() __attribute__((aligned(1024)));
  void bar4() __attribute__((aligned(1024)));
} c;

// CHECK: define{{.*}} void @_ZN1C4bar1Ev(%class.C* {{[^,]*}} %this) unnamed_addr [[NUW]] align 2
void C::bar1() { }

// CHECK: define{{.*}} void @_ZN1C4bar2Ev(%class.C* {{[^,]*}} %this) unnamed_addr [[NUW]] align 2
void C::bar2() { }

// CHECK: define{{.*}} void @_ZN1C4bar3Ev(%class.C* {{[^,]*}} %this) unnamed_addr [[NUW]] align 1024
void C::bar3() { }

// CHECK: define{{.*}} void @_ZN1C4bar4Ev(%class.C* {{[^,]*}} %this) [[NUW]] align 1024
void C::bar4() { }

// PR6635
// CHECK-LABEL: define{{.*}} i32 @_Z5test1v()
int test1() { return 10; }
// CHECK at top of file
extern "C" int test2() __attribute__((alias("_Z5test1v")));

// CHECK: attributes [[NUW]] = { {{.*}}noinline nounwind{{.*}} }
