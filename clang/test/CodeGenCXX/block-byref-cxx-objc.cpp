// RUN: %clang_cc1 %s -emit-llvm -o - -fblocks | FileCheck %s
// rdar://8594790

struct A {
	int x;
	A(const A &);
	A();
	~A();
};

int main()
{
	__block A BYREF_VAR;
        ^{ BYREF_VAR.x = 1234; };
	return 0;
}

// CHECK: define internal void @__Block_byref_object_copy_
// CHECK: call {{.*}} @_ZN1AC1ERKS_
// CHECK: define internal void @__Block_byref_object_dispose_
// CHECK: call {{.*}} @_ZN1AD1Ev
// CHECK: define internal void @__copy_helper_block_
// CHECK: call void @_Block_object_assign
// CHECK: define internal void @__destroy_helper_block_
// CHECK: call void @_Block_object_dispose

// rdar://problem/11135650
namespace test1 {
  struct A { int x; A(); ~A(); };

  void test() {
    return;
    __block A a;
  }
}
