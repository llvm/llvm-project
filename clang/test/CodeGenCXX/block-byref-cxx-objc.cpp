// RUN: %clang_cc1 %s -std=c++11 -emit-llvm -triple %itanium_abi_triple -o - -fblocks -fexceptions | FileCheck %s

struct A {
	int x;
	A(const A &);
	A();
	~A() noexcept(false);
};

struct B {
	int x;
	B(const B &);
	B();
	~B();
};

int testA() {
	__block A a0, a1;
  ^{ a0.x = 1234; a1.x = 5678; };
	return 0;
}

// CHECK-LABEL: define internal void @__Block_byref_object_copy_
// CHECK: call {{.*}} @_ZN1AC1ERKS_
// CHECK-LABEL: define internal void @__Block_byref_object_dispose_
// CHECK: call {{.*}} @_ZN1AD1Ev

// CHECK-LABEL: define linkonce_odr hidden void @__copy_helper_block_e{{4|8}}_{{20|32}}rc{{24|40}}rc(
// CHECK: call void @_Block_object_assign(
// CHECK: invoke void @_Block_object_assign(
// CHECK: call void @_Block_object_dispose({{.*}}) #[[NOUNWIND_ATTR:[0-9]+]]

// CHECK-LABEL: define linkonce_odr hidden void @__destroy_helper_block_e{{4|8}}_{{20|32}}rd{{24|40}}rd(
// CHECK: invoke void @_Block_object_dispose(
// CHECK: call void @_Block_object_dispose(
// CHECK: invoke void @_Block_object_dispose(

int testB() {
	__block B b0, b1;
  ^{ b0.x = 1234; b1.x = 5678; };
	return 0;
}

// CHECK-LABEL: define internal void @__Block_byref_object_copy_
// CHECK: call {{.*}} @_ZN1BC1ERKS_
// CHECK-LABEL: define internal void @__Block_byref_object_dispose_
// CHECK: call {{.*}} @_ZN1BD1Ev

// CHECK-NOT: define{{.*}}@__copy_helper_block
// CHECK: define linkonce_odr hidden void @__destroy_helper_block_e{{4|8}}_{{20|32}}r{{24|40}}r(

// CHECK: attributes #[[NOUNWIND_ATTR]] = {{{.*}}nounwind{{.*}}}
namespace test1 {
  struct A { int x; A(); ~A(); };

  void test() {
    return;
    __block A a;
  }
}
